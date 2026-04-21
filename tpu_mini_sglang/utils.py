from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from contextlib import suppress
from functools import wraps
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import psutil
import zmq

if TYPE_CHECKING:
    from tpu_mini_sglang.model_config import ModelConfig
    from tpu_mini_sglang.server_args import ServerArgs

PERF_LOGGER_NAME = "tpu_mini_sglang.perf"


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def kill_process_tree(include_parent: bool = True):
    """Kill all child process and optionally the process itself."""
    itself = psutil.Process(os.getpid())

    children = itself.children(recursive=True)
    for child in children:
        with suppress(psutil.NoSuchProcess):
            child.kill()

    if include_parent:
        itself.kill()
        sys.exit(0)


def get_zmq_socket(context: zmq.Context, socket_type: int, endpoint: str, bind: bool) -> zmq.Socket:
    # zmq.SocketType is an int enum type but e.g. zmq.PUSH is an int
    # We annotate socket_type with int so that mypy doesn't complain

    socket = context.socket(socket_type)

    # High Water Mark settings allow for unlimited queue
    # BUFfer settings use OS default socket buffer size
    if socket_type == zmq.PUSH:
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, -1)
    elif socket_type == zmq.PULL:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, -1)
    else:
        raise ValueError(f"Unsupported socket type {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket


def get_jax_dtype(config_dtype) -> jnp.dtype:
    config_dtype = str(config_dtype)
    if "torch" in config_dtype:
        config_dtype = config_dtype.split(".")[-1]
    return jnp.dtype(config_dtype)


def approximate_model_size(model_config: ModelConfig, dtype_size: int):
    # Conservatively estimate the memory footprint of a model
    mlp_params = (
        model_config.num_layers
        * model_config.hidden_size
        * model_config.intermediate_size
        * 3  # Assume MLP layer is a GLU
    )
    attention_params = (
        model_config.num_layers
        * model_config.hidden_size
        * model_config.head_dim
        * (
            2 * model_config.num_heads + 2 * model_config.num_kv_heads  # From q/o, k/v
        )
    )
    embedding_params = 2 * model_config.hidden_size * model_config.vocab_size
    return dtype_size * (mlp_params + attention_params + embedding_params)


def get_paddings(min_padding: int, max_padding: int):
    # Generates all powers of two between min and max, inclusive
    return [
        1 << i
        for i in range(0, max_padding.bit_length() + 1)
        if min_padding <= (1 << i) <= max_padding
    ]


def configure_logger(server_args: ServerArgs, prefix: str = ""):
    log_format = f"[%(asctime)s {prefix}%(name)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger(PERF_LOGGER_NAME).setLevel(server_args.perf_log_level.upper())


def log_runtime(name: str | None = None, jax_sync=False):
    # If perf_log_level is set to debug, logs runtimes of decorated functions
    # Add jax_sync=True for functions returning a jax pytree
    # since otherwise we would just measure jax dispatch, not execution
    perf_logger = logging.getLogger(PERF_LOGGER_NAME)

    def decorator(func):
        logname = name if name is not None else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            do_perf = perf_logger.isEnabledFor(logging.DEBUG)
            if do_perf:
                start_time = time.perf_counter()
            result = func(*args, **kwargs)
            if do_perf:
                if jax_sync:
                    jax.block_until_ready(result)
                end_time = time.perf_counter()
                perf_logger.debug(
                    "%s runtime: %.6f s",
                    logname,
                    end_time - start_time,
                )

            return result

        return wrapper

    return decorator


class _JaxCompileLogFilter(logging.Filter):
    # perf_logger MUST NOT be a child of any jax_loggers or this will infinite loop
    def __init__(self, function_names: list[str]):
        super().__init__()
        self.perf_logger = logging.getLogger(PERF_LOGGER_NAME)
        prefix = "Finished XLA compilation of jit("
        suffix = ")"
        # Warn on primary compile lines
        self.warning_strings = {prefix + name + suffix for name in function_names}
        # Suppress other related messages
        self.suppress_strings = {
            "Finished tracing + transforming",
            "Compiling jit",
            "Finished jaxpr to MLIR module conversion jit",
            "Finished XLA compilation of jit",
        }

    def filter(self, record):
        msg = record.getMessage()
        if any(substr in msg for substr in self.warning_strings):
            self.perf_logger.warning("Unexpected JAX recompilation: %s", msg.splitlines()[0])
            return False  # Suppress output
        # Decide to suppress output or not
        return not any(substr in msg for substr in self.suppress_strings)


def activate_jax_log_compiles(function_names: list[str]):
    gate = _JaxCompileLogFilter(function_names)
    jax_loggers = {"jax._src.dispatch", "jax._src.interpreters.pxla"}
    for logger_name in jax_loggers:
        jax_logger = logging.getLogger(logger_name)
        if not any(isinstance(f, _JaxCompileLogFilter) for f in jax_logger.filters):
            jax_logger.addFilter(gate)
    jax.config.update("jax_log_compiles", True)


def get_padded_head_dim(original_head_dim: int):
    # ragged paged attention only takes head dims of 64 or div by 128
    # So we pad for compatibility. TPUs have 128 lanes,
    # so TPU kernels in general will want a head_dim divisible by 128
    # (or two heads of size 64 packed together) for efficiency
    if original_head_dim == 64:
        return original_head_dim
    else:
        return ((original_head_dim + 127) // 128) * 128


def reshape_and_pad_weight(pad_axis: int, target_shape: tuple[int, ...], weight: jax.Array):
    # Unpack weight into the target shape, padding the dimension pad_axis if needed
    assert 0 <= pad_axis < len(target_shape)
    unpadded_shape = list(target_shape)
    unpadded_shape[pad_axis] = -1
    unpadded_weight = weight.reshape(unpadded_shape)

    pad_spec = [(0, 0)] * unpadded_weight.ndim
    pad = target_shape[pad_axis] - unpadded_weight.shape[pad_axis]
    pad_spec[pad_axis] = (0, pad)

    out = jnp.pad(unpadded_weight, pad_spec)
    assert out.shape == tuple(target_shape)
    return out
