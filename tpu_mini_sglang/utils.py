import logging
import os
import sys
import traceback
from contextlib import suppress
from typing import TYPE_CHECKING

import jax.numpy as jnp
import psutil
import zmq

if TYPE_CHECKING:
    from tpu_mini_sglang.model_config import ModelConfig
    from tpu_mini_sglang.server_args import ServerArgs


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


def configure_logger(server_args: ServerArgs, prefix: str = ""):
    log_format = f"[%(asctime)s {prefix}%(name)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
