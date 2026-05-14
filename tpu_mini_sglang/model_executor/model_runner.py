import logging
from queue import Queue
from threading import Thread

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_mini_sglang.layers.attention_backends.base_attention_backend import BaseAttentionBackend
from tpu_mini_sglang.layers.attention_backends.native_attention import NativeAttention
from tpu_mini_sglang.layers.attention_backends.ragged_paged_attention import RaggedPagedAttention
from tpu_mini_sglang.layers.sampler import Sampler, get_jitted_sampler
from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
    GenerationBatchResult,
)
from tpu_mini_sglang.mem_cache.memory_pool import MHATokenToKVPool
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.model_executor.forward_batch_info import (
    ModelWorkerBatch,
    construct_forward_and_sampling_info,
)
from tpu_mini_sglang.models.model_loader import get_jitted_model
from tpu_mini_sglang.server_args import ServerArgs
from tpu_mini_sglang.utils import (
    approximate_model_size,
    get_paddings,
    resolve_future_token_ids,
    set_future_token_ids,
)

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        mesh: Mesh,
        kv_page_size: int,
    ):
        self.model_config = model_config
        self.mesh = mesh
        self.model_fn = get_jitted_model(config=model_config, mesh=self.mesh)
        sampler_graphdef, self.sampler_state = nnx.split(Sampler(mesh=self.mesh))
        self.sampler_fn = get_jitted_sampler(sampler_graphdef)
        self.enable_overlap = server_args.enable_overlap

        if self.enable_overlap:
            # Queues should have size at most 1
            # We use Queue for the blocking on get() behavior
            self.input_queue: Queue = Queue()
            self.output_queue: Queue = Queue()
            self.future_token_ids_map = jax.device_put(
                jnp.zeros((server_args.max_num_batched_requests + 1,), dtype=jnp.int32),
                NamedSharding(self.mesh, P(None)),
            )
            self.forward_thread = Thread(target=self._forward_thread_func, daemon=True)
            self.forward_thread.start()

        devices = jax.devices()
        attn_cls = RaggedPagedAttention if devices[0].platform == "tpu" else NativeAttention
        self.attn_backend: BaseAttentionBackend = attn_cls(
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            original_head_dim=model_config.original_head_dim,
            num_kv_heads=model_config.num_kv_heads,
            page_size=kv_page_size,
            mesh=self.mesh,
        )

        _token_paddings = get_paddings(
            server_args.min_num_batched_tokens, server_args.max_num_batched_tokens
        )
        _req_paddings = get_paddings(
            server_args.min_num_batched_requests, server_args.max_num_batched_requests
        )

        # Source of truth for what padding sizes to use for different ForwardModes
        # We use a single source of truth to avoid risk of JIT recompilation
        # Paddings are arrays of (token_padding, req_padding)
        self._prefill_paddings = [(x, _req_paddings[-1]) for x in _token_paddings]
        self._decode_paddings = [(x, x) for x in _req_paddings]

    def forward_batch_generation(
        self, cache: MHATokenToKVPool, batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        if self.enable_overlap:
            self.input_queue.put((cache, batch))
            # We use negative token ids to show that these are placeholders
            # which double as indices into future_token_ids_map
            true_batch_len = sum(batch.seq_lens > 0)
            next_token_ids = -1 - np.arange(true_batch_len, dtype=np.int32)
            return GenerationBatchResult(next_token_ids=next_token_ids.tolist())
        else:
            return GenerationBatchResult(
                next_token_ids=self._run_forward_batch_generation(cache, batch).tolist()
            )

    def _run_forward_batch_generation(
        self, cache: MHATokenToKVPool, batch: ModelWorkerBatch
    ) -> jax.Array:
        forward_batch, sampling_metadata = construct_forward_and_sampling_info(batch, self)

        # JIT boundary; model_fn is jit compiled
        cache.kv_buffer, full_logits = self.model_fn(cache.kv_buffer, forward_batch)

        last_token_loc = jnp.cumsum(forward_batch.extend_lens) - 1
        logits = full_logits[last_token_loc]  # Now has shape (padded_batch_len, vocab)

        # Unlike the model, the sampler is stateful (nnx.Rngs) so we need to update the state
        self.sampler_state, next_token_ids = self.sampler_fn(
            self.sampler_state, logits, sampling_metadata
        )

        # We use take :len(reqs) to only get the ids for real (non padding) sequences
        next_token_ids = next_token_ids[: sum(batch.seq_lens > 0)]

        next_token_ids.copy_to_host_async()

        return next_token_ids

    def resolve_last_result(self) -> GenerationBatchResult:
        next_token_ids = self.output_queue.get()
        return GenerationBatchResult(next_token_ids=next_token_ids.tolist())

    def _forward_thread_func(self):
        while True:
            # Blocks until we have a batch to run
            (cache, batch) = self.input_queue.get()
            batch.input_ids = resolve_future_token_ids(batch.input_ids, self.future_token_ids_map)

            next_token_ids = self._run_forward_batch_generation(cache, batch)

            self.future_token_ids_map = set_future_token_ids(
                next_token_ids, self.future_token_ids_map
            )
            self.output_queue.put(next_token_ids)

    def get_max_kv_tokens(
        self,
        kv_cache_dtype: jnp.dtype,
    ):
        HAIRCUT_FACTOR = 0.8  # Leave 20% of the memory for misc non-kv-cache uses
        dtype_itemsize = kv_cache_dtype.itemsize
        devices = jax.devices()
        platform = devices[0].platform
        if platform == "tpu":
            memory_per_device = [
                mem_stats["bytes_limit"] - mem_stats["bytes_in_use"]
                for device in devices
                for mem_stats in [device.memory_stats()]
            ]
        elif platform == "cpu":
            import psutil

            memory_per_device = [psutil.virtual_memory().available]
            # Without active workarounds, JAX on CPU will often upcast the entire KV cache
            # whenever we work with it, since most CPUs don't support quantization
            # Hence we leave enough memory for both the original quantized + f32 upcasted cache
            dtype_itemsize += jnp.dtype(jnp.float32).itemsize

            # We also give allowance to help the model fully fit in memory
            # We reserve space for both the original params (since on CPU, the weights could have
            # been swapped to virtual memory) and set of float32 upcasted params
            memory_per_device[0] -= approximate_model_size(
                self.model_config,
                jnp.dtype(jnp.float32).itemsize + self.model_config.dtype.itemsize,
            )
        else:
            raise NotImplementedError("This library only supports running on CPU and TPU")

        bytes_per_token = (
            self.model_config.num_layers
            * self.model_config.num_kv_heads
            * self.model_config.head_dim
            * dtype_itemsize
            * 2
        )
        # Since the kv cache will be sharded by head, and thus evenly across the mesh,
        # we take the min memory to be safe
        device_bytes_free = min(memory_per_device) * len(devices)
        cache_size = int(device_bytes_free * HAIRCUT_FACTOR) // bytes_per_token
        logger.info(
            "Found %d bytes free and creating KV cache of size %d, allocating %d bytes per token",
            device_bytes_free,
            cache_size,
            bytes_per_token,
        )
        return cache_size

    def get_pad_lengths(
        self, num_tokens: int, num_reqs: int, forward_mode: ForwardMode
    ) -> tuple[int, int]:
        paddings = self._decode_paddings if forward_mode.is_decode() else self._prefill_paddings

        # Finds smallest padding in paddings that can handle the given input
        padding = next(((x, y) for (x, y) in paddings if x >= num_tokens and y >= num_reqs), None)
        if padding is None:
            raise RuntimeError(
                f"Could not find valid padding for {num_tokens} tokens and {num_reqs} requests.\n"
                f"Currently consider possible paddings: {paddings}."
            )
        total_toks, total_reqs = padding
        return total_toks - num_tokens, total_reqs - num_reqs

    def precompile(self, cache: MHATokenToKVPool, req_to_token: np.ndarray) -> None:
        # Mixed batches will be treated the same as prefill
        self._precompile_prefill(cache=cache, req_to_token=req_to_token)
        self._precompile_decode(cache=cache, req_to_token=req_to_token)

    def _precompile_prefill(self, cache: MHATokenToKVPool, req_to_token: np.ndarray) -> None:
        for num_tokens, num_reqs in self._prefill_paddings:
            synthetic_batch = ModelWorkerBatch.generate_synthetic(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                forward_mode=ForwardMode.PREFILL,
                req_to_token=req_to_token,
            )
            self._run_forward_batch_generation(cache=cache, batch=synthetic_batch)

    def _precompile_decode(self, cache: MHATokenToKVPool, req_to_token: np.ndarray) -> None:
        for num_tokens, num_reqs in self._decode_paddings:
            synthetic_batch = ModelWorkerBatch.generate_synthetic(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                forward_mode=ForwardMode.DECODE,
                req_to_token=req_to_token,
            )
            self._run_forward_batch_generation(cache=cache, batch=synthetic_batch)
