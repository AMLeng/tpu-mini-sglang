import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_mini_sglang.layers.attention_backends.native_attention import NativeAttention
from tpu_mini_sglang.layers.sampler import Sampler, get_jitted_sampler
from tpu_mini_sglang.managers.schedule_batch import ScheduleBatch
from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
    GenerationBatchResult,
)
from tpu_mini_sglang.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch
from tpu_mini_sglang.models.model_loader import get_jitted_model
from tpu_mini_sglang.server_args import ServerArgs
from tpu_mini_sglang.utils import approximate_model_size, get_paddings

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        mesh: Mesh,
        kv_page_size: int = 1,
    ):
        self.model_config = model_config
        self.mesh = mesh
        self.model_fn = get_jitted_model(config=model_config, mesh=self.mesh)
        sampler_graphdef, self.sampler_state = nnx.split(Sampler(mesh=self.mesh))
        self.sampler_fn = get_jitted_sampler(sampler_graphdef)
        self.attn_backend = NativeAttention(
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            num_kv_heads=model_config.num_kv_heads,
        )

        min_token_paddings = 64
        min_batch_paddings = 4
        _token_paddings = get_paddings(min_token_paddings, server_args.max_num_batched_tokens)
        _req_paddings = get_paddings(min_batch_paddings, server_args.max_num_batched_requests)

        # Source of truth for what padding sizes to use for different ForwardModes
        # We use a single source of truth to avoid risk of JIT recompilation
        # Paddings are arrays of (token_padding, req_padding)
        self._prefill_paddings = [(x, _req_paddings[-1]) for x in _token_paddings]
        self._decode_paddings = [(x, x) for x in _req_paddings]

    def forward_batch_generation(
        self, cache: MHATokenToKVPool, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        forward_batch = ForwardBatch.init_new(batch, self)

        # We must call this beforehand to make the attention backend aware
        # of the token->slot mappings for each request
        self.attn_backend.init_forward_metadata(forward_batch)

        # JIT boundary; model_fn is jit compiled
        cache.kv_buffer, full_logits = self.model_fn(cache.kv_buffer, forward_batch)

        last_token_loc = jnp.cumsum(forward_batch.extend_lens) - 1
        logits = full_logits[last_token_loc]  # Now has shape (padded_batch_len, vocab)

        # Unlike the model, the sampler is stateful (nnx.Rngs) so we need to update the state
        self.sampler_state, next_token_ids = self.sampler_fn(
            self.sampler_state, logits, forward_batch.sampling_metadata
        )

        # We use take :len(reqs) to only get the ids for real (non padding) sequences
        next_token_ids = next_token_ids[: len(batch.reqs)]

        return GenerationBatchResult(next_token_ids=next_token_ids.tolist())

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

    def precompile(self, cache: MHATokenToKVPool, req_to_token_pool: ReqToTokenPool) -> None:
        # Mixed batches will be treated the same as prefill
        self._precompile_prefill(cache=cache, req_to_token_pool=req_to_token_pool)
        self._precompile_decode(cache=cache, req_to_token_pool=req_to_token_pool)

    def _precompile_prefill(
        self, cache: MHATokenToKVPool, req_to_token_pool: ReqToTokenPool
    ) -> None:
        for num_tokens, num_reqs in self._prefill_paddings:
            synthetic_batch = ScheduleBatch.generate_synthetic(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                forward_mode=ForwardMode.PREFILL,
                req_to_token_pool=req_to_token_pool,
            )
            self.forward_batch_generation(cache=cache, batch=synthetic_batch)

    def _precompile_decode(
        self, cache: MHATokenToKVPool, req_to_token_pool: ReqToTokenPool
    ) -> None:
        for num_tokens, num_reqs in self._decode_paddings:
            synthetic_batch = ScheduleBatch.generate_synthetic(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                forward_mode=ForwardMode.DECODE,
                req_to_token_pool=req_to_token_pool,
            )
            self.forward_batch_generation(cache=cache, batch=synthetic_batch)
