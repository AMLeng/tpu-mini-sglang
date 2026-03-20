import logging

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from tpu_mini_sglang.layers.attention_backends.native_attention import NativeAttention
from tpu_mini_sglang.managers.scheduler_struct import (
    GenerationBatchResult,
    ScheduleBatch,
)
from tpu_mini_sglang.mem_cache.memory_pool import MHATokenToKVPool
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch
from tpu_mini_sglang.models.model_loader import get_jitted_model
from tpu_mini_sglang.utils import approximate_model_size

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        mesh: Mesh,
        kv_page_size: int = 1,
    ):
        self.model_config = model_config
        self.mesh = mesh
        self.model_fn = get_jitted_model(config=model_config, mesh=self.mesh)
        self.attn_backend = NativeAttention(
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            num_kv_heads=model_config.num_kv_heads,
        )

    def forward_batch_generation(
        self, cache: MHATokenToKVPool, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        forward_batch = ForwardBatch.init_new(batch, self)

        # We must call this beforehand to make the attention backend aware
        # of the token->slot mappings for each request
        self.attn_backend.init_forward_metadata(forward_batch)

        # JIT boundary; model_fn is jit compiled
        cache.kv_buffer, full_logits = self.model_fn(cache.kv_buffer, forward_batch)

        # We use take :len(reqs) to only get the logits for real (non padding) sequences
        last_token_loc = jnp.cumsum(forward_batch.extend_lens) - 1
        logits = full_logits[last_token_loc][: len(batch.reqs)]

        next_token_ids = jnp.argmax(logits, axis=-1)  # Greedy sampling for now

        return GenerationBatchResult(next_token_ids=next_token_ids.tolist())

    def get_max_kv_tokens(
        self,
        kv_cache_dtype: jnp.dtype,
    ):
        HAIRCUT_FACTOR = 0.9  # Leave 10% of the memory for misc non-kv-cache uses
        dtype_itemsize = kv_cache_dtype.itemsize
        devices = jax.devices()
        platform = devices[0].platform
        if platform == "tpu":
            memory_per_device = [
                mem_stats["byte_limit"] - mem_stats["bytes_in_use"]
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

            # Finally, use a harsher haircut factor for good measure, to take care of other
            # situations where a CPU might create additional unexpected copies
            HAIRCUT_FACTOR = 0.8
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
