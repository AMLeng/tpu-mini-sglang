from abc import ABC, abstractmethod

import jax
from flax import nnx

from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch


class BaseAttentionBackend(ABC, nnx.Module):
    @abstractmethod
    def __call__(
        self,
        q: jax.Array,  # (num_tokens, num_heads, head_dim)
        k: jax.Array,  # (num_kv_tokens, num_kv_heads, head_dim)
        v: jax.Array,  # (num_kv_tokens, num_kv_heads, head_dim)
        forward_batch: ForwardBatch,
    ):
        pass

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass
