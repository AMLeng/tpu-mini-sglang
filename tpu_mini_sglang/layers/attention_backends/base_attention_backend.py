from abc import ABC, abstractmethod

import jax
from flax import nnx
from jax.sharding import Mesh

from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch


class BaseAttentionBackend(ABC, nnx.Module):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        original_head_dim: int,
        head_dim: int,
        num_kv_heads: int,
        page_size: int,
        mesh: Mesh,
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        kv_cache: jax.Array,
        q: jax.Array,  # (num_tokens, num_heads, head_dim)
        k: jax.Array,  # (num_kv_tokens, num_kv_heads, head_dim)
        v: jax.Array,  # (num_kv_tokens, num_kv_heads, head_dim)
        forward_batch: ForwardBatch,
    ):
        pass

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Called before the JIT boundary to let each attention backend
        # compute batch-dependent metadata (e.g., static pytree fields that control
        # recompilation) without adding backend-specific fields to ForwardBatch.
        pass
