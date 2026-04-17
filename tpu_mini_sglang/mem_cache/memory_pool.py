from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.tree_util import register_pytree_node_class

from tpu_mini_sglang.kernels.ragged_paged_attention.kernel import (
    get_kv_cache_shape as get_kv_shape,
)
from tpu_mini_sglang.kernels.ragged_paged_attention.kernel_hd64 import (
    get_kv_cache_shape as get_kv_shape_64,
)
from tpu_mini_sglang.sharding import RPA_CACHE_SHARDING, ShardingAxisName


class ReqToTokenPool:
    """
    ReqToTokenPool maps a request to the locations of its tokens in the KV cache
    It is a (max_running_requests, max_context_len) tensor with some allocation logic
    It is purely CPU side bookkeeping
    Other core KV cache classes:
    TokenToKVPoolAllocator tracks which pages of the KV cache are available
    TokenToKVPool holds the actual caches
    """

    def __init__(
        self,
        max_running_requests: int,
        max_context_len: int,
    ):
        self.req_to_token = np.zeros((max_running_requests, max_context_len), dtype=np.int32)
        self.clear()

    def clear(self):
        self.free_slots = np.arange(self.req_to_token.shape[0], dtype=np.int32)

    def available_size(self):
        return len(self.free_slots)

    def write(
        self, indices: tuple[int, slice] | tuple[np.ndarray, np.ndarray], values: np.ndarray
    ) -> None:
        self.req_to_token[indices] = values

    def read(self, req_idx: int, length: int) -> np.ndarray:
        return self.req_to_token[req_idx, :length].copy()

    def free(self, free_index: list[int]):
        self.free_slots = np.concatenate([self.free_slots, np.asarray(free_index, dtype=np.int32)])

    def alloc(self, need_size: int) -> np.ndarray | None:
        if need_size > self.available_size():
            return None
        ret = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return ret


@register_pytree_node_class
class MHATokenToKVPool:
    """
    TokenToKVPool holds the actual caches, which are JAX arrays sharded on the TPUs
    On the device, we index directly into the k/v buffers with indices from ReqToTokenPool
    Other core KV cache classes:
    ReqToTokenPool maps a request to the locations of its tokens in the KV cache
    TokenToKVPoolAllocator tracks which pages of the KV cache are available
    """

    def __init__(
        self,
        max_cache_size: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        mesh: Mesh,
        dtype: np.dtype,
    ):
        # We initialize to the shape preferred by ragged paged attention by default
        # However, each backend fully owns its own cache, and is the only reader/writer
        # In particular, other backends are free to reshape before/after; we just use this
        # default since RPA is the primary kernel used
        cache_shape_fn = get_kv_shape_64 if head_dim == 64 else get_kv_shape

        cache_pages = max_cache_size // page_size
        # We use cache_pages + 1 because we need an additional dummy page
        # to be the write target for KV results of padding tokens
        cache_shape = cache_shape_fn(
            total_num_pages=cache_pages + 1,
            page_size=page_size,
            actual_num_kv_heads=num_kv_heads,
            actual_head_dim=head_dim,
            kv_dtype=jnp.dtype(dtype),
        )
        if cache_shape[-2] == 1:
            # For the sharding to work properly, each shard must contain full KV heads
            # This will be fine as long as the following condition holds
            assert num_kv_heads % mesh.shape[ShardingAxisName.ATTN_HEAD] == 0
        self.kv_buffer = [
            jax.jit(
                lambda: jnp.zeros(cache_shape, dtype=dtype),
                out_shardings=NamedSharding(mesh, RPA_CACHE_SHARDING),
            )()
            for _ in range(num_layers)
        ]

    def tree_flatten(self):
        children = (self.kv_buffer,)
        aux_data: dict[str, Any] = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # We don't want to construct a new kv cache so we do this to avoid the __init__ call
        obj = object.__new__(cls)
        obj.kv_buffer = children[0]
        return obj
