from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class

from tpu_mini_sglang.sharding import ShardingAxisName


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
        cache_size: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        mesh: Mesh,
        dtype: np.dtype,
    ):
        # We use cache_size + page_size because we need an additional dummy page
        # to be the write target for KV results of padding tokens
        # For the sharding to work properly, KV caches need to be interleaved
        # I.e. the entry for each token looks like [K0,V0,K1,V1,...]
        # This is where the "2" in the shape comes from
        self.kv_buffer = [
            jax.jit(
                lambda: jnp.zeros((cache_size + page_size, num_kv_heads, 2, head_dim), dtype=dtype),
                out_shardings=NamedSharding(
                    mesh, PartitionSpec(None, ShardingAxisName.ATTN_HEAD, None, None)
                ),
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
