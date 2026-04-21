import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_mini_sglang.kernels.ragged_paged_attention.kernel import ragged_paged_attention
from tpu_mini_sglang.kernels.ragged_paged_attention.kernel_hd64 import ragged_paged_attention_hd64
from tpu_mini_sglang.layers.attention_backends.base_attention_backend import BaseAttentionBackend
from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
)
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch
from tpu_mini_sglang.sharding import RPA_CACHE_SHARDING, ShardingAxisName


class RaggedPagedAttention(BaseAttentionBackend):
    def __init__(
        self,
        num_heads: int,
        original_head_dim: int,
        head_dim: int,
        num_kv_heads: int,
        page_size: int,
        mesh: Mesh,
    ):
        # static pytree values
        self.head_dim = head_dim  # h
        self.scaling = original_head_dim**-0.5
        self.page_size = page_size
        self.mesh = mesh
        self._decode_mask = jax.device_put(
            np.array([1, 1, 1], dtype=np.int32), device=NamedSharding(self.mesh, PartitionSpec())
        )
        self._prefill_mask = jax.device_put(
            np.array([0, 0, 1], dtype=np.int32), device=NamedSharding(self.mesh, PartitionSpec())
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        q: jax.Array,  # (num_tokens, num_heads, head_dim)
        k: jax.Array,  # (num_tokens, num_kv_heads, head_dim)
        v: jax.Array,  # (num_tokens, num_kv_heads, head_dim)
        forward_batch: ForwardBatch,
    ):
        attn_function = (
            ragged_paged_attention_hd64 if self.head_dim == 64 else ragged_paged_attention
        )

        attn_output, updated_cache = jax.shard_map(
            functools.partial(attn_function, sm_scale=self.scaling),
            mesh=self.mesh,
            in_specs=(
                PartitionSpec(None, ShardingAxisName.ATTN_HEAD, None),
                PartitionSpec(None, ShardingAxisName.ATTN_HEAD, None),
                PartitionSpec(None, ShardingAxisName.ATTN_HEAD, None),
                RPA_CACHE_SHARDING,
                PartitionSpec(None),
                PartitionSpec(None),
                PartitionSpec(None),
                PartitionSpec(None),
            ),
            out_specs=(
                PartitionSpec(None, ShardingAxisName.ATTN_HEAD, None),
                RPA_CACHE_SHARDING,
            ),
            check_vma=False,
        )(
            q,
            k,
            v,
            kv_cache,
            forward_batch.seq_lens,
            self.page_indices,
            self.cu_q_lens,
            self.distribution,
        )

        return (
            updated_cache,
            attn_output,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # RPA takes a flattened list of page indices rather than slot indices
        self.page_indices = forward_batch.req_to_token[..., :: self.page_size] // self.page_size
        self.page_indices = self.page_indices.reshape(-1)
        # We pad rather than concatenate to preserve the sharding
        self.cu_q_lens = jnp.pad(jnp.cumsum(forward_batch.extend_lens), ((1, 0),))

        num_seqs = jnp.sum(forward_batch.seq_lens > 0, dtype=jnp.int32)
        # The prefill mask will handle any sort of batch;
        # what is important is that all-decode batches get the optimized decode path
        mask = (
            self._decode_mask
            if forward_batch.forward_mode == ForwardMode.DECODE
            else self._prefill_mask
        )
        self.distribution = num_seqs * mask
