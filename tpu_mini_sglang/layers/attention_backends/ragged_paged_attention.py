import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from tpu_mini_sglang.kernels.ragged_paged_attention.kernel import ragged_paged_attention
from tpu_mini_sglang.kernels.ragged_paged_attention.kernel_hd64 import ragged_paged_attention_hd64
from tpu_mini_sglang.layers.attention_backends.base_attention_backend import BaseAttentionBackend
from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
)
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch
from tpu_mini_sglang.sharding import RPA_CACHE_SHARDING, ShardingAxisName


class RaggedPagedAttention(BaseAttentionBackend):
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
        self.cu_q_lens = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(forward_batch.extend_lens)]
        )

        num_seqs = jnp.sum(forward_batch.seq_lens > 0, dtype=jnp.int32)
        # The non-decode case can handle any sort of batch
        mask = (
            jnp.array([1, 1, 1], dtype=jnp.int32)
            if forward_batch.forward_mode == ForwardMode.DECODE
            else jnp.array([0, 0, 1], dtype=jnp.int32)
        )
        self.distribution = num_seqs * mask
