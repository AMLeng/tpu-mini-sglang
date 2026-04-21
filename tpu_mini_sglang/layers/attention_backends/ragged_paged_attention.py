import functools
from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass

from tpu_mini_sglang.kernels.ragged_paged_attention.kernel import ragged_paged_attention
from tpu_mini_sglang.kernels.ragged_paged_attention.kernel_hd64 import ragged_paged_attention_hd64
from tpu_mini_sglang.layers.attention_backends.base_attention_backend import (
    BaseAttentionBackend,
    BaseAttentionMetadata,
)
from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
)
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch, ModelWorkerBatch
from tpu_mini_sglang.sharding import RPA_CACHE_SHARDING, ShardingAxisName


@register_dataclass
@dataclass
class RaggedPagedAttentionMetadata(BaseAttentionMetadata):
    page_indices: jax.typing.ArrayLike
    cu_q_lens: jax.typing.ArrayLike
    distribution: jax.typing.ArrayLike


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

    def __call__(
        self,
        kv_cache: jax.Array,
        q: jax.Array,  # (num_tokens, num_heads, head_dim)
        k: jax.Array,  # (num_tokens, num_kv_heads, head_dim)
        v: jax.Array,  # (num_tokens, num_kv_heads, head_dim)
        forward_batch: ForwardBatch,
    ):
        assert isinstance(forward_batch.attn_metadata, RaggedPagedAttentionMetadata)
        metadata = forward_batch.attn_metadata
        attn_function = (
            ragged_paged_attention_hd64 if self.head_dim == 64 else ragged_paged_attention
        )

        attn_output, updated_cache = jax.shard_map(
            functools.partial(attn_function, sm_scale=self.scaling),
            mesh=self.mesh,
            in_specs=(
                P(None, ShardingAxisName.ATTN_HEAD, None),
                P(None, ShardingAxisName.ATTN_HEAD, None),
                P(None, ShardingAxisName.ATTN_HEAD, None),
                RPA_CACHE_SHARDING,
                P(None),
                P(None),
                P(None),
                P(None),
            ),
            out_specs=(
                P(None, ShardingAxisName.ATTN_HEAD, None),
                RPA_CACHE_SHARDING,
            ),
            check_vma=False,
        )(
            q,
            k,
            v,
            kv_cache,
            forward_batch.seq_lens,
            metadata.page_indices,
            metadata.cu_q_lens,
            metadata.distribution,
        )

        return (
            updated_cache,
            attn_output,
        )

    def get_forward_metadata(self, batch: ModelWorkerBatch) -> RaggedPagedAttentionMetadata:
        # RPA takes a flattened list of page indices rather than slot indices
        page_indices = batch.req_to_token[..., :: self.page_size] // self.page_size
        page_indices = page_indices.reshape(-1)

        cu_q_lens = np.pad(np.cumsum(batch.extend_lens), ((1, 0),))

        num_seqs = np.sum(batch.seq_lens > 0)
        # The prefill mask will handle any sort of batch;
        # what is important is that all-decode batches get the optimized decode path
        mask = (
            np.array([1, 1, 1], dtype=np.int32)
            if batch.forward_mode == ForwardMode.DECODE
            else np.array([0, 0, 1], dtype=np.int32)
        )

        return RaggedPagedAttentionMetadata(
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=num_seqs * mask,
        )
