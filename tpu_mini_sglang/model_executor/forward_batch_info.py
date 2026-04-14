from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass

from tpu_mini_sglang.managers.schedule_batch import ScheduleBatch
from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
)
from tpu_mini_sglang.sampling.sampling_batch_info import SamplingMetadata
from tpu_mini_sglang.sampling.sampling_params import TOP_K_ALL

if TYPE_CHECKING:
    from tpu_mini_sglang.layers.attention_backends.base_attention_backend import (
        BaseAttentionBackend,
    )
    from tpu_mini_sglang.model_executor.model_runner import ModelRunner


# We need register_dataclass since ForwardBatch is passed to the jit-ed model function
# Registering it allows it to be passed through the jit boundary as a pytree
# Note that with register_dataclass, non-data fields must be marked explicitly
# Otherwise, jax will trace everything as children
@register_dataclass
@dataclass
class ForwardBatch:
    # A ForwardBatch should contain only TPU structures to pass into the model
    input_ids: jax.Array
    positions: jax.Array
    seq_lens: jax.Array

    # KV Cache info
    req_to_token: jax.Array  # (padded_batch_len, max_context_len) page table
    out_cache_loc: jax.Array

    attn_backend: BaseAttentionBackend

    extend_lens: jax.Array

    sampling_metadata: SamplingMetadata

    forward_mode: ForwardMode = field(metadata={"static": True})

    @classmethod
    def init_new(cls, batch: ScheduleBatch, model_runner: ModelRunner):
        input_ids: list[int] = []
        positions: list[int] = []
        seq_lens: list[int] = []
        extend_lens: list[int] = []
        req_pool_indices: list[int] = []

        temperature: list[float] = []
        top_p: list[float] = []
        top_k: list[int] = []

        # Gather request information together
        for req in batch.reqs:
            # Slice ids and positions to only include the uncached portion
            full_ids = req.req_info.origin_input_ids + req.output_ids
            full_positions = range(len(full_ids))
            req_ids = full_ids[-req.extend_len :]
            req_positions = full_positions[-req.extend_len :]

            input_ids.extend(req_ids)
            positions.extend(req_positions)
            extend_lens.append(req.extend_len)

            seq_lens.append(len(full_ids))
            req_pool_indices.append(req.req_pool_idx)

            temperature.append(req.req_info.sampling_params.temperature)
            top_p.append(req.req_info.sampling_params.top_p)
            top_k.append(req.req_info.sampling_params.top_k)

        # Collapses MIXED to PREFILL since it can be handled the same way
        forward_mode = ForwardMode.DECODE if batch.forward_mode.is_decode() else ForwardMode.PREFILL
        # Add padding to prevent excessive JAX jits
        # Since we must recompile every time the input is a different size
        query_pad_len, batch_pad_len = model_runner.get_pad_lengths(
            len(input_ids), len(seq_lens), forward_mode
        )

        # Construct arrays on the tpu(s)
        jax_input_ids = jnp.array(input_ids + query_pad_len * [0])
        jax_positions = jnp.array(positions + query_pad_len * [0])
        # Padding for out_cache_loc *MUST* be 0 since it the 0 page of the KV cache is specifically
        # reserved for junk padding token writes
        jax_out_cache_loc = jnp.array(list(batch.out_cache_loc) + query_pad_len * [0])
        jax_seq_lens = jnp.array(seq_lens + batch_pad_len * [0])
        jax_extend_lens = jnp.array(extend_lens + batch_pad_len * [0])
        padded_req_pool_indices = np.array(req_pool_indices + batch_pad_len * [0])

        # Pad sampling params with no-op sentinel values
        jax_temperature = jnp.array(temperature + batch_pad_len * [1.0])
        jax_top_p = jnp.array(top_p + batch_pad_len * [1.0])
        jax_top_k = jnp.array(top_k + batch_pad_len * [TOP_K_ALL])

        # We gather the relevant rows of req_to_token on CPU and only transfer the ones we need
        return cls(
            input_ids=jax_input_ids,
            positions=jax_positions,
            seq_lens=jax_seq_lens,
            req_to_token=jnp.array(batch.req_to_token[padded_req_pool_indices]),
            extend_lens=jax_extend_lens,
            out_cache_loc=jax_out_cache_loc,
            attn_backend=model_runner.attn_backend,
            forward_mode=forward_mode,
            sampling_metadata=SamplingMetadata(
                temperature=jax_temperature,
                top_p=jax_top_p,
                top_k=jax_top_k,
                do_greedy=all(tk == 1 for tk in top_k),
            ),
        )
