from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from tpu_mini_sglang.managers.scheduler_struct import (
    ScheduleBatch,
)

if TYPE_CHECKING:
    from tpu_mini_sglang.layers.attention_backends.base_attention_backend import (
        BaseAttentionBackend,
    )
    from tpu_mini_sglang.model_executor.model_runner import ModelRunner


# We need register_dataclass since ForwardBatch is passed to the jit-ed model function
# Registering it allows it to be passed through the jit boundary as a pytree
@register_dataclass
@dataclass
class ForwardBatch:
    # With register_dataclass, non-data fields must be marked explicitly
    # Otherwise, jax will trace everything as children
    input_ids: jax.Array
    positions: jax.Array
    seq_lens: jax.Array

    attn_backend: BaseAttentionBackend

    extend_start_loc: jax.Array

    @classmethod
    def init_new(cls, batch: ScheduleBatch, model_runner: ModelRunner):
        input_ids: list[int] = []
        positions: list[int] = []
        seq_lens: list[int] = []

        # Gather request information together
        for req in batch.reqs:
            ids = req.origin_input_ids + req.output_ids
            seq_len = len(ids)

            input_ids.extend(ids)
            positions.extend(range(seq_len))
            seq_lens.append(seq_len)

        # Add padding to prevent excessive JAX jits
        # Since we must recompile every time the input is a different size
        ID_CHUNK_SIZE = 256
        pad_len = (ID_CHUNK_SIZE - (len(input_ids) % ID_CHUNK_SIZE)) % ID_CHUNK_SIZE
        BATCH_CHUNK_SIZE = 4
        batch_pad_len = (BATCH_CHUNK_SIZE - (len(seq_lens) % BATCH_CHUNK_SIZE)) % BATCH_CHUNK_SIZE

        # Construct arrays on the tpu(s)
        jax_input_ids = jnp.array(input_ids + pad_len * [0])
        jax_positions = jnp.array(positions + pad_len * [0])
        jax_seq_lens = jnp.array(seq_lens + batch_pad_len * [0])

        return cls(
            input_ids=jax_input_ids,
            positions=jax_positions,
            seq_lens=jax_seq_lens,
            extend_start_loc=jnp.cumsum(jax_seq_lens),
            attn_backend=model_runner.attn_backend,
        )
