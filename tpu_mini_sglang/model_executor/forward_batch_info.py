from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
from jax.tree_util import register_dataclass

if TYPE_CHECKING:
    from tpu_mini_sglang.layers.attention_backends.base_attention_backend import (
        BaseAttentionBackend,
    )


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
