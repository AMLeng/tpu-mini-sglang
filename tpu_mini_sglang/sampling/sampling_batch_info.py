from dataclasses import dataclass, field

import jax
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class SamplingMetadata:
    temperature: jax.Array
    top_p: jax.Array
    top_k: jax.Array
    do_greedy: bool = field(metadata={"static": True})
