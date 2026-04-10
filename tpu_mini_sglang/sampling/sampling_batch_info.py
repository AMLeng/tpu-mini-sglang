from dataclasses import dataclass

import jax
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class SamplingMetadata:
    temperature: jax.Array
    top_p: jax.Array
    top_k: jax.Array
    # If do_greedy was marked static, since SamplingMetadata lives on the forward batch,
    # changing it would cause recompilation of both the sampler *and* the model
    do_greedy: bool  # Not marked static, so will get converted to a scalar jax array
