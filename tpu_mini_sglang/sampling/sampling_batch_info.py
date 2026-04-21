from dataclasses import dataclass

import jax
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class SamplingMetadata:
    temperature: jax.Array
    top_p: jax.Array
    top_k: jax.Array
    # all_greedy is a scalar jax array; it cannot be a static bool, since then
    # changing it would cause recompilation (of both the sampler and the model)
    all_greedy: jax.Array
