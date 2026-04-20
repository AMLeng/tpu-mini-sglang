from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_mini_sglang.layers.binary_search import topk_mask, topp_mask
from tpu_mini_sglang.sampling.sampling_batch_info import SamplingMetadata


class Sampler(nnx.Module):
    # We keep Sampler as a layer for consistency with SGLang and SGLang-JAX
    def __init__(self, mesh: Mesh):
        self.rngs = nnx.Rngs(0)
        self.mesh = mesh

    def __call__(self, logits: jax.Array, sampling_metadata: SamplingMetadata) -> jax.Array:
        # logits has shape (padded_batch_size, vocab_size)

        key = self.rngs()

        def _do_greedy(logits: jax.Array):
            return jnp.argmax(logits, axis=-1)

        def _do_sampling(logits: jax.Array):
            # Force unsharding on vocab axis, as required for efficient top_p/top_k masking
            logits = jax.lax.with_sharding_constraint(
                logits, NamedSharding(self.mesh, PartitionSpec(None, None))
            )

            # This sampling order diverges from CUDA SGLang
            # but mimics SGLang-JAX and tpu-inference, which both
            # divide by temperature after successively applying the masks
            # Masks are safe to apply since the sentinel values (TOP_K_ALL, 1.0) will cause no-ops

            logits = topk_mask(logits, sampling_metadata.top_k, replace_val=-1e12)

            logits = topp_mask(logits, sampling_metadata.top_p, replace_val=-1e12)

            logits /= sampling_metadata.temperature[:, None]
            next_token_ids = jax.random.categorical(key, logits, axis=-1)

            return next_token_ids

        # We use jax.lax.cond so that both branches are compiled during warmup
        return jax.lax.cond(sampling_metadata.do_greedy, _do_greedy, _do_sampling, logits)


def get_jitted_sampler(graphdef) -> Callable:
    @jax.jit
    def apply_sampler(state, *args, **kwargs):
        sampler = nnx.merge(graphdef, state)
        with jax.named_scope("sampling"):
            output = sampler(*args, **kwargs)
        _, new_state = nnx.split(sampler)
        return new_state, output

    return apply_sampler
