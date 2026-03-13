import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


class Llama3RotaryEmbedding(nnx.Module):
    def __init__(
        self,
        head_dim: int,
        base_freq: float,
        max_position_embeddings: int,
        rope_scaling: dict[str, Any] | None,
    ):
        self.base_freq = base_freq

        # Number of positions we generated cos sin cache for
        self.max_position_embeddings = max_position_embeddings

        # Number of head dimensions the rotary embedding is applied in
        self.rotary_dim = head_dim

        # Scaling config for Llama3.1 and later
        self.rope_scaling = rope_scaling

    def _compute_freq(self) -> jax.Array:
        # RoPE frequencies are sometimes called "inverse" frequencies
        # Mathematically, they function as frequencies
        # In particular they are inversely proportional to wavelength
        freq = 1.0 / (
            self.base_freq
            ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )

        # Llama3 case with vanilla RoPE
        if self.rope_scaling is None:
            return freq

        # Llama3.1 case
        # low frequencies are untouched, high frequencies are divided by scale_factor
        # frequencies in between the two are smoothly interpolated

        scale_factor = self.rope_scaling["factor"]
        high_freq_factor = self.rope_scaling["high_freq_factor"]
        low_freq_factor = self.rope_scaling["low_freq_factor"]
        original_max_position_embeddings = self.rope_scaling["original_max_position_embeddings"]

        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor

        wavelen = 2 * math.pi / freq
        smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )

        high_freq = jnp.where(wavelen < high_freq_wavelen, freq, 0)
        low_freq = jnp.where(wavelen > low_freq_wavelen, freq / scale_factor, 0)
        mid_freq = jnp.where(
            (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
            (1 - smooth) * freq / scale_factor + smooth * freq,
            0,
        )
        new_freq = high_freq + low_freq + mid_freq
        return new_freq

    def _compute_cos_sin_cache(self) -> jax.Array:
        # Use float32 for numerical stability
        freq = self._compute_freq()
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freq = jnp.einsum("i,j->ij", t, freq)
        cos = jnp.cos(freq)
        sin = jnp.sin(freq)
        return jnp.concatenate((cos, sin), axis=-1)

    def __call__(
        self, positions: jax.Array, q: jax.Array, k: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # q, k have shape (n_tokens, kv_heads, head_dim), while positions has shape (n_tokens,)
        # cos_sin_cache has shape (max_position_embeddings, rotary_dim)
        # Right now, we have only implemented the case rotary_dim == head_dim

        # Computed every call because this should get folded in by jax.jit()
        # Cannot be computed in the init since initialization happens through nnx.eval_shape,
        # which would not actually compute the cache values
        cos_sin_cache = self._compute_cos_sin_cache()

        # Llama3 models use NeoX-style, where on the last axis, the first half is treated as real
        # and the second half is treated as imaginary, for a total of head_dim/2 complex dimensions
        cos, sin = jnp.split(cos_sin_cache[positions], 2, axis=-1)
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        q_real, q_imaginary = jnp.split(q, 2, axis=-1)
        k_real, k_imaginary = jnp.split(k, 2, axis=-1)
        q_rot = jnp.concatenate(
            (cos * q_real - sin * q_imaginary, sin * q_real + cos * q_imaginary), axis=-1
        )
        k_rot = jnp.concatenate(
            (cos * k_real - sin * k_imaginary, sin * k_real + cos * k_imaginary), axis=-1
        )

        # We explicitly computed RoPE in float32 for numerical stability,
        # and thus need to explicitly cast to ensure the correct output dtypes
        return q_rot.astype(q.dtype), k_rot.astype(k.dtype)
