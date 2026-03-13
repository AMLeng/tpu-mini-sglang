import jax
import jax.numpy as jnp
from flax import nnx


class NaiveAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
    ):
        self.num_heads = num_heads  # n
        self.num_kv_heads = num_kv_heads  # k
        self.num_groups = num_heads // num_kv_heads  # g
        self.head_dim = head_dim  # h
        self.scaling = self.head_dim**-0.5

    def __call__(
        self,
        q: jax.Array,  # (num_tokens, num_heads, head_dim)
        k: jax.Array,  # (num_kv_tokens, num_kv_heads, head_dim)
        v: jax.Array,  # (num_kv_tokens, num_kv_heads, head_dim)
    ):
        num_tokens = q.shape[0]  # t
        num_kv_tokens = k.shape[0]  # s

        q = q.reshape(-1, self.num_kv_heads, self.num_groups, self.head_dim)

        # Our einsums implicitly allow for transposition
        # We trust jax to handle the work for us rather than transposing explicitly
        attn_scores = jnp.einsum("tkgh,skh->kgts", q, k) * self.scaling
        mask = jnp.arange(num_tokens)[:, None] >= jnp.arange(num_kv_tokens)[None, :]
        attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum("kgts,skh->tkgh", attn_probs, v)
        return attn_output.reshape(-1, self.num_heads, self.head_dim)
