import jax
import jax.numpy as jnp

from tpu_mini_sglang.layers.attention_backends.base_attention_backend import BaseAttentionBackend
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch


class NativeAttention(BaseAttentionBackend):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
    ):
        # static pytree values
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
        forward_batch: ForwardBatch,
    ):
        num_tokens = len(forward_batch.input_ids)  # t, s

        q = q.reshape(-1, self.num_kv_heads, self.num_groups, self.head_dim)

        # Our einsums implicitly allow for transposition
        # We trust jax to handle the work for us rather than transposing explicitly
        attn_scores = jnp.einsum("tkgh,skh->kgts", q, k) * self.scaling

        # Compute giant cross-sequence causal mask
        indices = jnp.arange(num_tokens)
        causal_mask = indices[:, None] >= indices[None, :]

        # Assign an id to each sequence, and the same sequence id to all padding tokens
        # We do this by counting how many sequence boundaries each token has passed
        cum_lens = jnp.cumsum(forward_batch.seq_lens)
        seq_ids = jnp.sum(indices[:, None] >= cum_lens[None, :], axis=1)
        # mask so that only tokens in the same sequence attend to one another
        sequence_mask = seq_ids[:, None] == seq_ids[None, :]

        mask = causal_mask & sequence_mask

        # The 2d mask broadcasts correctly since we have t,s as the last dimensions
        attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum("kgts,skh->tkgh", attn_probs, v)
        return attn_output.reshape(-1, self.num_heads, self.head_dim)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass
