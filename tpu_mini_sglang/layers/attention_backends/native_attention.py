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
        kv_cache: jax.Array,
        q: jax.Array,  # (num_tokens, num_heads, head_dim)
        k: jax.Array,  # (num_tokens, num_kv_heads, head_dim)
        v: jax.Array,  # (num_tokens, num_kv_heads, head_dim)
        forward_batch: ForwardBatch,
    ):
        # update kv cache with newly computed values
        kv_cache = kv_cache.at[forward_batch.out_cache_loc, :, 0].set(k)
        kv_cache = kv_cache.at[forward_batch.out_cache_loc, :, 1].set(v)

        k_cache = kv_cache[:, :, 0]  # (cache_size, num_kv_heads, head_dim)
        v_cache = kv_cache[:, :, 1]  # (cache_size, num_kv_heads, head_dim)

        num_query_tokens = len(forward_batch.input_ids)  # t
        num_kv_tokens = self.padded_total_num_tokens  # s

        def make_seq_ids(sequence_lengths: jax.Array, total_padded_length: int):
            # Assign an id to tokens in each sequence, and a shared sequence id for padding tokens
            # The padding token sequence id does not match any real token sequence id
            # We do this by counting how many sequence boundaries each token has passed
            cum_lens = jnp.cumsum(sequence_lengths)
            indices = jnp.arange(total_padded_length)
            return jnp.sum(indices[:, None] >= cum_lens[None, :], axis=1)

        def make_ragged_arange(sequence_lengths: jax.Array, total_padded_length: int):
            # Compute ragged sequence-by-sequence arange for the sequence lengths
            # seq_start_indices[seq_ids] will go out of bounds on padding tokens,
            # but JAX clamps indices so this will just continue the last sequence's count up
            seq_ids = make_seq_ids(sequence_lengths, total_padded_length)
            seq_start_indices = jnp.cumsum(sequence_lengths) - sequence_lengths
            unadjusted_indices = jnp.arange(total_padded_length)
            return unadjusted_indices - seq_start_indices[seq_ids]

        # This ragged_arange will allow us to read out data from the kv cache
        kv_seq_ids = make_seq_ids(forward_batch.seq_lens, num_kv_tokens)
        kv_ragged_arange = make_ragged_arange(forward_batch.seq_lens, num_kv_tokens)

        # Copy necessary data from k_cache, v_cache
        req_pool_indices = forward_batch.req_pool_indices[kv_seq_ids]
        kv_cache_indices = forward_batch.req_to_token[req_pool_indices, kv_ragged_arange]
        cached_k = k_cache[kv_cache_indices]  # (s, num_kv_heads, head_dim)
        cached_v = v_cache[kv_cache_indices]  # (s, num_kv_heads, head_dim)

        # Begin actual attention computation
        q = q.reshape(-1, self.num_kv_heads, self.num_groups, self.head_dim)

        # Our einsums implicitly allow for transposition
        # We trust jax to handle the work for us rather than transposing explicitly
        attn_scores = jnp.einsum("tkgh,skh->kgts", q, cached_k) * self.scaling

        query_seq_ids = make_seq_ids(forward_batch.extend_lens, num_query_tokens)
        query_ragged_arange = make_ragged_arange(forward_batch.extend_lens, num_query_tokens)
        # shift to take into account that query tokens come after all prefix tokens
        shift = forward_batch.seq_lens[query_seq_ids] - forward_batch.extend_lens[query_seq_ids]
        query_shifted_positions = query_ragged_arange + shift

        # Compute giant causal mask with ragged arrays for each sequence
        causal_mask = query_shifted_positions[:, None] >= kv_ragged_arange[None, :]  # (t, s)
        # separately mask so that only tokens in the same sequence attend to one another
        sequence_mask = query_seq_ids[:, None] == kv_seq_ids[None, :]  # (t, s)
        mask = causal_mask & sequence_mask  # (t, s)

        # The 2d mask broadcasts correctly since we have t,s as the last dimensions
        attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum("kgts,skh->tkgh", attn_probs, cached_v)
        return kv_cache, attn_output.reshape(-1, self.num_heads, self.head_dim)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        sum_seq_lens: int = jnp.sum(forward_batch.seq_lens).item()
        PAD_LEN = 256
        pad_len = (PAD_LEN - (sum_seq_lens % PAD_LEN)) % PAD_LEN
        # static pytree value that causes recompilation upon change (hence why we pad)
        self.padded_total_num_tokens = sum_seq_lens + pad_len
