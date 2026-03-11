from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig

from tpu_mini_sglang.layers.attention import Attention
from tpu_mini_sglang.layers.rotary_embedding import Llama3RotaryEmbedding
from tpu_mini_sglang.layers.swiglu import SwiGLU as LlamaMLP
from tpu_mini_sglang.models.model_base import ModelBase
from tpu_mini_sglang.sharding import ShardingAxisName

# Should never actually be applied, since all models are loaded abstractly with nnx.eval_shape
_dummy_init = nnx.initializers.uniform()


class LlamaAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        mesh: Mesh,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,  # Required by the API contract of underlying default nnx modules
    ):
        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (hidden_size, num_heads, head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _dummy_init, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rngs,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, num_kv_heads, head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _dummy_init, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rngs,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, num_kv_heads, head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _dummy_init, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rngs,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (num_heads, head_dim, hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _dummy_init, (ShardingAxisName.ATTN_HEAD, None, None)
            ),
            rngs=rngs,
        )
        self.rotary_embedding = Llama3RotaryEmbedding(
            head_dim=head_dim,
            base_freq=rope_theta,
            max_position_embeddings=max_position_embeddings,
            rope_scaling=rope_scaling,
        )
        self.attention = Attention(
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            mesh=mesh,
        )

    def _forward_prepare(self, positions, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q, k = self.rotary_embedding(positions, q, k)
        return q, k, v

    def __call__(self, x: jax.Array, positions: jax.Array):
        q, k, v = self._forward_prepare(positions, x)
        output = self.attention(q, k, v)
        return self.o_proj(output)


class LlamaDecoderLayer(nnx.Module):
    def __init__(self, config: LlamaConfig, mesh: Mesh, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.attention = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, dtype=dtype, rngs=rngs)

        self.attention_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, rngs=rngs
        )
        self.mlp_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, rngs=rngs
        )

    def __call__(self, x: jax.Array, positions: jax.Array):
        # We upcast the input to float32 before the norm for numerical stability, matching SGLang
        normed_x = self.attention_norm(x.astype(jnp.float32)).astype(x.dtype)
        h = x + self.attention(normed_x, positions)
        normed_h = self.mlp_norm(h.astype(jnp.float32)).astype(h.dtype)
        out = h + self.mlp(normed_h)
        return out


class LlamaModel(nnx.Module):
    def __init__(self, config: LlamaConfig, mesh: Mesh, dtype: jnp.dtype):
        rngs = nnx.Rngs(0)
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(_dummy_init, (ShardingAxisName.VOCAB, None)),
            rngs=rngs,
        )
        self.layers = nnx.data(
            [LlamaDecoderLayer(config, mesh, dtype, rngs) for i in range(config.num_hidden_layers)]
        )
        self.norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, rngs=rngs
        )

    def __call__(self, input_ids: jax.Array, positions: jax.Array):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        # We upcast the input to float32 before the norm for numerical stability, matching SGLang
        return self.norm(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)


class LlamaForCausalLM(ModelBase):
    def __init__(self, config: LlamaConfig, mesh: Mesh):
        dtype = jnp.bfloat16
        self.model = LlamaModel(config, mesh, dtype)
        # the language model head is just the unembedding matrix
        self.lm_head = nnx.Param(
            _dummy_init(jax.random.key(0), (config.hidden_size, config.vocab_size), dtype),
            sharding=(None, ShardingAxisName.VOCAB),
        )

    def __call__(self, input_ids: jax.Array, positions: jax.Array):
        hidden_states = self.model(input_ids, positions)
        return jnp.dot(hidden_states, self.lm_head.value)

    def load_weights(self, weights: Iterable[tuple[str, jax.Array]]):
        pass
