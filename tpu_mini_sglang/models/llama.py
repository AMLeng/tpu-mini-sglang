from collections.abc import Iterable
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig

from tpu_mini_sglang.layers.attention import Attention
from tpu_mini_sglang.layers.rotary_embedding import Llama3RotaryEmbedding
from tpu_mini_sglang.layers.swiglu import SwiGLU as LlamaMLP
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch
from tpu_mini_sglang.models.model_base import ModelBase
from tpu_mini_sglang.sharding import ShardingAxisName
from tpu_mini_sglang.utils import get_jax_dtype, get_padded_head_dim, reshape_and_pad_weight

# Should never actually be applied, since all models are loaded abstractly with nnx.eval_shape
_dummy_init = nnx.initializers.uniform()


class LlamaAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        original_head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        mesh: Mesh,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,  # Required by the API contract of underlying default nnx modules
    ):
        head_dim = get_padded_head_dim(original_head_dim)
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
            rotary_dim=original_head_dim,
            base_freq=rope_theta,
            max_position_embeddings=max_position_embeddings,
            rope_scaling=rope_scaling,
        )
        self.attention = Attention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mesh=mesh,
        )

    def _forward_prepare(self, positions, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q, k = self.rotary_embedding(positions, q, k)
        return q, k, v

    def __call__(self, kv_cache: jax.Array, x: jax.Array, forward_batch: ForwardBatch):
        q, k, v = self._forward_prepare(forward_batch.positions, x)
        new_kv_cache, output = self.attention(kv_cache, q, k, v, forward_batch)
        return new_kv_cache, self.o_proj(output)


class LlamaDecoderLayer(nnx.Module):
    def __init__(self, config: LlamaConfig, mesh: Mesh, dtype: jnp.dtype, rngs: nnx.Rngs):
        original_head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.attention = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            original_head_dim=original_head_dim,
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

    def __call__(self, kv_cache: jax.Array, x: jax.Array, forward_batch: ForwardBatch):
        # We upcast the input to float32 before the norm for numerical stability, matching SGLang
        normed_x = self.attention_norm(x.astype(jnp.float32)).astype(x.dtype)
        new_kv_cache, attention_output = self.attention(kv_cache, normed_x, forward_batch)
        h = x + attention_output
        normed_h = self.mlp_norm(h.astype(jnp.float32)).astype(h.dtype)
        out = h + self.mlp(normed_h)
        return new_kv_cache, out


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

    def __call__(self, kv_caches: list[jax.Array], forward_batch: ForwardBatch):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, hidden_states = layer(kv_cache, hidden_states, forward_batch)
            kv_caches[i] = kv_cache
        # We upcast the input to float32 before the norm for numerical stability, matching SGLang
        return kv_caches, self.norm(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)


class LlamaForCausalLM(ModelBase):
    def __init__(self, config: LlamaConfig, mesh: Mesh):
        dtype = get_jax_dtype(getattr(config, "dtype", "bfloat16"))
        self.config = config
        self.model = LlamaModel(config, mesh, dtype)
        self.mesh = mesh

        # the language model head does unembedding
        if not config.tie_word_embeddings:
            self.lm_head = nnx.Param(
                _dummy_init(jax.random.key(0), (config.hidden_size, config.vocab_size), dtype),
                sharding=(None, ShardingAxisName.VOCAB),
            )

    def __call__(self, kv_caches: list[jax.Array], forward_batch: ForwardBatch):
        kv_caches, hidden_states = self.model(kv_caches, forward_batch)
        if self.config.tie_word_embeddings:
            return kv_caches, self.model.embed_tokens.attend(hidden_states)
        else:
            return kv_caches, jnp.dot(hidden_states, self.lm_head.value)

    def load_weights(self, weights: Iterable[tuple[str, jax.Array]]):
        # Construct a dict of flattened field names to Params
        flat_state = {
            ".".join(str(x) for x in field_name_tuple): cast(nnx.Variable, param)
            for field_name_tuple, param in nnx.to_flat_state(nnx.state(self))
        }

        # The loop happens in three steps:
        # - Rename the Huggingface weight keys
        # - Transpose and reshape
        # - Error check and assign
        for key, weight in weights:
            # Hard coded param renaming logic
            # Handle weight suffixes
            key = key.replace("embed_tokens.weight", "embed_tokens.embedding")
            key = key.replace("norm.weight", "norm.scale")
            key = key.replace("proj.weight", "proj.kernel")
            key = key.replace("lm_head.weight", "lm_head")
            # Handle layer names
            key = key.replace("post_attention_layernorm", "mlp_norm")
            key = key.replace("input_layernorm", "attention_norm")
            key = key.replace("self_attn", "attention")

            # Handle all transpositions and reshapings
            # Transpose all linear weights
            if key.endswith(".kernel"):
                weight = weight.T
            if key == "lm_head":
                weight = weight.T
            # Reshape all four attention projections
            if any(f"attention.{x}_proj" in key for x in "qkvo"):
                # Axis of head_dim in the target_shape, since we must pad head_dim
                pad_axis = 1 if "attention.o_proj" in key else 2
                weight = reshape_and_pad_weight(
                    pad_axis=pad_axis,
                    target_shape=flat_state[key].value.shape,
                    weight=weight,
                )

            # Error checking
            val = flat_state[key].value
            if val.dtype != weight.dtype:
                raise RuntimeError(
                    f"Type mismatch when assigning weight of type {weight.dtype} "
                    f"to field {key} of type {val.dtype}. Note that quantization "
                    "is not yet supported by this library."
                )
            if val.shape != weight.shape:
                raise RuntimeError(
                    f"Shape mismatch when assigning weight of shape {weight.shape} "
                    f"to field {key} of shape {val.shape}."
                )

            # We loaded all the weights on CPU, so before assignment we apply the stored sharding
            # Otherwise, we could just clobber our sharded variables with the unsharded weights
            flat_state[key].value = jax.device_put(
                weight, nnx.get_named_sharding(flat_state[key], self.mesh).value
            )

            # Keep track of which weights have been assigned by removing them from the dict
            del flat_state[key]

        if flat_state:
            raise RuntimeError(
                f"Model parameters missing weights from the loaded checkpoint: {flat_state.keys()}"
            )
