import jax
from flax import nnx
from jax.sharding import Mesh

from tpu_mini_sglang.layers.attention_backends.naive_attention import NaiveAttention


class Attention(nnx.Module):
    """Thin wrapper to call the attention backend from the ForwardBatch"""

    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int, mesh: Mesh):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.mesh = mesh
        self.attention_backend = NaiveAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
        )

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ):
        return self.attention_backend(
            q,
            k,
            v,
        )
