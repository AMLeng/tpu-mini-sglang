import jax
from flax import nnx
from jax.sharding import Mesh

from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch


class Attention(nnx.Module):
    """Thin wrapper to call the attention backend from the ForwardBatch"""

    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int, mesh: Mesh):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.mesh = mesh

    def __call__(
        self,
        kv_cache: jax.Array,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
    ):
        return forward_batch.attn_backend(
            kv_cache,
            q,
            k,
            v,
            forward_batch,
        )
