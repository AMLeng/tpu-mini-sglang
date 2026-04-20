import jax
import jax.numpy as jnp
from flax import nnx
from jax import nn

from tpu_mini_sglang.sharding import ShardingAxisName

# Should never actually be applied, since all models are loaded abstractly with nnx.eval_shape
_dummy_init = nnx.initializers.uniform()


class SwiGLU(nnx.Module):
    def __init__(
        self,
        hidden_size: int,  # input/output size for the layer
        intermediate_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,  # Required by the API contract of default nnx modules
    ):
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(_dummy_init, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(_dummy_init, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rngs,
        )

        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(_dummy_init, (ShardingAxisName.MLP_TENSOR, None)),
            rngs=rngs,
        )

        self.activation = nn.silu

    def __call__(self, x: jax.Array):
        with jax.named_scope("gate_proj"):
            gate = self.gate_proj(x)
        with jax.named_scope("up_proj"):
            up = self.up_proj(x)
        fuse = self.activation(gate) * up
        with jax.named_scope("down_proj"):
            return self.down_proj(fuse)
