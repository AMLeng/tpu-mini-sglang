import jax.numpy as jnp
from jax.sharding import Mesh

from tpu_mini_sglang.layers.attention_backends.native_attention import NativeAttention
from tpu_mini_sglang.managers.scheduler_struct import (
    GenerationBatchResult,
    ScheduleBatch,
)
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch
from tpu_mini_sglang.models.model_loader import get_jitted_model


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        mesh: Mesh,
    ):
        self.mesh = mesh
        self.model_fn = get_jitted_model(config=model_config, mesh=self.mesh)
        self.attn_backend = NativeAttention(
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            num_kv_heads=model_config.num_kv_heads,
        )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        forward_batch = ForwardBatch.init_new(batch, self)

        self.attn_backend.init_forward_metadata(forward_batch)

        # extend_start_loc is the position right after the last token for each sequence
        # We use this to get the final logit, and then take :len(reqs) to only get the
        # tokens for real (non padding) sequences
        logits = self.model_fn(forward_batch)[forward_batch.extend_start_loc - 1][: len(batch.reqs)]

        next_token_ids = jnp.argmax(logits, axis=-1)  # Greedy sampling for now

        return GenerationBatchResult(next_token_ids=next_token_ids.tolist())
