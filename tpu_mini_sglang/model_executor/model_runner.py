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
        input_ids: list[int] = []
        positions: list[int] = []
        seq_lens: list[int] = []

        # Gather request information together
        for req in batch.reqs:
            ids = req.origin_input_ids + req.output_ids
            seq_len = len(ids)

            input_ids.extend(ids)
            positions.extend(range(seq_len))
            seq_lens.append(seq_len)

        # Add padding to prevent excessive JAX jits
        # Since we must recompile every time the input is a different size
        ID_CHUNK_SIZE = 256
        pad_len = (ID_CHUNK_SIZE - (len(input_ids) % ID_CHUNK_SIZE)) % ID_CHUNK_SIZE
        BATCH_CHUNK_SIZE = 4
        batch_pad_len = (BATCH_CHUNK_SIZE - (len(seq_lens) % BATCH_CHUNK_SIZE)) % BATCH_CHUNK_SIZE

        # Construct arrays on the tpu(s)
        jax_input_ids = jnp.array(input_ids + pad_len * [0])
        jax_positions = jnp.array(positions + pad_len * [0])
        jax_seq_lens = jnp.array(seq_lens + batch_pad_len * [0])

        forward_batch = ForwardBatch(
            input_ids=jax_input_ids,
            positions=jax_positions,
            seq_lens=jax_seq_lens,
            extend_start_loc=jnp.cumsum(jax_seq_lens),
            attn_backend=self.attn_backend,
        )

        self.attn_backend.init_forward_metadata(forward_batch)

        # extend_start_loc is the position right after the last token for each sequence
        # We use this to get the final logit, and then take :len(reqs) to only get the
        # tokens for real (non padding) sequences
        logits = self.model_fn(forward_batch)[forward_batch.extend_start_loc - 1][: len(batch.reqs)]

        next_token_ids = jnp.argmax(logits, axis=-1)  # Greedy sampling for now

        return GenerationBatchResult(next_token_ids=next_token_ids.tolist())
