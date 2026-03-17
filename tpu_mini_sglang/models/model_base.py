from abc import ABC, abstractmethod
from collections.abc import Iterable

import jax
from flax import nnx
from jax.sharding import Mesh
from transformers import PretrainedConfig

from tpu_mini_sglang.model_executor.forward_batch_info import ForwardBatch


class ModelBase(ABC, nnx.Module):
    @abstractmethod
    def __init__(self, config: PretrainedConfig, mesh: Mesh) -> None:
        """Constructs the model from a Huggingface config"""
        pass

    @abstractmethod
    def __call__(self, forward_batch: ForwardBatch) -> jax.Array:
        """Calls the model with the given position info, returning logits"""
        pass

    @abstractmethod
    def load_weights(self, weights: Iterable[tuple[str, jax.Array]]) -> None:
        """Logic to take an iterator of (name,weights) and load them into the model"""
        pass
