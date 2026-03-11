from transformers import PretrainedConfig

from tpu_mini_sglang.models.llama import LlamaForCausalLM
from tpu_mini_sglang.models.model_base import ModelBase

_MODEL_REGISTRY = {"LlamaForCausalLM": LlamaForCausalLM}


def get_model_architecture(config: PretrainedConfig) -> type[ModelBase]:
    for arch in (architectures := getattr(config, "architectures", [])):
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(f"Model architectures {architectures} not supported.")
