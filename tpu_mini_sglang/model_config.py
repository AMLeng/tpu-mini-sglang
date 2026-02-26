from transformers import AutoConfig

from tpu_mini_sglang.server_args import ServerArgs


class ModelConfig:
    def __init__(self, model_path: str, context_length: int | None = None):
        self.model_path = model_path
        self.hf_text_config = AutoConfig.from_pretrained(self.model_path)
        self.context_len = context_length or getattr(
            self.hf_text_config,
            "max_position_embeddings",
            2048,  # Use a conservative fallback
        )

    @classmethod
    def from_server_args(cls, server_args: ServerArgs):
        return cls(model_path=server_args.model_path)
