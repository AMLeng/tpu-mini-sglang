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

        # set self.hf_eos_token_id
        eos_ids = getattr(self.hf_text_config, "eos_token_id", None)
        if eos_ids is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        else:
            eos_ids = set()
        self.hf_eos_token_id = eos_ids

        self.bos_token_id = getattr(self.hf_text_config, "bos_token_id", 0)

        self.vocab_size = self.hf_text_config.vocab_size

    @classmethod
    def from_server_args(cls, server_args: ServerArgs):
        return cls(model_path=server_args.model_path)
