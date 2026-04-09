from pydantic import BaseModel, Field, field_validator, model_validator

TOP_K_ALL = 1 << 30  # Larger than any vocab size


class SamplingParams(BaseModel):
    max_new_tokens: int = 1 << 30  # Effectively unlimited default, clamped by the scheduler
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    ignore_eos: bool = False
    logit_bias: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_greedy(self):
        # Fall back to greedy for temperature ~0
        # Uses same threshold as tpu-inference and SGLang-JAX
        if self.temperature < 1e-5:
            self.temperature = 1.0
            self.top_k = 1
        return self

    @field_validator("top_k")
    @classmethod
    def set_topk(cls, value):
        return TOP_K_ALL if value == -1 else value
