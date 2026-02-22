import uuid

from pydantic import BaseModel, Field

from tpu_mini_sglang.sampling.sampling_params import SamplingParams


class GenerateRequest(BaseModel):
    rid: str | None = None
    text: str | None = None
    input_ids: list[int] | None = None
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)

    def normalize_input(self):
        if self.text is None and self.input_ids is None:
            raise ValueError("GenerateRequest has missing request content")
        if self.text is not None and self.input_ids is not None:
            raise ValueError(
                "In a GenerateRequest only one of text or input_ids should be provided"
            )
        if self.rid is None:
            self.rid = uuid.uuid4().hex
