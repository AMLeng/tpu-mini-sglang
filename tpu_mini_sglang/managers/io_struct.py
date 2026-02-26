import uuid
from dataclasses import dataclass
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from tpu_mini_sglang.sampling.sampling_params import SamplingParams


class GenerateRequest(BaseModel):
    # Externally facing through the /generate endpoint
    rid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str | None = None
    input_ids: list[int] | None = None
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)

    @model_validator(mode="after")
    def normalize_input(self):
        if self.text is None and self.input_ids is None:
            raise ValueError("GenerateRequest has missing request content")
        if self.text is not None and self.input_ids is not None:
            raise ValueError(
                "In a GenerateRequest only one of text or input_ids should be provided"
            )
        return self


@dataclass
class TokenizedGenerateRequest:
    # Internal tokenized request
    rid: str
    input_ids: list[int]
    sampling_params: SamplingParams


class ResponseMetadataDict(TypedDict):
    id: str
    finish_reason: Literal["stop", "length", "abort"] | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


class ResponseDict(TypedDict):
    text: str
    output_ids: list[int]
    meta_info: ResponseMetadataDict
