from dataclasses import dataclass, field

from tpu_mini_sglang.managers.io_struct import FinishReason
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.sampling.sampling_params import SamplingParams


@dataclass
class ReqState:
    rid: str
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    eos_token_ids: set[int]
    vocab_size: int
    output_ids: list[int] = field(default_factory=list)
    finished_reason: FinishReason | None = None


@dataclass
class ScheduleBatch:
    reqs: list[ReqState]
    batch_id: int
    model_config: ModelConfig


@dataclass
class GenerationBatchResult:
    next_token_ids: list[int] | None
    batch_id: int
