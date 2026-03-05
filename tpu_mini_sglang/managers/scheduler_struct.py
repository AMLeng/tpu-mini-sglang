from dataclasses import dataclass, field

from tpu_mini_sglang.managers.io_struct import FinishReason
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.sampling.sampling_params import SamplingParams


@dataclass
class ReqState:
    rid: str

    # inputs
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    eos_token_ids: set[int]
    vocab_size: int

    # outputs
    output_ids: list[int] = field(default_factory=list)
    finished_reason: FinishReason | None = None
    send_token_offset: int = 0

    def check_finished(self):
        if self.finished_reason is not None:
            return
        if (
            self.sampling_params.max_new_tokens
            and len(self.output_ids) >= self.sampling_params.max_new_tokens
        ):
            self.finished_reason = "length"
            return
        if not self.sampling_params.ignore_eos and self.output_ids[-1] in self.eos_token_ids:
            self.finished_reason = "stop"
            return


@dataclass
class ScheduleBatch:
    reqs: list[ReqState]
    model_config: ModelConfig


@dataclass
class GenerationBatchResult:
    next_token_ids: list[int]
