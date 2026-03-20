from dataclasses import dataclass, field

import numpy as np

from tpu_mini_sglang.managers.io_struct import FinishReason
from tpu_mini_sglang.mem_cache.allocator import TokenToKVPoolAllocator
from tpu_mini_sglang.mem_cache.memory_pool import ReqToTokenPool
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.sampling.sampling_params import SamplingParams


@dataclass(kw_only=True)
class ReqState:
    rid: str

    # inputs
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    eos_token_ids: set[int]
    vocab_size: int

    # kv cache state
    req_pool_idx: int | None = None
    prefix_len: int = 0
    extend_len: int

    # outputs
    output_ids: list[int] = field(default_factory=list)
    finished_reason: FinishReason | None = None
    send_token_offset: int = 0

    @classmethod
    def init_new(
        cls,
        rid: str,
        origin_input_ids: list[int],
        sampling_params: SamplingParams,
        eos_token_ids: set[int],
        vocab_size: int,
    ):
        # Initialize as though no prefix was matched and we just extended 0
        return cls(
            rid=rid,
            origin_input_ids=origin_input_ids,
            sampling_params=sampling_params,
            eos_token_ids=eos_token_ids,
            vocab_size=vocab_size,
            extend_len=0,
        )

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
    # Unlike original SGLang, we enforce that ScheduleBatch is entirely CPU-side
    # Any structures needed on the TPU are constructed/copied over later,
    # when we construct the ForwardBatch from the ScheduleBatch
    reqs: list[ReqState]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    model_config: ModelConfig
    out_cache_loc: np.ndarray | None

    @classmethod
    def init_new(
        cls,
        reqs: list[ReqState],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        model_config: ModelConfig,
    ):
        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            model_config=model_config,
            out_cache_loc=None,
        )

    # Function for mixed extend/decode
    def prepare(self):
        need_req_slot = [r for r in self.reqs if r.req_pool_idx is None]
        req_pool_indices = self.req_to_token_pool.alloc(len(need_req_slot))
        if req_pool_indices is None:
            raise RuntimeError("Ran out of running request slots.")
        for req, req_pool_idx in zip(need_req_slot, req_pool_indices, strict=True):
            req.req_pool_idx = req_pool_idx

        for req in self.reqs:
            req.prefix_len += req.extend_len
            req.extend_len = len(req.origin_input_ids) + len(req.output_ids) - req.prefix_len
        total_extend_len = sum([r.extend_len for r in self.reqs])

        # Allocate actual cache
        self.out_cache_loc = self.token_to_kv_pool_allocator.alloc(total_extend_len)
        if self.out_cache_loc is None:
            raise RuntimeError("Ran out of kv cache slots.")

        # Update req_to_token_pool information
        pt = 0
        for i, req in enumerate(self.reqs):
            # We use pt to step through out_cache_loc
            # Since out_cache_loc is a flattened list of length total_extend_len
            assert req.req_pool_idx is not None
            self.req_to_token_pool.write(
                (
                    req.req_pool_idx,
                    slice(req.prefix_len, req.prefix_len + req.extend_len),
                ),
                self.out_cache_loc[pt : pt + req.extend_len],
            )
            pt += req.extend_len


@dataclass
class GenerationBatchResult:
    next_token_ids: list[int]
