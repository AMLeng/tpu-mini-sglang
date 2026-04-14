from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Self

import numpy as np

from tpu_mini_sglang.managers.io_struct import FinishReason
from tpu_mini_sglang.mem_cache.tree_node import TreeNode
from tpu_mini_sglang.sampling.sampling_params import SamplingParams


class ForwardMode(IntEnum):
    PREFILL = auto()  # Matches SGLang "EXTEND"; we use the more common terminology
    DECODE = auto()  # Decode one token
    MIXED = auto()  # Mixed prefill and decode

    def is_prefill(self):
        return self == ForwardMode.PREFILL

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    @classmethod
    def merge(cls, a, b):
        if a == b:
            return a
        if {a, b} <= {cls.PREFILL, cls.DECODE, cls.MIXED}:
            return cls.MIXED
        raise ValueError(f"Cannot merge {a} and {b}")


@dataclass(frozen=True)
class ReqInfo:
    # Original SGLang uses a single Req class which is mutated in-place many times
    # I found this hard to follow and refactored
    # ReqInfo contains the immutable input state for a request
    rid: str
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    stream: bool


@dataclass
class PrefillReqState:
    # PrefillReqState has information for a prefill request after KV cache prefix matching
    req_info: ReqInfo

    # kv cache state
    extend_len: int  # Number of new tokens we will extend the kv cache by
    prefix_indices: np.ndarray
    last_node: TreeNode
    # Our tree_matched_len corresponds directly to SGLang's cache_protected_len
    tree_matched_len: int  # In sync with last_node; number of tokens matched in the RadixCache

    prefill_unfinished: bool  # Prefill will not finish this round (for a chunked req)
    req_pool_idx: int | None = None  # Should be none except for at most one previously chunked req


@dataclass
class PreparedReqState:
    # PreparedReqState is a request in a ScheduleBatch that has been prepared for prefill/decode
    req_info: ReqInfo

    # kv cache state
    req_pool_idx: int
    extend_len: int
    last_node: TreeNode
    tree_matched_len: int

    prefill_unfinished: bool  # Prefill will not finish this round (for a chunked req)

    # Output information
    output_ids: list[int] = field(default_factory=list)
    send_token_offset: int = 0

    @classmethod
    def init_prefill_req(
        cls,
        req: PrefillReqState,
        req_pool_idx: int,
    ) -> Self:
        return cls(
            req_info=req.req_info,
            req_pool_idx=req_pool_idx,
            extend_len=req.extend_len,
            last_node=req.last_node,
            tree_matched_len=req.tree_matched_len,
            prefill_unfinished=req.prefill_unfinished,
        )

    @classmethod
    def init_prefill_req_from_chunked(
        cls,
        req: PrefillReqState,
    ) -> Self:
        assert req.req_pool_idx is not None
        return cls(
            req_info=req.req_info,
            req_pool_idx=req.req_pool_idx,
            extend_len=req.extend_len,
            last_node=req.last_node,
            tree_matched_len=req.tree_matched_len,
            prefill_unfinished=req.prefill_unfinished,
        )

    @classmethod
    def init_decode_req(
        cls,
        req: ProcessedReqState,
    ) -> Self:
        return cls(
            req_info=req.req_info,
            req_pool_idx=req.req_pool_idx,
            extend_len=1,  # Decode always extends by one
            last_node=req.last_node,
            tree_matched_len=req.tree_matched_len,
            prefill_unfinished=False,
            output_ids=req.output_ids,
            send_token_offset=req.send_token_offset,
        )


@dataclass
class ProcessedReqState:
    # ProcessedReqState is the result of a request after running through the model
    req_info: ReqInfo

    # kv cache state
    req_pool_idx: int
    last_node: TreeNode
    tree_matched_len: int

    output_ids: list[int]
    send_token_offset: int
    finished_reason: FinishReason | None

    @classmethod
    def process_req(cls, req: PreparedReqState, new_token: int, eos_token_ids: set[int]) -> Self:
        # We mutate the prepared req here, but this is fine since we should never touch it again
        req.output_ids.append(new_token)
        obj = cls(
            req_info=req.req_info,
            req_pool_idx=req.req_pool_idx,
            output_ids=req.output_ids,
            last_node=req.last_node,
            tree_matched_len=req.tree_matched_len,
            send_token_offset=req.send_token_offset,
            finished_reason=None,
        )
        info = obj.req_info

        # Set finished reason if applicable
        if (
            info.sampling_params.max_new_tokens
            and len(obj.output_ids) >= info.sampling_params.max_new_tokens
        ):
            obj.finished_reason = "length"
            return obj
        if not info.sampling_params.ignore_eos and obj.output_ids[-1] in eos_token_ids:
            obj.finished_reason = "stop"
            return obj
        return obj


@dataclass
class GenerationBatchResult:
    next_token_ids: list[int]
