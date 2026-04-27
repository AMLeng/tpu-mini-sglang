from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto

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
    # ReqInfo contains the immutable input state for a request
    rid: str
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    stream: bool


@dataclass
class ReqState:
    # ReqState contains a bunch of mutable fields; to make this easier to reason about compared
    # to SGLang, we have the convention that ReqState is only ever mutated by its own methods.
    req_info: ReqInfo

    # Number of new tokens we will extend the kv cache by; varies in prefill, always 1 in decode
    extend_len: int

    # information matched from the RadixCache
    prefix_indices: np.ndarray
    last_node: TreeNode
    # Our tree_matched_len corresponds directly to SGLang's cache_protected_len
    tree_matched_len: int  # In sync with last_node; number of tokens matched in the RadixCache

    # Only None for a brief time before the first prepare prefill
    _req_pool_idx: int | None = field(init=False, default=None)

    # Prefill-only info
    prefill_unfinished: bool  # Prefill will not finish this round (for a chunked req)

    # Output information
    output_ids: list[int] = field(default_factory=list)
    send_token_offset: int = 0
    finished_reason: FinishReason | None = None

    @property
    def has_req_pool_idx(self) -> bool:
        return self._req_pool_idx is not None

    def set_req_pool_idx(self, req_pool_idx: int):
        assert self._req_pool_idx is None
        self._req_pool_idx = req_pool_idx

    @property
    def req_pool_idx(self) -> int:
        assert self._req_pool_idx is not None
        return self._req_pool_idx

    def set_prefill_extend(self, extend_len: int, prefill_truncated: bool):
        self.extend_len = extend_len
        self.prefill_unfinished = prefill_truncated

    def update_cached_prefix(self, new_last_node: TreeNode, new_indices: np.ndarray, prefill: bool):
        self.last_node = new_last_node
        self.tree_matched_len = len(new_indices)

        if prefill:
            self.prefix_indices = new_indices
            # extend_len and prefill_unfinished will be overwritten by the chunking logic later
            # Right now we write both values as though we will fully finish prefill the next pass
            self.set_prefill_extend(len(self.req_info.origin_input_ids) - len(new_indices), False)

    def prepare_decode(self):
        self.extend_len = 1

    def add_output_token(self, new_token: int, eos_token_ids: set[int]):
        self.output_ids.append(new_token)
        self._check_finished(eos_token_ids)

    def _check_finished(self, eos_token_ids: set[int]):
        if (
            self.req_info.sampling_params.max_new_tokens
            and len(self.output_ids) >= self.req_info.sampling_params.max_new_tokens
        ):
            self.finished_reason = "length"
            return
        if not self.req_info.sampling_params.ignore_eos and self.output_ids[-1] in eos_token_ids:
            self.finished_reason = "stop"
            return

    def mark_streamed(self) -> None:
        self.send_token_offset = len(self.output_ids)


@dataclass
class GenerationBatchResult:
    next_token_ids: list[int]
