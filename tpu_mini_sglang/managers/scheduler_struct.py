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

    # Input ids corresponding to allocated kv cache slots
    # That invariant should be valid everywhere outside of prepare for prefill/decode
    # Can be shorter than origin_input_ids during chunked prefill
    # Can be different from origin_input_ids + output_ids during decode for overlap scheduling,
    # due to placeholder tokens that are added to kv_token_ids but not to output_ids
    kv_token_ids: list[int] = field(init=False, default_factory=list)

    # information matched from the RadixCache
    prefix_indices: np.ndarray
    last_node: TreeNode
    # Our tree_matched_len corresponds directly to SGLang's cache_protected_len
    tree_matched_len: int  # In sync with last_node; number of tokens matched in the RadixCache

    # Only None for a brief time before the first prepare prefill
    _req_pool_idx: int | None = field(init=False, default=None)

    # Prefill-only info
    # The number of times we have chunked this req; we use an int so that even during overlap
    # scheduling, this can be tracked correctly
    is_chunked: int
    _chunked_processing_started: bool = field(init=False, default=False)

    # Output information
    output_ids: list[int] = field(default_factory=list)
    send_token_offset: int = 0
    finished_reason: FinishReason | None = None
    previously_finished: bool = field(init=False, default=False)

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
        if prefill_truncated:
            self.is_chunked += 1

    def mark_chunked_partly_processed(self) -> None:
        # We process all requests in two steps (finalize and process in the scheduler),
        # which happen in different orders in overlap/non-overlap scheduling.
        # We use this method and the chunked_processing_started bool to only decrement
        # is_chunked when we are finished with both processing steps.
        assert self.is_chunked > 0
        if self._chunked_processing_started:
            self.is_chunked -= 1
        self._chunked_processing_started = not self._chunked_processing_started

    def update_cached_prefix(self, new_last_node: TreeNode, new_indices: np.ndarray, prefill: bool):
        self.last_node = new_last_node
        self.tree_matched_len = len(new_indices)

        # We only need the prefix_indices during prefill
        if prefill:
            self.prefix_indices = new_indices

    def prepare_prefill(self):
        self.kv_token_ids = self.req_info.origin_input_ids[
            : self.tree_matched_len + self.extend_len
        ]

    def prepare_decode(self, placeholder_id: int | None = None):
        if placeholder_id is None:
            self.kv_token_ids.append(self.output_ids[-1])
        else:
            self.kv_token_ids.append(placeholder_id)
        self.extend_len = 1

    def add_output_token(self, new_token: int, eos_token_ids: set[int]):
        self.output_ids.append(new_token)
        if self.kv_token_ids[-1] < 0:
            # Placeholder was from results of batch N, and inserted in the prepare for N+1.
            # There will only be one placeholder because we finish processing batch N before
            # we begin preparing batch N+2.
            self.kv_token_ids[-1] = new_token
        self._check_finished(eos_token_ids)

    def _check_finished(self, eos_token_ids: set[int]):
        if self.finished_reason is not None:
            self.previously_finished = True
            return
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
