from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from tpu_mini_sglang.managers.io_struct import FinishReason
from tpu_mini_sglang.mem_cache.allocator import TokenToKVPoolAllocator
from tpu_mini_sglang.mem_cache.memory_pool import ReqToTokenPool
from tpu_mini_sglang.mem_cache.radix_cache import RadixCache, TreeNode
from tpu_mini_sglang.sampling.sampling_params import SamplingParams


@dataclass(frozen=True)
class ReqInfo:
    # Original SGLang uses a single Req class which is mutated in-place many times
    # I found this hard to follow and refactored
    # ReqInfo contains the immutable input state for a request
    rid: str
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    eos_token_ids: set[int]
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
    def process_req(cls, req: PreparedReqState, new_token: int) -> Self:
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
        if not info.sampling_params.ignore_eos and obj.output_ids[-1] in info.eos_token_ids:
            obj.finished_reason = "stop"
            return obj
        return obj


@dataclass
class ScheduleBatch:
    # Unlike original SGLang, we enforce that ScheduleBatch is entirely CPU-side
    # Any structures needed on the TPU are constructed/copied over later,
    # when we construct the ForwardBatch from the ScheduleBatch
    # We now merge the prepare logic into ScheduleBatch, so that a ScheduleBatch is always
    # fully prepared to run and only stale for a brief moment immediately after running
    reqs: list[PreparedReqState]
    out_cache_loc: np.ndarray

    # Should always be the same reference; this is just a convenient way to pass it to ForwardBatch
    req_to_token: np.ndarray

    @classmethod
    def prepare_for_prefill(
        cls,
        reqs: list[PrefillReqState],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        tree_cache: RadixCache,
    ) -> Self:
        # Allocates and writes KV and ReqToTokenPool caches, creates ScheduleBatch

        # Only chunked reqs should have a req_pool_idx assigned
        chunked_reqs = [r for r in reqs if r.req_pool_idx is not None]

        need_req_slot = [r for r in reqs if r.req_pool_idx is None]
        req_pool_indices = req_to_token_pool.alloc(len(need_req_slot))
        if req_pool_indices is None:
            raise RuntimeError("Ran out of running request slots.")

        prepared_reqs = [
            PreparedReqState.init_prefill_req(r, req_pool_idx)
            for r, req_pool_idx in zip(need_req_slot, req_pool_indices, strict=True)
        ] + [PreparedReqState.init_prefill_req_from_chunked(req) for req in chunked_reqs]
        # Created in parallel to prepared_reqs
        prefill_reqs = need_req_slot + chunked_reqs

        total_extend_len = sum(r.extend_len for r in prepared_reqs)

        # Allocate actual cache
        tree_cache.ensure_free_size(total_extend_len)
        out_cache_loc = token_to_kv_pool_allocator.alloc(total_extend_len)
        if out_cache_loc is None:
            raise RuntimeError("Ran out of kv cache slots.")

        # Update req_to_token_pool information
        pt = 0
        for prefill_req, req in zip(prefill_reqs, prepared_reqs, strict=True):
            # We use pt to step through out_cache_loc
            # Since out_cache_loc is a flattened list of length total_extend_len
            prefix_indices = prefill_req.prefix_indices
            prefix_len = len(prefix_indices)
            # Chunked requests already had their prefix indices written previously
            if prefill_req.req_pool_idx is None:
                req_to_token_pool.write(
                    (
                        req.req_pool_idx,
                        slice(0, len(prefix_indices)),
                    ),
                    prefix_indices,
                )
            req_to_token_pool.write(
                (
                    req.req_pool_idx,
                    slice(prefix_len, prefix_len + req.extend_len),
                ),
                out_cache_loc[pt : pt + req.extend_len],
            )
            pt += req.extend_len

        return cls(
            reqs=prepared_reqs,
            out_cache_loc=out_cache_loc,
            req_to_token=req_to_token_pool.req_to_token,
        )

    @classmethod
    def prepare_for_decode(
        cls,
        reqs: list[ProcessedReqState],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        tree_cache: RadixCache,
    ) -> Self:
        prepared_reqs = [PreparedReqState.init_decode_req(r) for r in reqs]
        total_extend_len = sum(r.extend_len for r in prepared_reqs)

        # Allocate actual cache
        tree_cache.ensure_free_size(total_extend_len)
        out_cache_loc = token_to_kv_pool_allocator.alloc(total_extend_len)
        if out_cache_loc is None:
            raise RuntimeError("Ran out of kv cache slots.")

        req_pool_indices = np.asarray([r.req_pool_idx for r in prepared_reqs])
        seq_lens = np.asarray(
            [len(r.req_info.origin_input_ids) + len(r.output_ids) for r in prepared_reqs]
        )

        # Update req_to_token_pool information
        req_to_token_pool.write(
            (req_pool_indices, seq_lens - 1),
            out_cache_loc,
        )

        return cls(
            reqs=prepared_reqs,
            out_cache_loc=out_cache_loc,
            req_to_token=req_to_token_pool.req_to_token,
        )

    def merge_batch(self, other: ScheduleBatch):
        self.reqs.extend(other.reqs)
        # Safe since both batches are prepared/have kv cache slots allocated on construction
        self.out_cache_loc = np.concatenate([self.out_cache_loc, other.out_cache_loc])


@dataclass
class GenerationBatchResult:
    next_token_ids: list[int]
