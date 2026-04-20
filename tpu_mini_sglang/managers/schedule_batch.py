from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
    PrefillReqState,
    PreparedReqState,
    ProcessedReqState,
    ReqInfo,
)
from tpu_mini_sglang.mem_cache.allocator import BaseTokenToKVPoolAllocator
from tpu_mini_sglang.mem_cache.memory_pool import ReqToTokenPool
from tpu_mini_sglang.mem_cache.radix_cache import RadixCache
from tpu_mini_sglang.mem_cache.tree_node import TreeNode
from tpu_mini_sglang.sampling.sampling_params import SamplingParams


@dataclass
class ScheduleBatch:
    # Unlike original SGLang, we enforce that ScheduleBatch is entirely CPU-side
    # Any structures needed on the TPU are constructed/copied over later,
    # when we construct the ForwardBatch from the ScheduleBatch
    # We now merge the prepare logic into ScheduleBatch, so that a ScheduleBatch is always
    # fully prepared to run and only stale for a brief moment immediately after running
    reqs: list[PreparedReqState]
    out_cache_loc: np.ndarray

    forward_mode: ForwardMode

    # Should always be the same reference; this is just a convenient way to pass it to ForwardBatch
    req_to_token: np.ndarray

    @classmethod
    def prepare_for_prefill(
        cls,
        reqs: list[PrefillReqState],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: RadixCache,
    ) -> Self:
        # Allocates and writes KV and ReqToTokenPool caches, creates ScheduleBatch

        # Only chunked reqs should have a req_pool_idx assigned
        chunked_reqs = [r for r in reqs if r.req_pool_idx is not None]

        need_req_slot = [r for r in reqs if r.req_pool_idx is None]
        req_pool_indices = req_to_token_pool.alloc(len(need_req_slot))
        if req_pool_indices is None:
            raise RuntimeError("Ran out of running request slots.")

        # Update req_to_token_pool with our prefix indices for non-chunked reqs
        # Chunked requests already had their prefix indices written previously
        # so we don't need to loop over them here
        for prefill_req, req_pool_idx in zip(need_req_slot, req_pool_indices, strict=True):
            prefix_indices = prefill_req.prefix_indices
            req_to_token_pool.write(
                (
                    req_pool_idx,
                    slice(0, len(prefix_indices)),
                ),
                prefix_indices,
            )

        prepared_reqs = [
            PreparedReqState.init_prefill_req(r, req_pool_idx)
            for r, req_pool_idx in zip(need_req_slot, req_pool_indices, strict=True)
        ] + [PreparedReqState.init_prefill_req_from_chunked(req) for req in chunked_reqs]

        # Created in parallel to prepared_reqs
        prefix_lens = np.asarray([len(r.prefix_indices) for r in need_req_slot + chunked_reqs])

        # Allocate actual cache
        total_extend_len = sum(r.extend_len for r in prepared_reqs)
        tree_cache.ensure_free_size(total_extend_len)
        if token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = token_to_kv_pool_allocator.alloc(total_extend_len)
        else:
            extend_lens = np.asarray([r.extend_len for r in prepared_reqs])
            out_cache_loc = token_to_kv_pool_allocator.alloc_prefill(
                prefix_lens=prefix_lens,
                seq_lens=prefix_lens + extend_lens,
            )
        if out_cache_loc is None:
            raise RuntimeError("Ran out of kv cache slots.")

        # Update req_to_token_pool information
        pt = 0
        for i, req in enumerate(prepared_reqs):
            # We use pt to step through out_cache_loc
            # Since out_cache_loc is a flattened list of length total_extend_len
            req_to_token_pool.write(
                (
                    req.req_pool_idx,
                    slice(prefix_lens[i], prefix_lens[i] + req.extend_len),
                ),
                out_cache_loc[pt : pt + req.extend_len],
            )
            pt += req.extend_len

        return cls(
            reqs=prepared_reqs,
            out_cache_loc=out_cache_loc,
            forward_mode=ForwardMode.PREFILL,
            req_to_token=req_to_token_pool.req_to_token,
        )

    @classmethod
    def prepare_for_decode(
        cls,
        reqs: list[ProcessedReqState],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: RadixCache,
    ) -> Self:
        prepared_reqs = [PreparedReqState.init_decode_req(r) for r in reqs]
        total_extend_len = sum(r.extend_len for r in prepared_reqs)

        req_pool_indices = np.asarray([r.req_pool_idx for r in prepared_reqs])
        seq_lens = np.asarray(
            [len(r.req_info.origin_input_ids) + len(r.output_ids) for r in prepared_reqs]
        )

        # Allocate actual cache
        tree_cache.ensure_free_size(total_extend_len)
        if token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = token_to_kv_pool_allocator.alloc(total_extend_len)
        else:
            out_cache_loc = token_to_kv_pool_allocator.alloc_decode(
                prev_cache_loc=req_to_token_pool.req_to_token[req_pool_indices, seq_lens - 2]
            )
        if out_cache_loc is None:
            raise RuntimeError("Ran out of kv cache slots.")

        # Update req_to_token_pool information
        req_to_token_pool.write(
            (req_pool_indices, seq_lens - 1),
            out_cache_loc,
        )

        return cls(
            reqs=prepared_reqs,
            out_cache_loc=out_cache_loc,
            forward_mode=ForwardMode.DECODE,
            req_to_token=req_to_token_pool.req_to_token,
        )

    @classmethod
    def generate_synthetic(
        cls,
        num_tokens: int,
        num_reqs: int,
        forward_mode: ForwardMode,
        req_to_token_pool: ReqToTokenPool,
    ):
        dummy_node = TreeNode(
            key=[],
            value=np.array([], dtype=int),
            parent=None,
            children={},
            lock_count=0,
            last_access_time=0,
        )
        # The first request has all num_tokens tokens, all other requests have no tokens
        req_infos = [
            ReqInfo(
                rid=str(i),
                origin_input_ids=(i == 0) * num_tokens * [0],
                sampling_params=SamplingParams(),
                stream=False,
            )
            for i in range(num_reqs)
        ]

        prepared_reqs = [
            PreparedReqState(
                req_info=req_info,
                req_pool_idx=i,
                extend_len=len(req_info.origin_input_ids),
                last_node=dummy_node,  # Should never be used by anything
                tree_matched_len=0,  # Should never be used by anything
                prefill_unfinished=False,  # Should never be used by anything
            )
            for i, req_info in enumerate(req_infos)
        ]

        total_extend_len = sum(r.extend_len for r in prepared_reqs)
        # We use np.zeros because the 0 cache location is for writing padding/trash
        return cls(
            reqs=prepared_reqs,
            out_cache_loc=np.zeros((total_extend_len,), dtype=np.int32),
            forward_mode=forward_mode,
            req_to_token=req_to_token_pool.req_to_token,
        )

    def merge_batch(self, other: ScheduleBatch):
        self.reqs.extend(other.reqs)
        # Safe since both batches are prepared/have kv cache slots allocated on construction
        self.out_cache_loc = np.concatenate([self.out_cache_loc, other.out_cache_loc])
        self.forward_mode = ForwardMode.merge(self.forward_mode, other.forward_mode)
