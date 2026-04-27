from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from tpu_mini_sglang.managers.scheduler_struct import (
    ForwardMode,
    ReqState,
)
from tpu_mini_sglang.mem_cache.allocator import BaseTokenToKVPoolAllocator
from tpu_mini_sglang.mem_cache.memory_pool import ReqToTokenPool
from tpu_mini_sglang.mem_cache.radix_cache import RadixCache


@dataclass
class ScheduleBatch:
    # Unlike original SGLang, we enforce that ScheduleBatch is entirely CPU-side
    # Any structures needed on the TPU are constructed/copied over later,
    # when we construct the ForwardBatch from the ScheduleBatch
    # We now merge the prepare logic into ScheduleBatch, so that a ScheduleBatch is always
    # fully prepared to run and only stale for a brief moment immediately after running
    reqs: list[ReqState]
    out_cache_loc: np.ndarray

    forward_mode: ForwardMode

    # Should always be the same reference; this is just a convenient way to pass it to ForwardBatch
    req_to_token: np.ndarray

    @classmethod
    def prepare_for_prefill(
        cls,
        reqs: list[ReqState],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: RadixCache,
    ) -> Self:
        # Allocates and writes KV and ReqToTokenPool caches, creates ScheduleBatch

        # Alloc a req_pool_idx to every request which needs one

        # Requests which already have one should be in chunked prefill
        need_req_slot = [r for r in reqs if not r.has_req_pool_idx]
        req_pool_indices = req_to_token_pool.alloc(len(need_req_slot))
        if req_pool_indices is None:
            raise RuntimeError("Ran out of running request slots.")
        for req, req_pool_idx in zip(need_req_slot, req_pool_indices, strict=True):
            req.set_req_pool_idx(req_pool_idx)

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

        # Correct formula for required_pages since the prefix must be page-aligned
        extend_lens = np.asarray([r.extend_len for r in reqs])
        required_pages = np.sum(
            (extend_lens + token_to_kv_pool_allocator.page_size - 1)
            // token_to_kv_pool_allocator.page_size
        ).item()

        # Allocate actual cache
        tree_cache.ensure_free_size(required_pages * token_to_kv_pool_allocator.page_size)
        if token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = token_to_kv_pool_allocator.alloc(np.sum(extend_lens).item())
        else:
            prefix_lens = np.asarray([len(r.prefix_indices) for r in reqs])
            out_cache_loc = token_to_kv_pool_allocator.alloc_prefill(
                prefix_lens=prefix_lens,
                seq_lens=prefix_lens + extend_lens,
            )
        if out_cache_loc is None:
            raise RuntimeError("Ran out of kv cache slots.")

        # Update req_to_token_pool information for the kv cache allocation
        pt = 0
        for i, req in enumerate(reqs):
            # We use pt to step through out_cache_loc
            # Since out_cache_loc is a flattened list of length total_extend_len
            prefix_len = len(req.prefix_indices)
            req_to_token_pool.write(
                (
                    req.req_pool_idx,
                    slice(prefix_len, prefix_len + req.extend_len),
                ),
                out_cache_loc[pt : pt + req.extend_len],
            )
            pt += req.extend_len

        return cls(
            reqs=reqs,
            out_cache_loc=out_cache_loc,
            forward_mode=ForwardMode.PREFILL,
            req_to_token=req_to_token_pool.req_to_token,
        )

    @classmethod
    def prepare_for_decode(
        cls,
        reqs: list[ReqState],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: RadixCache,
    ) -> Self:
        for req in reqs:
            req.prepare_decode()

        req_pool_indices = np.asarray([r.req_pool_idx for r in reqs])
        seq_lens = np.asarray([len(r.req_info.origin_input_ids) + len(r.output_ids) for r in reqs])
        # Implicitly assumes extend_len == 1
        # The new, uncached token is at position seq_lens - 1
        required_pages = np.sum((seq_lens - 1) % token_to_kv_pool_allocator.page_size == 0).item()

        # Allocate actual cache
        tree_cache.ensure_free_size(token_to_kv_pool_allocator.page_size * required_pages)
        if token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = token_to_kv_pool_allocator.alloc(required_pages)
        else:
            out_cache_loc = token_to_kv_pool_allocator.alloc_decode(
                prev_cache_loc=req_to_token_pool.req_to_token[req_pool_indices, seq_lens - 2],
            )
        if out_cache_loc is None:
            raise RuntimeError("Ran out of kv cache slots.")

        # Update req_to_token_pool information
        req_to_token_pool.write(
            (req_pool_indices, seq_lens - 1),
            out_cache_loc,
        )

        return cls(
            reqs=reqs,
            out_cache_loc=out_cache_loc,
            forward_mode=ForwardMode.DECODE,
            req_to_token=req_to_token_pool.req_to_token,
        )

    def merge_batch(self, other: ScheduleBatch):
        self.reqs.extend(other.reqs)
        # Safe since both batches are prepared/have kv cache slots allocated on construction
        self.out_cache_loc = np.concatenate([self.out_cache_loc, other.out_cache_loc])
        self.forward_mode = ForwardMode.merge(self.forward_mode, other.forward_mode)
