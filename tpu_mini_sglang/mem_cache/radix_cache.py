import heapq
import time

import numpy as np

from tpu_mini_sglang.managers.scheduler_struct import (
    PrefillReqState,
    PreparedReqState,
    ProcessedReqState,
)
from tpu_mini_sglang.mem_cache.allocator import TokenToKVPoolAllocator
from tpu_mini_sglang.mem_cache.memory_pool import ReqToTokenPool
from tpu_mini_sglang.mem_cache.tree_node import TreeNode


class RadixCache:
    def __init__(
        self,
        page_size: int,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    ):
        self.page_size = page_size
        self.root = TreeNode(
            key=[],
            value=np.array([], dtype=int),
            parent=None,
            children={},
            lock_count=1,  #  Root should never be evicted
            last_access_time=time.monotonic(),
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        #  Set of all leaves (no children) with lock_count == 0, maintained for efficiency
        self._evictable_leaves: set[TreeNode] = set()
        #  Total value length of all nodes with lock_count == 0, maintained for efficiency
        self._evictable_size: int = 0

    def shared_prefix_len(self, first: list[int], second: list[int]) -> int:
        # Computes page-aligned shared prefix length for the two lists
        shared = 0
        max_page_start = min(len(first), len(second)) - self.page_size
        while shared <= max_page_start:
            if any(first[shared + i] != second[shared + i] for i in range(self.page_size)):
                break
            shared += self.page_size
        return shared

    def _get_child_key(self, token_ids: list[int]):
        # The children dict maps from a tuple of first page of ids to nodes
        # so we use this helper method to easily get the key
        return tuple(token_ids[: self.page_size])

    def _split_node(self, node: TreeNode, split_len: int):
        # Splits node in the tree so that the new intermediate node holds key[:split_len]
        # and the old node becomes its child with key[split_len:]
        # We mutate node in place so that existing references to it are still valid
        # most importantly, we preserve req.last_node references from prior prefix matches
        assert node.parent is not None  # The root node cannot be split
        mid_node = TreeNode(
            key=node.key[:split_len],
            value=node.value[:split_len],
            parent=node.parent,
            children={self._get_child_key(node.key[split_len:]): node},
            lock_count=node.lock_count,
            last_access_time=node.last_access_time,
        )
        node.parent.children[self._get_child_key(mid_node.key)] = mid_node
        node.key = node.key[split_len:]
        node.value = node.value[split_len:]
        node.parent = mid_node
        return mid_node

    def match_prefix(self, token_ids: list[int]) -> tuple[np.ndarray, TreeNode]:
        values = []
        current_node = self.root
        current_tail = token_ids
        access_time = time.monotonic()
        current_node.last_access_time = access_time
        while (key := self._get_child_key(current_tail)) in current_node.children:
            current_node = current_node.children[key]
            current_node.last_access_time = access_time

            prefix_len = self.shared_prefix_len(current_node.key, current_tail)
            if prefix_len < len(current_node.key):
                # Split so that len(current_node.key) == prefix_len
                current_node = self._split_node(current_node, prefix_len)

            current_tail = current_tail[prefix_len:]
            values.append(current_node.value)

        value = np.concatenate(values) if len(values) > 0 else np.array([], dtype=int)
        return value, current_node

    def inc_lock_count(self, node: TreeNode) -> int:  # Returns delta in evictable size
        self._evictable_leaves.discard(node)
        size = 0
        cur_node: TreeNode | None = node
        while cur_node is not None:
            if cur_node.lock_count == 0:
                size -= len(cur_node.value)
            cur_node.lock_count += 1
            cur_node = cur_node.parent

        # Update cached eviction info
        self._evictable_size += size
        return size

    def dec_lock_count(self, node: TreeNode) -> int:  # Returns delta in evictable size
        size = 0
        cur_node: TreeNode | None = node
        while cur_node is not None:
            cur_node.lock_count -= 1
            if cur_node.lock_count == 0:
                size += len(cur_node.value)
            cur_node = cur_node.parent

        # Update cached eviction info
        if len(node.children) == 0 and node.lock_count == 0:
            self._evictable_leaves.add(node)
        self._evictable_size += size
        return size

    def _insert(self, token_ids: list[int], value: np.ndarray) -> tuple[int, np.ndarray, TreeNode]:
        assert len(token_ids) % self.page_size == 0

        # The tree may already have inserted the token ids with different values
        # e.g. when two requests in the same batch extended in the same way
        # so we prefix match against the existing tree and take those values as
        # the canonical ones, and then only insert values that go beyond the prefix
        tree_values, current_node = self.match_prefix(token_ids)

        rematched_prefix_len = len(tree_values)
        new_ids = token_ids[rematched_prefix_len:]
        new_values = value[rematched_prefix_len:]

        # Add remaining tail to the tree as a new node
        if len(new_ids) > 0:
            new_node = TreeNode(
                key=new_ids,
                value=new_values,
                parent=current_node,
                children={},
                lock_count=0,
                last_access_time=current_node.last_access_time,
            )
            current_node.children[self._get_child_key(new_ids)] = new_node
            # Update cached eviction info
            self._evictable_leaves.discard(current_node)
            self._evictable_leaves.add(new_node)
            self._evictable_size += len(new_node.value)

            # Update values as though we prefixed matched through the new node
            tree_values = np.concatenate([tree_values, new_values])
            current_node = new_node

        return rematched_prefix_len, tree_values, current_node

    def cache_chunked_req(self, req: PreparedReqState) -> PrefillReqState:
        # Returning the prefix-matched req for further prefill
        # Chunked prefill is always page-aligned except for the final chunk,
        # so this method should never be called on a req with an unaligned tail
        # In particular, the below correctly gets the prefilled subset of token_ids
        token_ids = req.req_info.origin_input_ids[: req.tree_matched_len + req.extend_len]

        aligned_token_count = len(token_ids)
        assert aligned_token_count % self.page_size == 0
        cache_indices = self.req_to_token_pool.read(req.req_pool_idx, aligned_token_count)

        rematched_prefix_len, new_indices, new_last_node = self._insert(token_ids, cache_indices)

        # We must free [tree_matched_len:rematched_prefix_len], since those are tokens which
        # the current req computed kv cache for, but which were already in the tree when
        # we inserted above, and thus our current request has references to non-canonical stale
        # kv cache entries for those tokens. new_value will have the updated canonical values
        # from prefix_matching with the tree, so we need to use those instead.
        self.token_to_kv_pool_allocator.free(
            cache_indices[req.tree_matched_len : rematched_prefix_len]
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.tree_matched_len, rematched_prefix_len)),
            new_indices[req.tree_matched_len : rematched_prefix_len],
        )

        # Update locks and req state
        self.inc_lock_count(new_last_node)
        self.dec_lock_count(req.last_node)

        # extend_len and prefill_unfinished will be overwritten by the chunking logic later
        # Right now we write both values as though we will fully finish prefill the next pass
        return PrefillReqState(
            req_info=req.req_info,
            extend_len=len(req.req_info.origin_input_ids) - len(new_indices),
            prefix_indices=new_indices,
            last_node=new_last_node,
            tree_matched_len=len(new_indices),
            prefill_unfinished=False,
            req_pool_idx=req.req_pool_idx,
        )

    def cache_unfinished_req(self, req: ProcessedReqState) -> None:
        token_ids = req.req_info.origin_input_ids + req.output_ids
        unaligned_token_count = len(token_ids)
        cache_indices = self.req_to_token_pool.read(req.req_pool_idx, unaligned_token_count)

        # We cannot cache loose tokens that don't form a full cache page
        aligned_token_count = unaligned_token_count - (unaligned_token_count % self.page_size)
        rematched_prefix_len, new_indices, new_last_node = self._insert(
            token_ids[:aligned_token_count], cache_indices[:aligned_token_count]
        )

        # We must free [tree_matched_len:rematched_prefix_len], since those are tokens which
        # the current req computed kv cache for, but which were already in the tree when
        # we inserted above, and thus our current request has references to non-canonical stale
        # kv cache entries for those tokens. new_value will have the updated canonical values
        # from prefix_matching with the tree, so we need to use those instead.
        self.token_to_kv_pool_allocator.free(
            cache_indices[req.tree_matched_len : rematched_prefix_len]
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.tree_matched_len, rematched_prefix_len)),
            new_indices[req.tree_matched_len : rematched_prefix_len],
        )

        # Update locks and req state
        self.inc_lock_count(new_last_node)
        self.dec_lock_count(req.last_node)
        req.last_node = new_last_node
        req.tree_matched_len = len(new_indices)

    def cache_finished_req(self, req: ProcessedReqState) -> None:
        token_ids = req.req_info.origin_input_ids + req.output_ids
        unaligned_token_count = len(token_ids)
        cache_indices = self.req_to_token_pool.read(req.req_pool_idx, unaligned_token_count)

        # We cannot cache loose tokens that don't form a full cache page
        aligned_token_count = unaligned_token_count - (unaligned_token_count % self.page_size)
        rematched_prefix_len, new_indices, new_last_node = self._insert(
            token_ids[:aligned_token_count], cache_indices[:aligned_token_count]
        )

        # We must free [tree_matched_len:rematched_prefix_len], since those are tokens which
        # the current req computed kv cache for, but which were already in the tree when
        # we inserted above, and thus our current request has references to non-canonical stale
        # kv cache entries for those tokens. new_value will have the updated canonical values
        # from prefix_matching with the tree, so we need to use those instead.
        self.token_to_kv_pool_allocator.free(
            cache_indices[req.tree_matched_len : rematched_prefix_len]
        )

        # For finished reqs, we can't store the unaligned tail in the tree and must just free it
        self.token_to_kv_pool_allocator.free(
            cache_indices[aligned_token_count:unaligned_token_count]
        )

        # Update locks and req state
        self.dec_lock_count(req.last_node)

    def evictable_size(self) -> int:
        return self._evictable_size

    def _delete_leaf(self, node: TreeNode) -> None:
        # Core operation
        key = self._get_child_key(node.key)
        assert node.parent is not None  # The root node is never passed to delete_leaf
        del node.parent.children[key]

        # Update cached eviction info
        self._evictable_leaves.discard(node)
        parent = node.parent
        if len(parent.children) == 0 and parent.lock_count == 0:
            self._evictable_leaves.add(parent)

    def _evict(self, num_tokens: int) -> None:
        # Evict at least num_tokens

        evictable_heap = [(node.last_access_time, node) for node in list(self._evictable_leaves)]
        heapq.heapify(evictable_heap)

        num_evicted = 0
        while num_evicted < num_tokens:
            assert evictable_heap, "Eviction heap is unexpectedly empty, this should not happen."
            _, oldest_node = heapq.heappop(evictable_heap)

            self.token_to_kv_pool_allocator.free(oldest_node.value)
            num_evicted += len(oldest_node.value)
            self._delete_leaf(oldest_node)
            if oldest_node.parent in self._evictable_leaves:
                last_access = oldest_node.parent.last_access_time
                heapq.heappush(evictable_heap, (last_access, oldest_node.parent))

        self._evictable_size -= num_evicted

    def ensure_free_size(self, num_tokens: int) -> None:
        available_size = self.token_to_kv_pool_allocator.available_size()
        if available_size < num_tokens:
            self._evict(num_tokens - available_size)
