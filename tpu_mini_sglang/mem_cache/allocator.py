import numpy as np


class BaseTokenToKVPoolAllocator:
    """
    TokenToKVPoolAllocator tracks which pages of the KV cache are available
    Because it purely does tracking/management, it stays in the CPU
    Other core KV cache classes:
    ReqToTokenPool maps a request to the locations of its tokens in the KV cache
    TokenToKVPool holds the actual caches
    """

    def __init__(self, num_tokens: int, page_size: int):
        self.num_pages: int = num_tokens // page_size
        self.page_size: int = page_size
        self.clear()

    def clear(self):
        # In the actual cache, slot 0 is used for writing dummy values from padding tokens
        self.free_pages = np.arange(1, self.num_pages + 1, dtype=np.int32)

    def available_size(self):
        # Available size in tokens; implicit multiply by page size of 1
        return len(self.free_pages) * self.page_size

    def free(self, free_index: np.ndarray):
        # np.concatenate always makes a copy
        pages_to_free = free_index // self.page_size
        self.free_pages = np.concatenate([self.free_pages, np.unique(pages_to_free)])

    def alloc(self, need_size: int) -> np.ndarray | None:
        raise NotImplementedError

    def alloc_prefill(self, prefix_lens: np.ndarray, seq_lens: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError

    def alloc_decode(self, prev_cache_loc: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    TokenToKVPoolAllocator tracks which pages of the KV cache are available
    Because it purely does tracking/management, it stays in the CPU
    This class hardcodes an implicit page size of 1, matching the SGLang equivalent
    Other core KV cache classes:
    ReqToTokenPool maps a request to the locations of its tokens in the KV cache
    TokenToKVPool holds the actual caches
    """

    def available_size(self):
        # Available size in tokens; implicit multiply by page size of 1
        return len(self.free_pages)

    def free(self, free_index: np.ndarray):
        # np.concatenate always makes a copy
        self.free_pages = np.concatenate([self.free_pages, free_index])

    def alloc(self, need_size: int) -> np.ndarray | None:
        if need_size > self.available_size():
            return None
        ret = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return ret


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    PagedTokenToKVPoolAllocator tracks which pages of the KV cache are available
    Because it purely does tracking/management, it stays in the CPU
    This class allows arbitrary page sizes
    Other core KV cache classes:
    ReqToTokenPool maps a request to the locations of its tokens in the KV cache
    TokenToKVPool holds the actual caches
    """

    def alloc_prefill(self, prefix_lens: np.ndarray, seq_lens: np.ndarray) -> np.ndarray | None:
        # We start by allocating all the pages we need
        assert np.all(prefix_lens % self.page_size == 0)
        seq_lens_ceil = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
        needed_size = np.sum(seq_lens_ceil - prefix_lens).item()
        if needed_size > self.available_size():
            return None
        needed_pages = needed_size // self.page_size
        new_pages = self.free_pages[:needed_pages]
        self.free_pages = self.free_pages[needed_pages:]

        # We have two separate pieces to write:
        # Part 1: Full pages for the bulk of the prefill
        # Part 2: Partial last page for unaligned tail
        # Note that prefixes are always padded to an even page_size

        extend_lens = seq_lens - prefix_lens
        extend_num_tokens = np.sum(extend_lens)
        out_cache_loc = np.empty((extend_num_tokens,), dtype=np.int32)

        write_offsets = np.cumsum(extend_lens) - extend_lens

        part_1_pages = (seq_lens - prefix_lens) // self.page_size
        new_page_counts = (seq_lens_ceil - prefix_lens) // self.page_size
        new_page_offsets = np.cumsum(new_page_counts) - new_page_counts
        # Part 1: Full pages
        for i in range(len(write_offsets)):
            write_loc = slice(write_offsets[i], write_offsets[i] + self.page_size * part_1_pages[i])
            page_ids = new_pages[new_page_offsets[i] : new_page_offsets[i] + part_1_pages[i]]
            # arange(self.page_size) within each individual page
            write_value = (
                np.arange(self.page_size)[None, :] + self.page_size * page_ids[:, None]
            ).ravel()
            out_cache_loc[write_loc] = write_value
            write_offsets[i] += part_1_pages[i] * self.page_size
        if np.sum(part_1_pages) * self.page_size == extend_num_tokens:
            return out_cache_loc

        part_2_slots = extend_lens - self.page_size * part_1_pages
        # Part 2: Unaligned tail
        for i in range(len(write_offsets)):
            if part_2_slots[i] == 0:
                continue
            write_loc = slice(write_offsets[i], write_offsets[i] + part_2_slots[i])
            page_id = new_pages[new_page_offsets[i] + part_1_pages[i]]
            write_value = np.arange(part_2_slots[i]) + self.page_size * page_id
            out_cache_loc[write_loc] = write_value

        return out_cache_loc

    def alloc_decode(self, prev_cache_loc: np.ndarray) -> np.ndarray | None:
        next_cache_loc = prev_cache_loc + 1
        needs_new_page = np.mod(next_cache_loc, self.page_size) == 0
        need_size = np.sum(needs_new_page).item()
        if need_size > self.available_size():
            return None
        if need_size == 0:
            return next_cache_loc
        new_pages = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        page_idx = np.cumsum(needs_new_page) - 1
        return np.where(needs_new_page, self.page_size * new_pages[page_idx], next_cache_loc)
