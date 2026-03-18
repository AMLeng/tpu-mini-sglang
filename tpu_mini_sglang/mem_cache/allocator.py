import numpy as np


class TokenToKVPoolAllocator:
    """
    TokenToKVPoolAllocator tracks which pages of the KV cache are available
    Because it purely does tracking/management, it stays in the CPU
    This one hardcodes an implicit page size of 1, matching the SGLang equivalent
    Other core KV cache classes:
    ReqToTokenPool maps a request to the locations of its tokens in the KV cache
    TokenToKVPool holds the actual caches
    """

    def __init__(self, size: int):
        self.size = size
        self.clear()

    def clear(self):
        # In the actual cache, slot 0 is used for writing dummy values from padding tokens
        self.free_pages = np.arange(1, self.size + 1, dtype=np.int32)

    def available_size(self):
        return len(self.free_pages)

    def free(self, free_index: np.ndarray):
        self.free_pages = np.concatenate([self.free_pages, np.array(free_index)])

    def alloc(self, need_size: int):
        if need_size > self.available_size():
            return None
        ret = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return ret
