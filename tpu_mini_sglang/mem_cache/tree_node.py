from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(eq=False)
class TreeNode:
    key: list[int]
    value: np.ndarray  # len(value) == len(key)
    parent: Self | None
    children: dict[tuple[int, ...], Self]
    lock_count: int
    last_access_time: float

    # Necessary to avoid type errors with heapq
    # heapq compares tuple elements left-to-right; when last_access_time ties in evictable_heap
    # it falls through to comparing TreeNode objects, which would TypeError without this method.
    def __lt__(self, other: Self):
        return self.last_access_time < other.last_access_time
