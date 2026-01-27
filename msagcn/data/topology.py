from __future__ import annotations

from typing import List, Tuple

NUM_HAND_JOINTS = 21
NUM_HAND_NODES = 42  # 21 left + 21 right

HAND_EDGES_ONE = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def hand_edges_42() -> List[Tuple[int, int]]:
    e = []
    e += HAND_EDGES_ONE
    e += [(a + 21, b + 21) for (a, b) in HAND_EDGES_ONE]
    return e


POSE_KEEP_DEFAULT = [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_EDGE_PAIRS_ABS = [
    (11, 13), (12, 14), (13, 15), (14, 16),
    (23, 11), (24, 12), (11, 12), (23, 24),
    (0, 9), (0, 10),
]
CROSS_EDGE_PAIRS_ABS = [
    ("LWRIST", 15), ("RWRIST", 16),
    ("LWRIST", 13), ("RWRIST", 14),
    ("LWRIST", 11), ("RWRIST", 12),
]  # wrists <-> pose wrists/elbows/shoulders
