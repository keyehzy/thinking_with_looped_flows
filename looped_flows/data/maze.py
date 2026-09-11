"""Maze-Hard (30x30) from Wang et al. (2025), following TRM's preprocessing.

Tokens: 0 = pad, 1 = wall '#', 2 = empty ' ', 3 = start 'S', 4 = goal 'G', 5 = path 'o'.
Walls, start and goal are fixed by the problem; the interpolant runs over the empty
cells, which become either path or remain empty. Training uses the 8 dihedral
transforms of the grid as augmentation.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .common import Task, TaskSpec

CHARS = "# SGo"
SIDE = 30
SEQ_LEN = SIDE * SIDE
VOCAB = len(CHARS) + 1
EMPTY = CHARS.index(" ") + 1


def _load_csv(path: str):
    lut = np.zeros(256, dtype=np.int64)
    for i, ch in enumerate(CHARS):
        lut[ord(ch)] = i + 1
    qs, ans = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            qs.append(lut[np.frombuffer(row[1].encode(), dtype=np.uint8)])
            ans.append(lut[np.frombuffer(row[2].encode(), dtype=np.uint8)])
    return np.stack(qs), np.stack(ans)


def dihedral_transform(grid: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of the 8 symmetries of the square to a [.., H, W] array."""
    if tid >= 4:
        grid = np.flip(grid, axis=-1)
    return np.rot90(grid, k=tid % 4, axes=(-2, -1))


def inverse_dihedral_transform(grid: np.ndarray, tid: int) -> np.ndarray:
    grid = np.rot90(grid, k=-(tid % 4), axes=(-2, -1))
    if tid >= 4:
        grid = np.flip(grid, axis=-1)
    return grid


class MazeTask(Task):
    spec = TaskSpec(name="maze", vocab_size=VOCAB, seq_len=SEQ_LEN, mixer="attention", L_cycles=4)

    def __init__(self, root: str = "data/raw/maze-30x30-hard-1k", test_limit: int | None = None, augment: bool = True):
        self.train_q, self.train_a = _load_csv(os.path.join(root, "train.csv"))
        self.test_q, self.test_a = _load_csv(os.path.join(root, "test.csv"))
        if test_limit is not None:
            self.test_q, self.test_a = self.test_q[:test_limit], self.test_a[:test_limit]
        self.augment = augment

    def _encode(self, q: np.ndarray, a: np.ndarray) -> dict:
        return self.to_batch(q, a, q != EMPTY)

    def sample_train(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, len(self.train_q), size=n)
        q, a = self.train_q[idx], self.train_a[idx]
        if self.augment:
            q = q.reshape(n, SIDE, SIDE).copy()
            a = a.reshape(n, SIDE, SIDE).copy()
            tids = rng.integers(0, 8, size=n)
            q = np.stack([dihedral_transform(q[i], tids[i]) for i in range(n)]).reshape(n, SEQ_LEN)
            a = np.stack([dihedral_transform(a[i], tids[i]) for i in range(n)]).reshape(n, SEQ_LEN)
        return self._encode(np.ascontiguousarray(q), np.ascontiguousarray(a))

    def num_test(self, limit: int | None = None) -> int:
        return len(self.test_q) if limit is None else min(limit, len(self.test_q))

    def test_batches(self, batch_size: int, limit: int | None = None):
        n = self.num_test(limit)
        for s in range(0, n, batch_size):
            yield self._encode(self.test_q[s : s + batch_size], self.test_a[s : s + batch_size])
