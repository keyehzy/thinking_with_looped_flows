"""Sudoku-Extreme (Wang et al., 2025) with TRM's preprocessing: 1,000 training
puzzles, on-the-fly validity-preserving augmentations, full test set.

Tokens: 0 = pad, 1 = blank cell, 2..10 = digits 1..9. Cells given by the puzzle
are fixed; the interpolant runs over the blank cells only.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch

from .common import Task, TaskSpec

SEQ_LEN = 81
VOCAB = 11


def _load_csv(path: str, limit: int | None = None):
    qs, ans = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            q, a = row[1], row[2]
            qs.append(np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8) - ord("0"))
            ans.append(np.frombuffer(a.encode(), dtype=np.uint8) - ord("0"))
            if limit is not None and len(qs) >= limit:
                break
    return np.stack(qs).astype(np.int64), np.stack(ans).astype(np.int64)


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, rng: np.random.Generator):
    """Random digit relabeling, band/stack and within-band row/column permutations, transposition."""
    digit_map = np.concatenate(([0], rng.permutation(np.arange(1, 10))))
    bands = rng.permutation(3)
    rows = np.concatenate([bands[i] * 3 + rng.permutation(3) for i in range(3)])
    stacks = rng.permutation(3)
    cols = np.concatenate([stacks[i] * 3 + rng.permutation(3) for i in range(3)])
    transpose = rng.random() < 0.5

    def apply(x):
        x = x.reshape(9, 9)
        if transpose:
            x = x.T
        x = x[rows][:, cols]
        return digit_map[x].reshape(81)

    return apply(board), apply(solution)


class SudokuTask(Task):
    spec = TaskSpec(name="sudoku", vocab_size=VOCAB, seq_len=SEQ_LEN, mixer="mlp", L_cycles=6)

    def __init__(self, root: str = "data/raw/sudoku-extreme", num_train: int = 1000, test_limit: int | None = None, augment: bool = True, seed: int = 42):
        train_q, train_a = _load_csv(os.path.join(root, "train.csv"))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(train_q))[:num_train]
        self.train_q, self.train_a = train_q[idx], train_a[idx]
        self.test_q, self.test_a = _load_csv(os.path.join(root, "test.csv"), limit=test_limit)
        self.augment = augment

    def _encode(self, q: np.ndarray, a: np.ndarray) -> dict:
        inputs = q + 1
        labels = a + 1
        given = q != 0
        return self.to_batch(inputs, labels, given)

    def sample_train(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, len(self.train_q), size=n)
        qs, ans = [], []
        for i in idx:
            q, a = self.train_q[i], self.train_a[i]
            if self.augment:
                q, a = shuffle_sudoku(q, a, rng)
            qs.append(q)
            ans.append(a)
        return self._encode(np.stack(qs), np.stack(ans))

    def num_test(self, limit: int | None = None) -> int:
        return len(self.test_q) if limit is None else min(limit, len(self.test_q))

    def test_batches(self, batch_size: int, limit: int | None = None):
        n = self.num_test(limit)
        for s in range(0, n, batch_size):
            yield self._encode(self.test_q[s : s + batch_size], self.test_a[s : s + batch_size])
