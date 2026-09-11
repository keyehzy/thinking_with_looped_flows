"""N-Queens multi-solution benchmark, following the construction of GRAM (Baek et al., 2026).

Each puzzle hides ``n-3``..``n-1`` queens of a complete n-queens solution (5-7 on
8x8, 7-9 on 10x10) and asks for a completion. Puzzles are formed from all
solutions and all such subsets, de-duplicated, and split into train/test.
A puzzle may admit several completions; training targets are drawn uniformly
from the compatible solutions.

Tokens: 0 = pad, 1 = empty, 2 = queen. The shown queens are fixed entries.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from .common import Task, TaskSpec

EMPTY, QUEEN = 1, 2
VOCAB = 3


def all_solutions(n: int) -> np.ndarray:
    """All n-queens solutions as [S, n] arrays of column indices per row."""
    sols = []
    cols = [-1] * n

    def rec(r, used_c, used_d1, used_d2):
        if r == n:
            sols.append(list(cols))
            return
        for c in range(n):
            if c in used_c or (r - c) in used_d1 or (r + c) in used_d2:
                continue
            cols[r] = c
            rec(r + 1, used_c | {c}, used_d1 | {r - c}, used_d2 | {r + c})

    rec(0, frozenset(), frozenset(), frozenset())
    return np.array(sols, dtype=np.int64)


def board_from_cols(cols: np.ndarray, n: int) -> np.ndarray:
    b = np.full((n, n), EMPTY, dtype=np.int64)
    b[np.arange(n), cols] = QUEEN
    return b


def is_valid_board(board: np.ndarray, n: int) -> bool:
    board = board.reshape(n, n)
    if not np.isin(board, (EMPTY, QUEEN)).all():
        return False
    rows, cols = np.nonzero(board == QUEEN)
    if len(rows) != n or len(set(cols.tolist())) != n or len(set(rows.tolist())) != n:
        return False
    if len(set((rows - cols).tolist())) != n or len(set((rows + cols).tolist())) != n:
        return False
    return True


class NQueensTask(Task):
    def __init__(self, n: int = 8, test_fraction: float = 0.1, seed: int = 0, test_limit: int | None = None):
        self.n = n
        self.spec = TaskSpec(name=f"nqueens{n}", vocab_size=VOCAB, seq_len=n * n, mixer="attention", L_cycles=4, sigma=1.0)
        self.solutions = all_solutions(n)  # [S, n]
        puzzles: dict[frozenset, set] = {}
        for si, cols in enumerate(self.solutions):
            queens = [(r, int(cols[r])) for r in range(n)]
            for hide in range(n - 3, n):
                for shown in combinations(queens, n - hide):
                    puzzles.setdefault(frozenset(shown), set()).add(si)
        keys = sorted(puzzles, key=lambda k: sorted(k))
        rng = np.random.default_rng(seed)
        rng.shuffle(keys)
        n_test = int(round(len(keys) * test_fraction))
        self.test_puzzles = [(sorted(k), sorted(puzzles[k])) for k in keys[:n_test]]
        self.train_puzzles = [(sorted(k), sorted(puzzles[k])) for k in keys[n_test:]]
        if test_limit is not None:
            self.test_puzzles = self.test_puzzles[:test_limit]

    def _encode_puzzle(self, shown, solution_idx: int | None):
        n = self.n
        inp = np.full((n, n), EMPTY, dtype=np.int64)
        for r, c in shown:
            inp[r, c] = QUEEN
        given = inp == QUEEN
        label = board_from_cols(self.solutions[solution_idx], n) if solution_idx is not None else inp
        return inp.reshape(-1), label.reshape(-1), given.reshape(-1)

    def sample_train(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, len(self.train_puzzles), size=n)
        inputs, labels, given = [], [], []
        for i in idx:
            shown, sols = self.train_puzzles[i]
            a, b, c = self._encode_puzzle(shown, sols[rng.integers(len(sols))])
            inputs.append(a)
            labels.append(b)
            given.append(c)
        return self.to_batch(np.stack(inputs), np.stack(labels), np.stack(given))

    def num_test(self, limit: int | None = None) -> int:
        return len(self.test_puzzles) if limit is None else min(limit, len(self.test_puzzles))

    def test_batches(self, batch_size: int, limit: int | None = None):
        m = self.num_test(limit)
        for s in range(0, m, batch_size):
            chunk = self.test_puzzles[s : s + batch_size]
            enc = [self._encode_puzzle(shown, sols[0]) for shown, sols in chunk]
            batch = self.to_batch(np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc]), np.stack([e[2] for e in enc]))
            batch["num_solutions"] = [len(sols) for _, sols in chunk]
            batch["shown"] = [shown for shown, _ in chunk]
            yield batch

    def evaluate_samples(self, batch: dict, samples: np.ndarray) -> dict:
        """samples: [B, num_samples, L] token arrays. Returns per-problem accuracy and coverage."""
        n = self.n
        acc, cov = [], []
        for b in range(samples.shape[0]):
            shown = batch["shown"][b]
            valid = set()
            first_ok = False
            for j in range(samples.shape[1]):
                board = samples[b, j]
                ok = is_valid_board(board, n) and all(board[r * n + c] == QUEEN for r, c in shown)
                if j == 0:
                    first_ok = ok
                if ok:
                    valid.add(board.tobytes())
            acc.append(float(first_ok))
            cov.append(len(valid) / batch["num_solutions"][b])
        return {"accuracy": acc, "coverage": cov}
