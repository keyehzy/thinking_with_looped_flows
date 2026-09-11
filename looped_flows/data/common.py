"""Task interface shared by all benchmarks.

A task exposes token-level problems as fixed-length sequences. Each batch is a
dict of CPU tensors:

    inputs      [B, L] long   problem tokens (fixed grid entries, or the separately
                              embedded problem for ARC / graph colouring)
    labels      [B, L] long   solution tokens, IGNORE_LABEL_ID at padding
    puzzle_ids  [B]    long   puzzle-embedding identifier (0 when unused)
    given_mask  [B, L] bool   positions whose value is fixed by the problem; the
                              interpolant is built over the remaining positions

The vocabulary always reserves index 0 for padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..losses import IGNORE_LABEL_ID

PAD = 0


@dataclass
class TaskSpec:
    name: str
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int = 1
    problem_tokens: bool = False  # embed ``inputs`` separately and add to the projected x_t
    mixer: str = "attention"
    L_cycles: int = 4
    sigma: float | None = None  # default noise scale, None -> 1/sqrt(|V|)

    @property
    def default_sigma(self) -> float:
        return self.sigma if self.sigma is not None else 1.0 / self.vocab_size**0.5


class Task:
    spec: TaskSpec

    def sample_train(self, n: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def test_batches(self, batch_size: int, limit: int | None = None):
        """Yields test batches (dicts, possibly with extra task-specific keys)."""
        raise NotImplementedError

    def num_test(self, limit: int | None = None) -> int:
        raise NotImplementedError

    def to_batch(self, inputs: np.ndarray, labels: np.ndarray, given_mask: np.ndarray, puzzle_ids: np.ndarray | None = None) -> dict:
        labels = labels.astype(np.int64).copy()
        labels[labels == PAD] = IGNORE_LABEL_ID
        n = inputs.shape[0]
        return {
            "inputs": torch.from_numpy(inputs.astype(np.int64)),
            "labels": torch.from_numpy(labels),
            "puzzle_ids": torch.from_numpy((puzzle_ids if puzzle_ids is not None else np.zeros(n)).astype(np.int64)),
            "given_mask": torch.from_numpy(given_mask.astype(bool)),
        }


def batch_to(batch: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def slice_batch(batch: dict, idx) -> dict:
    return {k: (v[idx] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v) for k, v in batch.items()}
