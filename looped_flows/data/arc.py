"""ARC-AGI-1 / ARC-AGI-2 following the preprocessing and evaluation protocol of
TRM / HRM.

* Every input/output grid is placed on a 30x30 canvas and flattened to 900
  tokens: 0 = pad, 1 = EOS (the row below and the column right of the grid),
  2..11 = colours 0..9.
* Each puzzle receives ``num_aug`` random augmentations (dihedral transform +
  colour permutation with black fixed) and every (puzzle, augmentation) pair has
  its own puzzle identifier whose embedding is learned. Demonstration pairs of
  evaluation puzzles are part of the training set (their test pairs are not).
* Training additionally applies a random translation of the grids on the canvas.
* At test time each test input is predicted under ``num_test_aug`` augmentations,
  predictions are mapped back and the two most voted outputs form the pass@2
  prediction. The score is the official ARC one: mean over tasks of the fraction
  of test inputs solved.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from ..flow import SampleConfig, sample, sample_best_q
from .common import Task, TaskSpec, batch_to
from .maze import dihedral_transform, inverse_dihedral_transform

SIDE = 30
SEQ_LEN = SIDE * SIDE
PAD, EOS, COLOR0 = 0, 1, 2
VOCAB = COLOR0 + 10


@dataclass
class Puzzle:
    name: str
    train: list[tuple[np.ndarray, np.ndarray]]  # pairs used for training
    test: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # held-out pairs (evaluation puzzles)
    augs: np.ndarray | None = None  # [num_aug, 11]: dihedral id followed by a colour permutation
    id_offset: int = 0  # puzzle identifier of augmentation a is id_offset + a


def _load_json_puzzles(path: str) -> list[tuple[str, dict]]:
    files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    return [(os.path.splitext(os.path.basename(f))[0], json.load(open(f))) for f in files]


def _pairs(examples) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array(e["input"], dtype=np.int64), np.array(e["output"], dtype=np.int64)) for e in examples]


def _puzzle_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def augment_grid(grid: np.ndarray, aug: np.ndarray) -> np.ndarray:
    tid, perm = int(aug[0]), aug[1:]
    return perm[dihedral_transform(grid, tid)]


def inverse_augment_grid(grid: np.ndarray, aug: np.ndarray) -> np.ndarray:
    tid, perm = int(aug[0]), aug[1:]
    inv = np.argsort(perm)
    return inverse_dihedral_transform(inv[grid], tid)


def encode_grid(grid: np.ndarray, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Place a grid on the canvas with EOS markers; returns a [900] token array."""
    canvas = np.zeros((SIDE, SIDE), dtype=np.int64)
    r, c = offset
    h, w = grid.shape
    canvas[r : r + h, c : c + w] = grid + COLOR0
    if r + h < SIDE:
        canvas[r + h, c : c + w] = EOS
    if c + w < SIDE:
        canvas[r : r + h, c + w] = EOS
    return canvas.reshape(-1)


def decode_grid(seq: np.ndarray) -> np.ndarray:
    """Read the grid at the top-left corner of the canvas back (colours; cells that
    are not colours become 0)."""
    canvas = seq.reshape(SIDE, SIDE)
    h = next((i for i in range(SIDE) if canvas[i, 0] < COLOR0), SIDE)
    w = next((j for j in range(SIDE) if canvas[0, j] < COLOR0), SIDE)
    h, w = max(h, 1), max(w, 1)
    grid = canvas[:h, :w] - COLOR0
    return np.clip(grid, 0, 9)


class ARCTask(Task):
    def __init__(
        self,
        sources: list[dict] | None = None,
        num_aug: int = 1000,
        num_test_aug: int = 1000,
        seed: int = 42,
        test_limit: int | None = None,
        translate: bool = True,
    ):
        """``sources`` is a list of ``{"path": dir, "role": "train" | "test"}``. Puzzles
        from a ``test`` source contribute their demonstration pairs to training and
        their test pairs to evaluation."""
        if sources is None:
            sources = [
                {"path": "data/raw/ARC-AGI/data/training", "role": "train"},
                {"path": "data/raw/ARC-AGI/data/evaluation", "role": "test"},
                {"path": "data/raw/ConceptARC/corpus", "role": "train"},
            ]
        self.translate = translate
        self.num_aug = num_aug
        self.num_test_aug = min(num_test_aug, num_aug)
        rng = np.random.default_rng(seed)
        seen: set[str] = set()
        self.puzzles: list[Puzzle] = []
        self.test_puzzles: list[Puzzle] = []
        for src in sources:
            for name, data in _load_json_puzzles(src["path"]):
                h = _puzzle_hash(data)
                if h in seen:
                    continue
                seen.add(h)
                if src["role"] == "test":
                    p = Puzzle(name=name, train=_pairs(data["train"]), test=_pairs(data["test"]))
                    self.test_puzzles.append(p)
                else:
                    p = Puzzle(name=name, train=_pairs(data["train"]) + _pairs(data["test"]))
                self.puzzles.append(p)
        if test_limit is not None:
            self.test_puzzles = self.test_puzzles[:test_limit]

        next_id = 1  # identifier 0 is the blank embedding
        for p in self.puzzles:
            p.augs = self._sample_augmentations(p, rng)
            p.id_offset = next_id
            next_id += num_aug
        self.num_puzzle_identifiers = next_id
        self.spec = TaskSpec(
            name="arc", vocab_size=VOCAB, seq_len=SEQ_LEN, num_puzzle_identifiers=next_id, problem_tokens=True, mixer="attention", L_cycles=4
        )
        # flat index of training examples, sampled uniformly
        self.train_index = [(pi, ei) for pi, p in enumerate(self.puzzles) for ei in range(len(p.train))]

    def _sample_augmentations(self, p: Puzzle, rng: np.random.Generator) -> np.ndarray:
        """Distinct (dihedral, colour permutation) pairs; the first is the identity."""
        augs = [np.concatenate(([0], np.arange(10)))]
        seen = {augs[0].tobytes()}
        tries = 0
        while len(augs) < self.num_aug and tries < self.num_aug * 5:
            tries += 1
            perm = np.concatenate(([0], rng.permutation(np.arange(1, 10))))
            a = np.concatenate(([rng.integers(8)], perm))
            if a.tobytes() in seen:
                continue
            seen.add(a.tobytes())
            augs.append(a)
        while len(augs) < self.num_aug:  # tiny puzzles may not have enough distinct augmentations
            augs.append(augs[rng.integers(len(augs))])
        return np.stack(augs)

    # ------------------------------------------------------------- training
    def _encode_pair(self, inp: np.ndarray, out: np.ndarray, rng: np.random.Generator | None):
        offset = (0, 0)
        if rng is not None and self.translate:
            h = max(inp.shape[0], out.shape[0])
            w = max(inp.shape[1], out.shape[1])
            offset = (int(rng.integers(0, SIDE - h + 1)), int(rng.integers(0, SIDE - w + 1)))
        return encode_grid(inp, offset), encode_grid(out, offset)

    def sample_train(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, len(self.train_index), size=n)
        inputs, labels, ids = [], [], []
        for i in idx:
            pi, ei = self.train_index[i]
            p = self.puzzles[pi]
            a = int(rng.integers(self.num_aug))
            inp, out = p.train[ei]
            aug = p.augs[a]
            x, y = self._encode_pair(augment_grid(inp, aug), augment_grid(out, aug), rng)
            inputs.append(x)
            labels.append(y)
            ids.append(p.id_offset + a)
        inputs = np.stack(inputs)
        return self.to_batch(inputs, np.stack(labels), np.zeros_like(inputs, dtype=bool), np.array(ids))

    # ----------------------------------------------------------- evaluation
    def num_test(self, limit: int | None = None) -> int:
        n = sum(len(p.test) for p in self.test_puzzles)
        return n if limit is None else min(n, limit)

    def test_batches(self, batch_size: int, limit: int | None = None):
        """Yields batches of augmented test inputs with bookkeeping for voting."""
        items = []
        count = 0
        for pi, p in enumerate(self.test_puzzles):
            for ei, (inp, out) in enumerate(p.test):
                if limit is not None and count >= limit:
                    break
                count += 1
                for a in range(self.num_test_aug):
                    items.append((pi, ei, a))
        for s in range(0, len(items), batch_size):
            chunk = items[s : min(s + batch_size, len(items))]
            inputs, labels, ids = [], [], []
            for pi, ei, a in chunk:
                p = self.test_puzzles[pi]
                inp, out = p.test[ei]
                aug = p.augs[a]
                x, y = self._encode_pair(augment_grid(inp, aug), augment_grid(out, aug), None)
                inputs.append(x)
                labels.append(y)
                ids.append(p.id_offset + a)
            inputs = np.stack(inputs)
            batch = self.to_batch(inputs, np.stack(labels), np.zeros_like(inputs, dtype=bool), np.array(ids))
            batch["keys"] = chunk
            yield batch

    @torch.no_grad()
    def evaluate(self, model, sample_cfg: SampleConfig, batch_size: int, device, limit: int | None = None, num_trajectories: int = 1, progress=None) -> dict:
        model.eval()
        votes: dict[tuple[int, int], Counter] = {}
        done = 0
        for batch in self.test_batches(batch_size, limit):
            b = batch_to(batch, device)
            res = sample(model, b, sample_cfg) if num_trajectories == 1 else sample_best_q(model, b, sample_cfg, num_trajectories, max_batch=batch_size)
            tokens = res.tokens.cpu().numpy()
            for (pi, ei, a), seq in zip(batch["keys"], tokens):
                grid = inverse_augment_grid(decode_grid(seq), self.test_puzzles[pi].augs[a])
                votes.setdefault((pi, ei), Counter())[grid.tobytes() + bytes(grid.shape)] += 1
            done += len(batch["keys"])
            if progress is not None:
                progress(done)
        per_task: dict[int, list[float]] = {}
        pass1 = []
        for (pi, ei), counter in votes.items():
            out = self.test_puzzles[pi].test[ei][1]
            key = out.tobytes() + bytes(out.shape)
            top = [k for k, _ in counter.most_common(2)]
            per_task.setdefault(pi, []).append(float(key in top))
            pass1.append(float(top[0] == key))
        task_scores = [float(np.mean(v)) for v in per_task.values()]
        return {
            "pass@2": float(np.mean(task_scores)) if task_scores else 0.0,
            "pass@1": float(np.mean(pass1)) if pass1 else 0.0,
            "num_tasks": len(task_scores),
            "num_examples": len(pass1),
        }
