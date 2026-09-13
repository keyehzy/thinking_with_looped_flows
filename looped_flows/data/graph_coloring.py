"""Graph 3-colouring multi-solution benchmark (Erdős–Rényi graphs with 8 or 10
vertices that admit a proper 3-colouring), following GRAM (Baek et al., 2026).

Sequence layout: the problem is the flattened n x n adjacency matrix (tokens
1 = no edge, 2 = edge), embedded separately and added to the projected flow
state. The solution is laid out on the same n x n grid, with every entry of row
i carrying the colour of vertex i (tokens 3, 4, 5). Vertex colours are decoded
by majority vote over the row. Training targets are drawn uniformly from the
valid colourings of the graph.
"""

from __future__ import annotations

import numpy as np

from .common import Task, TaskSpec

NO_EDGE, EDGE = 1, 2
COLOR0 = 3
NUM_COLORS = 3
VOCAB = COLOR0 + NUM_COLORS


def all_colorings(adj: np.ndarray, num_colors: int = NUM_COLORS) -> list[tuple[int, ...]]:
    n = adj.shape[0]
    out = []
    colors = [-1] * n

    def rec(v):
        if v == n:
            out.append(tuple(colors))
            return
        for c in range(num_colors):
            if all(colors[u] != c for u in range(v) if adj[v, u]):
                colors[v] = c
                rec(v + 1)
        colors[v] = -1

    rec(0)
    return out


def random_colorable_graph(n: int, p: float, rng: np.random.Generator):
    while True:
        upper = rng.random((n, n)) < p
        adj = np.triu(upper, 1)
        adj = (adj | adj.T).astype(np.int64)
        cols = all_colorings(adj)
        if cols:
            return adj, cols


def count_conflicts(adj: np.ndarray, colors: np.ndarray) -> int:
    same = colors[:, None] == colors[None, :]
    return int((np.triu(adj, 1) * same).sum())


class GraphColoringTask(Task):
    def __init__(self, n: int = 8, edge_prob: float = 0.35, num_train: int = 20000, num_test: int = 1000, seed: int = 0, test_limit: int | None = None):
        self.n = n
        self.spec = TaskSpec(
            name=f"graphcoloring{n}", vocab_size=VOCAB, seq_len=n * n, mixer="attention", L_cycles=4, problem_tokens=True,
            sigma=1.0 if n == 8 else None,
        )
        rng = np.random.default_rng(seed)
        seen = set()
        graphs = []
        while len(graphs) < num_train + num_test:
            adj, cols = random_colorable_graph(n, edge_prob, rng)
            key = adj.tobytes()
            if key in seen:
                continue
            seen.add(key)
            graphs.append((adj, cols))
        self.test_graphs = graphs[:num_test]
        self.train_graphs = graphs[num_test:]
        if test_limit is not None:
            self.test_graphs = self.test_graphs[:test_limit]

    def _encode(self, adj: np.ndarray, coloring) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        inp = np.where(adj > 0, EDGE, NO_EDGE).reshape(-1)
        label = (np.asarray(coloring, dtype=np.int64)[:, None] + COLOR0).repeat(n, axis=1).reshape(-1)
        return inp, label

    def sample_train(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, len(self.train_graphs), size=n)
        inputs, labels = [], []
        for i in idx:
            adj, cols = self.train_graphs[i]
            a, b = self._encode(adj, cols[rng.integers(len(cols))])
            inputs.append(a)
            labels.append(b)
        inputs, labels = np.stack(inputs), np.stack(labels)
        return self.to_batch(inputs, labels, np.zeros_like(inputs, dtype=bool))

    def num_test(self, limit: int | None = None) -> int:
        return len(self.test_graphs) if limit is None else min(limit, len(self.test_graphs))

    def test_batches(self, batch_size: int, limit: int | None = None):
        m = self.num_test(limit)
        for s in range(0, m, batch_size):
            chunk = self.test_graphs[s : min(s + batch_size, m)]
            enc = [self._encode(adj, cols[0]) for adj, cols in chunk]
            inputs = np.stack([e[0] for e in enc])
            batch = self.to_batch(inputs, np.stack([e[1] for e in enc]), np.zeros_like(inputs, dtype=bool))
            batch["adj"] = [adj for adj, _ in chunk]
            batch["colorings"] = [cols for _, cols in chunk]
            yield batch

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Majority vote over each row; returns [n] colour indices, -1 for a row without a colour token."""
        n = self.n
        rows = tokens.reshape(n, n)
        out = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            counts = np.bincount(rows[i], minlength=VOCAB)[COLOR0:]
            if counts.sum() > 0:
                out[i] = int(counts.argmax())
        return out

    def evaluate_samples(self, batch: dict, samples: np.ndarray) -> dict:
        """samples: [B, num_samples, L]. Returns per-graph conflicts (of the most frequent
        complete colouring) and coverage of the valid colourings."""
        conflicts, coverage = [], []
        for b in range(samples.shape[0]):
            adj, valid_set = batch["adj"][b], set(batch["colorings"][b])
            decoded = [self.decode(samples[b, j]) for j in range(samples.shape[1])]
            complete = [tuple(d.tolist()) for d in decoded if (d >= 0).all()]
            if complete:
                keys, counts = np.unique(np.array(complete), axis=0, return_counts=True)
                best = keys[counts.argmax()]
                conflicts.append(count_conflicts(adj, best))
            else:
                conflicts.append(int(np.triu(adj, 1).sum()))
            found = {c for c in complete if c in valid_set}
            coverage.append(len(found) / len(valid_set))
        return {"conflicts": conflicts, "coverage": coverage}
