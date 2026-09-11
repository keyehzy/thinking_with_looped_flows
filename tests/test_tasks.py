import numpy as np
import pytest

from looped_flows.data.graph_coloring import GraphColoringTask, all_colorings, count_conflicts
from looped_flows.data.maze import dihedral_transform, inverse_dihedral_transform
from looped_flows.data.nqueens import NQueensTask, QUEEN, all_solutions, board_from_cols, is_valid_board
from looped_flows.data.sudoku import shuffle_sudoku


def sudoku_valid(sol):
    g = sol.reshape(9, 9)
    for i in range(9):
        assert sorted(g[i]) == list(range(1, 10))
        assert sorted(g[:, i]) == list(range(1, 10))
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            assert sorted(g[r : r + 3, c : c + 3].reshape(-1)) == list(range(1, 10))


def test_sudoku_augmentation_preserves_validity():
    sol = np.array([int(ch) for ch in "583427169974136528216859374792364851351298746648715293865971432137642985429583617"])
    board = sol.copy()
    board[np.random.default_rng(0).random(81) < 0.6] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        b, s = shuffle_sudoku(board, sol, rng)
        sudoku_valid(s)
        assert ((b == 0) | (b == s)).all()
        assert (b == 0).sum() == (board == 0).sum()


def test_dihedral_inverse():
    g = np.arange(12).reshape(3, 4)
    for tid in range(8):
        assert (inverse_dihedral_transform(dihedral_transform(g, tid), tid) == g).all()


def test_nqueens_solutions_and_puzzles():
    assert len(all_solutions(8)) == 92
    task = NQueensTask(n=8, test_fraction=0.1)
    for cols in task.solutions[:5]:
        assert is_valid_board(board_from_cols(cols, 8), 8)
    batch = task.sample_train(16, np.random.default_rng(0))
    assert batch["inputs"].shape == (16, 64)
    for i in range(16):
        assert is_valid_board(batch["labels"][i].numpy(), 8)
        given = batch["given_mask"][i]
        assert (batch["inputs"][i][given] == QUEEN).all()
        assert (batch["labels"][i][given] == QUEEN).all()
    shown_counts = [len(shown) for shown, _ in task.test_puzzles]
    assert set(shown_counts) <= {1, 2, 3}


def test_nqueens_eval_metrics():
    task = NQueensTask(n=8, test_fraction=0.1, test_limit=4)
    batch = next(task.test_batches(4))
    labels = batch["labels"].numpy()
    samples = np.repeat(labels[:, None], 3, axis=1)
    m = task.evaluate_samples(batch, samples)
    assert m["accuracy"] == [1.0] * 4
    assert all(0 < c <= 1 for c in m["coverage"])


def test_graph_coloring_generation_and_eval():
    task = GraphColoringTask(n=8, edge_prob=0.35, num_train=20, num_test=4)
    adj, cols = task.train_graphs[0]
    assert cols == all_colorings(adj)
    for c in cols:
        assert count_conflicts(adj, np.array(c)) == 0
    batch = task.sample_train(4, np.random.default_rng(0))
    assert batch["inputs"].shape == (4, 64)
    assert batch["given_mask"].sum() == 0
    tb = next(task.test_batches(4))
    samples = np.repeat(tb["labels"].numpy()[:, None], 5, axis=1)
    m = task.evaluate_samples(tb, samples)
    assert m["conflicts"] == [0] * 4
    assert all(c > 0 for c in m["coverage"])


@pytest.mark.skipif(not __import__("os").path.isdir("data/raw/ARC-AGI/data"), reason="ARC data not downloaded")
def test_arc_roundtrip():
    from looped_flows.data.arc import ARCTask, augment_grid, decode_grid, encode_grid, inverse_augment_grid

    task = ARCTask(num_aug=4, num_test_aug=2, test_limit=3)
    assert task.num_puzzle_identifiers == 1 + 4 * len(task.puzzles)
    rng = np.random.default_rng(0)
    batch = task.sample_train(8, rng)
    assert batch["inputs"].shape == (8, 900)
    assert (batch["puzzle_ids"] > 0).all()
    grid = rng.integers(0, 10, size=(5, 7))
    for aug in task.puzzles[0].augs:
        assert (inverse_augment_grid(augment_grid(grid, aug), aug) == grid).all()
    assert (decode_grid(encode_grid(grid)) == grid).all()
    tb = next(task.test_batches(4))
    assert len(tb["keys"]) == 4
