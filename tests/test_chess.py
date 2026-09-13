import random

import chess
import numpy as np
import pytest

from looped_flows.data.chess import (
    BOARD_LEN,
    NO_MOVE,
    SEQ_LEN,
    VOCAB,
    ChessTask,
    apply_move_tokens,
    decode_board,
    decode_move,
    encode_board,
    encode_move,
    mirror_files,
    normalize,
    pad_moves,
    write_split,
)


def random_positions(num_games: int, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(num_games):
        board = chess.Board()
        for _ in range(rng.randint(5, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            yield board.copy(), move
            board.push(move)


def test_encoding_roundtrip_and_token_moves():
    for board, move in random_positions(60):
        nb, nm = normalize(board, move)
        tokens, castling, ep = encode_board(nb)
        back = decode_board(tokens, castling, ep)
        assert back.board_fen() == nb.board_fen()
        assert back.castling_rights == nb.castling_rights
        assert nm in back.legal_moves
        nb.push(nm)
        expected = encode_board(nb)[0]
        got = apply_move_tokens(tokens[None], np.array([ep], dtype=np.uint8), encode_move(nm)[None])[0]
        assert (expected == got).all(), nb.fen()


def test_file_mirror_preserves_legality():
    checked = 0
    for board, move in random_positions(60, seed=1):
        nb, nm = normalize(board, move)
        tokens, castling, ep = encode_board(nb)
        if castling != 0:
            continue
        mb, mep, mm = mirror_files(tokens[None], np.array([ep], dtype=np.uint8), encode_move(nm)[None, None])
        mirrored = decode_board(mb[0], 0, int(mep[0]))
        m2 = decode_move(mm[0, 0])
        assert m2 in mirrored.legal_moves
        mirrored.push(m2)
        assert (encode_board(mirrored)[0] == apply_move_tokens(mb, mep, mm[:, 0])[0]).all()
        checked += 1
    assert checked > 100


def _write_dataset(root, positions):
    boards, castling, ep, moves, mate_in, fens = [], [], [], [], [], []
    for board, move in positions:
        nb, nm = normalize(board, move)
        t, c, e = encode_board(nb)
        boards.append(t)
        castling.append(c)
        ep.append(e)
        moves.append([encode_move(nm)])
        mate_in.append(0)
        fens.append(board.fen())
    for split in ("train", "test"):
        write_split(str(root / split), np.stack(boards), castling, ep, pad_moves(moves), mate_in, fens)


def test_chess_task_batches_and_metrics(tmp_path):
    positions = list(random_positions(4, seed=2))[:50]
    _write_dataset(tmp_path, positions)
    task = ChessTask(str(tmp_path), target="any", augment=True)
    assert task.spec.vocab_size == VOCAB and task.spec.seq_len == SEQ_LEN and task.spec.problem_tokens
    batch = task.sample_train(32, np.random.default_rng(0))
    assert batch["inputs"].shape == (32, SEQ_LEN)
    assert batch["given_mask"].sum() == 0
    assert (batch["labels"][:, BOARD_LEN:] < 0).all()  # meta positions are ignored in the loss
    assert (batch["inputs"][:, :BOARD_LEN] >= 1).all() and (batch["inputs"] < VOCAB).all()

    tb = next(task.test_batches(16))
    labels = tb["labels"].numpy()
    perfect = np.repeat(labels[:, None], 3, axis=1)
    m = task.evaluate_samples(tb, perfect)
    assert m["accuracy"] == [1.0] * 16 and m["legal"] == [1.0] * 16 and m["majority"] == [1.0] * 16
    # an unreachable board is neither legal nor correct
    broken = perfect.copy()
    broken[:, :, 0] = 12
    m = task.evaluate_samples(tb, broken)
    assert m["accuracy"] == [0.0] * 16 and m["legal"] == [0.0] * 16
    # a legal but different move counts as legal only
    b0 = decode_board(tb["board"][0], int(tb["castling"][0]), int(tb["ep"][0]))
    good = decode_move(tb["moves"][0][0])
    other = next(mv for mv in b0.legal_moves if mv != good)
    b0.push(other)
    alt = np.concatenate([encode_board(b0)[0], np.zeros(2, dtype=np.uint8)])
    samples = np.stack([alt, labels[0], labels[0]])[None]
    m = task.evaluate_samples({k: v[:1] for k, v in tb.items()}, samples)
    assert m["legal"] == [1.0] and m["accuracy"] == [0.0] and m["pass_at_k"] == [1.0] and m["majority"] == [1.0]


def test_pad_moves():
    m = pad_moves([[np.array([1, 2, 0], dtype=np.uint8)], [np.array([3, 4, 0], dtype=np.uint8), np.array([5, 6, 5], dtype=np.uint8)]])
    assert m.shape == (2, 2, 3)
    assert m[0, 1, 0] == NO_MOVE and m[1, 1, 2] == 5


@pytest.mark.skipif(not __import__("os").path.isdir("data/chess/mate_small/train"), reason="chess data not prepared")
def test_mate_dataset_labels_are_mates():
    task = ChessTask("data/chess/mate_small")
    ex = task.train.get(np.arange(100))
    for i in range(100):
        board = decode_board(ex["board"][i], int(ex["castling"][i]), int(ex["ep"][i]))
        good = [decode_move(m) for m in ex["moves"][i] if m[0] != NO_MOVE]
        assert good and all(m in board.legal_moves for m in good)
        for m in good:
            b = board.copy()
            b.push(m)
            assert b.is_checkmate() == (ex["mate_in"][i] == 1)
