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


def test_king_takes_rook_castling_is_canonicalised():
    """The Lichess eval database writes castling as king-takes-rook (e1h1); python-chess accepts
    it in `legal_moves` but apply_move_tokens only understands the e1g1 form, and reachable()
    only ever yields e1g1. Both conventions must land on the same encoded move."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from prepare_chess import make_example

    fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    board = chess.Board(fen)
    for ktr, std in (("e1h1", "e1g1"),):
        a = make_example(board, [chess.Move.from_uci(ktr)], 0)
        b = make_example(board, [chess.Move.from_uci(std)], 0)
        assert (a["moves"][0] == b["moves"][0]).all(), f"{ktr} did not canonicalise to {std}"
        # and the encoded target must be a board an actual legal move can produce
        task_board = decode_board(a["board"], a["castling"], a["ep"])
        after = apply_move_tokens(a["board"][None], np.array([a["ep"]], dtype=np.uint8), a["moves"][0][None])[0]
        assert after.tobytes() in {k for k in ChessTask.reachable(task_board)}


def test_queenside_king_takes_rook_castling():
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from prepare_chess import make_example
    board = chess.Board(fen)
    for ktr, std in (("e1a1", "e1c1"), ("e1h1", "e1g1")):
        a = make_example(board, [chess.Move.from_uci(ktr)], 0)
        b = make_example(board, [chess.Move.from_uci(std)], 0)
        assert (a["moves"][0] == b["moves"][0]).all(), f"{ktr} != {std}"
        tb = decode_board(a["board"], a["castling"], a["ep"])
        after = apply_move_tokens(a["board"][None], np.array([a["ep"]], dtype=np.uint8), a["moves"][0][None])[0]
        assert after.tobytes() in set(ChessTask.reachable(tb)), f"{ktr}: target unreachable"


def _pvs(*pairs):
    return [(chess.Move.from_uci(u), s) for u, s in pairs]


def test_mates_are_scored_on_their_own_scale():
    """Folding a mate into centipawns puts mate-in-1 and mate-in-2 one unit apart, so a 30 cp
    margin used to accept any mate within ~30 plies -- and non-mating moves never mix in."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from prepare_chess import MATE_SCORE, acceptable_from_pvs

    M = MATE_SCORE
    pvs = _pvs(("a1a2", M - 1), ("b1b2", M - 5), ("c1c2", M - 25), ("d1d2", 900))
    moves, mate_in = acceptable_from_pvs(pvs, margin=30)
    assert mate_in == 1
    assert [m.uci() for m in moves] == ["a1a2"], "only the fastest mate by default"

    moves, _ = acceptable_from_pvs(pvs, margin=30, mate_margin=4)
    assert [m.uci() for m in moves] == ["a1a2", "b1b2"], "mate_margin counts plies, not centipawns"

    moves, _ = acceptable_from_pvs(pvs, margin=30, mate_margin=100)
    assert "d1d2" not in [m.uci() for m in moves], "a merely-winning move is never an acceptable mate"

    # non-mate positions keep the centipawn margin
    moves, mate_in = acceptable_from_pvs(_pvs(("a1a2", 120), ("b1b2", 100), ("c1c2", 60)), margin=30)
    assert mate_in == 0 and [m.uci() for m in moves] == ["a1a2", "b1b2"]


def test_writer_slices_are_disjoint_and_cover(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from prepare_chess import Writer

    keys = [f"pos{i}" for i in range(1000)]
    got = []
    for i in range(4):
        w = Writer(str(tmp_path / f"s{i}"), 0.0, shard=(i, 4))
        got.append({k for k in keys if w.add(k, {"moves": [], "x": k})})
    assert sum(len(g) for g in got) == len(keys), "shards must cover every key"
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (got[i] & got[j]), "shards must be disjoint"

    # skip acts as a sequential cursor
    a = Writer(str(tmp_path / "a"), 0.0)
    for k in keys:
        a.add(k, {"moves": [], "x": k})
    b = Writer(str(tmp_path / "b"), 0.0, skip=600)
    kept_b = [k for k in keys if b.add(k, {"moves": [], "x": k})]
    assert a.count == 1000 and b.count == 400 and kept_b == keys[600:]


def test_evaldb_scores_are_converted_from_white_pov():
    """lichess_db_eval reports cp/mate from White's point of view. Treating them as
    side-to-move relative inverts every Black-to-move position, picking the worst move."""
    import json, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from prepare_chess import evaldb_example

    # Black to move; White mates in 1 after g8f8, so a Black player must avoid it.
    fen = "2r1r1k1/5p2/6R1/4NQp1/P7/1q5P/5PP1/4R1K1 b - -"
    rec = {"fen": fen, "evals": [{"depth": 30, "knodes": 1, "pvs": [
        {"mate": 1, "line": "g8f8"}, {"mate": 2, "line": "g8h8"}, {"mate": 15, "line": "f7g6"}]}]}
    _, ex = evaldb_example(json.dumps(rec), 20, 30)
    board = chess.Board(fen)
    chosen = [decode_move(m) for m in ex["moves"]]
    # the label is stored in the mirrored (White-to-move) frame, so mirror back to compare
    back = [chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
            for m in chosen]
    assert all(m in board.legal_moves for m in back)
    assert chess.Move.from_uci("g8f8") not in back, "picked the move that lets White mate fastest"
    assert chess.Move.from_uci("f7g6") in back, "should prefer the longest resistance"
    assert ex["mate_in"] == 0, "the mover is not the one delivering mate"
