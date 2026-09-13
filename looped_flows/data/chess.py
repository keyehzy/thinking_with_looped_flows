"""Chess as a looped-flow task: predict the board after the best move.

Two datasets share this task: mate-in-N positions from the Lichess puzzle
database and general positions labelled by Stockfish. Both are produced by
``scripts/prepare_chess.py`` as a directory of ``.npy`` arrays per split.

Sequence layout (66 positions, ``problem_tokens=True`` as for graph colouring):
positions 0..63 hold the squares a1..h8, position 64 the castling rights and
position 65 the en-passant file. The position is normalised so that the side to
move is always White (Black-to-move positions are mirrored and colour-swapped),
so no side-to-move token is needed. The target is the board after the move on
the same 64 positions; the two meta positions are padding in the target. The
move is recovered from the predicted board by matching it against the boards
reachable by a legal move.

Tokens: 0 = pad, 1 = empty, 2..7 = white P N B R Q K, 8..13 = black p n b r q k,
14..29 = castling rights (4-bit mask KQkq), 30..38 = en-passant file (none, a..h).
"""

from __future__ import annotations

import os
from collections import Counter

import chess
import numpy as np

from .common import Task, TaskSpec

EMPTY = 1
WHITE0, BLACK0 = 2, 8  # token of a white / black pawn; other piece types follow in chess order
CASTLING0 = 14
EP0 = 30
VOCAB = EP0 + 9
BOARD_LEN = 64
SEQ_LEN = BOARD_LEN + 2
NO_MOVE = 255  # padding value in move arrays

ARRAYS = ("board", "castling", "ep", "moves", "mate_in")


# ------------------------------------------------------------------ encoding
def encode_board(board: chess.Board) -> tuple[np.ndarray, int, int]:
    """Tokens [64] (a1..h8), castling mask (0..15) and ep file (0 = none, 1..8 = a..h).
    The piece placement is encoded as is; call :func:`normalize` first so White is to move."""
    tokens = np.full(BOARD_LEN, EMPTY, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        base = WHITE0 if piece.color == chess.WHITE else BLACK0
        tokens[sq] = base + piece.piece_type - 1
    castling = (
        (1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        | (2 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        | (4 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        | (8 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    ep = 0 if board.ep_square is None else chess.square_file(board.ep_square) + 1
    return tokens, castling, ep


def decode_board(tokens: np.ndarray, castling: int, ep: int) -> chess.Board:
    """Inverse of :func:`encode_board` (White to move)."""
    board = chess.Board(None)
    pieces = {}
    for sq in range(BOARD_LEN):
        t = int(tokens[sq])
        if WHITE0 <= t < WHITE0 + 6:
            pieces[sq] = chess.Piece(t - WHITE0 + 1, chess.WHITE)
        elif BLACK0 <= t < BLACK0 + 6:
            pieces[sq] = chess.Piece(t - BLACK0 + 1, chess.BLACK)
    board.set_piece_map(pieces)
    board.turn = chess.WHITE
    rights = 0
    if castling & 1:
        rights |= chess.BB_H1
    if castling & 2:
        rights |= chess.BB_A1
    if castling & 4:
        rights |= chess.BB_H8
    if castling & 8:
        rights |= chess.BB_A8
    board.castling_rights = rights
    board.ep_square = chess.square(ep - 1, 5) if ep > 0 else None
    return board


def normalize(board: chess.Board, move: chess.Move | None = None):
    """Mirror the position (ranks flipped, colours swapped) when Black is to move, so that
    the side to move is always White. Returns the normalised board and move."""
    if board.turn == chess.WHITE:
        return board, move
    board = board.mirror()
    if move is not None:
        move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
    return board, move


def encode_move(move: chess.Move) -> np.ndarray:
    return np.array([move.from_square, move.to_square, move.promotion or 0], dtype=np.uint8)


def decode_move(m: np.ndarray) -> chess.Move:
    return chess.Move(int(m[0]), int(m[1]), int(m[2]) or None)


def apply_move_tokens(board: np.ndarray, ep: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Board tokens after a White move, computed directly on token arrays (batched).
    board [B, 64], ep [B], moves [B, 3] -> [B, 64]. Handles captures, promotion,
    castling (rook relocation) and en-passant captures."""
    out = board.copy()
    B = board.shape[0]
    rows = np.arange(B)
    frm, to, promo = moves[:, 0].astype(np.int64), moves[:, 1].astype(np.int64), moves[:, 2].astype(np.int64)
    piece = out[rows, frm].copy()
    out[rows, frm] = EMPTY
    is_pawn = piece == WHITE0
    is_king = piece == WHITE0 + 5
    # en-passant capture: pawn moves diagonally onto the empty ep square
    ep_sq = np.where(ep > 0, 40 + (ep.astype(np.int64) - 1), -1)
    ep_cap = is_pawn & (to == ep_sq) & (out[rows, to] == EMPTY)
    out[rows[ep_cap], to[ep_cap] - 8] = EMPTY
    # castling: king moves two files, the rook jumps over it
    ks = is_king & (to - frm == 2)
    qs = is_king & (frm - to == 2)
    out[rows[ks], frm[ks] + 3] = EMPTY
    out[rows[ks], frm[ks] + 1] = WHITE0 + 3
    out[rows[qs], frm[qs] - 4] = EMPTY
    out[rows[qs], frm[qs] - 1] = WHITE0 + 3
    placed = np.where(promo > 0, WHITE0 + promo - 1, piece).astype(np.uint8)
    out[rows, to] = placed
    return out


def mirror_files(board: np.ndarray, ep: np.ndarray, moves: np.ndarray):
    """Reflect files a<->h. Only valid for positions without castling rights."""
    board = board.reshape(-1, 8, 8)[:, :, ::-1].reshape(-1, BOARD_LEN)
    ep = np.where(ep > 0, 9 - ep, 0).astype(ep.dtype)
    m = moves.copy()
    valid = m[..., 0] != NO_MOVE
    for i in range(2):
        sq = m[..., i].astype(np.int64)
        m[..., i] = np.where(valid, (sq & ~7) | (7 - (sq & 7)), NO_MOVE).astype(np.uint8)
    return board, ep, m


# ------------------------------------------------------------------- dataset
def write_split(path: str, board, castling, ep, moves, mate_in, fens=None):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "board.npy"), np.asarray(board, dtype=np.uint8))
    np.save(os.path.join(path, "castling.npy"), np.asarray(castling, dtype=np.uint8))
    np.save(os.path.join(path, "ep.npy"), np.asarray(ep, dtype=np.uint8))
    np.save(os.path.join(path, "moves.npy"), np.asarray(moves, dtype=np.uint8))
    np.save(os.path.join(path, "mate_in.npy"), np.asarray(mate_in, dtype=np.uint8))
    if fens is not None:
        np.save(os.path.join(path, "fen.npy"), np.asarray(fens, dtype=str))


def pad_moves(move_lists: list[list[np.ndarray]], width: int | None = None) -> np.ndarray:
    """[N, M, 3] uint8 with NO_MOVE padding; the first move of each row is the preferred target."""
    width = width or max(len(m) for m in move_lists)
    out = np.full((len(move_lists), width, 3), NO_MOVE, dtype=np.uint8)
    for i, ms in enumerate(move_lists):
        for j, m in enumerate(ms[:width]):
            out[i, j] = m
    return out


class ChessSplit:
    def __init__(self, path: str, limit: int | None = None):
        self.arrays = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r") for k in ARRAYS}
        n = len(self.arrays["board"])
        self.n = n if limit is None else min(limit, n)

    def __len__(self):
        return self.n

    def get(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        return {k: np.asarray(v[idx]) for k, v in self.arrays.items()}


class ChessTask(Task):
    """``root`` contains ``train/`` and ``test/`` splits written by ``scripts/prepare_chess.py``.

    ``target``: ``any`` draws the training target uniformly from the acceptable moves
    (all mating moves, or every engine move within the labelling margin); ``best`` always
    uses the first (preferred) move. ``augment`` mirrors files a<->h with probability 0.5
    on positions without castling rights.
    """

    spec = TaskSpec(name="chess", vocab_size=VOCAB, seq_len=SEQ_LEN, mixer="attention", L_cycles=4, problem_tokens=True)

    def __init__(self, root: str = "data/chess/mate", target: str = "any", augment: bool = True, test_limit: int | None = None):
        assert target in ("any", "best")
        self.root = root
        self.target = target
        self.augment = augment
        self.train = ChessSplit(os.path.join(root, "train"))
        self.test = ChessSplit(os.path.join(root, "test"), test_limit)

    def _encode(self, ex: dict, move_idx: np.ndarray) -> dict:
        board, castling, ep, moves = ex["board"], ex["castling"], ex["ep"], ex["moves"]
        n = board.shape[0]
        chosen = moves[np.arange(n), move_idx]
        after = apply_move_tokens(board, ep, chosen)
        inputs = np.concatenate([board, (CASTLING0 + castling)[:, None], (EP0 + ep)[:, None]], axis=1).astype(np.int64)
        labels = np.concatenate([after, np.zeros((n, 2), dtype=np.uint8)], axis=1).astype(np.int64)
        return self.to_batch(inputs, labels, np.zeros_like(inputs, dtype=bool))

    def sample_train(self, n: int, rng: np.random.Generator) -> dict:
        ex = self.train.get(np.sort(rng.integers(0, len(self.train), size=n)))
        if self.augment:
            flip = (rng.random(n) < 0.5) & (ex["castling"] == 0)
            b, e, m = mirror_files(ex["board"][flip], ex["ep"][flip], ex["moves"][flip])
            ex["board"], ex["ep"], ex["moves"] = ex["board"].copy(), ex["ep"].copy(), ex["moves"].copy()
            ex["board"][flip], ex["ep"][flip], ex["moves"][flip] = b, e, m
        num = (ex["moves"][:, :, 0] != NO_MOVE).sum(1)
        move_idx = (rng.random(n) * num).astype(np.int64) if self.target == "any" else np.zeros(n, dtype=np.int64)
        return self._encode(ex, move_idx)

    def num_test(self, limit: int | None = None) -> int:
        return len(self.test) if limit is None else min(limit, len(self.test))

    def test_batches(self, batch_size: int, limit: int | None = None):
        m = self.num_test(limit)
        for s in range(0, m, batch_size):
            ex = self.test.get(np.arange(s, min(s + batch_size, m)))
            batch = self._encode(ex, np.zeros(len(ex["board"]), dtype=np.int64))
            batch.update(ex)
            yield batch

    # ---------------------------------------------------------------- metrics
    @staticmethod
    def reachable(board: chess.Board) -> dict[bytes, chess.Move]:
        """Map from encoded board-after-move to the legal move producing it."""
        out = {}
        for mv in board.legal_moves:
            board.push(mv)
            out[encode_board(board)[0].tobytes()] = mv
            board.pop()
        return out

    def evaluate_samples(self, batch: dict, samples: np.ndarray) -> dict:
        """samples: [B, num_samples, L]. Per position: ``legal`` (fraction of samples that
        decode to a legal move), ``accuracy`` (first sample is an acceptable move),
        ``pass_at_k`` (any sample), ``majority`` (most frequent legal move), and the
        first-sample accuracy broken down by mate depth (``acc_mate<k>``)."""
        out: dict[str, list] = {"legal": [], "accuracy": [], "pass_at_k": [], "majority": []}
        for b in range(samples.shape[0]):
            board = decode_board(batch["board"][b], int(batch["castling"][b]), int(batch["ep"][b]))
            table = self.reachable(board)
            good = {decode_move(m) for m in batch["moves"][b] if m[0] != NO_MOVE}
            decoded = [table.get(samples[b, j, :BOARD_LEN].astype(np.uint8).tobytes()) for j in range(samples.shape[1])]
            legal = [d for d in decoded if d is not None]
            oks = [d in good for d in decoded]
            out["legal"].append(len(legal) / len(decoded))
            out["accuracy"].append(float(oks[0]))
            out["pass_at_k"].append(float(any(oks)))
            out["majority"].append(float(Counter(legal).most_common(1)[0][0] in good) if legal else 0.0)
            k = int(batch["mate_in"][b])
            if k > 0:
                out.setdefault(f"acc_mate{k}", []).append(float(oks[0]))
        return out
