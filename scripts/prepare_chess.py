"""Build chess datasets for ``looped_flows.data.chess.ChessTask``.

Every mode writes ``<out>/train`` and ``<out>/test`` (split deterministically by
position/puzzle id) with the arrays ``board, castling, ep, moves, mate_in, fen``.

  # Stage 1: mate-in-N from the Lichess puzzle database (no engine needed)
  uv run python scripts/prepare_chess.py puzzles data/raw/lichess/lichess_db_puzzle.csv.zst data/chess/mate \\
      --mate-in 1,2,3,4,5

  # Stage 2a: general positions labelled by a local Stockfish (positions from PGNs or a FEN list)
  uv run python scripts/prepare_chess.py stockfish data/chess/stockfish --pgn data/raw/lichess/games.pgn.zst \\
      --engine stockfish --depth 14 --multipv 4 --margin 30 --workers 16 --limit 2000000

  # Stage 2b: convert Lichess' precomputed Stockfish evaluations (lichess_db_eval.jsonl.zst)
  uv run python scripts/prepare_chess.py evaldb data/raw/lichess/lichess_db_eval.jsonl.zst data/chess/evaldb \\
      --min-depth 20 --margin 30 --limit 5000000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
import chess.pgn
import numpy as np
from tqdm import tqdm

from looped_flows.data.chess import encode_board, encode_move, normalize, pad_moves, write_split

MATE_SCORE = 100_000
EVAL_DB_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def default_workers() -> int:
    """CPUs this process may actually use. ``os.cpu_count()`` reports the host's cores, which
    inside a cpu-quota'd container badly oversubscribes the pool -- enough, with an engine
    loading a ~100 MB NNUE net per worker, to push the UCI handshake past its timeout."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2
            quota, period = f.read().split()
            if quota != "max":
                n = min(n, max(1, int(int(quota) / int(period))))
    except OSError:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:  # cgroup v1
                quota = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except OSError:
            pass
    return max(1, n)


def shard_arg(text: str) -> tuple[int, int]:
    i, n = (int(x) for x in text.split("/", 1))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"--shard {text}: need 0 <= I < N")
    return i, n


# ----------------------------------------------------------------- utilities
def open_text(path: str):
    """Open a local file or an http(s) URL, transparently decompressing ``.zst``.

    The eval database is ~22 GB compressed, so streaming it straight from
    database.lichess.org avoids ever landing it on disk; combined with ``--limit`` only the
    prefix actually needed is transferred."""
    if path.startswith(("http://", "https://")):
        import urllib.request

        f = urllib.request.urlopen(path)  # noqa: S310 - fixed, user-supplied database URL
    else:
        f = open(path, "rb")
    if path.endswith(".zst"):
        import zstandard

        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8", errors="replace")
    return io.TextIOWrapper(f, encoding="utf-8", errors="replace")


def is_test(key: str, fraction: float) -> bool:
    return (zlib.crc32(key.encode()) % 100_000) < int(fraction * 100_000)


def canonical_castling(board: chess.Board, move: chess.Move) -> chess.Move:
    """python-chess accepts castling both as e1g1 and as the king-takes-rook e1h1 (the form the
    Lichess eval database writes in its PV lines), and ``move in board.legal_moves`` is true for
    either. Only the first form satisfies the ``to - from == 2`` test in ``apply_move_tokens``,
    and only it compares equal to what ``ChessTask.reachable`` yields, so canonicalise here --
    the one place every mode funnels through."""
    if board.is_castling(move):
        return chess.Move.from_uci(board.uci(move, chess960=False))
    return move


def make_example(board: chess.Board, moves: list[chess.Move], mate_in: int) -> dict:
    """Encode a position and its acceptable moves (first = preferred) in the White-to-move frame."""
    nb, _ = normalize(board)
    tokens, castling, ep = encode_board(nb)
    enc = [encode_move(normalize(board, canonical_castling(board, m))[1]) for m in moves]
    return {"board": tokens, "castling": castling, "ep": ep, "moves": enc, "mate_in": mate_in, "fen": board.fen()}


class Writer:
    """Collects examples and splits them train/test.

    ``shard=(i, n)`` keeps only the positions whose key hashes into shard ``i`` of ``n``, and
    ``skip`` drops that many accepted positions before collecting. Either one carves disjoint
    slices out of the same source -- shards are spread over the whole database, while skip is
    a cheap sequential cursor for "the next N after the ones I already built". ``count`` is
    what the callers compare against ``--limit``, so the limit counts kept positions."""

    def __init__(self, out: str, test_fraction: float, skip: int = 0, shard: tuple[int, int] | None = None):
        self.out = out
        self.test_fraction = test_fraction
        self.skip = skip
        self.shard = shard
        self.seen = 0
        self.count = 0
        self.splits = {"train": [], "test": []}

    def add(self, key: str, ex: dict) -> bool:
        if self.shard is not None:
            i, n = self.shard
            # salted so shard membership is independent of the train/test hash
            if zlib.crc32(f"shard:{key}".encode()) % n != i:
                return False
        self.seen += 1
        if self.seen <= self.skip:
            return False
        self.splits["test" if is_test(key, self.test_fraction) else "train"].append(ex)
        self.count += 1
        return True

    def finish(self):
        width = max((len(e["moves"]) for s in self.splits.values() for e in s), default=1)
        for name, exs in self.splits.items():
            if not exs:
                print(f"warning: empty {name} split")
                continue
            write_split(
                os.path.join(self.out, name),
                np.stack([e["board"] for e in exs]),
                [e["castling"] for e in exs],
                [e["ep"] for e in exs],
                pad_moves([e["moves"] for e in exs], width),
                [e["mate_in"] for e in exs],
                [e["fen"] for e in exs],
            )
            print(f"{name}: {len(exs)} positions -> {os.path.join(self.out, name)}")


# ------------------------------------------------------------------- puzzles
def puzzle_examples(row: dict, wanted: set[int]) -> list[tuple[str, dict]]:
    themes = row["Themes"].split()
    if not any(t.startswith("mateIn") for t in themes):
        return []
    board = chess.Board(row["FEN"])
    uci = row["Moves"].split()
    board.push_uci(uci[0])
    solver_moves = uci[1::2]
    depth = len(solver_moves)  # the theme "mateIn5" means five or more; the line length is exact
    final = board.copy()
    try:
        for mv in uci[1:]:
            final.push_uci(mv)
    except ValueError:
        return []
    if not final.is_checkmate():
        return []
    out = []
    for i, mv in enumerate(solver_moves):
        remaining = depth - i
        move = chess.Move.from_uci(mv)
        accept = [move]
        if remaining == 1:
            for alt in board.legal_moves:
                if alt != move:
                    board.push(alt)
                    if board.is_checkmate():
                        accept.append(alt)
                    board.pop()
        if remaining in wanted:
            out.append((row["PuzzleId"], make_example(board, accept, remaining)))
        board.push(move)
        if 2 * i + 2 < len(uci):
            board.push_uci(uci[2 * i + 2])
    return out


def _puzzle_chunk(args):
    rows, wanted = args
    return [e for r in rows for e in puzzle_examples(r, wanted)]


def cmd_puzzles(a):
    wanted = {int(x) for x in a.mate_in.split(",")}
    writer = Writer(a.out, a.test_fraction, a.skip, a.shard)
    reader = csv.DictReader(open_text(a.csv))

    def chunks():
        buf = []
        for row in reader:
            if a.min_rating and int(row["Rating"]) < a.min_rating:
                continue
            buf.append({k: row[k] for k in ("PuzzleId", "FEN", "Moves", "Themes")})
            if len(buf) == 2000:
                yield buf, wanted
                buf = []
        if buf:
            yield buf, wanted

    with Pool(a.workers) as pool:
        for exs in tqdm(pool.imap(_puzzle_chunk, chunks()), unit="chunk"):
            for key, ex in exs:
                writer.add(key, ex)
            if a.limit and writer.count >= a.limit:
                pool.terminate()
                break
    writer.finish()


# ------------------------------------------------------------------ positions
def iter_pgn_positions(path: str, skip_plies: int, sample_prob: float, seed: int):
    rng = random.Random(seed)
    with open_text(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply + 1 >= skip_plies and not board.is_game_over() and rng.random() < sample_prob:
                    yield board.fen()


def iter_fen_file(path: str):
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def is_mate_score(score: int) -> bool:
    """The mover delivers mate (folded score MATE_SCORE - plies)."""
    return score > MATE_SCORE - 1000


def is_mated_score(score: int) -> bool:
    """The mover gets mated (folded score -MATE_SCORE + plies of resistance)."""
    return score < -MATE_SCORE + 1000


def acceptable_from_pvs(pvs: list[tuple[chess.Move, int]], margin: int, mate_margin: int = 0) -> tuple[list[chess.Move], int]:
    """pvs: (move, score from the mover's view, mate distances folded into MATE_SCORE-ply).

    Mates are ranked on their own scale rather than through ``margin``. Folding a mate into the
    centipawn scale puts mate-in-1 and mate-in-2 one "centipawn" apart, so a 30 cp margin used to
    accept every mate within ~30 plies of the best -- on 64% of mate-in-1 positions it admitted a
    move that does not mate at all. When the best move mates, only moves that also mate within
    ``mate_margin`` plies of the fastest are acceptable, and a merely-winning move never is.
    Otherwise the usual centipawn ``margin`` applies."""
    pvs = sorted(pvs, key=lambda p: -p[1])
    best = pvs[0][1]
    if is_mate_score(best):
        best_ply = MATE_SCORE - best
        moves = [m for m, s in pvs if is_mate_score(s) and MATE_SCORE - s <= best_ply + mate_margin]
        return moves, min(best_ply, 255)
    if is_mated_score(best):
        # Every move loses; rank by how long they resist, not by a centipawn margin (which on the
        # folded scale would accept walking into the fastest mate).
        best_ply = best + MATE_SCORE
        moves = [m for m, s in pvs if is_mated_score(s) and s + MATE_SCORE >= best_ply - mate_margin]
        return moves, 0
    moves = [m for m, s in pvs if s >= best - margin]
    return moves, 0


_engine = None


def _engine_init(path: str, threads: int, hash_mb: int, timeout: float | None):
    global _engine
    # Modern Stockfish loads a ~100 MB NNUE network before answering "uciok", which routinely
    # takes longer than python-chess' 10 s default -- 17 s here with the labelling pool already
    # running, and the whole run then dies with TimeoutError. The same value bounds ``configure``;
    # ``analyse`` is unbounded for depth / node limits, so this only covers startup.
    _engine = chess.engine.SimpleEngine.popen_uci(path, timeout=timeout)
    _engine.configure({"Threads": threads, "Hash": hash_mb})


def _engine_label(args):
    fen, depth, nodes, multipv, margin, mate_margin = args
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    limit = chess.engine.Limit(depth=depth) if nodes is None else chess.engine.Limit(nodes=nodes)
    infos = _engine.analyse(board, limit, multipv=multipv)
    pvs = []
    for info in infos:
        if "pv" not in info or "score" not in info:
            continue
        pvs.append((info["pv"][0], info["score"].relative.score(mate_score=MATE_SCORE)))
    if not pvs:
        return None
    moves, mate_in = acceptable_from_pvs(pvs, margin, mate_margin)
    return fen, make_example(board, moves, mate_in)


def cmd_stockfish(a):
    if a.pgn:
        fens = iter_pgn_positions(a.pgn, a.skip_plies, a.sample_prob, a.seed)
    elif a.fens:
        fens = iter_fen_file(a.fens)
    else:
        raise SystemExit("stockfish mode needs --pgn or --fens")
    writer = Writer(a.out, a.test_fraction, a.skip, a.shard)
    jobs = ((fen, a.depth, a.nodes, a.multipv, a.margin, a.mate_margin) for fen in fens)
    with Pool(a.workers, initializer=_engine_init, initargs=(a.engine, a.threads, a.hash, a.engine_timeout)) as pool:
        for res in tqdm(pool.imap_unordered(_engine_label, jobs, chunksize=8), unit="pos"):
            if res is None:
                continue
            writer.add(*res)
            if a.limit and writer.count >= a.limit:
                pool.terminate()
                break
    writer.finish()


# --------------------------------------------------------------------- evaldb
def evaldb_example(line: str, min_depth: int, margin: int, mate_margin: int = 0):
    rec = json.loads(line)
    evals = [e for e in rec["evals"] if e.get("depth", 0) >= min_depth]
    if not evals:
        return None
    ev = max(evals, key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))
    board = chess.Board(rec["fen"])
    pvs = []
    for pv in ev["pvs"]:
        try:
            move = chess.Move.from_uci(pv["line"].split()[0])
        except (ValueError, IndexError):
            continue
        if move not in board.legal_moves:
            continue
        if "mate" in pv:
            m = pv["mate"]
            score = MATE_SCORE - m if m > 0 else -MATE_SCORE - m
        else:
            score = pv["cp"]
        # The Lichess eval database reports cp and mate from WHITE's point of view, whereas
        # acceptable_from_pvs (and the engine path, which uses score.relative) works in the
        # mover's frame. Without this flip every Black-to-move position selected the move that
        # is best for White, i.e. the worst legal move: 0 of 665 sampled "mate in 1" labels
        # with Black to move actually mated, against 532 of 532 with White to move.
        if board.turn == chess.BLACK:
            score = -score
        pvs.append((move, score))
    if not pvs:
        return None
    moves, mate_in = acceptable_from_pvs(pvs, margin, mate_margin)
    return rec["fen"], make_example(board, moves, mate_in)


def _evaldb_chunk(args):
    lines, min_depth, margin, mate_margin = args
    out = []
    for line in lines:
        ex = evaldb_example(line, min_depth, margin, mate_margin)
        if ex is not None:
            out.append(ex)
    return out


def cmd_evaldb(a):
    writer = Writer(a.out, a.test_fraction, a.skip, a.shard)

    def chunks():
        buf = []
        with open_text(a.jsonl) as f:
            for line in f:
                buf.append(line)
                if len(buf) == 2000:
                    yield buf, a.min_depth, a.margin, a.mate_margin
                    buf = []
        if buf:
            yield buf, a.min_depth, a.margin, a.mate_margin

    with Pool(a.workers) as pool:
        for exs in tqdm(pool.imap(_evaldb_chunk, chunks()), unit="chunk"):
            for key, ex in exs:
                writer.add(key, ex)
            if a.limit and writer.count >= a.limit:
                pool.terminate()
                break
    writer.finish()


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--test-fraction", type=float, default=0.02)
        p.add_argument("--limit", type=int, default=None, help="stop after this many kept positions")
        p.add_argument("--skip", type=int, default=0, help="drop this many kept positions first; with --limit this walks successive slices of one source")
        p.add_argument("--shard", type=shard_arg, default=None, metavar="I/N",
                       help="keep only shard I of N (hashed on the position key), for disjoint slices spread over the whole source")
        p.add_argument("--workers", type=int, default=default_workers(), help="worker processes (defaults to the CPUs actually available to this process/cgroup)")

    p = sub.add_parser("puzzles", help="mate-in-N positions from lichess_db_puzzle.csv(.zst)")
    p.add_argument("csv")
    p.add_argument("out")
    p.add_argument("--mate-in", default="1,2,3,4,5")
    p.add_argument("--min-rating", type=int, default=0)
    common(p)

    p = sub.add_parser("stockfish", help="label positions with a UCI engine")
    p.add_argument("out")
    p.add_argument("--pgn", default=None, help="PGN file (.zst ok); positions are sampled from the games")
    p.add_argument("--fens", default=None, help="text file with one FEN per line (.zst ok)")
    p.add_argument("--skip-plies", type=int, default=10)
    p.add_argument("--sample-prob", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", default="stockfish")
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--nodes", type=int, default=None, help="node limit instead of depth")
    p.add_argument("--multipv", type=int, default=4)
    p.add_argument("--margin", type=int, default=30, help="centipawns within the best move that count as acceptable")
    p.add_argument("--mate-margin", type=int, default=0, help="when the best move mates, also accept mates this many plies slower (0 = only the fastest)")
    p.add_argument("--threads", type=int, default=1, help="engine threads per worker")
    p.add_argument("--hash", type=int, default=64, help="engine hash (MB) per worker")
    p.add_argument("--engine-timeout", type=float, default=120.0, help="seconds to allow for engine startup (NNUE load); 0 = wait forever")
    common(p)

    p = sub.add_parser("evaldb", help="convert lichess_db_eval.jsonl(.zst); accepts a local path or an http(s) URL streamed on the fly")
    p.add_argument("jsonl", help=f"local .jsonl(.zst) path, or a URL (default source: {EVAL_DB_URL})")
    p.add_argument("out")
    p.add_argument("--min-depth", type=int, default=20)
    p.add_argument("--margin", type=int, default=30)
    p.add_argument("--mate-margin", type=int, default=0, help="when the best move mates, also accept mates this many plies slower (0 = only the fastest)")
    common(p)

    a = ap.parse_args()
    if getattr(a, "engine_timeout", None) == 0:
        a.engine_timeout = None
    {"puzzles": cmd_puzzles, "stockfish": cmd_stockfish, "evaldb": cmd_evaldb}[a.mode](a)


if __name__ == "__main__":
    main()
