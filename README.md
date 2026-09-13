# Thinking with Looped Flows

PyTorch implementation of *Thinking with Looped Flows* (Suleymanzade et al., arXiv:2609.11801):
a looped reasoning model whose recurrence is trained with local denoising objectives at
progressively decreasing noise levels, and whose inference integrates the resulting
probability flow coupled with the recurrent state.

## Paper -> code map

| Paper | Code |
| --- | --- |
| Stateful denoiser on the TRM architecture (App. A) | `looped_flows/model.py`, `looped_flows/layers.py` |
| Interpolant, time samplers (sorted / random-start) | `looped_flows/flow.py` |
| Training, Alg. 1 (stop-gradient recurrence, shared noise, ACT, pseudotargets) | `looped_flows/train.py` |
| Inference, Alg. 2 (stochastic backtracking sampler, gamma) and best-Q ensembling | `looped_flows/flow.py` |
| StableMax cross-entropy, ACT BCE | `looped_flows/losses.py` |
| Adam-atan2, signSGD puzzle embeddings, warmup/cosine, EMA | `looped_flows/optim.py` |
| Convergence / spurious-attractor analysis (Sec. 4.1) | `looped_flows/evaluate.py` |
| Sudoku-Extreme, Maze-Hard, ARC-AGI-1/2, N-Queens, Graph Colouring | `looped_flows/data/` |
| Table 5 hyperparameters | `configs/*.yaml` |
| Chess (not in the paper): mate-in-N puzzles, Stockfish-labelled positions | `looped_flows/data/chess.py`, `scripts/prepare_chess.py`, `configs/chess_*.yaml` |

The denoiser is one TRM supervision step: a 2-layer, width-512 network `F` (SwiGLU 1536,
post RMSNorm; MLP-Mixer for Sudoku, 8-head RoPE attention otherwise) applied as
`l <- F(l + h + e_t)` (m times) then `h <- F(h + l)`, three cycles, gradients through the
last cycle only. `e_t` is a bias-free projection of the noisy state `x_t` (plus an embedded
problem for ARC / graph colouring, plus a prepended puzzle-embedding token) scaled by
`sqrt(512)` and added to a `1 -> 512 -> 512` SiLU time embedding. The prediction is decoded
from `h`; the ACT head reads the first position of `h`.

Training keeps a persistent batch ("carry") exactly like TRM: every optimisation step runs one
denoiser call per slot, slots halt when the ACT head fires (or after `k = 16` steps) and are
refilled with a new problem, noise sample and time grid. Per problem the noise `x0` and the
target are shared across steps; timesteps are sorted so noise decreases along the recurrence.

## Setup

```bash
uv sync                                    # Python 3.12 + torch (MPS on Apple silicon, CUDA on Linux)
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("sapientinc/sudoku-extreme", repo_type="dataset", local_dir="data/raw/sudoku-extreme")
snapshot_download("sapientinc/maze-30x30-hard-1k", repo_type="dataset", local_dir="data/raw/maze-30x30-hard-1k")
EOF
./scripts/download_arc.sh                  # ARC-AGI-1, ARC-AGI-2, ConceptARC
uv run pytest                              # unit tests
```

N-Queens and Graph Colouring are generated on the fly (no download).

## Training and evaluation

```bash
uv run python scripts/train.py configs/sudoku.yaml
uv run python scripts/train.py configs/arc1.yaml eval.save_dir=runs/arc1_seed1 train.seed=1
uv run python scripts/evaluate.py runs/sudoku/checkpoint.pt sample.n_steps=128 sample.gamma=5 eval.num_trajectories=5
uv run python scripts/evaluate.py runs/sudoku/checkpoint.pt --analysis eval.limit=65000   # convergence statistics
```

Any config key can be overridden as `section.key=value`. Useful knobs:

* `sample.n_steps`, `sample.gamma` — inference grid and stochasticity (Table 5).
* `eval.num_trajectories=5` — best-Q ensembling. `eval.num_samples=20` — multi-solution tasks.
* `train.time_sampler` (`sorted` / `random_start`), `train.pseudotargets`, `train.sigma`.
* Ablations: `model.time_conditioning=false`, `train.use_interpolant=false`,
  `train.decreasing_noise=false`, `train.share_noise=false`, `sample.gamma=0` (ODE).
* `task.num_aug` / `task.num_test_aug` (ARC), `task.test_limit`, `eval.limit` to shrink runs.

Configs follow Table 5 (batch 768, lr 1e-4 with 2k warmup, Adam-atan2 (0.9, 0.95), grad clip 1,
EMA 0.999, k = 16, lambda = 0.5, exploration 0.1, bf16 forward). `train.total_steps` is not
specified in the paper; the defaults (50k-200k) are ballpark and should be tuned per task.

## Local test on this machine (Apple M4, MPS)

`configs/nqueens8.yaml` with `train.batch_size=128 train.total_steps=2000` trains the full
7M model at ~1.4 s/step (about 50 minutes; log in `runs/local_nqueens8.log`). Evaluated with
20 samples per puzzle on 64 test puzzles, 32 SDE steps, gamma = 5:

| step | 500 | 1000 | 1500 | 2000 |
| --- | --- | --- | --- | --- |
| coverage of valid completions | 0.0 | 0.0 | 0.016 | 0.138 |
| first-sample accuracy | 0.0 | 0.0 | 0.0 | 0.0 |

This is ~0.3% of the paper's training budget (batch 768 for hours on an H100), so it only shows
that the pipeline learns; converged numbers need the GPU runs. A 20-sample evaluation of 64
puzzles takes ~4 minutes on MPS, so keep `eval.limit` small locally.
For a quick shape/NaN check of any task:

```bash
uv run python scripts/train.py configs/maze.yaml train.total_steps=6 train.batch_size=8 eval.every=6 eval.limit=8 sample.n_steps=4
```

## GPU deployment notes

* All configs run unchanged on CUDA (`device: auto`). Batch 768 fits comfortably on a 24 GB GPU
  for Sudoku / N-Queens / Graph Colouring; Maze and ARC (900 tokens) may need `train.batch_size`
  reduced on smaller cards.
* ARC with `num_aug=1000` allocates ~500M float32 puzzle-embedding parameters (as in TRM). Full
  ARC evaluation predicts each test input under 1000 augmentations; lower `task.num_test_aug`
  for intermediate evaluations.
* The Sudoku test set has 423k puzzles; `eval.limit` controls how many are used during training.
* The trainer is single-process. For multi-GPU, wrap `Trainer.train_step` in DDP or run seeds in parallel.

## Chess

Chess is posed as "predict the board after the best move": the position is the fixed problem
(embedded like the graph-colouring adjacency matrix) and the target is the resulting board on the
same 64 positions, so no move vocabulary is needed. Positions are normalised so the side to move is
always White (Black-to-move positions are mirrored and colour-swapped); two extra tokens carry
castling rights and the en-passant file. Sequence length 66, vocabulary 39, attention mixer.
The predicted board is matched against the boards reachable by a legal move, which gives the
`legal` rate, first-sample `accuracy`, `pass_at_k` over 20 samples, `majority` (vote over legal
samples) and a per-depth breakdown `acc_mate<k>`. See `looped_flows/data/chess.py`.

Two stages share the task and datasets are built by `scripts/prepare_chess.py`:

```bash
uv add chess zstandard                                   # already in pyproject.toml
curl -L -o data/raw/lichess/lichess_db_puzzle.csv.zst https://database.lichess.org/lichess_db_puzzle.csv.zst

# Stage 1: mate-in-N from Lichess puzzles (every solver-to-move position of a mate puzzle;
# all mating moves are accepted on mate-in-1 positions). ~2.5M positions, a few minutes on 8 cores.
uv run python scripts/prepare_chess.py puzzles data/raw/lichess/lichess_db_puzzle.csv.zst data/chess/mate
uv run python scripts/train.py configs/chess_mate.yaml

# Stage 2: general positions labelled by Stockfish. Either run an engine over positions sampled
# from PGNs / a FEN list (multi-PV; moves within --margin centipawns of the best are accepted) ...
brew install stockfish   # or apt install stockfish / a cluster module
uv run python scripts/prepare_chess.py stockfish data/chess/stockfish --pgn games.pgn.zst \
    --depth 14 --multipv 4 --margin 30 --workers 32 --limit 5000000
# ... or convert Lichess' precomputed evaluations (no engine needed, depth 20+ labels). The
# database is ~22 GB compressed; pass the URL and it is streamed and decompressed on the fly,
# so with --limit only the prefix actually needed is transferred and nothing lands on disk:
uv run python scripts/prepare_chess.py evaldb \
    https://database.lichess.org/lichess_db_eval.jsonl.zst data/chess/stockfish \
    --min-depth 20 --limit 5000000
uv run python scripts/train.py configs/chess_stockfish.yaml --init-from runs/chess_mate/checkpoint.pt
```

### Slices

`--limit` counts kept positions, and `--skip` / `--shard` (on every mode) carve a source into
disjoint pieces, so a later stage can train on data the first one never saw:

```bash
# sequential cursor: the second slice continues where the first stopped
uv run python scripts/prepare_chess.py evaldb $EVAL_DB data/chess/sf_a --limit 5000000
uv run python scripts/prepare_chess.py evaldb $EVAL_DB data/chess/sf_b --skip 5000000 --limit 5000000

# or hashed shards, spread over the whole database rather than taken from the front
uv run python scripts/prepare_chess.py evaldb $EVAL_DB data/chess/sf_0 --shard 0/8 --limit 2000000
```

`--shard I/N` is salted separately from the train/test hash, so shards stay disjoint and each
still gets its own ~2% test split. Note `--limit` alone takes a **prefix** of the database, which
inherits whatever ordering the file has; `--shard` samples across all of it.

`--mate-margin` controls how much slower than the fastest mate a move may be and still count as
acceptable (default 0, i.e. only the fastest). Mates are ranked on their own scale, so a merely
winning move is never accepted in place of a mate, and when the mover is the one being mated the
acceptable moves are those that resist longest.

`task.target=any` draws the training target uniformly from the acceptable moves (as N-Queens draws
among its solutions); `task.target=best` always uses the engine's first choice. `task.augment`
mirrors files a-h on positions without castling rights (the only cheap symmetry chess has besides the
colour flip already applied by normalisation). Splits are deterministic by puzzle id / FEN hash
(`--test-fraction`, default 2%). Arrays are memory-mapped, so the Stockfish stage can use tens of
millions of positions (128 bytes each).

For a cluster, `scripts/train_chess.sbatch` wraps both stages (auto-resumes from the run's checkpoint):

```bash
sbatch scripts/train_chess.sbatch configs/chess_mate.yaml
sbatch scripts/train_chess.sbatch configs/chess_stockfish.yaml --init-from runs/chess_mate/checkpoint.pt
```

Local check on the M4 (`train.batch_size=128 train.total_steps=1500 train.warmup_steps=200`, ~1.6 s/step,
256 test positions, 5 samples, 16 SDE steps, gamma = 5; log in `runs/local_chess_mate.log`):

| step | 500 | 1000 | 1500 |
| --- | --- | --- | --- |
| mean squares wrong vs target (1 sample) | ~16 | ~2 | - |
| legal-move rate | 0.0 | 0.0 | 0.002 |
| pass@5 (acceptable move among 5 samples) | 0.0 | 0.0 | 0.008 |

At step 1000 the model copies the position and vacates the right origin square in two thirds of
the positions but rarely places the piece correctly; the first legal mating moves appear at step
1500. As with the N-Queens local run this is far below the paper's budget and only shows the
pipeline learns; converged numbers need the GPU runs.

Quick pipeline check without the full data: `--limit 20000` on the prepare step, then the usual
6-step smoke command with `task.root=data/chess/mate_small`.

## Assumptions and departures from the paper

* **Multi-solution datasets.** GRAM's exact N-Queens / Graph-Colouring files are not public here,
  so they are regenerated as described: N-Queens puzzles hide `n-3..n-1` queens from every
  solution (all subsets, de-duplicated, 90/10 split); Graph Colouring uses Erdős–Rényi graphs
  (`edge_prob` 0.35 / 0.30) filtered to be 3-colourable, 20k train / 1k test. Coverage counts
  distinct valid colourings literally (not up to colour permutation).
* **Graph colouring layout.** The problem is the flattened `n x n` adjacency matrix; the solution is
  laid out on the same grid with row `i` holding vertex `i`'s colour, decoded by row majority.
* **Sudoku / Maze / N-Queens problem encoding.** Only the projected `x_t` is fed; given entries are
  fixed to their one-hot value and the interpolant runs over the remaining entries (App. A).
* **Probabilities.** The denoiser output `x_hat` used in the flow update is the StableMax
  distribution (the one trained by the loss).
* **ARC.** Follows HRM/TRM: 30x30 canvas with EOS markers, dihedral + colour-permutation
  augmentations with per-augmentation puzzle identifiers, random translation during training,
  ConceptARC added to ARC-1 training, pass@2 by majority vote over augmentations, scored per task.
* **StableMax** is evaluated with clamped branches so a logit of exactly 1.0 (common in bf16)
  cannot produce NaN gradients.
