#!/usr/bin/env bash
# Slurm template for the chess runs. Submit from the repository root:
#   sbatch scripts/train_chess.sbatch configs/chess_mate.yaml
#   sbatch scripts/train_chess.sbatch configs/chess_stockfish.yaml --init-from runs/chess_mate/checkpoint.pt
# Extra "section.key=value" overrides are passed through to scripts/train.py.
# Adjust the partition / account / time to the cluster; one GPU is enough (the trainer is single-process).
#SBATCH --job-name=looped-chess
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm-%j.out
set -euo pipefail

CONFIG=$1
shift
RUN_NAME=$(basename "$CONFIG" .yaml)
SAVE_DIR=${SAVE_DIR:-runs/$RUN_NAME}
mkdir -p "$SAVE_DIR"

# Resume automatically when a checkpoint exists (e.g. after a preemption / requeue).
RESUME=()
if [ -f "$SAVE_DIR/checkpoint.pt" ] && [[ "$*" != *"--init-from"* ]]; then
  RESUME=(--resume "$SAVE_DIR/checkpoint.pt")
fi

uv sync --frozen
uv run python scripts/train.py "$CONFIG" eval.save_dir="$SAVE_DIR" "${RESUME[@]}" "$@"
