#!/usr/bin/env bash
# Keep a training run alive on a single box: relaunch from the run's checkpoint if the
# process dies before train.total_steps. The Slurm template auto-resumes on requeue;
# this is the equivalent for a plain machine, where an external kill (session teardown,
# OOM reaper) otherwise ends the run silently -- a stage 2 run was lost that way at step
# 20700 of 300000 and went unnoticed for 15 hours.
#
#   scripts/supervise_train.sh configs/chess_stockfish.yaml runs/my_run task.target=any
#
# Arg 1 is the config, arg 2 the run directory (also used as eval.save_dir), and any
# further args are passed to scripts/train.py as dotted overrides. Set INIT_FROM to
# warm-start the first attempt from another run's EMA weights. Progress, restarts and
# exit codes are appended to <run dir>/supervisor.log.
set -uo pipefail
CFG=${1:?usage: supervise_train.sh <config> <run-dir> [overrides...]}
DIR=${2:?usage: supervise_train.sh <config> <run-dir> [overrides...]}
shift 2
mkdir -p "$DIR"
TOTAL=$(uv run python -c "
import sys,yaml
c=yaml.safe_load(open('$CFG')) or {}
ov=[a for a in sys.argv[1:] if a.startswith('train.total_steps=')]
print(int(ov[-1].split('=')[1]) if ov else c.get('train',{}).get('total_steps',100000))" "$@")

for attempt in $(seq 1 200); do
  STEP=$(python3 -c "
import json,os
p='$DIR/train_log.jsonl'
print(json.loads(open(p).read().strip().split(chr(10))[-1])['step'] if os.path.exists(p) and os.path.getsize(p) else 0)" 2>/dev/null || echo 0)
  if [ "${STEP:-0}" -ge "$TOTAL" ]; then
    echo "[supervisor] $(date -Is) reached step $STEP >= $TOTAL, done" >> "$DIR/supervisor.log"; break
  fi
  if [ -f "$DIR/checkpoint.pt" ]; then
    ARGS=(--resume "$DIR/checkpoint.pt")
  elif [ -n "${INIT_FROM:-}" ]; then
    ARGS=(--init-from "$INIT_FROM")
  else
    ARGS=()
  fi
  echo "[supervisor] $(date -Is) attempt $attempt, last step ${STEP:-0}, args: ${ARGS[*]-none}" >> "$DIR/supervisor.log"
  uv run python scripts/train.py "$CFG" "eval.save_dir=$DIR" "$@" "${ARGS[@]+"${ARGS[@]}"}" >> "$DIR/train.out" 2>&1
  echo "[supervisor] $(date -Is) exited rc=$?" >> "$DIR/supervisor.log"
  sleep 10
done
