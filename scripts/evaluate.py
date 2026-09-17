"""Evaluate a checkpoint.  Usage:
  uv run python scripts/evaluate.py runs/sudoku/checkpoint.pt [sample.n_steps=128 sample.gamma=5 eval.limit=1000 ...]
Use --analysis to also report recurrence convergence statistics (single-solution tasks)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from looped_flows.config import load_config
from looped_flows.evaluate import convergence_analysis
from looped_flows.experiment import build_task_from_config, load_model_from_checkpoint, pick_device, resolve_sample_config, run_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--raw-weights", action="store_true", help="use the raw weights instead of the EMA")
    ap.add_argument("--analysis", action="store_true")
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_path = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    cfg = load_config(cfg_path if os.path.exists(cfg_path) else None, args.overrides)
    device = pick_device(cfg.device)
    task = build_task_from_config(cfg)
    model, _ = load_model_from_checkpoint(args.checkpoint, device, use_ema=not args.raw_weights)
    sample_cfg = resolve_sample_config(task, cfg)
    # The sampler draws its noise from the global generator, which is seeded from OS entropy at
    # process start, so the same checkpoint scored twice gave different numbers. Pin it here (in
    # the script, not in run_evaluation, so evaluating mid-training does not disturb the
    # training RNG stream); pass train.seed=N to draw a different set of trajectories.
    torch.manual_seed(cfg.train.seed)
    metrics = run_evaluation(model, task, cfg, device, sample_cfg)
    if args.analysis:
        metrics["analysis"] = convergence_analysis(model, task, sample_cfg, cfg.eval.batch_size, device, limit=cfg.eval.limit)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
