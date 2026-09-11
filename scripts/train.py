"""Train a looped flow.  Usage: uv run python scripts/train.py configs/sudoku.yaml [key=value ...]"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from looped_flows.config import load_config
from looped_flows.experiment import build_model_config, build_task_from_config, log_json, pick_device, run_evaluation, save_checkpoint
from looped_flows.model import LoopedFlowDenoiser
from looped_flows.train import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.total_steps=1000")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = pick_device(cfg.device)
    save_dir = cfg.eval.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f)

    task = build_task_from_config(cfg)
    model = LoopedFlowDenoiser(build_model_config(task, cfg))
    print(f"task={task.spec.name} seq_len={task.spec.seq_len} vocab={task.spec.vocab_size} params={model.num_parameters()/1e6:.2f}M device={device}")
    trainer = Trainer(task, model, cfg.train, device)
    if args.resume:
        trainer.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False))
        print(f"resumed from {args.resume} at step {trainer.step}")

    t0 = time.time()
    while trainer.step < cfg.train.total_steps:
        trainer.train_step()
        step = trainer.step
        if step % cfg.train.log_every == 0:
            stats = trainer.stats.flush()
            elapsed = time.time() - t0
            print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()) + f" ({elapsed/step:.3f}s/step)", flush=True)
            log_json(os.path.join(save_dir, "train_log.jsonl"), {"step": step, **stats, "elapsed": elapsed})
        if cfg.eval.every > 0 and (step % cfg.eval.every == 0 or step == cfg.train.total_steps):
            save_checkpoint(trainer, cfg, os.path.join(save_dir, "checkpoint.pt"))
            metrics = run_evaluation(trainer.ema_model(), task, cfg, device)
            print(f"[eval step {step}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)
            log_json(os.path.join(save_dir, "eval_log.jsonl"), {"step": step, **metrics})
    save_checkpoint(trainer, cfg, os.path.join(save_dir, "checkpoint.pt"))


if __name__ == "__main__":
    main()
