"""Evaluation routines: exact-match accuracy (Sudoku, Maze), multi-solution
metrics (N-Queens, Graph Colouring) and recurrence-convergence analysis."""

from __future__ import annotations

import numpy as np
import torch

from .data.common import Task, batch_to
from .flow import SampleConfig, sample, sample_best_q, sample_repeated
from .losses import sequence_correct
from .model import LoopedFlowDenoiser


@torch.no_grad()
def evaluate_exact_match(model: LoopedFlowDenoiser, task: Task, cfg: SampleConfig, batch_size: int, device, limit: int | None = None, num_trajectories: int = 1, progress=None) -> dict:
    model.eval()
    correct = 0
    total = 0
    for batch in task.test_batches(batch_size, limit):
        b = batch_to(batch, device)
        res = sample(model, b, cfg) if num_trajectories == 1 else sample_best_q(model, b, cfg, num_trajectories, max_batch=batch_size)
        ok = sequence_correct(res.tokens, b["labels"])
        correct += int(ok.sum())
        total += ok.numel()
        if progress is not None:
            progress(total)
    return {"accuracy": correct / max(1, total), "num_examples": total}


@torch.no_grad()
def evaluate_multi_solution(model: LoopedFlowDenoiser, task: Task, cfg: SampleConfig, batch_size: int, device, num_samples: int = 20, limit: int | None = None, progress=None) -> dict:
    """``batch_size`` counts sequences per sampler call; each problem is sampled ``num_samples`` times."""
    model.eval()
    agg: dict[str, list] = {}
    total = 0
    problems_per_batch = max(1, batch_size // num_samples)
    for batch in task.test_batches(problems_per_batch, limit):
        b = batch_to(batch, device)
        samples = sample_repeated(model, b, cfg, num_samples, max_batch=batch_size)[0].cpu().numpy()
        metrics = task.evaluate_samples(batch, samples)
        for k, v in metrics.items():
            agg.setdefault(k, []).extend(v)
        total += samples.shape[0]
        if progress is not None:
            progress(total)
    out = {"num_examples": total}
    for k, v in agg.items():
        out[k] = float(np.sum(v)) if k == "conflicts" else float(np.mean(v))
    return out


@torch.no_grad()
def convergence_analysis(model: LoopedFlowDenoiser, task: Task, cfg: SampleConfig, batch_size: int, device, limit: int | None = None, tail: int = 8, threshold: float = 0.05) -> dict:
    """Stepwise relative residual R_i = ||z_i - z_{i-1}|| / ||z_{i-1}|| of the recurrent
    state h; a run is converged when the mean of its last ``tail`` residuals is
    below ``threshold * E[R_1]``. Failures are split into non-convergence and
    spurious attractors (converged but wrong)."""
    model.eval()
    cfg = SampleConfig(**{**cfg.__dict__, "record_states": True})
    residuals, correct = [], []
    for batch in task.test_batches(batch_size, limit):
        b = batch_to(batch, device)
        res = sample(model, b, cfg)
        residuals.append(res.residuals.cpu())
        correct.append(sequence_correct(res.tokens, b["labels"]).cpu())
    R = torch.cat(residuals)
    ok = torch.cat(correct)
    converged = R[:, -tail:].mean(1) < threshold * R[:, 0].mean()
    fail = ~ok
    n_fail = int(fail.sum())
    return {
        "accuracy": ok.float().mean().item(),
        "failure_rate": fail.float().mean().item(),
        "nonconverged_among_failures": float((fail & ~converged).sum() / max(1, n_fail)),
        "spurious_attractor_among_failures": float((fail & converged).sum() / max(1, n_fail)),
        "mean_R1": R[:, 0].mean().item(),
        "mean_R_last": R[:, -tail:].mean().item(),
    }
