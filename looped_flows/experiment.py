"""Glue between configs, tasks, models and evaluation."""

from __future__ import annotations

import dataclasses
import json
import os
import time

import torch

from .config import ExperimentConfig
from .data import Task, build_task
from .evaluate import evaluate_exact_match, evaluate_multi_solution
from .flow import SampleConfig
from .model import DenoiserConfig, LoopedFlowDenoiser
from .train import Trainer


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_task_from_config(cfg: ExperimentConfig) -> Task:
    kwargs = dict(cfg.task)
    name = kwargs.pop("name")
    return build_task(name, **kwargs)


def build_model_config(task: Task, cfg: ExperimentConfig) -> DenoiserConfig:
    m = cfg.model
    spec = task.spec
    return DenoiserConfig(
        vocab_size=spec.vocab_size,
        seq_len=spec.seq_len,
        num_puzzle_identifiers=spec.num_puzzle_identifiers,
        problem_tokens=spec.problem_tokens,
        hidden_size=m.hidden_size,
        num_heads=m.num_heads,
        expansion=m.expansion,
        num_layers=m.num_layers,
        H_cycles=m.H_cycles,
        L_cycles=m.L_cycles if m.L_cycles is not None else spec.L_cycles,
        mixer=m.mixer if m.mixer is not None else spec.mixer,
        puzzle_emb_ndim=m.puzzle_emb_ndim,
        time_conditioning=m.time_conditioning,
        forward_dtype=m.forward_dtype,
    )


def resolve_sample_config(task: Task, cfg: ExperimentConfig) -> SampleConfig:
    s = dataclasses.replace(cfg.sample)
    s.sigma = cfg.train.sigma if cfg.train.sigma is not None else task.spec.default_sigma
    s.prob_kind = cfg.train.loss_kind
    return s


def run_evaluation(model: LoopedFlowDenoiser, task: Task, cfg: ExperimentConfig, device, sample_cfg: SampleConfig | None = None, limit: int | None = None) -> dict:
    sample_cfg = sample_cfg or resolve_sample_config(task, cfg)
    limit = cfg.eval.limit if limit is None else limit
    t0 = time.time()
    if hasattr(task, "evaluate_samples"):
        out = evaluate_multi_solution(model, task, sample_cfg, cfg.eval.batch_size, device, num_samples=cfg.eval.num_samples, limit=limit)
    elif hasattr(task, "evaluate"):
        out = task.evaluate(model, sample_cfg, cfg.eval.batch_size, device, limit=limit, num_trajectories=cfg.eval.num_trajectories)
    else:
        out = evaluate_exact_match(model, task, sample_cfg, cfg.eval.batch_size, device, limit=limit, num_trajectories=cfg.eval.num_trajectories)
    out["eval_seconds"] = time.time() - t0
    out["n_steps"] = sample_cfg.n_steps
    out["gamma"] = sample_cfg.gamma
    out["H_cycles"] = sample_cfg.H_cycles if sample_cfg.H_cycles is not None else model.cfg.H_cycles
    return out


def save_checkpoint(trainer: Trainer, cfg: ExperimentConfig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = trainer.state_dict()
    sd["experiment_config"] = dataclasses.asdict(cfg)
    torch.save(sd, path)


def load_model_from_checkpoint(path: str, device, use_ema: bool = True) -> tuple[LoopedFlowDenoiser, dict]:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model = LoopedFlowDenoiser(DenoiserConfig(**sd["model_config"]))
    model.load_state_dict(sd["ema"] if use_ema else sd["model"])
    return model.to(device).eval(), sd


def log_json(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
