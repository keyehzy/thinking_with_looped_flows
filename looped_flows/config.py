"""Experiment configuration: YAML files with dotted-key command-line overrides."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml

from .flow import SampleConfig
from .train import TrainConfig


@dataclass
class EvalConfig:
    every: int = 10_000
    batch_size: int = 512
    limit: int | None = None
    num_trajectories: int = 1
    num_samples: int = 20  # multi-solution tasks
    save_dir: str = "runs/default"


@dataclass
class ModelOverrides:
    """Fields of DenoiserConfig that a config file may set; the rest come from the task."""

    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    num_layers: int = 2
    H_cycles: int = 3
    L_cycles: int | None = None
    mixer: str | None = None
    puzzle_emb_ndim: int = 512
    time_conditioning: bool = True
    forward_dtype: str = "bfloat16"
    compile: bool = False  # torch.compile the shared core (and the carry update in training)


@dataclass
class ExperimentConfig:
    task: dict[str, Any] = field(default_factory=lambda: {"name": "sudoku"})
    model: ModelOverrides = field(default_factory=ModelOverrides)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "auto"


def _coerce(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _from_dict(cls, d: dict):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if dataclasses.is_dataclass(f.type) if isinstance(f.type, type) else False:
            v = _from_dict(f.type, v)
        kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | None, overrides: list[str] = ()) -> ExperimentConfig:
    raw: dict = {}
    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for ov in overrides:
        key, _, value = ov.partition("=")
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    cfg = ExperimentConfig(
        task=raw.get("task", {"name": "sudoku"}),
        model=_from_dict(ModelOverrides, raw.get("model", {})),
        train=_from_dict(TrainConfig, raw.get("train", {})),
        sample=_from_dict(SampleConfig, raw.get("sample", {})),
        eval=_from_dict(EvalConfig, raw.get("eval", {})),
        device=raw.get("device", "auto"),
    )
    if isinstance(cfg.train.betas, list):
        cfg.train.betas = tuple(cfg.train.betas)
    return cfg


def to_dict(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)
