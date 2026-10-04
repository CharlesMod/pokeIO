"""Run configuration (trainer/vec/policy), separate from the game spec.

YAML files under configs/ map onto these dataclasses; any value can be
overridden on the command line with ``--set section.key=value``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VecConfig:
    num_workers: int = 48  # processes; 0 = in-process SerialVec
    envs_per_worker: int = 6
    batch_workers: int = 24  # workers returned per recv (half = double-buffering)
    pin_cpus: bool = False
    pin_offset: int = 0


@dataclass
class PolicyConfig:
    # (out_channels, kernel, stride) per conv; puffer's tile-sized stack.
    conv: list[list[int]] = dc_field(default_factory=lambda: [[32, 8, 2], [64, 4, 2], [64, 3, 2]])
    bits_hidden: int = 128
    hidden: int = 512
    layer_norm: bool = True


@dataclass
class TrainConfig:
    total_steps: int = 2_000_000_000
    rollout_len: int = 128  # steps per env per update (multiple of bptt)
    bptt: int = 16
    minibatch_seqs: int = 128  # sequences of length bptt per minibatch (=2048 steps)
    epochs: int = 3
    lr: float = 2.0e-4
    anneal_lr: bool = False
    adam_eps: float = 1.0e-5
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip: float = 0.1
    vf_clip: float = 0.1  # in normalized-value units; 0 disables
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.0  # 0 disables early stop
    value_norm: bool = True
    amp: bool = False  # fp16 autocast (P100 has fast fp16, no bf16)
    compile: bool = False  # torch.compile needs sm_70+ for Inductor; off for Pascal
    device: str = "cuda"
    seed: int = 1
    run_dir: str = "runs"
    run_name: str = ""
    checkpoint_every: int = 50  # updates
    log_every: int = 1  # updates
    resume: str = ""  # path to a run dir or checkpoint file


@dataclass
class DashConfig:
    enabled: bool = True
    host: str = "0.0.0.0"  # reachable from other machines (e.g. over Tailscale)
    port: int = 8600
    wall_size: int = 16  # agents shown live on the wall
    hero_history: int = 240  # focused-agent frames buffered for replay
    saliency_every_s: float = 2.0


@dataclass
class Config:
    spec: str = "games/pokemon_yellow/spec.yaml"
    vec: VecConfig = dc_field(default_factory=VecConfig)
    policy: PolicyConfig = dc_field(default_factory=PolicyConfig)
    train: TrainConfig = dc_field(default_factory=TrainConfig)
    dash: DashConfig = dc_field(default_factory=DashConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(old: Any, value: str) -> Any:
    if isinstance(old, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(float(value)) if "e" in value.lower() else int(value)
    if isinstance(old, float):
        return float(value)
    if isinstance(old, (list, dict)):
        return yaml.safe_load(value)
    return value


def _merge(obj: Any, data: dict[str, Any], where: str = "") -> None:
    names = {f.name for f in fields(obj)}
    for k, v in (data or {}).items():
        if k not in names:
            raise KeyError(f"unknown config key {where}{k}")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _merge(cur, v, f"{where}{k}.")
        elif isinstance(v, str) and not isinstance(cur, str):
            setattr(obj, k, _coerce(cur, v))  # e.g. YAML reads `lr: 2e-4` as a string
        elif isinstance(cur, float) and isinstance(v, int) and not isinstance(v, bool):
            setattr(obj, k, float(v))
        else:
            setattr(obj, k, v)


def load_config(paths: list[str | Path] | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    for p in paths or []:
        with open(p) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        key, _, value = ov.partition("=")
        obj: Any = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"unknown config key {key}")
        setattr(obj, parts[-1], _coerce(getattr(obj, parts[-1]), value))
    return cfg
