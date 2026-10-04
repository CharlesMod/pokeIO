"""Held-out evaluation: milestone reach rates from the start state(s).

pokemonred_puffer had no held-out eval; progress was read off training curves
where the swarm teleports agents forward. Here every eval episode starts
from the spec's start states with the swarm and hard resets off, so the numbers
measure what the *policy* can do from scratch:

  * per milestone: fraction of episodes reaching it (+ bootstrap 95% CI) and
    median steps to reach it (Pleines et al. 2025 report this the same way);
  * mean distinct cells / rooms / return;
  * optional ``blind`` mode zeroes the screen channel. If milestone rates don't
    drop, the policy isn't using vision (the failure mode of pokeIO v1).

With ``workers > 0`` the env count is ``workers * (episodes // workers)``.
Cells/rooms are counted from each step's ``aux`` (room, x, y), so they work
for in-process and worker envs alike; specs without a position report none.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from pokeio.config import VecConfig
from pokeio.env import POLICY_KEYS
from pokeio.spec import GameSpec
from pokeio.train import load_policy
from pokeio.vec import make_vec


def _bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@torch.no_grad()
def evaluate(
    spec: GameSpec,
    checkpoint: str | Path,
    episodes: int = 32,
    steps: int = 20000,
    greedy: bool = False,
    workers: int = 0,
    blind: bool = False,
    device: str = "cpu",
    seed: int = 12345,
) -> dict:
    spec = copy.deepcopy(spec)
    spec.swarm.enabled = False
    spec.episode.hard_reset_steps = 0
    spec.episode.stall_reset_steps = 0
    spec.episode.memory_reset_steps = 10**12  # one memory for the whole eval episode
    policy, _ = load_policy(checkpoint, device)

    if workers > 0:
        k = max(1, episodes // workers)
        vc = VecConfig(num_workers=workers, envs_per_worker=k, batch_workers=workers)
    else:
        vc = VecConfig(num_workers=0, envs_per_worker=episodes)
    vec = make_vec(spec, vc, seed=seed)
    n = vec.num_envs
    reached: list[dict[str, int]] = [dict() for _ in range(n)]
    returns = np.zeros(n)
    seen: list[set[tuple[int, int, int]]] = [set() for _ in range(n)]
    dev = torch.device(device)
    H, C = policy.initial_state(n, dev)
    starts = torch.ones(n, device=dev)
    vec.reset()
    try:
        for _ in range(steps + 1):
            b = vec.recv()
            ids = b.env_ids
            returns[ids] += b.rewards
            for info in b.infos:
                for m, s in info.get("milestones", {}).items():
                    reached[info["env_id"]].setdefault(m, s)
            for i, aux in zip(ids, b.obs["aux"]):
                if aux[0] >= 0:
                    seen[i].add((int(aux[0]), int(aux[1]), int(aux[2])))
            obs = {k: torch.from_numpy(b.obs[k]).to(dev) for k in POLICY_KEYS}
            if blind:
                obs["pixels"][:, 0] = 0
            idt = torch.from_numpy(ids).to(dev)
            starts[idt] = torch.maximum(starts[idt], torch.from_numpy((b.dones != 0).astype(np.float32)).to(dev))
            logits, _, (h, c) = policy.step(obs, (H[idt], C[idt]), starts[idt])
            H[idt], C[idt] = h, c
            starts[idt] = 0
            a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            vec.send(a.cpu().numpy().astype(np.int32), ids)
    finally:
        vec.close()

    names = [m.name for m in spec.milestones]
    out = {"episodes": n, "steps": steps, "greedy": greedy, "blind": blind, "milestones": {}}
    for m in names:
        hit = np.array([m in r for r in reached], dtype=float)
        when = [r[m] for r in reached if m in r]
        lo, hi = _bootstrap_ci(hit)
        out["milestones"][m] = {
            "rate": float(hit.mean()),
            "ci95": [lo, hi],
            "median_steps": float(np.median(when)) if when else None,
        }
    out["return_mean"] = float(returns.mean())
    if any(seen):
        out["cells_mean"] = float(np.mean([len(c) for c in seen]))
        out["rooms_mean"] = float(np.mean([len({r for r, _, _ in c}) for c in seen]))
    return out


def print_report(r: dict) -> None:
    print(f"episodes={r['episodes']} steps={r['steps']} greedy={r['greedy']} blind={r['blind']}")
    for m, d in r["milestones"].items():
        ms = "-" if d["median_steps"] is None else f"{d['median_steps']:.0f}"
        print(f"  {m:28s} {d['rate']*100:5.1f}%  [{d['ci95'][0]*100:4.0f},{d['ci95'][1]*100:4.0f}]  median {ms}")
    line = f"  return {r['return_mean']:.2f}"
    if "cells_mean" in r:
        line += f"  cells {r['cells_mean']:.0f}  rooms {r['rooms_mean']:.1f}"
    print(line)


def save_report(r: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(r, f, indent=2)
