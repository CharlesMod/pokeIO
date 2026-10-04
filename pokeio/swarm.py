"""Swarm: population-level Go-Explore.

Every env reports ``info["frontier"] = {score, state}`` when its frontier score
(``Progress.state_score`` or milestone count) rises by ``min_delta`` over its own
best. When a report beats the global best, a ``fraction`` of the population is
moved to that save state and adopts it as their restart point.

puffer called this the single biggest stabilizer ("plagued us for months"
without it). Differences from puffer's v2:
  * The trigger is a spec-defined score, not a hand-curated "required events" list.
  * ``fraction`` < 1 keeps part of the population off the frontier for diversity.
  * Migrated envs get a done=2 marker, so their LSTM state is zeroed and the
    unexecuted action is masked out of the loss (puffer kept stale hidden state).
  * Every frontier state is archived to disk, and the best one is restored on resume.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from pokeio.spec import Swarm as SwarmSpec


class SwarmCoordinator:
    def __init__(self, cfg: SwarmSpec, num_envs: int, archive_dir: Path, seed: int = 0):
        self.cfg = cfg
        self.num_envs = num_envs
        self.dir = Path(archive_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.best_score = float("-inf")
        self.best_state: bytes | None = None
        self.candidate: tuple[float, bytes, int] | None = None  # (score, state, env_id)
        self.last_migration_step = -(10**18)
        self.migrations = 0
        self.last_src = -1

    def observe(self, info: dict) -> None:
        f = info.get("frontier")
        if not f or not self.cfg.enabled:
            return
        score = float(f["score"])
        if score < self.best_score + self.cfg.min_delta:  # -inf + delta stays -inf
            return
        if self.candidate is None or score > self.candidate[0]:
            self.candidate = (score, f["state"], int(info["env_id"]))

    def maybe_migrate(self, global_step: int) -> dict[int, bytes]:
        """Return {env_id: state} assignments to load, or {} if nothing to do."""
        if self.candidate is None:
            return {}
        if global_step - self.last_migration_step < self.cfg.min_interval_steps:
            return {}
        score, state, src = self.candidate
        self.candidate = None
        self.best_score, self.best_state = score, state
        self.last_migration_step = global_step
        self.migrations += 1
        self.last_src = src
        self._archive(score, state, src, global_step)
        others = [i for i in range(self.num_envs) if i != src]
        k = round(self.cfg.fraction * len(others))
        targets = self.rng.sample(others, k) if k < len(others) else others
        return {i: state for i in targets}

    def _archive(self, score: float, state: bytes, src: int, step: int) -> None:
        name = f"frontier_{self.migrations:04d}_score{score:.2f}"
        (self.dir / f"{name}.state").write_bytes(state)
        with open(self.dir / "index.jsonl", "a") as f:
            f.write(json.dumps({"file": f"{name}.state", "score": score, "env_id": src,
                                "global_step": step, "time": time.time()}) + "\n")

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> dict:
        return {"best_score": self.best_score, "best_state": self.best_state,
                "migrations": self.migrations}

    def load_state_dict(self, d: dict) -> None:
        self.best_score = d.get("best_score", float("-inf"))
        self.best_state = d.get("best_state")
        self.migrations = d.get("migrations", 0)

    def resume_assignments(self) -> dict[int, bytes]:
        """On resume, put the whole population back on the stored frontier."""
        if self.best_state is None:
            return {}
        return {i: self.best_state for i in range(self.num_envs)}
