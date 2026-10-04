"""LiveHub: everything the training wall shows, fed directly by the trainer.

Cost discipline: per step the trainer hands over the batch it already has.
The hub copies packed frames only for the ~16 agents on the wall, updates a
vectorized population heatmap, and appends one record for the focused agent.
Everything heavier (JSON encoding, PNG-free base64) happens on the server
thread when the browser polls.

Nothing here is mocked: every number on the page traces back to a trainer value.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from pokeio.env import ObsSpec
from pokeio.spec import GameSpec

_MAX_ROOM = 256


def _b64(a: np.ndarray | bytes) -> str:
    return base64.b64encode(a if isinstance(a, bytes) else np.ascontiguousarray(a).tobytes()).decode()


class LiveHub:
    def __init__(self, spec: GameSpec, obs_spec: ObsSpec, num_envs: int, run_dir: Path,
                 wall_size: int = 16, hero_history: int = 240, saliency_every_s: float = 2.0):
        self.spec, self.obs_spec, self.N = spec, obs_spec, num_envs
        self.run_dir = Path(run_dir)
        self.lock = threading.Lock()
        self.t0 = time.time()
        n = min(wall_size, num_envs)
        self.wall_ids = np.unique(np.linspace(0, num_envs - 1, n).round().astype(np.int64))
        self.wall_set = set(self.wall_ids.tolist())
        self.frames: dict[int, dict] = {}
        self.hero = int(self.wall_ids[0])
        self.follow_frontier = True
        self.hero_ring: deque[dict] = deque(maxlen=hero_history)
        self.hero_seq = 0
        self.saliency: dict | None = None
        self.saliency_every_s = saliency_every_s
        self._last_saliency = 0.0
        self.heat: dict[int, np.ndarray] = {}
        self.room_first: dict[int, tuple[int, int]] = {}  # room -> (global_step, env)
        self.events: deque[dict] = deque(maxlen=300)
        self.milestones: dict[str, dict] = {
            m.name: {"first_step": None, "first_time": None, "env": None, "hits": 0}
            for m in spec.milestones
        }
        self.history: list[dict] = []
        self.frontier: list[dict] = []
        self._frontier_frames: dict[int, np.ndarray] = {}
        self.scores = np.zeros(num_envs, dtype=np.float32)
        self.rooms_now = np.full(num_envs, -1, dtype=np.int32)
        self.global_step = 0
        self.update = 0
        self.sps = 0.0
        self.stop_requested = False
        self.status = "starting"
        self.names = {int(k): str(v) for k, v in (spec.display.get("room_names") or {}).items()}
        self.event("start", f"run started: {len(self.wall_ids)} agents on the wall, {num_envs} in the population")

    # ------------------------------------------------------------------ helpers
    def room_name(self, r: int) -> str:
        return self.names.get(int(r), f"room {int(r)}")

    def event(self, kind: str, text: str, env: int | None = None) -> None:
        self.events.append({"t": time.time() - self.t0, "step": self.global_step, "kind": kind,
                            "text": text, "env": env})

    # ------------------------------------------------------------------ trainer hooks
    def on_step(self, ids: np.ndarray, obs: dict, actions: np.ndarray, probs: np.ndarray,
                values: np.ndarray, rewards: np.ndarray, dones: np.ndarray, global_step: int) -> None:
        aux = obs["aux"]
        with self.lock:
            self.global_step = global_step
            self.scores[ids] = aux[:, 4] / 100.0
            self.rooms_now[ids] = aux[:, 0]
            # population heatmap of visited cells
            live = (aux[:, 0] >= 0) & (aux[:, 3] == 0)
            if live.any():
                a = aux[live]
                for room in np.unique(a[:, 0]):
                    sel = a[a[:, 0] == room]
                    xs = np.clip(sel[:, 1], 0, _MAX_ROOM - 1)
                    ys = np.clip(sel[:, 2], 0, _MAX_ROOM - 1)
                    g = self.heat.get(int(room))
                    if g is None:
                        g = self.heat[int(room)] = np.zeros((32, 32), dtype=np.uint32)
                        env = int(ids[live][a[:, 0] == room][0])
                        self.room_first[int(room)] = (global_step, env)
                        if len(self.heat) > 1:
                            self.event("room", f"new area discovered: {self.room_name(room)}", env)
                    h, w = g.shape
                    need_h, need_w = int(ys.max()) + 1, int(xs.max()) + 1
                    if need_h > h or need_w > w:
                        ng = np.zeros((max(h, min(_MAX_ROOM, 1 << (need_h - 1).bit_length())),
                                       max(w, min(_MAX_ROOM, 1 << (need_w - 1).bit_length()))), np.uint32)
                        ng[:h, :w] = g
                        g = self.heat[int(room)] = ng
                    np.add.at(g, (ys, xs), 1)
            # follow the frontier: focus the highest-scoring agent
            if self.follow_frontier:
                best = int(np.argmax(self.scores))
                if self.scores[best] > self.scores[self.hero]:  # hysteresis: no flicker on ties
                    self.hero = best
                    self.hero_ring.clear()
            for j, e in enumerate(ids.tolist()):
                if e in self.wall_set or e == self.hero:
                    rec = {
                        "pix": obs["pixels"][j].tobytes(),
                        "a": int(actions[j]),
                        "p": [round(float(x), 3) for x in probs[j]],
                        "v": float(values[j]),
                        "r": float(rewards[j]),
                        "done": int(dones[j]),
                        "aux": aux[j].tolist(),
                    }
                    if e in self.wall_set:
                        self.frames[e] = rec
                    if e == self.hero:
                        self.hero_seq += 1
                        self.hero_ring.append({"seq": self.hero_seq, "env": e, **rec})

    def want_saliency(self, ids: np.ndarray) -> int | None:
        """Index into ``ids`` of the focused agent if a saliency map is due."""
        if time.time() - self._last_saliency < self.saliency_every_s:
            return None
        hit = np.nonzero(ids == self.hero)[0]
        return int(hit[0]) if len(hit) else None

    def on_saliency(self, env: int, sal: np.ndarray) -> None:
        """``sal``: (C, H, W) float magnitudes for one agent."""
        s = sal.sum(axis=0)
        s = s / (np.percentile(s, 99.5) + 1e-8)
        q = (np.clip(s, 0, 1) * 255).astype(np.uint8)
        with self.lock:
            self._last_saliency = time.time()
            self.saliency = {"env": env, "seq": self.hero_seq, "h": q.shape[0], "w": q.shape[1], "data": _b64(q)}

    def on_infos(self, ids: np.ndarray, obs: dict, infos: list[dict]) -> None:
        if not infos:
            return
        with self.lock:
            pos = {int(e): j for j, e in enumerate(ids.tolist())}
            for info in infos:
                e = int(info["env_id"])
                for m in info.get("milestones", {}):
                    rec = self.milestones.setdefault(m, {"first_step": None, "first_time": None, "env": None, "hits": 0})
                    rec["hits"] += 1
                    if rec["first_step"] is None:
                        rec.update(first_step=self.global_step, first_time=time.time() - self.t0, env=e)
                        self.event("milestone", f"FIRST EVER: {m.replace('_', ' ')}", e)
                    if e in self.frames:
                        self.frames[e]["flash"] = m
                if "frontier" in info and e in pos:
                    self._frontier_frames[e] = obs["pixels"][pos[e]].copy()

    def on_migration(self, score: float, src: int, n_moved: int) -> None:
        with self.lock:
            pix = self._frontier_frames.get(src)
            self.frontier.append({"step": self.global_step, "t": time.time() - self.t0, "score": score,
                                  "env": src, "moved": n_moved,
                                  "room": self.room_name(self.rooms_now[src]) if self.rooms_now[src] >= 0 else "",
                                  "pix": _b64(pix) if pix is not None else None})
            self.frontier = self.frontier[-48:]
            self.event("swarm", f"SWARM: {n_moved} agents jump to agent {src}'s frontier (score {score:.1f})", src)

    def on_update(self, update: int, global_step: int, sps: float, data: dict, act_counts: list[int]) -> None:
        with self.lock:
            self.update, self.global_step, self.sps = update, global_step, sps
            self.status = "training"
            rec = {"update": update, "step": global_step, "t": time.time() - self.t0, "sps": sps,
                   "actions": act_counts}
            rec.update({k: v for k, v in data.items() if isinstance(v, (int, float)) and np.isfinite(v)})
            self.history.append(rec)
            if len(self.history) > 4000:  # thin the old half, keep recent detail
                self.history = self.history[:2000:2] + self.history[2000:]

    def on_checkpoint(self, path: str) -> None:
        with self.lock:
            self.event("checkpoint", f"checkpoint saved: {Path(path).name}")

    # ------------------------------------------------------------------ controls
    def focus(self, env: int | None, follow: bool | None) -> None:
        with self.lock:
            if follow is not None:
                self.follow_frontier = follow
            if env is not None and 0 <= env < self.N:
                self.hero = env
                self.follow_frontier = False
                self.hero_ring.clear()

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.status = "stopping"
            self.event("stop", "stop requested from the wall: checkpointing and shutting down")

    # ------------------------------------------------------------------ snapshots
    def meta(self) -> dict:
        os_ = self.obs_spec
        d = self.spec.display
        return {
            "title": d.get("title", self.spec.name),
            "palette": d.get("palette", ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]),
            "fps": float(d.get("frames_per_second", 60.0)),
            "frames_per_action": self.spec.controls.frames_per_action,
            "buttons": self.spec.controls.buttons,
            "levels": os_.levels,
            "channels": os_.pixels[0],
            "h": os_.pixels[1],
            "w": os_.pixel_width,
            "bpp": os_.pixel_bpp,
            "num_envs": self.N,
            "wall_ids": self.wall_ids.tolist(),
            "milestone_order": [m.name for m in self.spec.milestones],
            "run": self.run_dir.name,
        }

    def snapshot_state(self) -> dict:
        with self.lock:
            heat = []
            for room, g in self.heat.items():
                nz = np.argwhere(g > 0)
                if len(nz) == 0:
                    continue
                (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1
                crop = g[y0:y1, x0:x1].astype(np.float64)
                img = (np.log1p(crop) / max(np.log1p(crop.max()), 1e-9) * 255).astype(np.uint8)
                step, env = self.room_first.get(room, (0, -1))
                heat.append({"room": room, "name": self.room_name(room), "w": int(x1 - x0), "h": int(y1 - y0),
                             "cells": int((g > 0).sum()), "visits": int(g.sum()), "first_step": step,
                             "occupants": int((self.rooms_now == room).sum()), "data": _b64(img)})
            heat.sort(key=lambda r: r["first_step"])
            order = np.argsort(-self.scores)[:10]
            leaders = [{"env": int(e), "score": float(self.scores[e]),
                        "room": self.room_name(self.rooms_now[e]) if self.rooms_now[e] >= 0 else "?"}
                       for e in order]
            return {
                "meta": self.meta(),
                "status": self.status,
                "uptime": time.time() - self.t0,
                "global_step": self.global_step,
                "update": self.update,
                "sps": self.sps,
                "hero": self.hero,
                "follow": self.follow_frontier,
                "history": self.history[-1500:],
                "milestones": self.milestones,
                "events": list(self.events)[-80:],
                "heat": heat,
                "total_cells": int(sum(int((g > 0).sum()) for g in self.heat.values())),
                "frontier": self.frontier,
                "leaders": leaders,
            }

    def snapshot_live(self, since: int = 0) -> dict:
        with self.lock:
            wall = {}
            for e, f in self.frames.items():
                wall[e] = {**{k: v for k, v in f.items() if k != "pix"}, "pix": _b64(f["pix"])}
                f.pop("flash", None)
            hero = [{**{k: v for k, v in r.items() if k != "pix"}, "pix": _b64(r["pix"])}
                    for r in self.hero_ring if r["seq"] > since]
            return {"global_step": self.global_step, "sps": self.sps, "hero": self.hero,
                    "follow": self.follow_frontier, "wall": wall, "hero_frames": hero[-120:],
                    "saliency": self.saliency, "status": self.status}

    def save(self) -> None:
        """Persist the state snapshot so `python -m pokeio dash --run` can show it offline."""
        snap = self.snapshot_state()
        tmp = self.run_dir / "dash_state.json.tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f)
        tmp.replace(self.run_dir / "dash_state.json")
