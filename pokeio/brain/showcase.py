"""System-1 live dashboard streaming — watch the brain train (#GUI).

Emits ``runs/<id>/live.json`` (the System-1 payload contract) so a dashboard can
show, in real time: the CHAMPION policy playing from boot (its GB screen + the
foveal percept it acts on — periphery / sharp fovea / motion + the gaze box), a
SWARM sample of the training envs, and live METRICS (frontier, reach, gate, entropy,
progress). Reuses the percept-render + atomic-writer helpers from
:mod:`pokeio.train.live` (built for the NEAT wall).

Thread-safe: the streamer holds its OWN cpu copy of the policy + a private showcase
env, and the trainer calls :meth:`sync` once per iteration to push fresh weights +
metrics under a lock. So the (background) render thread never races the optimizer,
and its inference stays off the training GPU.

live.json contract (kind="system1"):
  { iter, phase, run_id, ts,
    champion: { frame_w,frame_h, frame_b64, obs_res,obs_b64, fovea_res,fovea_b64,
                motion_b64, gaze{x01,y01,w01,h01}, action, buttons[9],
                map, map_id, progress{phi,party,level,events,badges,maps,started} },
    swarm: [ {id,w,h,b64}, ... ],
    metrics: { frontier,frontier_frac, reach,success_ema, entropy,
               gate_stochastic,gate_greedy, progress_mean, reward },
    history: { iter[], reach[], frontier_frac[], gate_stochastic[], entropy[] } }
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from pokeio.analytics.yellow import decode as decode_gamestate
from pokeio.brain.actor_critic import ActorCritic
from pokeio.emu.env import ACTIONS
from pokeio.emu.fleet import FovealEncoder, ReflexGaze
from pokeio.reward.from_boot import measure as measure_progress
from pokeio.train.live import (
    LiveWriter,
    _b64_block,
    _gaze_from_obs,
    _gaze_payload,
    _optical_blocks,
    b64_gray,
    downscale_swarm,
)

_H, _W = 144, 160
_STUCK = 60  # showcase self-restart after this many looping agent-steps


class BrainShowcase:
    """A private env replaying the current System-1 policy from a cold boot."""

    def __init__(self, rom_path: str, reset_state: str, *, grid: int = 12,
                 fovea_grid: int = 12, mem_grid: int = 96, obs_dim: int = 454,
                 greedy: bool = False, learned_gaze: bool = False) -> None:
        from pokeio.emu.env import PokeEnv

        self.env = PokeEnv(rom_path=rom_path, frame_skip=24)
        self.reset_state = reset_state
        self.enc = FovealEncoder(1, periph_grid=grid, fovea_native_px=48,
                                 fovea_grid=fovea_grid, n_ram=8, reflex_gaze=True,
                                 foveal_memory=mem_grid > 0, mem_grid=max(1, mem_grid))
        self._reflex = ReflexGaze.maybe(self.enc, 1)  # bottom-up saccade for the showcase
        # MUST match the trainer's architecture (learned_gaze adds the gaze head) or
        # sync_weights' load_state_dict rejects the extra keys.
        self.policy = ActorCritic(obs_dim, periph_grid=grid, fovea_grid=fovea_grid,
                                  canvas_grid=mem_grid, learned_gaze=learned_gaze).eval()
        self.greedy = bool(greedy)
        self._lock = threading.Lock()
        self.last_action = 8
        self._still = deque(maxlen=4)
        self._n_still = 0
        self._reset()

    def _reset(self) -> None:
        self.enc.reset(0)
        self.screen = self.env.reset(self.reset_state)
        self.wram = self.env.raw_wram()
        self.last_action = 8

    def sync_weights(self, state_dict) -> None:
        with self._lock:
            # strict=False: never let an arch drift (e.g. a gaze head the render policy
            # lacks) raise here — the dashboard is best-effort and must not touch training.
            self.policy.load_state_dict(
                {k: v.detach().cpu() for k, v in state_dict.items()}, strict=False)

    @torch.no_grad()
    def step(self) -> None:
        obs = self.enc.encode(0, self.screen, self.wram, button=int(self.last_action))
        self._last_obs = obs
        with self._lock:
            out = self.policy.act(torch.from_numpy(obs[None, :]).float(), greedy=self.greedy)
        self.last_action = int(out["buttons"][0])
        # reflex saccade: steer the sharp fovea toward motion for the NEXT encode
        # (one-step efference delay, same as the fleet). Learned gaze would blend here.
        if self._reflex is not None:
            z = np.zeros(1, np.float32)
            gdx, gdy = self._reflex.blend(obs[None, :], self.enc.o_proprio, z.copy(), z.copy())
            self.enc.update_gaze(0, float(gdx[0]), float(gdy[0]))
        self.screen, self.wram, _d, _i = self.env.step(self.last_action)
        # self-restart if wedged (a greedy policy can loop forever)
        sig = self.screen[::8, ::8].tobytes()
        if sig in self._still:
            self._n_still += 1
            if self._n_still >= _STUCK:
                self._reset()
        else:
            self._n_still = 0
        self._still.append(sig)

    def payload(self) -> dict:
        blocks = _optical_blocks(self._last_obs, self.enc)
        gy, gx = self.enc.gaze(0)
        buttons = [0] * 9
        if 0 <= self.last_action < 9:
            buttons[self.last_action] = 1
        m = measure_progress(self.wram)
        gs = decode_gamestate(self.wram)
        periph = np.clip(blocks["periph"], 0, 1) * 255.0
        pay = {
            "frame_w": _W, "frame_h": _H,
            "frame_b64": b64_gray(self.screen),
            "obs_res": int(self.enc.G), "obs_b64": b64_gray(periph),
            "fovea_res": int(self.enc.FG), "fovea_b64": _b64_block(blocks["fovea"]),
            "motion_b64": _b64_block(blocks["motion"]) if "motion" in blocks else "",
            "gaze": _gaze_payload(gy, gx, _H, _W, int(self.enc.F)),
            "action": int(self.last_action), "buttons": buttons,
            "map": gs.map_name, "map_id": int(gs.map_id),
            "progress": {"phi": round(m["progress_score"], 1), "party": m["party_count"],
                         "level": m["party_level"], "events": m["events"],
                         "badges": m["badges"], "maps": m["maps"], "started": m["started"]},
        }
        # THE CANVAS — the persisted-vision buffer the AI actually acts on (blurry
        # periphery + decaying sharp saccade stamps) + its staleness map.
        if getattr(self.enc, "foveal_memory", False):
            pay["canvas_res"] = int(self.enc.M)
            pay["canvas_b64"] = _b64_block(self.enc.mem_buffer(0))
            pay["stale_b64"] = _b64_block(self.enc.mem_staleness(0))
        return pay

    def close(self) -> None:
        self.env.close()


class BrainStreamer:
    """Background render loop: plays the showcase + samples the swarm + emits live.json.

    The trainer constructs one, calls :meth:`sync(policy, metrics)` each iteration,
    and :meth:`close` at the end. ``fleet`` is read best-effort for swarm frames.
    """

    def __init__(self, run_dir, fleet, *, rom_path: str, reset_state: str,
                 grid: int = 12, fovea_grid: int = 12, mem_grid: int = 96,
                 obs_dim: int = 454, hz: float = 4.0, swarm_cap: int = 24,
                 run_id: str = "brain", total_iters: int = 0,
                 learned_gaze: bool = False) -> None:
        self.fleet = fleet
        self.run_id = str(run_id)
        self.total_iters = int(total_iters)   # target iterations (UI progress-to-done bar)
        self.swarm_cap = int(swarm_cap)
        self.show = BrainShowcase(rom_path, reset_state, grid=grid,
                                  fovea_grid=fovea_grid, mem_grid=mem_grid, obs_dim=obs_dim,
                                  learned_gaze=learned_gaze)
        self.writer = LiveWriter(run_dir, hz=hz)
        self._select_path = Path(run_dir) / "select.json"   # focus-agent selection (/api/select)
        self._sel_sig = None
        self._sel_idx = -1
        self._metrics: dict = {}
        self._iter = 0
        self._phase = "training"
        self._hist = {k: deque(maxlen=200) for k in
                      ("iter", "reach", "frontier_frac", "gate_stochastic", "entropy")}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, name="brain-stream", daemon=True)
        self._t.start()

    def sync(self, policy, metrics: dict, iter_: int, phase: str = "training") -> None:
        """Push fresh weights + metrics from the trainer (once per iteration).

        Runs on the TRAINER's thread (the on_log callback), so any exception here would
        kill the run — the dashboard must NEVER do that.  Wrapped defensively."""
        try:
            self.show.sync_weights(policy.state_dict())
            with self._lock:
                self._metrics = dict(metrics)
                self._iter = int(iter_)
                self._phase = str(phase)
                self._hist["iter"].append(int(iter_))
                for k in ("reach", "frontier_frac", "gate_stochastic", "entropy"):
                    self._hist[k].append(round(float(metrics.get(k, 0.0)), 4))
        except Exception:
            pass  # best-effort telemetry; never propagate into training

    def _swarm(self) -> list:
        try:
            screens = self.fleet.arr["screens"]
            n = min(self.swarm_cap, screens.shape[0])
            return [{"id": i, "w": 40, "h": 36,
                     "b64": b64_gray(downscale_swarm(np.asarray(screens[i])))}
                    for i in range(n)]
        except Exception:
            return []

    def _read_select(self) -> int:
        """Cheap stat of select.json (written by /api/select); return the focused
        swarm env index, or -1 (= champion). Parses only when the file changes."""
        try:
            st = self._select_path.stat()
        except OSError:
            return -1
        sig = (st.st_mtime_ns, st.st_size)
        if sig != self._sel_sig:
            try:
                self._sel_idx = int(json.loads(self._select_path.read_bytes()).get("idx", -1))
                self._sel_sig = sig
            except Exception:
                pass
        return self._sel_idx

    def _focus_payload(self, idx: int) -> dict | None:
        """Full detail panel for a SELECTED swarm env, read from the fleet's shm
        (its live screen + encoded percept + action + progress) — same shape as the
        champion payload, so the hero can render the focused training agent instead."""
        try:
            arr = self.fleet.arr
            if idx < 0 or idx >= int(self.fleet.n_envs):
                return None
            screen = np.asarray(arr["screens"][idx])
            obs = np.asarray(arr["obs"][idx])
            action = int(arr["actions"][idx]) if "actions" in arr else 8
            enc = self.show.enc  # geometry-only decode of a stored obs (not per-env state)
            blocks = _optical_blocks(obs, enc)
            gy, gx = _gaze_from_obs(obs, enc)
            buttons = [0] * 9
            if 0 <= action < 9:
                buttons[action] = 1
            periph = np.clip(blocks["periph"], 0, 1) * 255.0
            pay = {
                "idx": int(idx), "frame_w": _W, "frame_h": _H,
                "frame_b64": b64_gray(screen),
                "obs_res": int(enc.G), "obs_b64": b64_gray(periph),
                "fovea_res": int(enc.FG), "fovea_b64": _b64_block(blocks["fovea"]),
                "motion_b64": _b64_block(blocks["motion"]) if "motion" in blocks else "",
                "gaze": _gaze_payload(gy, gx, _H, _W, int(enc.F)),
                "action": action, "buttons": buttons,
            }
            if "buffer" in blocks:   # the canvas this training agent acts on
                pay["canvas_res"] = int(enc.M)
                pay["canvas_b64"] = _b64_block(blocks["buffer"])
                pay["stale_b64"] = _b64_block(blocks["stale"])
            if "wram" in arr:
                w = np.asarray(arr["wram"][idx])
                m = measure_progress(w)
                gs = decode_gamestate(w)
                pay["map"] = gs.map_name
                pay["map_id"] = int(gs.map_id)
                pay["progress"] = {"phi": round(m["progress_score"], 1),
                                   "party": m["party_count"], "level": m["party_level"],
                                   "events": m["events"], "badges": m["badges"],
                                   "maps": m["maps"], "started": m["started"]}
            return pay
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.wait(0.03):
            if not self.writer.due():
                continue
            try:
                self.show.step()
                with self._lock:
                    metrics, it, phase = dict(self._metrics), self._iter, self._phase
                    hist = {k: list(v) for k, v in self._hist.items()}
                sel = self._read_select()
                focus = self._focus_payload(sel) if sel is not None and sel >= 0 else None
                payload = {
                    "kind": "system1", "run_id": self.run_id, "iter": it,
                    "total_iters": self.total_iters,
                    "phase": phase, "ts": round(time.monotonic(), 2),
                    "champion": self.show.payload(), "swarm": self._swarm(),
                    "focus": focus, "focus_idx": (sel if focus is not None else -1),
                    "metrics": metrics, "history": hist,
                }
                self.writer.write(payload)
            except Exception:
                pass  # the render loop must never kill training

    def close(self) -> None:
        self._stop.set()
        try:
            self._t.join(timeout=2)
        except Exception:
            pass
        self.writer.close()
        self.show.close()


__all__ = ["BrainShowcase", "BrainStreamer"]
