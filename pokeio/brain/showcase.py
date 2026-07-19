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

import threading
import time
from collections import deque

import numpy as np
import torch

from pokeio.analytics.yellow import decode as decode_gamestate
from pokeio.brain.actor_critic import ActorCritic
from pokeio.emu.env import ACTIONS
from pokeio.emu.fleet import FovealEncoder
from pokeio.reward.from_boot import measure as measure_progress
from pokeio.train.live import (
    LiveWriter,
    _b64_block,
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
                 obs_dim: int = 454, greedy: bool = False) -> None:
        from pokeio.emu.env import PokeEnv

        self.env = PokeEnv(rom_path=rom_path, frame_skip=24)
        self.reset_state = reset_state
        self.enc = FovealEncoder(1, periph_grid=grid, fovea_native_px=48,
                                 fovea_grid=0, n_ram=8)
        self.policy = ActorCritic(obs_dim, grid=grid).eval()  # cpu copy, synced by the trainer
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
            self.policy.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})

    @torch.no_grad()
    def step(self) -> None:
        obs = self.enc.encode(0, self.screen, self.wram, button=int(self.last_action))
        self._last_obs = obs
        with self._lock:
            out = self.policy.act(torch.from_numpy(obs[None, :]).float(), greedy=self.greedy)
        self.last_action = int(out["buttons"][0])
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
        return {
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

    def close(self) -> None:
        self.env.close()


class BrainStreamer:
    """Background render loop: plays the showcase + samples the swarm + emits live.json.

    The trainer constructs one, calls :meth:`sync(policy, metrics)` each iteration,
    and :meth:`close` at the end. ``fleet`` is read best-effort for swarm frames.
    """

    def __init__(self, run_dir, fleet, *, rom_path: str, reset_state: str,
                 grid: int = 12, obs_dim: int = 454, hz: float = 4.0,
                 swarm_cap: int = 24, run_id: str = "brain") -> None:
        self.fleet = fleet
        self.run_id = str(run_id)
        self.swarm_cap = int(swarm_cap)
        self.show = BrainShowcase(rom_path, reset_state, grid=grid, obs_dim=obs_dim)
        self.writer = LiveWriter(run_dir, hz=hz)
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
        """Push fresh weights + metrics from the trainer (once per iteration)."""
        self.show.sync_weights(policy.state_dict())
        with self._lock:
            self._metrics = dict(metrics)
            self._iter = int(iter_)
            self._phase = str(phase)
            self._hist["iter"].append(int(iter_))
            for k in ("reach", "frontier_frac", "gate_stochastic", "entropy"):
                self._hist[k].append(round(float(metrics.get(k, 0.0)), 4))

    def _swarm(self) -> list:
        try:
            screens = self.fleet.arr["screens"]
            n = min(self.swarm_cap, screens.shape[0])
            return [{"id": i, "w": 40, "h": 36,
                     "b64": b64_gray(downscale_swarm(np.asarray(screens[i])))}
                    for i in range(n)]
        except Exception:
            return []

    def _loop(self) -> None:
        while not self._stop.wait(0.03):
            if not self.writer.due():
                continue
            try:
                self.show.step()
                with self._lock:
                    metrics, it, phase = dict(self._metrics), self._iter, self._phase
                    hist = {k: list(v) for k, v in self._hist.items()}
                payload = {
                    "kind": "system1", "run_id": self.run_id, "iter": it,
                    "phase": phase, "ts": round(time.monotonic(), 2),
                    "champion": self.show.payload(), "swarm": self._swarm(),
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
