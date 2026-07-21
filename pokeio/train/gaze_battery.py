"""FROZEN-GAZE GENERALIZATION BATTERY — does the learned gaze generalize as a LOOKING policy?

brain5 trains a learned top-down saccade delta (Path B, #11) but only ever sees the
Pallet-Town opening. This battery measures whether that frozen gaze head transfers as a
*looking* policy to visual regimes it never trained on (free roaming, dense menu text,
and a zero-shot different game), against the reflex-only and trivial baselines.

CORE DESIGN INVARIANT (controlled comparison): buttons are driven OPEN-LOOP per stream —
the action trace is PRECOMPUTED (deterministic given --seed) before any controller runs,
so the world evolves identically for every controller and gaze can never influence what
happens on screen. One emulator pass per stream; every frame is fed to ALL controller
lanes, where each lane owns an INDEPENDENT FovealEncoder (the exact deployed brain
layout) plus its own gaze state. The ONLY thing that differs across lanes is the
parent-level saccade command; the encoder-internal sub-saccade reflex (part of the
deployed optical front-end, fleet.substep) runs identically in every lane, so lanes
compare *gaze policies under the trained front-end dynamics*, not different front-ends.
The 'center' lane additionally clamps the fovea to a fixed point after every gaze
integration — otherwise the internal reflex would drag it off its definition.

Cadence replicates training: S sub-saccades per action step derived from the GB clock
(derive_cadence -> S=2: update_gaze, substep(mid frame), encode(final frame)) so the
persisted-vision canvas dynamics match the regime the head was trained in.

Streams (game-specific scripting is FINE here — this is evaluation apparatus, not agent
machinery) vs lanes (100% domain-agnostic: every metric reads only encoder state,
priority maps, and screens — no Pokemon logic anywhere in a metric or controller):

  s0 pallet — pikachu_purestep.state + the demo actions verbatim (training visual dist).
  s1 wander — demo end state + seeded uniform-random walk (Route 1/grass/menus).
  s2 menus  — demo end state + a literal START/arrows/A/B script (dense text regime).
  s3 sml    — Super Mario Land cold boot + START presses + right/A-biased random
              (zero-shot cross-game; the RAM tail is the blind stride sample — the point).

Run (CPU-only, single env; the box trains on cuda:1 — do not disturb it):

    nice -n 19 .venv/bin/python -m pokeio.train.gaze_battery \
        --ckpt runs/brain5/brain.pt --out runs/gaze_battery.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from pokeio.brain.actor_critic import ActorCritic
from pokeio.brain.champion import _git_sha
from pokeio.brain.coupling import blind_delta
from pokeio.config import VisionConfig
from pokeio.emu.fleet import FovealEncoder, ReflexGaze, _area_matrix
from pokeio.emu.saccade_cadence import chunk_frames, derive_cadence
from pokeio.train.brain_loop import GB_FPS
from pokeio.vision.preprocess import ObsBuilder

_YELLOW_ROM = "roms/pokemon_yellow.gb"
_SML_ROM = "roms/super_mario_land.gb"
_PURESTEP = "assets/demo_pikachu/pikachu_purestep.state"
_DEMO_ACTIONS = "assets/demo_pikachu/demo_actions.npy"
_FRAME_SKIP = 24
_H, _W = 144, 160

STREAMS = {"s0": "pallet", "s1": "wander", "s2": "menus", "s3": "sml"}

# s2: literal menu-browsing script (ACTIONS ids: 0-3 arrows, 4=A, 5=B, 6=START, 8=NOOP).
# Opens the pause menu, dives into submenus, backs out, reopens — dense-text regime.
# Tiled to the requested step count; hand-written on purpose (evaluation apparatus).
_MENU_SCRIPT = [
    6, 8,                # START: open the pause menu
    1, 1, 0,             # cursor down/down/up over the entries
    4, 8,                # A: open the highlighted submenu
    1, 1, 4, 8,          # browse deeper, confirm
    5, 8,                # B: back out one level
    0, 0, 4, 8,          # up twice, open another entry
    5, 5, 8,             # B B: back toward the overworld
    6, 8,                # reopen the menu
    1, 4, 8,             # down, open
    2, 3, 0, 1,          # arrow noise inside the submenu
    5, 5, 6, 8,          # back out fully, toggle START once more
]

# s3: literal SML boot script (wait out the logo, tap START past the title), then a
# right+A-biased random walk (right=run, A=jump — enough to actually see level 1-1).
_SML_BOOT = [8] * 8 + [6, 8, 6, 8, 6, 8]
_SML_P = [0.02, 0.03, 0.05, 0.40, 0.25, 0.10, 0.0, 0.0, 0.15]  # ACTIONS order


# ==========================================================================
# Encoder + pure metric helpers (hermetically testable — no emulator, no torch)
# ==========================================================================
def make_encoder() -> FovealEncoder:
    """A fresh 1-env encoder in EXACTLY the deployed brain layout (build_fleet
    brain_loop.py + worker fleet.py: G=12, sharp-48px fovea folded into the 96x96
    persisted-vision canvas, reflex gaze on => dim=18888, o_proprio=18864). Each lane
    owns its own instance so gaze/canvas/EMA state never leaks across controllers."""
    return FovealEncoder(1, periph_grid=12, fovea_native_px=48, fovea_grid=0, n_ram=8,
                         reflex_gaze=True, foveal_memory=True, mem_grid=96)


def _priority_capture_at(priority: np.ndarray, P: int, gy: float, gx: float,
                         half: float, H: int = _H, W: int = _W) -> tuple[float, bool]:
    """(max priority over the fovea FOOTPRINT / priority argmax, footprint-covers-argmax)
    for a fixation at screen pixel ``(gy, gx)`` with fovea half-width ``half`` px.

    Footprint = the map cells whose CENTRE lies inside the FxF window — the exact
    ``_fovea_covers`` geometry at map resolution, and the same semantics change_capture
    uses.  Scoring the footprint (not the single fixated cell) is load-bearing: the
    deployed gaze clamp (fleet.update_gaze: gx in [24,136], gy in [24,120]) confines the
    fixated cell to r in [16,80], c in [14,81] of the 96x96 map, and the stamp zeroes
    staleness under the fovea (IOR), so a single-cell readout is floor-pinned to ~0 for
    every velocity-limited controller — the WINDOW is what the controller aims, and the
    window can cover border-band argmaxes the centre cell can never index.  The cell
    containing the fixation is always included (degenerate ``half`` stays well-defined).
    A flat map (max ~ 0) makes the ratio undefined — every fixation is trivially
    optimal — so return (1.0, True); callers that only want *informative* decisions
    should skip flat maps instead of counting."""
    pr = np.asarray(priority)
    pmax = float(pr.max())
    if pmax <= 1e-9:
        return 1.0, True
    # Fixated cell: the exact inverse of the encoder's soft-argmax readout mapping
    # (``tgt = (c+0.5)/P*2-1``, fleet._reflex_target).
    tx = gx / W * 2.0 - 1.0
    ty = gy / H * 2.0 - 1.0
    c = min(max(int((tx + 1.0) / 2.0 * P), 0), P - 1)
    r = min(max(int((ty + 1.0) / 2.0 * P), 0), P - 1)
    idx = np.arange(P)
    rmask = np.abs((idx + 0.5) * H / P - gy) <= half
    cmask = np.abs((idx + 0.5) * W / P - gx) <= half
    rmask[r] = True
    cmask[c] = True
    rm, cm = np.unravel_index(int(pr.argmax()), (P, P))
    ratio = float(pr[np.ix_(rmask, cmask)].max()) / pmax
    return ratio, bool(rmask[int(rm)] and cmask[int(cm)])


def _fovea_covers(gy: float, gx: float, cell: tuple[int, int],
                  G: int, H: int, W: int, half: int) -> bool:
    """Did a fovea centred at ``(gy, gx)`` cover periphery cell ``cell``?  Covered =
    the cell's CENTRE lies inside the fovea's half-width window (a 48px fovea spans
    ~4 periphery cells, so centre-containment is the honest 'looked at it' test)."""
    r, c = cell
    cy = (r + 0.5) * H / G
    cx = (c + 0.5) * W / G
    return abs(cy - gy) <= half and abs(cx - gx) <= half


class StreamTruth:
    """Lane-independent ground truth per action step.

    Everything here is computed from the raw shared screen — gaze-free by construction,
    so it is identical for every lane (the controlled-comparison requirement):
      * ``mem``      — the TRUE screen at canvas resolution: the same per-frame shade
        ranking the encoder perceives through (ObsBuilder.normalize_shades_f64) pushed
        through the same area-resample family the canvas stamps use (_area_matrix),
        at M=96.  canvas_err = L1(lane canvas, this).
      * ``chg_cell`` — the periphery-grid cell with the largest |frame delta| between
        consecutive FINAL frames (the per-action-step change to capture), or None when
        the screen is static.  Uses the encoder's own G=12 periphery resample.
    """

    def __init__(self, M: int = 96, G: int = 12):
        self._shade = ObsBuilder(VisionConfig(shades=4, screen_height=_H, screen_width=_W))
        self._rM = _area_matrix(_H, M)
        self._cM = _area_matrix(_W, M).T
        self._rG = _area_matrix(_H, G)
        self._cG = _area_matrix(_W, G).T
        self._prev: np.ndarray | None = None

    def begin(self, screen: np.ndarray) -> None:
        n = self._shade.normalize_shades_f64(screen)
        self._prev = (self._rG @ n @ self._cG).astype(np.float32)

    def step(self, final: np.ndarray) -> dict:
        n = self._shade.normalize_shades_f64(final)
        mem = (self._rM @ n @ self._cM).astype(np.float32)
        periph = (self._rG @ n @ self._cG).astype(np.float32)
        cell = None
        if self._prev is not None:
            d = np.abs(periph - self._prev)
            if float(d.max()) > 1e-6:
                cell = tuple(int(v) for v in np.unravel_index(int(d.argmax()), d.shape))
        self._prev = periph
        return {"mem": mem, "chg_cell": cell}


# ==========================================================================
# Gaze controllers — the ONLY thing that differs across lanes
# ==========================================================================
def _zero_cmd(obs: np.ndarray) -> tuple[float, float]:
    """The 'no top-down command' controller (center lane pairs it with a clamp)."""
    return 0.0, 0.0


def center_fixation(enc: FovealEncoder, prio, P) -> tuple[float, float]:
    """Fixation override pinning the fovea to the screen centre (the 'center' lane)."""
    return float(enc._cy), float(enc._cx)


class LearnedGaze:
    """The DEPLOYED Path-B command: greedy gaze mu seeded into ReflexGaze.blend —
    a bit-for-bit replica of rl_loop.BrainTrainer._gaze(obs, 1, learned=mu) followed by
    the worker's float() cast (fleet.py update_gaze call).  ``pure=True`` skips the
    reflex blend and applies the raw mu (isolates the learned head).  Locked by
    tests/test_gaze_battery.py::test_learned_lane_matches_deployed_blend — do not
    'simplify' the copies/casts, they are the deployed numerics."""

    def __init__(self, policy, o_proprio: int, gain: float, decay: float,
                 pure: bool = False):
        self.policy = policy
        self.o_proprio = int(o_proprio)
        self.reflex = None if pure else ReflexGaze(1, float(gain), float(decay))

    def __call__(self, obs: np.ndarray) -> tuple[float, float]:
        out = self.policy.act(torch.as_tensor(obs[None, :], dtype=torch.float32),
                              greedy=True)
        mu = out["gaze"].cpu().numpy().astype(np.float32)          # rl_loop._act cast
        ldx = np.ascontiguousarray(mu[:, 0], np.float32)           # rl_loop._gaze
        ldy = np.ascontiguousarray(mu[:, 1], np.float32)
        if self.reflex is None:                                    # learned-only lane
            return float(ldx[0]), float(ldy[0])
        gdx, gdy = self.reflex.blend(np.asarray(obs[None, :], np.float32),
                                     self.o_proprio, ldx.copy(), ldy.copy())
        return (float(gdx.astype(np.float32)[0]), float(gdy.astype(np.float32)[0]))


class ReflexOnlyGaze:
    """The brain4 gaze: ReflexGaze.blend seeded with zeros — rl_loop._gaze(obs, 1,
    learned=None) exactly (the learned seed is a zero vector, the reflex is added)."""

    def __init__(self, o_proprio: int, gain: float, decay: float):
        self.o_proprio = int(o_proprio)
        self.reflex = ReflexGaze(1, float(gain), float(decay))

    def __call__(self, obs: np.ndarray) -> tuple[float, float]:
        z = np.zeros(1, np.float32)
        gdx, gdy = self.reflex.blend(np.asarray(obs[None, :], np.float32),
                                     self.o_proprio, z.copy(), z.copy())
        return (float(gdx.astype(np.float32)[0]), float(gdy.astype(np.float32)[0]))


class RandomGaze:
    """Uniform random saccade command in [-1,1]^2 (own fixed rng) — magnitude-matched
    to the tanh-bounded regime the deployed commands live in (gain*tanh saturates on
    ~O(1) inputs), so it is a fair 'looks around aimlessly' baseline."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, obs: np.ndarray) -> tuple[float, float]:
        d = self.rng.uniform(-1.0, 1.0, size=2)
        return float(d[0]), float(d[1])


# ==========================================================================
# Lane — one (encoder, controller) pair stepped on the shared frame stream
# ==========================================================================
class Lane:
    """One controller lane: an independent deployed-layout FovealEncoder + a gaze
    source + metric accumulators.  ``step`` replicates the deployed per-action-step
    order EXACTLY (worker fleet.py: update_gaze BEFORE any frame -> substep on each
    intermediate frame -> encode on the final frame), with the controller command
    computed from the SAME obs the policy would act on (rl_loop rollout order).

    ``fixation`` (optional) is a post-integration override ``(enc, priority|None, P)
    -> (gy, gx)`` re-applied after update_gaze and after every substep — how 'center'
    (and test probes like an argmax-seeker) pin the fovea despite the encoder-internal
    sub-saccade reflex, which otherwise runs identically in every lane (deployed path).

    Metrics (all domain-agnostic — encoder state + shared screens only):
      canvas_err        mean L1(canvas, true screen at canvas res), all cells
      canvas_err_fresh  same over cells with staleness < 25% of the current max
      priority_capture  max priority over the realized fovea FOOTPRINT / priority
                        argmax, per decision (footprint scoring — the clamp makes a
                        single-cell readout floor-pinned; flat maps skipped — no
                        informative decision to score)
      argmax_hit_frac   fraction of decisions whose footprint covers the argmax cell
      salience_capture  same footprint ratio against the staleness-UNGATED salience
                        (a lane is not penalized for having just refreshed the
                        hotspot it correctly saccaded to)
      change_capture    did any fixation this step's saccade cycle cover the cell
                        with the largest inter-step frame delta
      mean_saccade_mag  mean |controller command| (pre-tanh, what update_gaze got)
    """

    def __init__(self, name: str, controller=None, fixation=None,
                 keep_obs_stride: int = 0):
        self.name = str(name)
        self.enc = make_encoder()
        self.controller = controller if controller is not None else _zero_cmd
        self.fixation = fixation
        self.keep_obs_stride = int(keep_obs_stride)
        self.obs: np.ndarray | None = None
        self.cmds: list[tuple[float, float]] = []
        self.obs_kept: list[np.ndarray] = []
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.t = 0
        self._cerr_sum = 0.0
        self._cerr_n = 0
        self._cfresh_sum = 0.0
        self._cfresh_n = 0
        self._prio_sum = 0.0
        self._prio_n = 0
        self._prio_hits = 0
        self._sal_sum = 0.0
        self._sal_n = 0
        self._chg_hits = 0
        self._chg_n = 0
        self._mag_sum = 0.0
        self.cmds = []
        self.obs_kept = []

    def _fix(self, prio, P: int) -> None:
        if self.fixation is None:
            return
        gy, gx = self.fixation(self.enc, prio, P)
        self.enc._gy[0] = float(gy)
        self.enc._gx[0] = float(gx)

    def begin(self, screen: np.ndarray, wram: np.ndarray | None) -> None:
        """Deployed reset round: enc.reset then the NOOP reset obs (fleet worker)."""
        self.enc.reset(0)
        self.obs = self.enc.encode(0, screen, wram, button=8)
        self._reset_metrics()

    def step(self, button: int, mids: list[np.ndarray], final: np.ndarray,
             wram: np.ndarray | None, truth: dict) -> None:
        """One action step on shared frames; pure-ish (touches only this lane)."""
        enc, obs = self.enc, self.obs
        assert obs is not None, "Lane.step before Lane.begin"
        if self.keep_obs_stride and (self.t % self.keep_obs_stride == 0):
            self.obs_kept.append(obs.copy())      # DECISION obs (what the policy saw)

        # Decision-time priority map: motion from the obs the controller acts on +
        # the encoder's current staleness (unchanged since that obs was encoded).
        # _priority_map is a pure read — nothing mutates.
        motion = obs[enc.o_motion:enc.o_motion_hi].reshape(enc.G, enc.G)
        P, prio = enc._priority_map(0, motion)
        informative = float(prio.max()) > 1e-9
        # Staleness-UNGATED salience: the same channels with the recency gate forced
        # to 1 (swap-in ones, restored immediately — a pure read overall), so a lane
        # is not penalized for having just refreshed the hotspot it saccaded to.
        if enc.foveal_memory:
            _saved_stale = enc._stale[0].copy()
            enc._stale[0].fill(1.0)
            _, sal = enc._priority_map(0, motion)
            enc._stale[0][:] = _saved_stale
        else:
            sal = prio                       # no canvas -> no gate: prio IS ungated
        sal_informative = float(sal.max()) > 1e-9

        # Controller command from the SAME obs (rl_loop order: act -> _gaze -> step).
        gdx, gdy = self.controller(obs)
        self.cmds.append((gdx, gdy))
        self._mag_sum += float(np.hypot(gdx, gdy))

        # (a) integrate the saccade BEFORE any frame (worker order), then override.
        enc.update_gaze(0, gdx, gdy)
        self._fix(prio if informative else None, P)
        gy, gx = enc.gaze(0)
        if informative:
            ratio, hit = _priority_capture_at(prio, P, gy, gx, enc._half,
                                              enc.H, enc.W)
            self._prio_sum += ratio
            self._prio_n += 1
            self._prio_hits += int(hit)
        if sal_informative:
            sratio, _ = _priority_capture_at(sal, P, gy, gx, enc._half,
                                             enc.H, enc.W)
            self._sal_sum += sratio
            self._sal_n += 1

        # (b) sub-saccades on the intermediate frames (canvas stamps + internal reflex).
        positions = [(gy, gx)]
        for m in mids:
            enc.substep(0, m)
            self._fix(prio if informative else None, P)
            positions.append(enc.gaze(0))

        # (c) final-frame encode -> the obs the next decision acts on.
        self.obs = enc.encode(0, final, wram, button=int(button))

        # Canvas fidelity vs the lane-independent truth.
        err = np.abs(enc._mem[0] - truth["mem"])
        self._cerr_sum += float(err.mean())
        self._cerr_n += 1
        stale = enc._stale[0]
        smax = float(stale.max())
        fresh = (stale < 0.25 * smax) if smax > 0.0 else np.ones(stale.shape, bool)
        if fresh.any():
            self._cfresh_sum += float(err[fresh].mean())
            self._cfresh_n += 1

        # Change capture: did any fixation of this saccade cycle cover the hot cell?
        cell = truth["chg_cell"]
        if cell is not None:
            covered = any(
                _fovea_covers(py, px, cell, enc.G, enc.H, enc.W, enc._half)
                for (py, px) in positions
            )
            self._chg_hits += int(covered)
            self._chg_n += 1
        self.t += 1

    def report(self) -> dict:
        def _mean(s, n):
            return (s / n) if n else None

        return {
            "n_steps": self.t,
            "canvas_err": _mean(self._cerr_sum, self._cerr_n),
            "canvas_err_fresh": _mean(self._cfresh_sum, self._cfresh_n),
            "priority_capture": _mean(self._prio_sum, self._prio_n),
            "argmax_hit_frac": _mean(float(self._prio_hits), self._prio_n),
            "salience_capture": _mean(self._sal_sum, self._sal_n),
            "change_capture": _mean(float(self._chg_hits), self._chg_n),
            "mean_saccade_mag": _mean(self._mag_sum, self.t),
        }


def run_stream(lanes: list[Lane], first: tuple, steps) -> list[int]:
    """Drive every lane over one shared stream; return the RECORDED button trace.

    ``first`` = (screen, wram) of the reset/restore frame; ``steps`` yields
    ``(button, mids, final, wram)`` per action step.  Buttons arrive precomputed in
    ``steps`` — lanes cannot influence them (the open-loop invariant); the returned
    trace is the witness the tests compare across lane subsets."""
    screen0, wram0 = first
    truth = StreamTruth()
    truth.begin(screen0)
    for lane in lanes:
        lane.begin(screen0, wram0)
    trace: list[int] = []
    for (button, mids, final, wram) in steps:
        trace.append(int(button))
        t = truth.step(final)
        for lane in lanes:
            lane.step(int(button), mids, final, wram, t)
    return trace


# ==========================================================================
# Streams — open-loop traces + emulator drivers (game-specific ON PURPOSE)
# ==========================================================================
def build_trace(key: str, frames: int, seed: int) -> list[int]:
    """Precompute a stream's button trace (pure, seed-deterministic, lane-blind —
    THE open-loop guarantee: the trace exists before any controller runs)."""
    n = int(frames)
    if key == "s0":     # the demo, verbatim (truncated to the budget; never padded)
        acts = np.asarray(np.load(_DEMO_ACTIONS)).astype(int).ravel()
        return [int(a) for a in acts[:n]]
    if key == "s1":     # uniform random walk over the full 9-action space
        rng = np.random.default_rng([int(seed), 1])
        return [int(a) for a in rng.integers(0, 9, size=n)]
    if key == "s2":     # the literal menu script, tiled
        reps = -(-n // len(_MENU_SCRIPT))
        return list((_MENU_SCRIPT * reps)[:n])
    if key == "s3":     # SML: boot script then right+A-biased random
        rng = np.random.default_rng([int(seed), 3])
        rest = max(0, n - len(_SML_BOOT))
        tail = [int(a) for a in rng.choice(9, size=rest, p=_SML_P)]
        return (list(_SML_BOOT) + tail)[:n]
    raise ValueError(f"unknown stream key {key!r}")


_END_BLOB: bytes | None = None


def _demo_end_blob() -> bytes:
    """The demo's full-depth save-state blob (cached: one replay serves s1 AND s2)."""
    global _END_BLOB
    if _END_BLOB is None:
        from pokeio.brain.replay import DemoTrajectory
        demo = DemoTrajectory()
        _END_BLOB = demo.blob_at(len(demo))
    return _END_BLOB


def open_stream(key: str, frames: int, seed: int):
    """Build (env, (first_screen, first_wram), trace) for one stream.  Lazy PyBoy
    import keeps the module core (lanes/metrics) emulator-free for the hermetic tests."""
    from pokeio.emu.env import PokeEnv

    trace = build_trace(key, frames, seed)
    if key == "s3":
        env = PokeEnv(_SML_ROM, frame_skip=_FRAME_SKIP)
        screen = env.reset(None)                        # cold boot
    else:
        env = PokeEnv(_YELLOW_ROM, frame_skip=_FRAME_SKIP)
        if key == "s0":
            screen = env.reset(_PURESTEP)
        else:                                            # s1/s2: demo end state
            env.load_state(_demo_end_blob())             # worker restore path
            env.pyboy.tick(1, True)
            screen = env._obs()
    return env, (screen, env.raw_wram()), trace


def emu_steps(env, trace: list[int], sub_steps: int):
    """Yield ``(button, mids, final, wram)`` per action step with the EXACT deployed
    frame advance (worker fleet.py: hold -> chunk_frames(frame_skip - retap, S) ->
    tick each chunk, sub-saccade frames are all but the last)."""
    for btn in trace:
        used = env.hold(int(btn))
        chunks = chunk_frames(env.frame_skip - used, sub_steps)
        mids = []
        screen = None
        for s in range(sub_steps):
            screen = env.tick_frames(chunks[s])
            if s < sub_steps - 1:
                mids.append(screen)
        yield int(btn), mids, screen, env.raw_wram()


# ==========================================================================
# Battery assembly
# ==========================================================================
def build_lanes(policy, seed: int, keep_obs_stride: int = 0) -> list[Lane]:
    """Fresh lanes for one stream (fresh encoder + fresh ReflexGaze per lane, per the
    clean-A/B isolation rule — game-level EMAs must not leak between controllers)."""
    lanes: list[Lane] = []
    if policy is not None and getattr(policy, "learned_gaze", False):
        l = Lane("learned", keep_obs_stride=keep_obs_stride)
        l.controller = LearnedGaze(policy, l.enc.o_proprio,
                                   l.enc.reflex_gain, l.enc.reflex_ema_decay)
        lanes.append(l)
        lp = Lane("learned_pure")
        lp.controller = LearnedGaze(policy, lp.enc.o_proprio,
                                    lp.enc.reflex_gain, lp.enc.reflex_ema_decay,
                                    pure=True)
        lanes.append(lp)
    lr = Lane("reflex", keep_obs_stride=keep_obs_stride)
    lr.controller = ReflexOnlyGaze(lr.enc.o_proprio,
                                   lr.enc.reflex_gain, lr.enc.reflex_ema_decay)
    lanes.append(lr)
    rnd = Lane("random", controller=RandomGaze(np.random.default_rng([int(seed), 7])))
    lanes.append(rnd)
    lanes.append(Lane("center", controller=_zero_cmd, fixation=center_fixation))
    return lanes


def load_policy(ckpt_path: str):
    """Load the brain checkpoint into a CPU ActorCritic (eval mode).

    The live trainer overwrites brain.pt every few iterations, so torch.load on the
    live file can catch a partial write — copy first, retry once on a torn read.
    Accepts the plain brain.pt format ('policy' key) and the champion format
    ('state_dict'); the gaze head's presence in the state dict decides learned_gaze."""
    def _read():
        fd, tmp = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            shutil.copy2(ckpt_path, tmp)
            return torch.load(tmp, map_location="cpu", weights_only=False)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    try:
        ck = _read()
    except Exception:
        time.sleep(2.0)
        ck = _read()
    sd = ck.get("policy") or ck.get("state_dict") or ck
    probe = make_encoder()
    policy = ActorCritic(probe.dim, periph_grid=probe.G, fovea_grid=probe.FG,
                         canvas_grid=probe.M,
                         learned_gaze="gaze_mu.weight" in sd)
    policy.load_state_dict(sd)
    policy.eval()
    return policy, int(ck.get("iter", -1)) if isinstance(ck, dict) else -1


def run_battery(ckpt: str, stream_keys: list[str], frames: int, seed: int) -> dict:
    """One emulator pass per stream, all lanes in lock-step; returns the output dict."""
    torch.set_num_threads(1)      # the box is training on cuda:1 — stay out of the way
    policy, ckpt_iter = load_policy(ckpt)
    S = derive_cadence(GB_FPS, _FRAME_SKIP).sub_steps    # the trained regime (S=2)
    results: dict = {}
    for key in stream_keys:
        name = STREAMS[key]
        env, first, trace = open_stream(key, frames, seed)
        # blind_delta only where the spec asks (learned/reflex on s0/s1): cap the
        # kept decision-obs at ~256 rows (18888-d each) to bound memory/compute.
        collect = key in ("s0", "s1")
        stride = max(1, len(trace) // 256) if collect else 0
        lanes = build_lanes(policy, seed, keep_obs_stride=stride)
        t0 = time.monotonic()
        print(f"[battery] stream {key}:{name}: {len(trace)} steps x "
              f"{len(lanes)} lanes (S={S})", flush=True)
        try:
            run_stream(lanes, first, emu_steps(env, trace, S))
        finally:
            env.close()
        rep: dict = {}
        pf = policy.numpy_policy_fn() if policy is not None else None
        for lane in lanes:
            r = lane.report()
            if pf is not None and lane.obs_kept and lane.name in ("learned", "reflex"):
                r["blind_delta"] = float(blind_delta(pf, np.stack(lane.obs_kept)))
            rep[lane.name] = r
        results[name] = rep
        print(f"[battery]   done in {time.monotonic() - t0:.0f}s", flush=True)
    return {
        "meta": {"ckpt": str(ckpt), "ckpt_iter": ckpt_iter, "frames": int(frames),
                 "seed": int(seed), "git_sha": _git_sha(), "sub_steps": int(S),
                 "streams": [STREAMS[k] for k in stream_keys]},
        "results": results,
    }


# ==========================================================================
# Reporting
# ==========================================================================
_COLS = [("n_steps", "steps", "d"), ("canvas_err", "canvas_err", "f"),
         ("canvas_err_fresh", "cerr_fresh", "f"), ("priority_capture", "prio_cap", "f"),
         ("argmax_hit_frac", "argmax_hit", "f"), ("salience_capture", "sal_cap", "f"),
         ("change_capture", "chg_cap", "f"), ("mean_saccade_mag", "|cmd|", "f"),
         ("blind_delta", "blind", "f")]


def format_table(results: dict) -> str:
    w_lane = max([12] + [len(l) for lanes in results.values() for l in lanes])
    head = "lane".ljust(w_lane) + "".join(f"  {h:>10}" for _, h, _ in _COLS)
    out = []
    for stream, lanes in results.items():
        out.append(f"=== {stream} ===")
        out.append(head)
        for lane, m in lanes.items():
            row = lane.ljust(w_lane)
            for key, _, kind in _COLS:
                v = m.get(key)
                if v is None:
                    row += f"  {'-':>10}"
                elif kind == "d":
                    row += f"  {int(v):>10d}"
                else:
                    row += f"  {v:>10.4f}"
            out.append(row)
        out.append("")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Frozen-gaze generalization battery (open-loop, multi-lane)")
    ap.add_argument("--ckpt", required=True, help="brain checkpoint (e.g. runs/brain5/brain.pt)")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--streams", default="s0,s1,s2,s3",
                    help=f"comma list of {sorted(STREAMS)} (default: all)")
    ap.add_argument("--frames", type=int, default=600,
                    help="action steps per stream (s0 is capped at the demo length)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    keys = [k.strip() for k in args.streams.split(",") if k.strip()]
    bad = [k for k in keys if k not in STREAMS]
    if bad:
        raise SystemExit(f"unknown streams {bad}; choose from {sorted(STREAMS)}")

    out = run_battery(args.ckpt, keys, args.frames, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[battery] wrote {out_path}")
    print(format_table(out["results"]))


if __name__ == "__main__":
    main()
