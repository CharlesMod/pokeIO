"""GameEnv: one emulator + spec-driven observation, reward and episode logic.

Step semantics (all from the spec, nothing game-specific here):
  1. press the chosen button for ``press_frames``, release, tick to
     ``frames_per_action`` without rendering;
  2. while ``wait_while`` holds (game ignoring input: text, cutscenes) tap
     ``wait_button`` and keep ticking — the agent never spends a decision there;
  3. render one frame, read memory once, compute rewards and the observation.

Episodes: the emulator is *not* reset on a timer. Every ``memory_reset_steps``
(jittered) the exploration memory is wiped while the game keeps running
("mini-episodes"). ``done`` is only raised when the emulator jumps to another
state (hard/stall reset, swarm migration), so the learner treats everything
else as one continuing stream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import numpy as np

from pokeio.exploration import CellMemory, InteractionNovelty, ScreenNovelty, VisitedMask
from pokeio.memory import Memory, parse_sym_file
from pokeio.platforms import make_platform
from pokeio.progress import Progress
from pokeio.screen import ScreenProcessor
from pokeio.spec import GameSpec, VectorFeature

DIRECTION_BUTTONS = ("up", "down", "left", "right")


@dataclass
class ObsSpec:
    """Shapes/dtypes of the flat observation arrays, plus decode metadata."""

    pixels: tuple[int, int, int]  # (channels, h, packed_w) uint8
    pixel_width: int  # unpacked width
    pixel_bpp: int  # bits per pixel in the packed array (8 = unpacked)
    levels: int
    bits: int  # number of packed bytes of binary features
    scalars: int  # float32 features
    # (num, embed, group) per categorical slot; slots of one spec feature share a table
    cats: list[tuple[int, int, int]] = dc_field(default_factory=list)
    n_actions: int = 0

    def arrays(self) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        """Every per-env array that travels through the vec env. ``aux`` is
        telemetry for the dashboard (room, x, y, paused, score*100) and is never
        shown to the policy."""
        return {
            "pixels": (self.pixels, np.dtype(np.uint8)),
            "bits": ((max(self.bits, 1),), np.dtype(np.uint8)),
            "scalars": ((max(self.scalars, 1),), np.dtype(np.float32)),
            "cats": ((max(len(self.cats), 1),), np.dtype(np.int32)),
            "aux": ((5,), np.dtype(np.int32)),
        }


POLICY_KEYS = ("pixels", "bits", "scalars", "cats")


def _vector_width(spec: GameSpec, v: VectorFeature) -> tuple[str, int]:
    f = spec.memory[v.field]
    n = f.length if f.is_block else f.count
    if v.kind == "bits":
        return "bits", n  # bytes
    if v.kind == "categorical":
        return "cats", n
    return "scalars", n


class GameEnv:
    def __init__(self, spec: GameSpec, env_id: int = 0, seed: int = 0, headless: bool = True):
        self.spec = spec
        self.env_id = env_id
        self.rng = random.Random(seed * 100003 + env_id)
        self.p = make_platform(spec, headless=headless)
        symbols = parse_sym_file(spec.symbols) if spec.symbols and spec.symbols.exists() else None
        self.mem = Memory(self.p, spec.memory, symbols)
        self.buttons = spec.controls.buttons
        ctl = spec.controls
        self._wait = ctl.wait_while if ctl.wait_while and self.mem.has(ctl.wait_while.field) else None

        self.screen = ScreenProcessor(
            self.p.height, self.p.width, spec.screen.downsample, spec.screen.levels, spec.screen.channel
        )
        pos = spec.position
        self.has_position = pos is not None and all(
            self.mem.has(f) for f in (pos.x, pos.y) + ((pos.room,) if pos.room else ())
        )
        ex = spec.exploration
        self.cells = CellMemory(ex.half_life_steps, ex.floor)
        self.mask = (
            VisitedMask(
                self.p.height, self.p.width, spec.screen.downsample, pos.cell_px,
                pos.player_cell_px, spec.screen.levels, pos.camera,
            )
            if self.has_position
            else None
        )
        self.interact = InteractionNovelty(ex.interaction_min_change)
        self.screen_novelty = ScreenNovelty(ex.screen_hash_grid, spec.screen.levels)
        self.progress = Progress(spec.terms, spec.milestones, self.mem)

        # vector features that resolved against this ROM/symbol table
        self.vector = [
            v for v in spec.vector if self.mem.has(v.field) and (v.denom is None or self.mem.has(v.denom))
        ]
        n_bits = n_scal = 0
        cats: list[tuple[int, int, int]] = []
        for gi, v in enumerate(self.vector):
            kind, n = _vector_width(spec, v)
            if kind == "bits":
                n_bits += n
            elif kind == "cats":
                cats += [(v.num, v.embed, gi)] * n
            else:
                n_scal += n
        n_scal += len(self.buttons)  # previous action, one-hot
        channels = 2 if self.has_position else 1
        self.obs_spec = ObsSpec(
            pixels=(channels, self.screen.h, self.screen.packed_w),
            pixel_width=self.screen.w,
            pixel_bpp=self.screen.bpp,
            levels=spec.screen.levels,
            bits=n_bits,
            scalars=n_scal,
            cats=cats,
            n_actions=len(self.buttons),
        )
        self._start_states = [Path(s).read_bytes() for s in spec.start_states if Path(s).exists()]
        self.frontier_state: bytes | None = None  # set by the swarm
        self.t = 0  # global step counter of this env (never reset; memory clock)
        self._reset_counters()

    # ------------------------------------------------------------------ state
    def _reset_counters(self) -> None:
        self.ep_step = 0
        self.mem_step = 0
        self.mem_horizon = self._jitter(self.spec.episode.memory_reset_steps)
        self.last_reward_step = 0
        self.prev_action = -1
        self.facing = 0
        self.ep_return = 0.0
        self.reward_parts: dict[str, float] = {}
        self.best_score = float("-inf")

    def _jitter(self, n: int) -> int:
        j = self.spec.episode.memory_reset_jitter
        return max(1, n + (self.rng.randint(-j, j) if j > 0 else 0))

    def _apply_patches(self) -> None:
        for p in self.spec.memory_patches:
            name, value = p["field"], int(p["value"])
            if not self.mem.has(name):
                continue
            cur = self.mem.get(name) & int(p.get("only_if_mask", 0xFF))
            allowed = p.get("only_if_in")
            if allowed is not None and cur not in [int(a) for a in allowed]:
                continue
            self.mem.write(name, value)

    def _wipe_memory(self) -> None:
        self.cells.wipe()
        self.interact.wipe()
        self.screen_novelty.wipe()
        self.mem_step = 0
        self.mem_horizon = self._jitter(self.spec.episode.memory_reset_steps)

    def reset(self, state: bytes | None = None) -> dict[str, np.ndarray]:
        """Load ``state`` (or the frontier / a start state) and start a fresh episode."""
        if state is None:
            state = self.frontier_state
        if state is None and self._start_states:
            state = self.rng.choice(self._start_states)
        if state is not None:
            self.p.load_state(state)
        self.mem.invalidate()
        self._apply_patches()
        jitter = self.spec.episode.start_jitter_frames
        if jitter > 0:
            self.p.tick(self.rng.randint(0, jitter), False)
        self.p.tick(1, True)
        self.mem.invalidate()
        self._reset_counters()
        self._wipe_memory()
        self.progress.rebase(self.t)
        self.best_score = self._score()
        self._levels = self.screen.levels_image(self.p.screen())
        self._pos = self._position()
        if self._pos is not None and not self._paused():
            self.cells.visit(*self._pos, self.t)
        return self._observe()

    # ------------------------------------------------------------------ helpers
    def _position(self) -> tuple[int, int, int] | None:
        if not self.has_position:
            return None
        pos = self.spec.position
        room = self.mem.get(pos.room) if pos.room else 0
        return room, self.mem.get(pos.x), self.mem.get(pos.y)

    def _paused(self) -> bool:
        pw = self.spec.position.pause_when if self.spec.position else None
        return pw is not None and self.mem.has(pw.field) and self.mem.check(pw)

    def _score(self) -> float:
        if self.spec.swarm.score == "milestones":
            return self.progress.milestone_score()
        return self.progress.state_score()

    def _press(self, button: str) -> None:
        ctl = self.spec.controls
        self.p.press(button)
        self.p.tick(ctl.press_frames, False)
        self.p.release(button)

    def _run_action(self, button: str) -> int:
        ctl = self.spec.controls
        self._press(button)
        self.p.tick(ctl.frames_per_action - ctl.press_frames - 1, False)
        waits = 0
        if self._wait is not None:
            self.mem.invalidate()
            while waits < ctl.wait_max_loops and self.mem.check(self._wait):
                self._press(ctl.wait_button)
                self.p.tick(ctl.frames_per_action - ctl.press_frames, False)
                self.mem.invalidate()
                waits += 1
        self.p.tick(1, True)
        self.mem.invalidate()
        return waits

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        button = self.buttons[action]
        before_levels = self._levels
        before_pos = self._pos
        waits = self._run_action(button)
        self.t += 1
        self.ep_step += 1
        self.mem_step += 1
        if button in DIRECTION_BUTTONS:
            self.facing = DIRECTION_BUTTONS.index(button)

        self._levels = self.screen.levels_image(self.p.screen())
        self._pos = self._position()
        ex = self.spec.exploration
        parts: dict[str, float] = {}

        if self._pos is not None and not self._paused():
            gain = self.cells.visit(*self._pos, self.t)
            if gain > 0:
                parts["explore"] = ex.cell_weight * gain
            if ex.interaction_weight > 0 and button in ex.interaction_buttons:
                moved = before_pos != self._pos
                hit = self.interact.observe(*self._pos, self.facing, moved, before_levels, self._levels)
                if hit:
                    parts["interact"] = ex.interaction_weight * hit
        if ex.screen_novelty_weight > 0:
            parts["screen_novelty"] = ex.screen_novelty_weight * self.screen_novelty.observe(self._levels)

        prog, prog_parts, new_milestones = self.progress.step(self.t)
        parts.update(prog_parts)
        reward = float(sum(parts.values()))
        for k, v in parts.items():
            self.reward_parts[k] = self.reward_parts.get(k, 0.0) + v
        self.ep_return += reward
        if prog > 0:
            self.last_reward_step = self.ep_step

        info: dict = {}
        if new_milestones:
            info["milestones"] = {m: self.ep_step for m in new_milestones}
        # frontier report for the swarm
        sw = self.spec.swarm
        if sw.enabled:
            score = self._score()
            if score >= self.best_score + sw.min_delta:
                self.best_score = score
                info["frontier"] = {"score": score, "state": self.p.save_state()}
        if waits:
            info["waits"] = waits

        # mini-episode bookkeeping
        ep = self.spec.episode
        if self.mem_step >= self.mem_horizon:
            info["episode"] = self.episode_stats()
            self._wipe_memory()
        done = False
        if (ep.hard_reset_steps and self.ep_step >= ep.hard_reset_steps) or (
            ep.stall_reset_steps and self.ep_step - self.last_reward_step >= ep.stall_reset_steps
        ):
            info.setdefault("episode", self.episode_stats())
            obs = self.reset()
            self.prev_action = -1
            return obs, reward, True, info

        self.prev_action = action
        return self._observe(), reward, done, info

    def episode_stats(self) -> dict:
        return {
            "return": self.ep_return,
            "length": self.ep_step,
            "cells": self.cells.unique,
            "rooms": len(self.cells.rooms),
            "milestones": dict(self.progress.reached),
            "parts": dict(self.reward_parts),
            "score": self._score(),
        }

    # ------------------------------------------------------------------ obs
    def _observe(self) -> dict[str, np.ndarray]:
        os_ = self.obs_spec
        pix = np.empty(os_.pixels, dtype=np.uint8)
        pix[0] = self.screen.pack(self._levels)
        if self.mask is not None:
            if self._pos is not None and not self._paused():
                m = self.mask.render(self.cells, *self._pos, self.t)
            else:
                m = self.mask.blank()
            pix[1] = self.screen.pack(m)

        bits = np.zeros(max(os_.bits, 1), dtype=np.uint8)
        scal = np.zeros(max(os_.scalars, 1), dtype=np.float32)
        cats = np.zeros(max(len(os_.cats), 1), dtype=np.int32)
        ib = isc = ic = 0
        for v in self.vector:
            f = self.spec.memory[v.field]
            if v.kind == "bits":
                raw = self.mem.block(v.field) if f.is_block else self.mem.array(v.field).astype(np.uint8)
                bits[ib : ib + len(raw)] = raw
                ib += len(raw)
            elif v.kind == "categorical":
                a = np.clip(self.mem.array(v.field), 0, v.num - 1)
                cats[ic : ic + len(a)] = a
                ic += len(a)
            elif v.kind == "ratio":
                num = self.mem.array(v.field).astype(np.float32)
                den = self.mem.array(v.denom).astype(np.float32)
                scal[isc : isc + len(num)] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                isc += len(num)
            else:
                a = self.mem.array(v.field).astype(np.float32) * v.scale
                scal[isc : isc + len(a)] = a
                isc += len(a)
        if self.prev_action >= 0:
            scal[isc + self.prev_action] = 1.0
        aux = np.full(5, -1, dtype=np.int32)
        if self._pos is not None:
            aux[0:3] = self._pos
            aux[3] = int(self._paused())
        aux[4] = int(round(self._score() * 100))
        return {"pixels": pix, "bits": bits, "scalars": scal, "cats": cats, "aux": aux}

    # ------------------------------------------------------------------ swarm / io
    def load_state(self, state: bytes) -> dict[str, np.ndarray]:
        """Swarm migration: jump to ``state`` and adopt it as this env's restart point."""
        self.frontier_state = state
        return self.reset(state)

    def save_state(self) -> bytes:
        return self.p.save_state()

    def render_rgb(self) -> np.ndarray:
        return np.array(self.p.screen())

    def close(self) -> None:
        self.p.close()
