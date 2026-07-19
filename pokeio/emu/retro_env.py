"""RetroPokeEnv — a libretro-backed drop-in for :class:`pokeio.emu.env.PokeEnv`.

The multi-console foundation (#36, the developmental-ladder North Star): same
observation / RAM / save-state / sticky-input contract as PokeEnv, but backed by a
fast native libretro core (:mod:`pokeio.emu.libretro`) that swaps per console
(gambatte GB/GBC, mgba GBA, parallel_n64 N64, …). It exposes a ``.pyboy`` SHIM
because ~30 downstream consumers reach through ``env.pyboy.save_state / load_state /
tick / memory / screen / button_*`` (the fleet taps, Go-Explore capture,
brain/replay); the shim contains the blast radius to this file.

Regime note (Fable): Pokémon Yellow runs CGB (color) under PyBoy today; this core
is forced to **DMG** so SYSTEM_RAM is the clean 8192-byte $C000-$DFFF block. That
makes the pixels grayscale — a different perceptual regime from the PyBoy stack, so
the demo + a trained policy must be re-derived on this core (a deliberate migration,
not a transparent swap). RAM taps + save-states + determinism DO port (validated).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np

from pokeio.emu.env import ACTIONS, WRAM_END, WRAM_START
from pokeio.emu.libretro import LibretroCore

_SCREEN_H, _SCREEN_W = 144, 160
_DPAD = frozenset(("up", "down", "left", "right"))
_TAP_GAP = 2

# libretro cores live here, fetched on demand (scripts/fetch_cores.py); gitignored.
CORES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets", "cores")


def core_path(name: str) -> str:
    """Absolute path to a fetched libretro core .so (e.g. 'gambatte_libretro.so')."""
    return os.path.join(CORES_DIR, name)


@dataclass(frozen=True)
class ConsoleSpec:
    """Per-console knobs (the ladder plug-point). Defaults = Game Boy / Gambatte-DMG."""

    core_path: str
    screen_h: int = _SCREEN_H
    screen_w: int = _SCREEN_W
    wram_base: int = WRAM_START           # SYSTEM_RAM index 0 -> this GB address
    options: dict | None = None           # libretro core-variable overrides

    @staticmethod
    def game_boy(core: str | None = None) -> "ConsoleSpec":
        """Game Boy / Gambatte spec. ``gambatte_gb_hwmode=GB`` forces DMG so
        SYSTEM_RAM is the clean 8192-byte $C000-$DFFF block. ``core`` defaults to the
        fetched ``assets/cores/gambatte_libretro.so``."""
        return ConsoleSpec(core_path=core or core_path("gambatte_libretro.so"),
                           options={"gambatte_gb_hwmode": "GB"})


class _Memory:
    """``env.pyboy.memory[addr]`` / ``[a:b]`` over SYSTEM_RAM (WRAM range only)."""

    def __init__(self, core: LibretroCore, base: int) -> None:
        self._core = core
        self._base = base

    def _ram(self) -> np.ndarray:
        return self._core.system_ram()

    def __getitem__(self, key):
        ram = self._ram()
        if isinstance(key, slice):
            lo = (key.start or self._base) - self._base
            hi = (key.stop if key.stop is not None else self._base + ram.size) - self._base
            return ram[lo:hi:key.step]
        i = int(key) - self._base
        return int(ram[i]) if 0 <= i < ram.size else 0


class _Screen:
    """``env.pyboy.screen.ndarray`` -> (H, W, 4) uint8 [R,G,B,255] (channel 0 = obs)."""

    def __init__(self, env: "RetroPokeEnv") -> None:
        self._env = env

    @property
    def ndarray(self) -> np.ndarray:
        rgb = self._env._core.framebuffer_rgb()  # (H,W,3)
        h, w = self._env.spec.screen_h, self._env.spec.screen_w
        if rgb.shape[:2] != (h, w):  # defensive: crop/pad to the console dims
            out = np.zeros((h, w, 3), np.uint8)
            hh, ww = min(h, rgb.shape[0]), min(w, rgb.shape[1])
            out[:hh, :ww] = rgb[:hh, :ww]
            rgb = out
        a = np.full((h, w, 1), 255, np.uint8)
        return np.concatenate([rgb, a], axis=-1)


class _PyBoyShim:
    """The ``env.pyboy`` surface the downstream consumers reach through."""

    def __init__(self, env: "RetroPokeEnv") -> None:
        self._env = env
        self.memory = _Memory(env._core, env.spec.wram_base)
        self.screen = _Screen(env)

    def tick(self, n: int, render: bool = True) -> bool:
        for _ in range(int(n)):
            self._env._core.run()
        return True

    def button_press(self, name: str) -> None:
        self._env._core.set_button(name, True)

    def button_release(self, name: str) -> None:
        self._env._core.set_button(name, False)

    def save_state(self, buf: io.BytesIO) -> None:
        buf.write(self._env._core.serialize())      # RAW (non-flushing) — fleet/goexplore path

    def load_state(self, buf: io.BytesIO) -> None:
        self._env._core.unserialize(buf.read())

    def stop(self, save: bool = False) -> None:
        self._env._core.close()


class RetroPokeEnv:
    """libretro-backed PokeEnv: same step/obs/RAM/save-state/sticky-input contract."""

    def __init__(self, rom_path: str, spec: ConsoleSpec, frame_skip: int = 24,
                 hold_frames: int = 8, sticky_input: bool = True) -> None:
        self.rom_path = str(rom_path)
        self.spec = spec
        self.frame_skip = int(frame_skip)
        self.hold_frames = int(hold_frames)
        self.sticky_input = bool(sticky_input)
        self._held: str | None = None
        self._core = LibretroCore(spec.core_path, rom_path, options=spec.options)
        self.pyboy = _PyBoyShim(self)

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        """Grayscale screen (H,W) uint8 = channel 0 (R) of the framebuffer."""
        return np.ascontiguousarray(self.pyboy.screen.ndarray[:, :, 0])

    def raw_wram(self) -> np.ndarray:
        """The WRAM block as a uint8 ndarray (copy), index i -> spec.wram_base + i."""
        return self._core.system_ram().copy()

    def wram_strided(self, stride: int = 64) -> np.ndarray:
        """``raw_wram()[::stride]`` — byte-identical to the strided fleet path."""
        return self._core.system_ram()[::stride].copy()

    # ---------------------------------------------------------------- input
    def _apply_input(self, name: str) -> int:
        if name == "noop":
            if self._held is not None:
                self._core.set_button(self._held, False)
                self._held = None
            return 0
        if name != self._held:
            if self._held is not None:
                self._core.set_button(self._held, False)
            self._core.set_button(name, True)
            self._held = name
            return 0
        if name in _DPAD:
            return 0
        # edge-read face button repeat: release, coast the gap, re-press
        self._core.set_button(name, False)
        self.pyboy.tick(_TAP_GAP, False)
        self._core.set_button(name, True)
        return _TAP_GAP

    def _advance_sticky(self, name: str) -> None:
        used = self._apply_input(name)
        remaining = self.frame_skip - used
        if remaining > 1:
            self.pyboy.tick(remaining - 1, False)
        if remaining >= 1:
            self.pyboy.tick(1, True)

    def release_all(self) -> None:
        self._core.release_all()
        self._held = None

    def _flush_input(self) -> None:
        self.release_all()
        self.pyboy.tick(1, False)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, state_path: str | None = None) -> np.ndarray:
        if state_path is not None:
            with open(state_path, "rb") as fh:
                self._core.unserialize(fh.read())
        self._held = None
        self._core.release_all()
        self.pyboy.tick(1, True)
        return self._obs()

    def step(self, action_idx: int):
        name = ACTIONS[action_idx]
        self._advance_sticky(name)
        info = {"action": name, "map_id": self.pyboy.memory[0xD35D]}
        return self._obs(), self.raw_wram(), False, info

    def step_fast(self, action_idx: int, wram_stride: int = 64):
        self._advance_sticky(ACTIONS[action_idx])
        return self._obs(), self.wram_strided(wram_stride), False

    def hold(self, action_idx: int) -> int:
        return self._apply_input(ACTIONS[action_idx])

    def tick_frames(self, n: int) -> np.ndarray:
        if n > 1:
            self.pyboy.tick(n - 1, False)
        if n >= 1:
            self.pyboy.tick(1, True)
        return self._obs()

    # ----------------------------------------------------------------- state io
    def save_state(self) -> bytes:
        self._flush_input()
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        return buf.getvalue()

    def load_state(self, data: bytes) -> None:
        self._core.unserialize(bytes(data))

    def close(self) -> None:
        self._core.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["RetroPokeEnv", "ConsoleSpec"]
