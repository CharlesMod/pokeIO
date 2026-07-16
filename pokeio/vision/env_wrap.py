"""VisionEnv — wraps PokeEnv so reset/step return the built observation dict.

This does NOT modify PokeEnv; it composes it. The raw grayscale screen and the
raw WRAM block stay reachable via passthroughs so the RAM miner (Phase 3) and
telemetry can still read them.

    reset(state_path) -> obs dict     (resets the ObsBuilder's motion memory)
    step(action)      -> (obs dict, ram, done, info)

``obs`` is the dict from ObsBuilder.build: coarse / fovea / motion / ram_aux.
"""

from __future__ import annotations

import numpy as np

from pokeio.config import Config
from pokeio.emu.env import PokeEnv
from pokeio.vision.preprocess import ObsBuilder


class VisionEnv:
    """A PokeEnv whose observations are the Phase 1 vision obs dict."""

    def __init__(
        self,
        rom_path: str | None = None,
        config: Config | None = None,
        frame_skip: int | None = None,
        hold_frames: int | None = None,
    ):
        self.config = config or Config()
        rom = rom_path or self.config.emu.rom_path
        fs = self.config.emu.frame_skip if frame_skip is None else frame_skip
        hf = (
            self.config.emu.button_hold_frames
            if hold_frames is None
            else hold_frames
        )
        self.env = PokeEnv(rom, frame_skip=fs, hold_frames=hf)
        self.builder = ObsBuilder(self.config)
        self._last_screen: np.ndarray | None = None

    # ------------------------------------------------------------------ lifecycle
    def reset(self, state_path: str | None = None) -> dict[str, np.ndarray]:
        """Reset the emulator (and the obs builder's motion memory)."""
        self.builder.reset()
        screen = self.env.reset(state_path)
        self._last_screen = screen
        return self.builder.build(screen, ram_vector=self.env.raw_wram())

    def step(self, action_idx: int):
        """Step the emulator; return (obs_dict, ram, done, info)."""
        screen, ram, done, info = self.env.step(action_idx)
        self._last_screen = screen
        obs = self.builder.build(screen, ram_vector=ram)
        return obs, ram, done, info

    # ------------------------------------------------------------------ passthrough
    def raw_screen(self) -> np.ndarray | None:
        """The most recent raw grayscale (144,160) uint8 screen."""
        return self._last_screen

    def raw_wram(self) -> np.ndarray:
        """The raw 8 KB WRAM block (for the RAM miner)."""
        return self.env.raw_wram()

    def save_state(self) -> bytes:
        return self.env.save_state()

    def load_state(self, data: bytes) -> None:
        self.env.load_state(data)

    def close(self) -> None:
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["VisionEnv"]
