"""PokeEnv — thin, deterministic Game Boy environment wrapper around PyBoy.

Phase 0 contract (see TODO.md):
  * observation  = raw grayscale screen ndarray, shape (144, 160), uint8
  * action space = Discrete(8): up, down, left, right, A, B, START, SELECT
  * frame-skip   ~= 24 ticks per action; button held ~8 frames then released
  * render=False during skipped frames; only the final frame of an action renders
  * done is always False for now (no episode-termination logic yet)

Determinism: given a save-state + a fixed action sequence, replay is identical.
Any queued input is flushed before a save so state snapshots are reproducible.
"""

from __future__ import annotations

import io

import numpy as np
from pyboy import PyBoy

# Discrete(8) action space -> PyBoy button names.
ACTIONS = ("up", "down", "left", "right", "a", "b", "start", "select")

# WRAM working block exposed for the (future) RAM miner: 0xC000-0xDFFF inclusive.
WRAM_START = 0xC000
WRAM_END = 0xE000  # exclusive; 0xE000 - 0xC000 == 8192 bytes
MAP_ID_ADDR = 0xD35E  # current map id (Pokemon Yellow); handy for state docs


class PokeEnv:
    """A single headless PyBoy instance with a gym-flavoured step/reset API."""

    def __init__(self, rom_path: str, frame_skip: int = 24, hold_frames: int = 8):
        if hold_frames >= frame_skip:
            raise ValueError("hold_frames must be < frame_skip")
        self.rom_path = rom_path
        self.frame_skip = int(frame_skip)
        self.hold_frames = int(hold_frames)
        self.pyboy = PyBoy(rom_path, window="null", sound_emulated=False)

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        """Raw grayscale screen (144, 160) uint8. Channel 0 of the DMG buffer."""
        # screen.ndarray is a live (144,160,4) view; copy so callers keep a stable frame.
        return np.array(self.pyboy.screen.ndarray[:, :, 0], dtype=np.uint8)

    def raw_wram(self) -> np.ndarray:
        """The 8 KB WRAM block 0xC000-0xDFFF as a uint8 ndarray (for the RAM miner)."""
        block = self.pyboy.memory[WRAM_START:WRAM_END]
        return np.frombuffer(bytes(block), dtype=np.uint8)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, state_path: str | None = None) -> np.ndarray:
        """Reset to a canonical state (if given) or boot fresh; return the observation."""
        if state_path is not None:
            with open(state_path, "rb") as fh:
                self.pyboy.load_state(fh)
        # settle one rendered frame so the returned obs is valid
        self.pyboy.tick(1, True)
        return self._obs()

    def step(self, action_idx: int):
        """Apply one action over `frame_skip` ticks. Returns (obs, ram, done, info)."""
        name = ACTIONS[action_idx]

        # Hold the button for hold_frames (skipped frames -> render=False).
        self.pyboy.button_press(name)
        self.pyboy.tick(self.hold_frames, False)
        self.pyboy.button_release(name)

        # Coast the remaining frames; render only the very last one.
        remaining = self.frame_skip - self.hold_frames
        if remaining > 1:
            self.pyboy.tick(remaining - 1, False)
        self.pyboy.tick(1, True)

        obs = self._obs()
        ram = self.raw_wram()
        done = False
        info = {"action": name, "map_id": self.pyboy.memory[MAP_ID_ADDR]}
        return obs, ram, done, info

    # ----------------------------------------------------------------- state io
    def _flush_input(self) -> None:
        """Clear any queued/held input so a following save_state is deterministic."""
        for name in ACTIONS:
            self.pyboy.button_release(name)
        # A single tick drains PyBoy's internal input queue into a settled state.
        self.pyboy.tick(1, False)

    def save_state(self) -> bytes:
        """Flush queued input, then return a serialized emulator state as bytes."""
        self._flush_input()
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        return buf.getvalue()

    def load_state(self, data: bytes) -> None:
        """Restore emulator state from bytes produced by save_state()."""
        self.pyboy.load_state(io.BytesIO(data))

    def close(self) -> None:
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
