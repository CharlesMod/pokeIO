"""PokeEnv — thin, deterministic Game Boy environment wrapper around PyBoy.

Phase 0 contract (see TODO.md):
  * observation  = raw grayscale screen ndarray, shape (144, 160), uint8
  * action space = Discrete(8): up, down, left, right, A, B, START, SELECT
  * frame-skip   ~= 24 ticks per action; button held ~8 frames then released
  * render=False during skipped frames; only the final frame of an action renders
  * done is always False for now (no episode-termination logic yet)

Input models (selected via ``sticky_input``):
  * sticky  (default): the chosen button is pressed and HELD DOWN across ticks;
    it is only released when the agent selects a DIFFERENT action. On an action
    change we release the previously-held button then press the new one. This is
    what makes frame_skip=1 register input at all: a held button spans a full
    frame boundary, whereas a press+release inside a single tick is invisible to
    the game. All 8 actions are buttons, so every action is "held-until-changed";
    there is no dedicated no-op/release action (see ``release_all`` if one is
    ever needed).
  * pulsed (sticky_input=False): the legacy behavior — press, hold hold_frames,
    release, coast the remainder. Requires hold_frames < frame_skip and only
    registers input when hold_frames >= 1 (so it is broken at frame_skip=1).

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

    def __init__(
        self,
        rom_path: str,
        frame_skip: int = 24,
        hold_frames: int = 8,
        sticky_input: bool = True,
    ):
        # hold_frames only governs the legacy "pulsed" path; it is irrelevant
        # (and unconstrained) in sticky mode, so only validate when it applies.
        if not sticky_input and hold_frames >= frame_skip:
            raise ValueError("hold_frames must be < frame_skip")
        self.rom_path = rom_path
        self.frame_skip = int(frame_skip)
        self.hold_frames = int(hold_frames)
        self.sticky_input = bool(sticky_input)
        # Name of the currently-held button in sticky mode (None == nothing held).
        self._held: str | None = None
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

    def wram_strided(self, stride: int = 64) -> np.ndarray:
        """A strided slice of the WRAM block: ``memory[0xC000:0xE000:stride]``.

        The hot path (novelty cell key + obs RAM bytes) only ever samples WRAM at
        a stride, so reading the full 8 KB and slicing wastes ~0.38 ms/step. The
        emulator's strided read materialises ~128 bytes directly (~0.01 ms). The
        default stride matches :attr:`NoveltyArchive.wram_stride`, so the returned
        vector is exactly ``raw_wram()[::stride]`` — the compact/full paths agree
        byte-for-byte.
        """
        block = self.pyboy.memory[WRAM_START:WRAM_END:stride]
        return np.frombuffer(bytes(block), dtype=np.uint8)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, state_path: str | None = None) -> np.ndarray:
        """Reset to a canonical state (if given) or boot fresh; return the observation."""
        if state_path is not None:
            with open(state_path, "rb") as fh:
                self.pyboy.load_state(fh)
        # A loaded state carries no held input; drop any bookkeeping so the first
        # sticky action re-presses cleanly.
        self._held = None
        # settle one rendered frame so the returned obs is valid
        self.pyboy.tick(1, True)
        return self._obs()

    def step(self, action_idx: int):
        """Apply one action over `frame_skip` ticks. Returns (obs, ram, done, info)."""
        name = ACTIONS[action_idx]

        if self.sticky_input:
            self._advance_sticky(name)
        else:
            self._advance_pulsed(name)

        obs = self._obs()
        ram = self.raw_wram()
        done = False
        info = {"action": name, "map_id": self.pyboy.memory[MAP_ID_ADDR]}
        return obs, ram, done, info

    def step_fast(self, action_idx: int, wram_stride: int = 64):
        """Slim hot-path step: advance one action, return ``(obs, wram64, done)``.

        Identical dynamics to :meth:`step` but (a) reads WRAM strided instead of
        the full 8 KB block and (b) skips the per-step ``info`` dict allocation.
        Used by the parallel worker fleet where map-id / action name are not
        needed on the hot path (measured ~3.10 ms -> ~2.41 ms/step on this box
        under load). ``done`` is always ``False`` for now (see class docstring).
        """
        name = ACTIONS[action_idx]
        if self.sticky_input:
            self._advance_sticky(name)
        else:
            self._advance_pulsed(name)
        return self._obs(), self.wram_strided(wram_stride), False

    def hold(self, action_idx: int) -> None:
        """Sticky-press an action's button without advancing a frame."""
        name = ACTIONS[action_idx]
        if name != self._held:
            if self._held is not None:
                self.pyboy.button_release(self._held)
            self.pyboy.button_press(name)
            self._held = name

    def tick_frames(self, n: int) -> np.ndarray:
        """Advance ``n`` game frames with the held input; render only the last.

        Sub-agent-step ticking for smooth spectator playback: ``hold()`` +
        ``frame_skip`` total ``tick_frames`` == one sticky :meth:`step`'s
        dynamics (rendering intermediate frames does not perturb emulation).
        Returns the observation after the last frame.
        """
        if n > 1:
            self.pyboy.tick(n - 1, False)
        if n >= 1:
            self.pyboy.tick(1, True)
        return self._obs()

    def _advance_sticky(self, name: str) -> None:
        """Hold `name` down across frame_skip ticks; only re-press on a change.

        On an action change we release the previously-held button and press the
        new one *before* ticking, so the newly-pressed button is down for the
        whole span (>= 1 full frame boundary) and the game actually reads it.
        A repeat of the same action leaves the button held — no release/press —
        which is exactly the continuous-hold movement the overworld needs.
        """
        if name != self._held:
            if self._held is not None:
                self.pyboy.button_release(self._held)
            self.pyboy.button_press(name)
            self._held = name
        # Advance the frame(s) with the button still held; render only the last.
        if self.frame_skip > 1:
            self.pyboy.tick(self.frame_skip - 1, False)
        self.pyboy.tick(1, True)

    def _advance_pulsed(self, name: str) -> None:
        """Legacy press/hold/release model (broken at frame_skip=1, hold=0)."""
        # Hold the button for hold_frames (skipped frames -> render=False).
        self.pyboy.button_press(name)
        if self.hold_frames > 0:
            self.pyboy.tick(self.hold_frames, False)
        self.pyboy.button_release(name)

        # Coast the remaining frames; render only the very last one.
        remaining = self.frame_skip - self.hold_frames
        if remaining > 1:
            self.pyboy.tick(remaining - 1, False)
        self.pyboy.tick(1, True)

    # ----------------------------------------------------------------- state io
    def release_all(self) -> None:
        """Release every button (sticky no-op) and forget the held action."""
        for name in ACTIONS:
            self.pyboy.button_release(name)
        self._held = None

    def _flush_input(self) -> None:
        """Clear any queued/held input so a following save_state is deterministic."""
        self.release_all()
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
