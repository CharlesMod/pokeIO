"""Game Boy / Game Boy Color via PyBoy (>= 2.0).

Speed notes (see docs/puffer-lessons.md):
  * window="null", sound off: ~20k+ frames/s per core.
  * Only the last frame of an action is rendered; all others tick with render=False.
  * ``cgb`` can force DMG mode for dual-mode carts (exact 4-shade output).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pokeio.spec import GameSpec


class GameBoy:
    height = 144
    width = 160

    def __init__(self, spec: "GameSpec", headless: bool = True):
        from pyboy import PyBoy

        kwargs = dict(
            window="null" if headless else "SDL2",
            sound_emulated=False,
            log_level="CRITICAL",
        )
        if "cgb" in spec.platform_options:
            kwargs["cgb"] = bool(spec.platform_options["cgb"])
        if spec.symbols is not None and spec.symbols.exists():
            kwargs["symbols"] = str(spec.symbols)
        self._pyboy = PyBoy(str(spec.rom), **kwargs)
        if not headless:
            self._pyboy.set_emulation_speed(int(spec.platform_options.get("speed", 6)))
        self._mem = self._pyboy.memory

    def press(self, button: str) -> None:
        if button != "noop":
            self._pyboy.button_press(button)

    def release(self, button: str) -> None:
        if button != "noop":
            self._pyboy.button_release(button)

    def tick(self, frames: int, render: bool) -> None:
        if frames > 0:
            self._pyboy.tick(frames, render)

    def screen(self) -> np.ndarray:
        # (144, 160, 4) RGBA view into PyBoy's buffer; drop alpha.
        return self._pyboy.screen.ndarray[:, :, :3]

    def read(self, addr: int, n: int = 1) -> np.ndarray:
        if n == 1:
            return np.array([self._mem[addr]], dtype=np.uint8)
        return np.asarray(self._mem[addr : addr + n], dtype=np.uint8)

    def write(self, addr: int, data: bytes | list[int]) -> None:
        data = list(data)
        if len(data) == 1:
            self._mem[addr] = data[0]
        else:
            self._mem[addr : addr + len(data)] = data

    def save_state(self) -> bytes:
        buf = io.BytesIO()
        self._pyboy.save_state(buf)
        return buf.getvalue()

    def load_state(self, data: bytes) -> None:
        self._pyboy.load_state(io.BytesIO(data))

    def symbol(self, name: str) -> int | None:
        try:
            _bank, addr = self._pyboy.symbol_lookup(name)
            return int(addr)
        except Exception:
            return None

    def close(self) -> None:
        self._pyboy.stop(save=False)
