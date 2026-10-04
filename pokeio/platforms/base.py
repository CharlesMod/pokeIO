from __future__ import annotations

from typing import Protocol

import numpy as np


class Platform(Protocol):
    """One emulator instance.

    ``screen()`` is only guaranteed fresh after a ``tick(..., render=True)``.
    Memory addresses are the console's CPU address space.
    """

    height: int
    width: int

    def press(self, button: str) -> None: ...

    def release(self, button: str) -> None: ...

    def tick(self, frames: int, render: bool) -> None: ...

    def screen(self) -> np.ndarray:
        """(H, W, 3) uint8 RGB of the last rendered frame."""
        ...

    def read(self, addr: int, n: int = 1) -> np.ndarray:
        """``n`` bytes starting at ``addr`` as a uint8 array."""
        ...

    def write(self, addr: int, data: bytes | list[int]) -> None: ...

    def save_state(self) -> bytes: ...

    def load_state(self, data: bytes) -> None: ...

    def symbol(self, name: str) -> int | None:
        """Address of a debug symbol, if the platform has a symbol table."""
        ...

    def close(self) -> None: ...
