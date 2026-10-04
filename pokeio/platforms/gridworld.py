"""A tiny deterministic fake console for tests and ROM-free smoke runs.

It mimics the structure that matters for the engine: rooms of 16-px cells, a
player position in RAM, interactable NPCs that set event bits and open a
"dialog" during which input is ignored, and doors gated on events. It renders
a 144x160 4-shade screen. RAM layout (exposed through ``read``/``write``)::

    0x00 room   0x01 x   0x02 y   0x03 facing   0x04 busy (input-ignored frames)
    0x10..0x13  event flags (32 bits)   0x20 events_count mirror
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pokeio.spec import GameSpec

ROOM_W, ROOM_H, CELL = 10, 9, 16
SHADES = np.array([255, 170, 85, 0], dtype=np.uint8)
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
FACING = {"down": 0, "up": 1, "left": 2, "right": 3}


def _build_world(n_rooms: int, seed: int):
    rng = np.random.default_rng(seed)
    rooms = []
    for r in range(n_rooms):
        grid = np.zeros((ROOM_H, ROOM_W), dtype=np.uint8)  # 0 floor, 1 wall
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
        for _ in range(6):
            grid[rng.integers(1, ROOM_H - 1), rng.integers(1, ROOM_W - 1)] = 1
        npc = (int(rng.integers(2, ROOM_W - 2)), int(rng.integers(2, ROOM_H - 2)))
        grid[npc[1], npc[0]] = 2
        door = (ROOM_W - 1, ROOM_H // 2)  # east door -> next room, gated on this room's event
        grid[door[1], door[0]] = 3 if r < n_rooms - 1 else 1
        grid[door[1], door[0] - 1] = 0
        rooms.append({"grid": grid, "npc": npc, "door": door})
    return rooms


class GridWorld:
    height = ROOM_H * CELL  # 144
    width = ROOM_W * CELL  # 160

    def __init__(self, spec: "GameSpec" | None = None, n_rooms: int = 4, seed: int = 0):
        opts = spec.platform_options if spec is not None else {}
        self.rooms = _build_world(int(opts.get("rooms", n_rooms)), int(opts.get("seed", seed)))
        self.ram = np.zeros(0x40, dtype=np.uint8)
        self.ram[0x01], self.ram[0x02] = 1, 1
        self._screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._render()

    # -- input -------------------------------------------------------------
    def press(self, button: str) -> None:
        if self.ram[0x04] > 0:
            return
        room = self.rooms[self.ram[0x00]]
        x, y = int(self.ram[0x01]), int(self.ram[0x02])
        if button in DIRS:
            self.ram[0x03] = FACING[button]
            dx, dy = DIRS[button]
            nx, ny = x + dx, y + dy
            tile = room["grid"][ny, nx]
            if tile == 0:
                self.ram[0x01], self.ram[0x02] = nx, ny
            elif tile == 3 and self._event(self.ram[0x00]):
                self.ram[0x00] += 1
                self.ram[0x01], self.ram[0x02] = 1, ROOM_H // 2
        elif button == "a":
            fx, fy = list(DIRS.values())[[1, 0, 2, 3][self.ram[0x03]]]
            if (x + fx, y + fy) == room["npc"]:
                self._set_event(int(self.ram[0x00]))
                self.ram[0x04] = 48  # dialog: input ignored for 48 frames

    def release(self, button: str) -> None:
        pass

    def tick(self, frames: int, render: bool) -> None:
        self.ram[0x04] = max(0, int(self.ram[0x04]) - frames)
        if render:
            self._render()

    # -- events --------------------------------------------------------------
    def _event(self, i: int) -> bool:
        return bool(self.ram[0x10 + i // 8] & (1 << (i % 8)))

    def _set_event(self, i: int) -> None:
        self.ram[0x10 + i // 8] |= 1 << (i % 8)
        self.ram[0x20] = int(np.unpackbits(self.ram[0x10:0x14]).sum())

    # -- screen ----------------------------------------------------------------
    def _render(self) -> None:
        room = self.rooms[self.ram[0x00]]
        shade = np.choose(np.minimum(room["grid"], 3), [0, 2, 1, 1]).astype(np.int64)
        img = SHADES[np.repeat(np.repeat(shade, CELL, 0), CELL, 1)]
        x, y = int(self.ram[0x01]), int(self.ram[0x02])
        img[y * CELL + 4 : y * CELL + 12, x * CELL + 4 : x * CELL + 12] = SHADES[3]
        if self.ram[0x04] > 0:  # dialog box
            img[-3 * CELL :, :] = SHADES[0]
            img[-3 * CELL, :] = SHADES[3]
        self._screen[:] = img[:, :, None]

    def screen(self) -> np.ndarray:
        return self._screen

    # -- memory / state -----------------------------------------------------
    def read(self, addr: int, n: int = 1) -> np.ndarray:
        return self.ram[addr : addr + n].copy()

    def write(self, addr: int, data) -> None:
        data = list(data)
        self.ram[addr : addr + len(data)] = data

    def save_state(self) -> bytes:
        return pickle.dumps(self.ram.copy())

    def load_state(self, data: bytes) -> None:
        self.ram[:] = pickle.loads(data)
        self._render()

    def symbol(self, name: str) -> int | None:
        return None

    def close(self) -> None:
        pass
