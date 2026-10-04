"""Generic exploration signals.

``CellMemory``: memory of visited (room, x, y) cells. Visiting a cell pays
``1 - strength_before``: 1.0 the first time, 0 afterwards. By default memory is
binary and wiped every mini-episode, which is what puffer's game-winning runs used.
With ``half_life_steps > 0`` strength decays instead (puffer's experimental
DecayWrapper), so stale regions become worth revisiting, up to ``1 - floor``.
That is O(1) per step via timestamps, and has no negative reward drift. The strength map,
cropped around the player and aligned to the screen, is the "visited mask"
observation that every successful Pokémon agent relied on.

``InteractionNovelty``: pressing an interaction button that visibly changes the
screen while standing still (dialog, sign, menu, item) at a new
(room, x, y, facing) pays once. A generic stand-in for puffer's per-game
sign/NPC/hidden-object hooks.

``ScreenNovelty``: count-based bonus over hashes of a coarse screen thumbnail.
The only exploration signal available to pixel-only specs (no position).
Weaker: animated tiles and sprites alias into new hashes.
"""

from __future__ import annotations

import math

import numpy as np

_NEVER = -1e18


class _RoomGrid:
    __slots__ = ("t",)

    def __init__(self, w: int = 32, h: int = 32):
        self.t = np.full((h, w), _NEVER, dtype=np.float64)

    def ensure(self, x: int, y: int) -> None:
        h, w = self.t.shape
        if x < w and y < h:
            return
        nh, nw = h, w
        while y >= nh:
            nh *= 2
        while x >= nw:
            nw *= 2
        g = np.full((nh, nw), _NEVER, dtype=np.float64)
        g[:h, :w] = self.t
        self.t = g

    def window(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        out = np.full((h, w), _NEVER, dtype=np.float64)
        gh, gw = self.t.shape
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + w, gw), min(y0 + h, gh)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = self.t[sy0:sy1, sx0:sx1]
        return out


class CellMemory:
    def __init__(self, half_life_steps: float, floor: float):
        self.decays = half_life_steps > 0
        self.log_decay = math.log(0.5) / half_life_steps if self.decays else 0.0
        self.floor = floor if self.decays else 1.0
        self.rooms: dict[int, _RoomGrid] = {}
        self.unique = 0  # distinct cells since the last wipe

    def wipe(self) -> None:
        self.rooms.clear()
        self.unique = 0

    def _strength(self, t_last: np.ndarray | float, t: float):
        s = np.exp(self.log_decay * (t - np.asarray(t_last)))
        s = np.maximum(s, self.floor)
        return np.where(np.asarray(t_last) <= _NEVER / 2, 0.0, s)

    def visit(self, room: int, x: int, y: int, t: float) -> float:
        """Mark the cell visited at step ``t``; return the refresh gain in [0, 1]."""
        if x < 0 or y < 0:
            return 0.0
        g = self.rooms.get(room)
        if g is None:
            g = self.rooms[room] = _RoomGrid()
        g.ensure(x, y)
        last = g.t[y, x]
        if last <= _NEVER / 2:
            self.unique += 1
            gain = 1.0
        else:
            gain = 1.0 - float(self._strength(last, t))
        g.t[y, x] = t
        return gain

    def strength_window(self, room: int, x0: int, y0: int, w: int, h: int, t: float) -> np.ndarray:
        g = self.rooms.get(room)
        if g is None:
            return np.zeros((h, w), dtype=np.float64)
        return self._strength(g.window(x0, y0, w, h), t)


class VisitedMask:
    """Renders CellMemory around the player into a screen-aligned image."""

    def __init__(
        self,
        screen_h: int,
        screen_w: int,
        downsample: int,
        cell_px: int,
        player_cell_px: tuple[int, int],
        levels: int,
        camera: str = "follow",
    ):
        self.fixed = camera == "fixed"
        self.c = max(cell_px // downsample, 1)
        self.ox = player_cell_px[0] // downsample
        self.oy = player_cell_px[1] // downsample
        self.h = -(-screen_h // downsample)
        self.w = -(-screen_w // downsample)
        self.nl = -(-self.ox // self.c)
        self.nr = -(-(self.w - self.ox) // self.c)
        self.nt = -(-self.oy // self.c)
        self.nb = -(-(self.h - self.oy) // self.c)
        self.levels = levels
        self.r0 = self.nt * self.c - self.oy
        self.c0 = self.nl * self.c - self.ox

    def render(self, mem: CellMemory, room: int, x: int, y: int, t: float) -> np.ndarray:
        if self.fixed:
            s = mem.strength_window(room, 0, 0, -(-self.w // self.c), -(-self.h // self.c), t)
            q = np.rint(s * (self.levels - 1)).astype(np.uint8)
            big = np.repeat(np.repeat(q, self.c, axis=0), self.c, axis=1)
            return big[: self.h, : self.w]
        s = mem.strength_window(room, x - self.nl, y - self.nt, self.nl + self.nr, self.nt + self.nb, t)
        q = np.rint(s * (self.levels - 1)).astype(np.uint8)
        big = np.repeat(np.repeat(q, self.c, axis=0), self.c, axis=1)
        return big[self.r0 : self.r0 + self.h, self.c0 : self.c0 + self.w]

    def blank(self) -> np.ndarray:
        return np.zeros((self.h, self.w), dtype=np.uint8)


class InteractionNovelty:
    def __init__(self, min_change: float):
        self.min_change = min_change
        self.seen: set[tuple[int, int, int, int]] = set()

    def wipe(self) -> None:
        self.seen.clear()

    def observe(
        self,
        room: int,
        x: int,
        y: int,
        facing: int,
        moved: bool,
        before: np.ndarray,
        after: np.ndarray,
    ) -> float:
        if moved:
            return 0.0
        changed = float(np.count_nonzero(before != after)) / before.size
        if changed < self.min_change:
            return 0.0
        key = (room, x, y, facing)
        if key in self.seen:
            return 0.0
        self.seen.add(key)
        return 1.0


class ScreenNovelty:
    def __init__(self, grid: tuple[int, int], levels: int):
        self.rows, self.cols = grid
        self.levels = levels
        self.counts: dict[bytes, int] = {}

    def wipe(self) -> None:
        self.counts.clear()

    def key(self, levels_img: np.ndarray) -> bytes:
        h, w = levels_img.shape
        rh, cw = h // self.rows, w // self.cols
        thumb = levels_img[: rh * self.rows, : cw * self.cols].reshape(self.rows, rh, self.cols, cw)
        thumb = np.rint(thumb.mean(axis=(1, 3))).astype(np.uint8)
        return thumb.tobytes()

    def observe(self, levels_img: np.ndarray) -> float:
        k = self.key(levels_img)
        n = self.counts.get(k, 0) + 1
        self.counts[k] = n
        return 1.0 / math.sqrt(n)
