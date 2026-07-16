"""Go-Explore novelty archive — the v1 reward backbone for pokeIO.

A *cell* is a coarse, deterministic fingerprint of the game state built from two
cheap signals:

  * the **screen**, downscaled to a tiny grid (~16x14) and quantized to a few
    brightness levels — captures "roughly what is on screen";
  * a **WRAM digest**, a strided + quantized slice of the 8 KB working RAM block
    (``env.raw_wram()``) — captures coarse game-memory state (map id, menu
    cursor, flags, ...) so two visually-similar frames in different game states
    land in different cells (a noisy-TV guard the pure-pixel hash lacks).

The archive keeps a **global set of seen cells** across the whole run.  A genome
episode's *fitness* is the number of cells it was the first to add to that global
set — i.e. how much new frontier that episode opened up (classic Go-Explore
novelty).  ``archive_delta`` is how much the frontier grew over a generation.

Everything here is stdlib + numpy; hashing is deterministic so runs are
reproducible given the same trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# GB screen native size (see emu/env.py).
_SCREEN_H = 144
_SCREEN_W = 160


def _area_matrix(in_size: int, out_size: int) -> np.ndarray:
    """(out_size, in_size) row-normalized area-overlap resample matrix.

    Mirrors ``vision.preprocess._area_matrix``; kept local so the reward stack
    has no import-time dependency on the vision package.
    """
    m = np.zeros((out_size, in_size), dtype=np.float64)
    scale = in_size / out_size
    for i in range(out_size):
        lo, hi = i * scale, (i + 1) * scale
        j0, j1 = int(np.floor(lo)), int(np.ceil(hi))
        for j in range(j0, min(j1, in_size)):
            overlap = min(hi, j + 1) - max(lo, j)
            if overlap > 0:
                m[i, j] = overlap
        s = m[i].sum()
        if s > 0:
            m[i] /= s
    return m


@dataclass
class NoveltyArchive:
    """A global Go-Explore cell archive.

    Parameters
    ----------
    screen_cells:
        (rows, cols) of the downscaled screen grid used for the pixel part of a
        cell key.  ~16x14 keeps cells coarse enough to generalize.
    screen_levels:
        brightness quantization levels for the screen grid (3-4 is plenty).
    wram_stride:
        take every ``wram_stride``-th WRAM byte for the memory digest.
    wram_levels:
        quantization levels for the WRAM digest bytes.
    """

    screen_cells: tuple[int, int] = (16, 14)
    screen_levels: int = 4
    wram_stride: int = 64
    wram_levels: int = 16

    seen: set[bytes] = field(default_factory=set)

    def __post_init__(self) -> None:
        rows, cols = self.screen_cells
        # Precompute area-resample matrices: (rows, H) and (W, cols).
        self._row_mat = _area_matrix(_SCREEN_H, rows)
        self._col_mat = _area_matrix(_SCREEN_W, cols).T
        self._gen_start_size = 0

    # ------------------------------------------------------------------ keys
    def _screen_digest(self, screen: np.ndarray) -> np.ndarray:
        """Downscale the raw (144,160) uint8 screen and quantize to few levels."""
        g = screen.astype(np.float64)
        small = self._row_mat @ g @ self._col_mat  # (rows, cols) in [0,255]
        q = np.floor(small / 256.0 * self.screen_levels).astype(np.int16)
        np.clip(q, 0, self.screen_levels - 1, out=q)
        return q.astype(np.uint8)

    def _wram_digest(self, wram: np.ndarray) -> np.ndarray:
        """Strided + quantized slice of the 8 KB WRAM block."""
        sl = wram[:: self.wram_stride]
        step = max(1, 256 // self.wram_levels)
        return (sl // step).astype(np.uint8)

    def cell_key(self, screen: np.ndarray, wram: np.ndarray) -> bytes:
        """Deterministic cell fingerprint from a (screen, wram) pair."""
        sd = self._screen_digest(screen).tobytes()
        wd = self._wram_digest(wram).tobytes()
        return sd + b"|" + wd

    # --------------------------------------------------------------- updates
    def add(self, key: bytes) -> bool:
        """Insert a cell key; return True iff it was newly discovered globally."""
        if key in self.seen:
            return False
        self.seen.add(key)
        return True

    def observe(self, screen: np.ndarray, wram: np.ndarray) -> bool:
        """Convenience: build the key and add it. True iff globally new."""
        return self.add(self.cell_key(screen, wram))

    # ----------------------------------------------------------- generation
    def begin_generation(self) -> None:
        """Snapshot the frontier size so :meth:`generation_delta` is meaningful."""
        self._gen_start_size = len(self.seen)

    @property
    def generation_delta(self) -> int:
        """New cells discovered since the last :meth:`begin_generation`."""
        return len(self.seen) - self._gen_start_size

    @property
    def size(self) -> int:
        """Total cells discovered so far (the whole-run frontier)."""
        return len(self.seen)


__all__ = ["NoveltyArchive"]
