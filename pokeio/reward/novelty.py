"""Per-wave novelty fitness accounting on top of the Go-Explore archive.

During a wave, every player (genome) steps its own emulator in lockstep.  A
player earns fitness for each cell it is the *first in the whole run* to add to
the shared :class:`~pokeio.reward.archive.NoveltyArchive`.  Because the archive's
``add`` returns True only on the global first sighting, revisits — within an
episode or by a later player — score nothing, which is exactly Go-Explore
frontier novelty.
"""

from __future__ import annotations

import numpy as np

from pokeio.reward.archive import NoveltyArchive


class WaveNovelty:
    """Tracks per-player novelty fitness for one evaluation wave."""

    def __init__(self, archive: NoveltyArchive, n_players: int) -> None:
        self.archive = archive
        self.fitness = np.zeros(n_players, dtype=np.float64)

    def observe(self, player_idx: int, screen: np.ndarray, wram: np.ndarray) -> bool:
        """Record one (screen, wram) for ``player_idx``. True iff globally new."""
        if self.archive.observe(screen, wram):
            self.fitness[player_idx] += 1.0
            return True
        return False


__all__ = ["WaveNovelty"]
