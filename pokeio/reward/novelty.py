"""Per-wave novelty fitness accounting on top of the Go-Explore archive.

The whole-run global first-come scheme starves selection pressure: once the
start-area cells are in the archive, almost every genome scores 0 and fitness is
a near-uniform (mostly-zero) signal that can neither differentiate genomes nor
seed speciation.  Three credit modes fix that to different degrees:

``rarity`` (default)
    A player earns, for each **distinct** cell it visits this episode, a weight
    ``floor + 1/sqrt(1 + prior_visits)`` where ``prior_visits`` is how many times
    that cell has been seen across the whole run so far.  A brand-new cell scores
    ~``floor + 1``; a well-trodden start cell scores ~``floor``.  So a genome that
    merely moves through several distinct states scores > 0 (median > 0, real
    spread), while genomes that reach rare / fresh frontier cells score much
    higher (exploration pressure).  This is the ``rarity_weighted`` + ``novelty_floor``
    intent from :class:`~pokeio.config.RewardConfig`.

``per_gen``
    +1 per distinct cell that belongs to *this generation's* new frontier
    (:meth:`~pokeio.reward.archive.NoveltyArchive.is_gen_new`).  Pure frontier
    credit — strong spread the moment new ground is opened, but sparse (median can
    fall to 0 once a generation stops finding new cells).

``global`` (legacy)
    +1 only for a cell that is the first in the whole run to be added.  Kept for
    A/B comparison; this is the near-uniform baseline the others replace.

In every mode the shared :class:`~pokeio.reward.archive.NoveltyArchive` tracks the
whole-run frontier (``archive_cells`` / ``archive_delta``) and per-cell visit
counts.
"""

from __future__ import annotations

import math

import numpy as np

from pokeio.reward.archive import NoveltyArchive

_MODES = ("rarity", "per_gen", "global")


class WaveNovelty:
    """Tracks per-player novelty fitness for one evaluation wave."""

    def __init__(
        self,
        archive: NoveltyArchive,
        n_players: int,
        mode: str = "rarity",
        floor: float = 0.01,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown novelty mode: {mode!r} (expected {_MODES})")
        self.archive = archive
        self.mode = mode
        self.floor = float(floor)
        self.fitness = np.zeros(n_players, dtype=np.float64)
        # cells this player has already been credited for this episode (all modes
        # give distinct-cell credit, so we must not double-count within a genome).
        self._credited: list[set[bytes]] = [set() for _ in range(n_players)]
        # The first cell a player reports is its SPAWN state (the fixed reset or
        # a Go-Explore frontier restore). It is recorded but never credited:
        # credit is for cells an agent REACHES, not the one it was handed —
        # otherwise a do-nothing genome restored into a rare frontier cell
        # collects near-full rarity credit for standing still (and such couch
        # potatoes were literally winning championships).
        self._spawn_seen: list[bool] = [False] * n_players
        # the most recent cell key / novelty flag per player (used by go-explore
        # to decide when to capture a restorable state).
        self.last_key: list[bytes | None] = [None] * n_players
        self.last_new: list[bool] = [False] * n_players
        # A7 — order-independent rarity baseline. ``archive.visit()`` is a shared
        # MONOTONIC counter, so within one evaluation wave the k-th co-visitor of
        # a fresh cell reads visit-count ``k-1`` and earns strictly less rarity
        # credit than the (k-1)-th: a pure array-slot bias (low player index
        # wins) that couples fitness to nothing but position. Instead we score
        # every co-visitor of a cell against ONE baseline — the cell's visit
        # count the first time it is reported this wave, which (because
        # ``archive.visit`` only increases) equals its count at wave start. All
        # co-visitors then share that baseline and earn EQUAL credit regardless
        # of index/order. This intentionally changes rarity semantics: repeat
        # visits to the same cell within a wave no longer see an escalating
        # visit count. (WaveNovelty is per-wave, so this snapshot is per-wave;
        # when the population fits one wave that is exactly the generation start.)
        self._visit_baseline: dict[bytes, int] = {}

    def _rarity_baseline(self, key: bytes, prior_visits: int) -> int:
        """Return the shared, order-independent visit baseline for ``key``.

        The first co-visitor of a cell this wave establishes the baseline; later
        co-visitors reuse it. ``min`` makes it robust to any caller ordering
        (``archive.visit`` is monotonic, so the first report already carries the
        lowest count, but we do not rely on that)."""
        base = self._visit_baseline.get(key)
        if base is None or prior_visits < base:
            base = int(prior_visits)
            self._visit_baseline[key] = base
        return base

    def observe(self, player_idx: int, screen: np.ndarray, wram: np.ndarray) -> bool:
        """Record one (screen, wram) for ``player_idx``.

        Returns True iff the observation earned this player novelty credit.
        """
        key = self.archive.cell_key(screen, wram)
        globally_new = self.archive.add(key)  # mutates whole-run + per-gen frontier
        prior_visits = self.archive.visit(key)  # visits before this one
        self.last_key[player_idx] = key
        self.last_new[player_idx] = globally_new

        if not self._spawn_seen[player_idx]:
            # Spawn cell: archive it, never credit it (see __init__).
            self._spawn_seen[player_idx] = True
            self._credited[player_idx].add(key)
            return False

        if self.mode == "global":
            if globally_new:
                self.fitness[player_idx] += 1.0
                return True
            return False

        # rarity / per_gen both give distinct-cell credit per genome.
        if key in self._credited[player_idx]:
            return False
        self._credited[player_idx].add(key)

        if self.mode == "per_gen":
            if self.archive.is_gen_new(key):
                self.fitness[player_idx] += 1.0
                return True
            return False

        # rarity: rare / fresh cells are worth more; every distinct cell > 0.
        # Score against the shared per-wave baseline (A7) so co-visitors of the
        # same fresh cell earn equal credit regardless of player index.
        base = self._rarity_baseline(key, prior_visits)
        self.fitness[player_idx] += self.floor + 1.0 / math.sqrt(1.0 + base)
        return True

    def observe_key(
        self,
        player_idx: int,
        key: bytes,
        globally_new: bool,
        prior_visits: int,
    ) -> bool:
        """Credit ``player_idx`` for a cell whose key was hashed elsewhere.

        The parallel worker fleet computes ``cell_key`` (the expensive screen /
        WRAM digest) inside each worker process and ships the key + the archive's
        ``add``/``visit`` outcomes across the barrier, so the parent only does the
        cheap credit bookkeeping here.  Semantics are identical to
        :meth:`observe`; the caller must have already invoked
        ``archive.add(key)`` (-> ``globally_new``) and ``archive.visit(key)``
        (-> ``prior_visits``) in that order so the per-generation frontier and
        visit counts stay consistent.
        """
        self.last_key[player_idx] = key
        self.last_new[player_idx] = globally_new

        if not self._spawn_seen[player_idx]:
            # Spawn cell: archive it, never credit it (see __init__).
            self._spawn_seen[player_idx] = True
            self._credited[player_idx].add(key)
            return False

        if self.mode == "global":
            if globally_new:
                self.fitness[player_idx] += 1.0
                return True
            return False

        if key in self._credited[player_idx]:
            return False
        self._credited[player_idx].add(key)

        if self.mode == "per_gen":
            if self.archive.is_gen_new(key):
                self.fitness[player_idx] += 1.0
                return True
            return False

        # Shared per-wave baseline (A7): order-independent rarity credit.
        base = self._rarity_baseline(key, prior_visits)
        self.fitness[player_idx] += self.floor + 1.0 / math.sqrt(1.0 + base)
        return True


__all__ = ["WaveNovelty"]
