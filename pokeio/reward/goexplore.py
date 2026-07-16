"""Go-Explore cell-restore archive for pokeIO.

The plain novelty archive (:mod:`pokeio.reward.archive`) answers *"has this cell
been seen?"*.  This module answers the complementary Go-Explore question: *"how
do I get BACK to a promising cell so exploration can push outward from the
frontier instead of always restarting at the fixed new-game state?"*

For every cell we choose to remember, we stash the **serialized emulator state**
(a PyBoy ``save_state`` blob) captured the moment the cell was first reached.  A
later episode can then :meth:`restore` that blob into an env and start exploring
from the frontier rather than from the start screen.

Cell selection follows the classic Go-Explore heuristic — a cell is *promising*
when it is **rarely visited** (low visit count), **freshly discovered** (recent),
**deep** (far from start, i.e. genuine frontier) and **seldom chosen** as a
restart before.  The weight formula combines those into a single sampling score.

Memory is bounded: at most ``capacity`` states are kept; when full, the
least-useful entry (frequently visited, shallow, stale) is evicted.  State blobs
are the only heavy payload, so the cap directly bounds RAM.

Capturing the state does **not** perturb the running episode: we serialize the
PyBoy core directly (``env.pyboy.save_state``) without the input-flush tick that
:meth:`PokeEnv.save_state` performs, so lockstep evaluation stays deterministic.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CellEntry", "GoExplore", "capture_state"]


# --------------------------------------------------------------------------
# non-perturbing state capture
# --------------------------------------------------------------------------
def capture_state(env) -> bytes:
    """Serialize the emulator state **without** advancing a frame.

    :meth:`pokeio.emu.env.PokeEnv.save_state` flushes queued input with an extra
    ``tick(1)`` — fine at episode boundaries but it would desync a player from
    its wave-mates if called mid-episode.  During a step the buttons are already
    released (``step`` ends on ``button_release`` + a render tick), so the input
    queue is settled and a direct core save is deterministic and side-effect free.
    """
    buf = io.BytesIO()
    env.pyboy.save_state(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# a remembered frontier cell
# --------------------------------------------------------------------------
@dataclass
class CellEntry:
    """One remembered cell + the emulator state to get back to it."""

    key: bytes
    state: bytes  # PyBoy save_state blob captured when first reached
    depth: int  # step index within the episode when first reached (frontier depth)
    gen_added: int  # generation the cell was first stored
    visits: int = 1  # times the cell has been reached (any player, any gen)
    selections: int = 0  # times chosen as a restart point
    gen_seen: int = 0  # most recent generation the cell was reached


# --------------------------------------------------------------------------
# the archive
# --------------------------------------------------------------------------
@dataclass
class GoExplore:
    """Bounded archive of frontier cells with restorable emulator states.

    Parameters
    ----------
    capacity:
        Maximum number of stored states.  Each PyBoy blob is tens of KB, so this
        directly bounds memory (default 2048 ~= a few hundred MB worst case).
    recency_halflife:
        Generations over which the recency bonus decays by half.
    depth_weight:
        How strongly to prefer deep (far-from-start) cells when sampling.
    rng:
        A ``numpy`` Generator for reproducible weighted sampling.
    """

    capacity: int = 2048
    recency_halflife: float = 4.0
    depth_weight: float = 0.25
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    cells: dict[bytes, CellEntry] = field(default_factory=dict)
    cur_gen: int = 0
    # lifetime counters for telemetry
    n_captured: int = 0
    n_evicted: int = 0
    n_restores: int = 0

    # ------------------------------------------------------------------ gen
    def begin_generation(self, gen: int) -> None:
        self.cur_gen = int(gen)

    # --------------------------------------------------------------- update
    def note(
        self,
        key: bytes,
        env,
        depth: int,
        *,
        globally_new: bool,
    ) -> None:
        """Record that ``key`` was reached by a player standing in ``env``.

        On the first sighting (``globally_new``) we capture and store the state
        (subject to capacity).  On a revisit we just bump the visit/recency
        bookkeeping so the sampler can de-prioritise well-trodden cells.
        """
        entry = self.cells.get(key)
        if entry is not None:
            entry.visits += 1
            entry.gen_seen = self.cur_gen
            return
        if not globally_new:
            # Cell exists in the novelty archive but we never stored a state for
            # it (e.g. it was evicted, or discovered before go-explore was on).
            return
        state = capture_state(env)
        self.n_captured += 1
        if len(self.cells) >= self.capacity:
            self._evict_one()
        self.cells[key] = CellEntry(
            key=key,
            state=state,
            depth=int(depth),
            gen_added=self.cur_gen,
            gen_seen=self.cur_gen,
        )

    def revisit(self, key: bytes) -> bool:
        """Bump visit/recency bookkeeping for an already-stored cell.

        Returns True if ``key`` was a known frontier cell (bookkeeping applied),
        False otherwise.  Split out of :meth:`note` for the parallel loop, where
        the parent knows ``globally_new`` and whether a state was captured in a
        worker, and only needs the revisit branch here.
        """
        entry = self.cells.get(key)
        if entry is None:
            return False
        entry.visits += 1
        entry.gen_seen = self.cur_gen
        return True

    def store_captured(self, key: bytes, state: bytes, depth: int) -> None:
        """Store a state blob captured in a worker for a freshly-discovered cell.

        The parallel fleet captures the emulator state inside the worker that
        first reached ``key`` (one barrier round after the cell is flagged
        globally-new, i.e. while the worker still sits in that exact state), then
        hands the bytes to the parent, which calls this.  Mirrors the capture
        branch of :meth:`note` (capacity-bounded, evicts the least-useful entry).
        """
        if key in self.cells:  # already stored (e.g. flagged twice); keep first
            return
        self.n_captured += 1
        if len(self.cells) >= self.capacity:
            self._evict_one()
        self.cells[key] = CellEntry(
            key=key,
            state=state,
            depth=int(depth),
            gen_added=self.cur_gen,
            gen_seen=self.cur_gen,
        )

    # ---------------------------------------------------------------- weight
    def _weight(self, e: CellEntry) -> float:
        """Sampling weight: rare, recent, deep and seldom-restarted cells win."""
        rarity = 1.0 / math.sqrt(e.visits)
        seldom = 1.0 / (1.0 + e.selections)
        age = max(0, self.cur_gen - e.gen_seen)
        recency = 0.5 ** (age / max(1e-6, self.recency_halflife))
        depth_bonus = 1.0 + self.depth_weight * math.log1p(max(0, e.depth))
        return rarity * seldom * recency * depth_bonus

    def _evict_one(self) -> None:
        """Drop the least-useful entry (inverse of the sampling weight)."""
        if not self.cells:
            return
        victim = min(self.cells.values(), key=self._weight)
        del self.cells[victim.key]
        self.n_evicted += 1

    # ---------------------------------------------------------------- sample
    def sample(self) -> CellEntry | None:
        """Weighted-random pick of a promising frontier cell (or ``None``)."""
        if not self.cells:
            return None
        entries = list(self.cells.values())
        w = np.array([self._weight(e) for e in entries], dtype=np.float64)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            idx = int(self.rng.integers(len(entries)))
        else:
            idx = int(self.rng.choice(len(entries), p=w / s))
        chosen = entries[idx]
        chosen.selections += 1
        return chosen

    def restore(self, env, entry: CellEntry) -> np.ndarray:
        """Load ``entry``'s state into ``env`` and return the fresh observation."""
        env.load_state(entry.state)
        env.pyboy.tick(1, True)  # settle one rendered frame, like PokeEnv.reset
        self.n_restores += 1
        return env._obs()

    # ------------------------------------------------------------------ misc
    @property
    def size(self) -> int:
        return len(self.cells)

    def stats(self) -> dict[str, float]:
        depths = [e.depth for e in self.cells.values()] or [0]
        return {
            "goexplore_states": float(len(self.cells)),
            "goexplore_captured": float(self.n_captured),
            "goexplore_evicted": float(self.n_evicted),
            "goexplore_restores": float(self.n_restores),
            "goexplore_max_depth": float(max(depths)),
        }
