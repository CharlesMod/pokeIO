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
least-useful entries are batch-evicted — stale never-re-reached one-offs go
first; reproducible (multi-visit), recent, and deep cells are kept.  State blobs
are the only heavy payload, so the cap directly bounds RAM.

Capturing the state does **not** perturb the running episode: we serialize the
PyBoy core directly (``env.pyboy.save_state``) without the input-flush tick that
:meth:`PokeEnv.save_state` performs, so lockstep evaluation stays deterministic.
"""

from __future__ import annotations

import io
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
    # CUMULATIVE distance from newgame: restore-parent depth + step index
    # within the discovering episode. Restore chains therefore accumulate —
    # a cell 10 chains deep reads ~10*episode_steps, not its local step index
    # — which is what lets sampling actually prefer the genuine frontier.
    depth: int
    gen_added: int  # generation the cell was first stored
    visits: int = 1  # times the cell has been reached (any player, any gen)
    selections: int = 0  # times chosen as a restart point
    gen_seen: int = 0  # most recent generation the cell was reached
    # Progress snapshot of the DISCOVERING trajectory at capture time (running
    # milestone maxes + maps-seen set — see ProgressReward.snapshot). A spawn
    # restored from this cell seeds its reward baseline from it, so handed
    # progress INCLUDING the trajectory's already-traversed maps is never
    # re-payable (teleport-decoupling). None for legacy/NEAT captures.
    progress: dict | None = None


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

    capacity: int = 16384
    recency_halflife: float = 4.0
    depth_weight: float = 0.25
    caps_per_round: float = 0.0  # max captures per swarm round (0 = unlimited)
    # --- backward-shift restore curriculum (docs/specs/boot-gauntlet.md D2) ---
    # When ``backward`` is on, restore sampling is restricted to cells in the
    # bottom depth-quantile ``backward_q`` (the shallow / early-game frontier),
    # so early generations respawn near newgame and the population re-earns the
    # opening before deep frontier spawns dominate. The training loop anneals
    # ``backward_q`` from ``restore_backward_q0`` toward 1.0 (full frontier) via
    # :meth:`set_backward_schedule`, called once per generation. ``backward_q ==
    # 1.0`` (the default) disables the mask — identical to the legacy sampler.
    backward: bool = False
    backward_q: float = 1.0
    # --- eviction / detachment guard (A6) ------------------------------------
    # The eviction keep-score used to be recency * visits * depth, so the fast
    # 0.5^(age/halflife) recency term crushed a proven deep hub to ~0 after a
    # dozen idle gens and batch-evicted it; because the novelty ``seen`` set is
    # append-only, an evicted hub can NEVER be re-captured and exploration depth
    # silently regresses on plateaus.  Two bounded guards below keep proven deep
    # frontier hubs while never letting them crowd out eviction capacity:
    evict_depth_floor: float = 1.0  # weight of an UNDECAYED proven-hub keep floor
    deep_exempt_frac: float = 0.125  # fraction of capacity exempt from eviction
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    cells: dict[bytes, CellEntry] = field(default_factory=dict)
    cur_gen: int = 0
    # lifetime counters for telemetry
    n_captured: int = 0
    n_evicted: int = 0
    n_restores: int = 0
    n_throttled: int = 0
    _cap_tokens: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cap_tokens = self._cap_burst()

    # ------------------------------------------------------- capture budget
    # A save_state blob costs ~47 ms inside the worker that produces it, and
    # the archive keeps only ``capacity`` states — so capturing every new cell
    # (tens of thousands per generation once agents explore) buys nothing but
    # archive churn and stalls.  The budget meters captures to roughly
    # ``caps_per_round`` per swarm round, which still refreshes a 2048-entry
    # archive faster than the recency halflife decays it.
    def _cap_burst(self) -> float:
        return max(2.0 * self.caps_per_round, 1.0)

    def feed_capture_budget(self, rounds: float) -> None:
        """Accrue capture tokens for ``rounds`` swarm-rounds of progress."""
        if self.caps_per_round > 0:
            self._cap_tokens = min(
                self._cap_burst(), self._cap_tokens + self.caps_per_round * rounds
            )

    def admit_capture(self) -> bool:
        """True if a state capture is within budget (consumes one token)."""
        if self.caps_per_round <= 0:
            return True
        if self._cap_tokens >= 1.0:
            self._cap_tokens -= 1.0
            return True
        self.n_throttled += 1
        return False

    # ------------------------------------------------------------------ gen
    def begin_generation(self, gen: int) -> None:
        self.cur_gen = int(gen)

    def set_backward_schedule(
        self, gen: int, q0: float, anneal_gens: int
    ) -> None:
        """Update the eligible bottom-depth quantile for ``gen`` (D2 curriculum).

        Linearly anneals ``backward_q`` from ``q0`` (gen 0) to 1.0 (full
        frontier) over ``anneal_gens`` generations. A no-op that pins
        ``backward_q = 1.0`` when :attr:`backward` is off, so the sampler stays
        byte-identical to the legacy (depth-weighted, no-mask) behaviour."""
        if not self.backward:
            self.backward_q = 1.0
            return
        anneal = max(1, int(anneal_gens))
        frac = min(1.0, max(0.0, float(gen) / anneal))
        q0 = float(q0)
        self.backward_q = float(min(1.0, q0 + (1.0 - q0) * frac))

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
        if not self.admit_capture():
            return
        state = capture_state(env)
        self.n_captured += 1
        if len(self.cells) >= self.capacity:
            self._evict_batch()
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

    def store_captured(self, key: bytes, state: bytes, depth: int,
                       progress: dict | None = None) -> None:
        """Store a state blob captured in a worker for a freshly-discovered cell.

        The parallel fleet captures the emulator state inside the worker that
        first reached ``key`` (one barrier round after the cell is flagged
        globally-new, i.e. while the worker still sits in that exact state), then
        hands the bytes to the parent, which calls this.  Mirrors the capture
        branch of :meth:`note` (capacity-bounded, evicts the least-useful entry).
        ``progress`` (optional) is the discovering trajectory's reward snapshot
        at capture time (see :class:`CellEntry`).
        """
        if key in self.cells:  # already stored (e.g. flagged twice); keep first
            return
        self.n_captured += 1
        if len(self.cells) >= self.capacity:
            self._evict_batch()
        self.cells[key] = CellEntry(
            key=key,
            state=state,
            depth=int(depth),
            gen_added=self.cur_gen,
            gen_seen=self.cur_gen,
            progress=progress,
        )

    # ---------------------------------------------------------------- weight
    def _fields(self, entries: list[CellEntry]):
        n = len(entries)
        v = np.fromiter((e.visits for e in entries), dtype=np.float64, count=n)
        sel = np.fromiter((e.selections for e in entries), dtype=np.float64, count=n)
        seen = np.fromiter((e.gen_seen for e in entries), dtype=np.float64, count=n)
        dep = np.fromiter((e.depth for e in entries), dtype=np.float64, count=n)
        return v, sel, seen, dep

    def _weights_vec(self, entries: list[CellEntry]) -> np.ndarray:
        """Sampling weights: rare, recent, deep, seldom-restarted cells win."""
        v, sel, seen, dep = self._fields(entries)
        rarity = 1.0 / np.sqrt(np.maximum(1.0, v))
        seldom = 1.0 / (1.0 + sel)
        age = np.maximum(0.0, self.cur_gen - seen)
        recency = 0.5 ** (age / max(1e-6, self.recency_halflife))
        depth_bonus = 1.0 + self.depth_weight * np.log1p(np.maximum(0.0, dep))
        return rarity * seldom * recency * depth_bonus

    def _weight(self, e: CellEntry) -> float:
        """Scalar sampling weight (kept for tests/telemetry)."""
        return float(self._weights_vec([e])[0])

    def _evict_batch(self) -> None:
        """Drop the least-useful ~capacity/64 entries in one vectorized pass.

        Eviction is NOT the inverse of the sampling weight: under that rule
        one-off never-re-reached cells look maximally "rare" and survive while
        proven-reproducible hub cells get squeezed out, so the archive decays
        toward unreproducible noise. Instead KEEP reproducible (multi-visit),
        recent, and deep cells; EVICT stale one-offs first. Batched because
        the old per-eviction O(n) min-scan ran thousands of times per
        generation once cell discovery outpaced capacity.

        Detachment guard (A6). The keep-score is no longer a bare
        ``recency * visits * depth`` product — that let the fast recency decay
        (0.5^(age/halflife)) out-rank a proven deep hub against a recent shallow
        one-off after ~a dozen idle gens, batch-evicting the hub for good. Two
        bounded guards restore stability:

        * an **additive, undecayed floor** ``evict_depth_floor * log1p(visits) *
          depth_term`` added to the recency-weighted component, so a proven
          (multi-visit AND deep) hub keeps a positive keep-score no matter how
          stale. The floor scales with ``log1p(visits)``, so a shallow one-off
          (visits=1, depth=0) contributes ~0 and is never protected by it;
        * a **bounded hard exemption**: the top-N deepest *proven* (visits>1)
          cells are never evicted. N is capped at ``deep_exempt_frac * capacity``
          AND at ``m - k`` so there are always >= k evictable cells — stale-but-
          deep cells therefore can never crowd out eviction capacity.
        """
        if not self.cells:
            return
        entries = list(self.cells.values())
        m = len(entries)
        k = max(1, self.capacity // 64)
        v, _sel, seen, dep = self._fields(entries)
        age = np.maximum(0.0, self.cur_gen - seen)
        recency = 0.5 ** (age / max(1e-6, self.recency_halflife))
        depth_term = 1.0 + self.depth_weight * np.log1p(np.maximum(0.0, dep))
        visits_log = np.log1p(v)
        # Recency-weighted component (differentiates cells while fresh) PLUS an
        # undecayed proven-hub floor so a stale deep hub is not crushed to ~0.
        keep_score = recency * (0.5 + visits_log) * depth_term + (
            max(0.0, self.evict_depth_floor) * visits_log * depth_term
        )
        # Bounded exemption: never evict the top-N deepest PROVEN cells.
        exempt_cap = min(int(self.capacity * self.deep_exempt_frac), m - k)
        if exempt_cap > 0:
            proven = np.nonzero(v > 1)[0]
            if proven.size:
                deepest = proven[np.argsort(-dep[proven])][:exempt_cap]
                keep_score = keep_score.copy()
                keep_score[deepest] = np.inf  # sort last -> never in bottom-k
        n_evict = min(k, m)
        for i in np.argsort(keep_score)[:n_evict]:
            del self.cells[entries[int(i)].key]
        self.n_evicted += n_evict

    # ---------------------------------------------------------------- sample
    def sample_many(self, k: int) -> list[CellEntry]:
        """``k`` weighted draws (with replacement) in ONE vectorized pass.

        The per-draw ``sample()`` recomputed every cell's weight per call —
        ~n_players * capacity python-level evaluations per generation. One
        weight pass per wave is indistinguishable statistically (selections
        feedback within a single wave is negligible) and ~100x cheaper.
        """
        if not self.cells or k <= 0:
            return []
        entries = list(self.cells.values())
        w = self._weights_vec(entries)
        # Backward-shift curriculum (D2): restrict eligibility to the bottom
        # ``backward_q`` depth-quantile (shallow cells) early, annealing to the
        # full frontier as ``backward_q`` -> 1.0. The rarity/recency/depth
        # weighting still applies WITHIN the shallow subset (so the deepest of
        # the currently-eligible cells is still preferred — the edge of what the
        # population has mastered). A no-op when backward is off / q>=1.0.
        if self.backward and self.backward_q < 1.0 and len(entries) > 1:
            dep = np.fromiter(
                (e.depth for e in entries), dtype=np.float64, count=len(entries)
            )
            thresh = np.quantile(dep, self.backward_q)
            masked = w * (dep <= thresh)
            if masked.sum() > 0.0:  # keep the mask only if it leaves live weight
                w = masked
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            idxs = self.rng.integers(len(entries), size=k)
        else:
            idxs = self.rng.choice(len(entries), size=k, p=w / s)
        out = []
        for i in idxs:
            e = entries[int(i)]
            e.selections += 1
            out.append(e)
        return out

    def sample(self) -> CellEntry | None:
        """Weighted-random pick of a promising frontier cell (or ``None``)."""
        got = self.sample_many(1)
        return got[0] if got else None

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
            "goexplore_throttled": float(self.n_throttled),
            "goexplore_max_depth": float(max(depths)),
            "goexplore_backward_q": float(self.backward_q),
        }
