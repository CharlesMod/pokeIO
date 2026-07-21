"""Basal-ganglia reward — dense, teleport-decoupled from-boot progress (#30).

THE #1 thing (``docs/specs/brain-architecture.md`` RED-TEAM REVISION): what killed
90 generations was a reward the policy could earn WITHOUT competent play. The fix,
made precise:

  * **Dense** — a per-step signal, not a sparse end-of-episode score. Every real
    step of progress (enter a new map, gain an event flag, get a Pokémon, level up,
    earn a badge) pays out the step it happens, so the gradient has a breadcrumb
    trail from boot to the first milestone. The demo path bedroom->house->Pallet->
    Oak's-lab->Pikachu pays +1,+1,+1 (maps) + event flags + 5 (party) along the way.
  * **Teleport-decoupled** — a Go-Explore restore JUMPS the game state; the progress
    it hands you is NOT reward you earned. On :meth:`reset` the per-env baseline is
    re-seeded FROM the restored state, so the policy is paid only for progress BEYOND
    the restore point. This is exactly what backward-robustification wants.
  * **Never an input** — this scalar goes to the critic ONLY. It is never written into
    the observation the actor reads (the miner-Goodhart trap). Enforced by keeping it
    out of the encoder; see :mod:`pokeio.brain.coupling` for the permanent guards.
  * **Coupled to competent play** — it reads the CORRECTED from-boot RAM milestones
    (:mod:`pokeio.reward.from_boot`, pixels-validated); the pointer-byte noise that
    let map-thrash fake progress is gone (see the from_boot module docstring).

Formulation. Let ``Phi(env)`` be the from-boot composite over the running MAX of
each milestone signal seen since the env's last reset, plus ``w_map`` per distinct
map entered since reset. Per step the reward is ``r_t = Phi_t - Phi_{t-1} >= 0``
(monotone: maxes never fall, the map set only grows — a faint that drops the party
does NOT claw back reward). Summed over an episode this telescopes to
``Phi_terminal - Phi_reset`` (potential-shaping form with gamma=1), i.e. the total
real progress made from the spawn point — teleport-invariant by construction.

Pure module: it reads WRAM snapshots and returns floats. The fleet plumbing that
supplies per-env WRAM lives in the RL loop, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pokeio.reward.from_boot import YELLOW, GameProgressSpec, decode_state


@dataclass
class _EnvProgress:
    """Per-env running progress state (the baseline the reward differences against)."""

    party_count: int = 0
    party_level: int = 0
    badges: int = 0
    events: int = 0
    counter_sum: int = 0
    maps: set = field(default_factory=set)   # distinct map-ids seen since reset
    phi: float = 0.0                          # last composite (the diff baseline)
    # Spawn-time milestone baseline (set at reset AFTER history+WRAM absorption):
    # the tier the env was HANDED, against which earned_milestone judges.
    spawn_party: int = 0
    spawn_badges: int = 0
    spawn_events: int = 0


class ProgressReward:
    """Dense, teleport-decoupled from-boot progress reward over N envs.

    Usage (per env, driven by the RL loop):
      * :meth:`reset(env_idx, wram)` at episode start / after a Go-Explore restore
        — re-seeds the baseline from the (possibly progressed) state, emits nothing.
      * :meth:`step(env_idx, wram) -> float` each step — the dense progress reward
        ``Phi_now - Phi_prev >= 0``.

    Batched helpers :meth:`reset_many` / :meth:`step_many` take aligned env indices +
    WRAM snapshots for the fleet. ``spec`` is the game-agnostic progress spec
    (default the corrected Yellow spec); the reward carries NO Yellow specifics of
    its own, so a second game plugs in its own spec unchanged.
    """

    def __init__(self, n_envs: int, spec: GameProgressSpec = YELLOW) -> None:
        self.n_envs = int(n_envs)
        self.spec = spec
        self._envs: dict[int, _EnvProgress] = {}

    # -- composite ---------------------------------------------------------
    def _phi(self, p: _EnvProgress) -> float:
        """Composite progress points from an env's running-max milestone state."""
        s = self.spec
        maps_beyond = max(0, len(p.maps) - 1)  # spawn map is not progress
        return float(
            s.w_map * maps_beyond
            + s.w_party * p.party_count
            + s.w_level * p.party_level
            + s.w_badge * p.badges
            + s.w_event * p.events
            + s.w_counter * p.counter_sum
        )

    def _absorb(self, p: _EnvProgress, wram: np.ndarray) -> None:
        """Fold one WRAM snapshot into the env's running-max milestone state."""
        d = decode_state(wram, self.spec)
        # monotone maxes: progress reached counts even if later lost (faint, etc.)
        if d["party_count"] > p.party_count:
            p.party_count = d["party_count"]
        if d["party_level"] > p.party_level:
            p.party_level = d["party_level"]
        if d["badges"] > p.badges:
            p.badges = d["badges"]
        if d["events"] > p.events:
            p.events = d["events"]
        if d["counter_sum"] > p.counter_sum:
            p.counter_sum = d["counter_sum"]
        p.maps.add(d["map_id"])

    # -- single-env API ----------------------------------------------------
    def reset(self, env_idx: int, wram: np.ndarray,
              history: dict | None = None) -> None:
        """Re-baseline env ``env_idx`` from ``wram`` (teleport-decoupled). No reward.

        Seeds the running maxes + the maps-seen set from the current (possibly
        restored/progressed) state, so the progress it was HANDED is never paid —
        only progress made from here on earns reward.

        ``history`` (optional) is the SPAWNING trajectory's progress snapshot
        (:meth:`snapshot` of the env that captured the spawn state, or the demo
        prefix at the restored depth): its maps-seen set + running maxes seed the
        baseline too. WRAM only holds the CURRENT map id, so without the history
        every map the spawning trajectory already traversed would be re-payable
        by backtracking through it — the maps term must be teleport-decoupled
        exactly like the running maxes.
        """
        p = _EnvProgress()
        if history:
            p.party_count = int(history.get("party_count", 0))
            p.party_level = int(history.get("party_level", 0))
            p.badges = int(history.get("badges", 0))
            p.events = int(history.get("events", 0))
            p.counter_sum = int(history.get("counter_sum", 0))
            p.maps = set(history.get("maps", ()))
        self._absorb(p, wram)
        p.phi = self._phi(p)
        # Spawn-relative milestone baseline (AFTER history+WRAM absorption):
        # everything up to here was HANDED, never a success.
        p.spawn_party = p.party_count
        p.spawn_badges = p.badges
        p.spawn_events = p.events
        self._envs[int(env_idx)] = p

    def step(self, env_idx: int, wram: np.ndarray) -> float:
        """Dense progress reward for env ``env_idx`` at ``wram``: ``Phi_now - Phi_prev``.

        ``>= 0`` by construction. If the env was never reset (no baseline), this
        first call baselines it and returns 0.0 (a step with no prior is not
        progress) — but the RL loop should always :meth:`reset` at episode start.
        """
        i = int(env_idx)
        p = self._envs.get(i)
        if p is None:  # no baseline -> treat as a reset, emit nothing
            self.reset(i, wram)
            return 0.0
        self._absorb(p, wram)
        phi_now = self._phi(p)
        r = phi_now - p.phi
        p.phi = phi_now
        return float(r)

    # -- batched API (fleet) ----------------------------------------------
    def reset_many(self, env_indices, wrams, histories=None) -> None:
        """Re-baseline several envs. ``wrams`` aligned with ``env_indices``;
        ``histories`` (optional) aligned spawning-trajectory snapshots (``None``
        entries = nothing handed, e.g. a boot spawn)."""
        if histories is None:
            for i, w in zip(env_indices, wrams):
                self.reset(i, w)
        else:
            for i, w, h in zip(env_indices, wrams, histories):
                self.reset(i, w, h)

    def step_many(self, env_indices, wrams) -> np.ndarray:
        """Dense reward for several envs. Returns a float32 array aligned to inputs."""
        return np.asarray(
            [self.step(i, w) for i, w in zip(env_indices, wrams)], dtype=np.float32
        )

    # -- introspection -----------------------------------------------------
    def progress(self, env_idx: int) -> dict:
        """The current baseline milestone state for env ``env_idx`` (for logging)."""
        p = self._envs.get(int(env_idx))
        if p is None:
            return {"phi": 0.0, "party_count": 0, "party_level": 0, "badges": 0,
                    "events": 0, "maps": 0, "counter_sum": 0}
        return {
            "phi": self._phi(p), "party_count": p.party_count,
            "party_level": p.party_level, "badges": p.badges, "events": p.events,
            "maps": len(p.maps), "counter_sum": p.counter_sum,
        }

    def started(self, env_idx: int) -> bool:
        """True iff env ``env_idx`` has reached the first real milestone (party>0)."""
        p = self._envs.get(int(env_idx))
        return bool(p is not None and (p.party_count > 0 or p.badges > 0))

    def snapshot(self, env_idx: int) -> dict | None:
        """The env's FULL progress state (running maxes + maps-seen set), for
        persisting alongside a captured spawn state — feed it back to
        :meth:`reset` as ``history`` when that state is later restored, so the
        capturing trajectory's progress (including its traversed maps) is part
        of the handed baseline, never re-payable."""
        p = self._envs.get(int(env_idx))
        if p is None:
            return None
        return {"party_count": p.party_count, "party_level": p.party_level,
                "badges": p.badges, "events": p.events,
                "counter_sum": p.counter_sum, "maps": set(p.maps)}

    def earned_milestone(self, env_idx: int) -> bool:
        """Spawn-relative MILESTONE crossing — the forward-regime outcome signal.

        A tier-0 spawn (no party/badge handed) succeeds only by EARNING
        ``started()`` (party>0 or a badge — the original BackwardCurriculum
        milestone). A spawn already handed that milestone reports against the
        NEXT unearned tier: a new badge or story event flag beyond its spawn
        baseline. Map transitions never count (walking out the spawn building
        is not a milestone), and restored progress can never be a success (it
        is folded into the spawn baseline at :meth:`reset`). The tiers come
        from the game-progress spec's own milestone taxonomy — no tuned bar.
        """
        p = self._envs.get(int(env_idx))
        if p is None:
            return False
        if p.spawn_party == 0 and p.spawn_badges == 0:
            return p.party_count > 0 or p.badges > 0
        return p.badges > p.spawn_badges or p.events > p.spawn_events


__all__ = ["ProgressReward"]
