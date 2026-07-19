"""Hippocampus — demo trajectory + backward-robustification curriculum (#31).

The Go-Explore archive from 90 generations holds NO milestone trajectories (it was
real maps but party=0 everywhere — see the tap-fix finding), so it cannot seed the
composition the plan hinges on. What CAN: the recorded first-milestone
demonstration ``assets/demo_pikachu`` (657 actions, newgame -> Pikachu).

Backward-robustification (Salimans & Chen 2018, "Learning Montezuma's Revenge from
a Single Demonstration" = Go-Explore Phase 2): a from-boot policy almost never
stumbles onto the ~506-step Oak sequence, but starting it a few steps before the
goal is trivial. So we start episodes from a point along the demo and RECEDE that
start toward boot as the policy masters each depth. When the frontier reaches 0 the
policy completes the whole milestone from a cold boot — which IS the gate.

Two pieces:
  * :class:`DemoTrajectory` — replays the demo deterministically and captures the
    emulator save-state blob at any depth (the restore seeds for the fleet). A
    save-state blob at depth k feeds ``fleet.reset_all(restore={env_idx: blob})``.
  * :class:`BackwardCurriculum` — a self-paced start-depth frontier. It recedes at a
    speed set by the policy's RECENT SUCCESS (a success EMA), so a mastered depth
    recedes fast and a hard depth stalls — no hand-set success threshold, matching
    the project's EMA-self-calibration idiom (the ``advance_gain`` is a curriculum
    PACE, like a learning rate, not a behaviour threshold).

The demo action ints are in the training action space (PokeEnv.step, frame_skip=24,
sticky input); replay is deterministic, so blobs are reproducible from the actions
+ the newgame state (no need to track the derived .state file).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_DEMO = Path("assets/demo_pikachu/demo_actions.npy")
_DEFAULT_ROM = Path("roms/pokemon_yellow.gb")
_DEFAULT_NEWGAME = Path("roms/yellow_newgame.state")


class DemoTrajectory:
    """A recorded action trajectory with on-demand emulator save-state capture.

    ``blob_at(depth)`` returns the serialized emulator state after replaying the
    first ``depth`` demo actions from the newgame state (depth 0 = the boot state).
    Blobs are cached, so a curriculum re-requesting the same depths is cheap after
    the first pass. Requires the ROM + newgame state on-box (raises if absent).
    """

    def __init__(self, actions_path=_DEFAULT_DEMO, rom_path=_DEFAULT_ROM,
                 newgame_state=_DEFAULT_NEWGAME) -> None:
        self.actions = np.asarray(np.load(str(actions_path))).astype(int).ravel()
        self.rom_path = str(rom_path)
        self.newgame_state = str(newgame_state)
        self._blobs: dict[int, bytes] = {}

    def __len__(self) -> int:
        return int(self.actions.size)

    def blob_at(self, depth: int) -> bytes:
        """Emulator save-state blob after the first ``depth`` demo actions (cached)."""
        d = int(max(0, min(depth, len(self))))
        if d in self._blobs:
            return self._blobs[d]
        from pokeio.emu.env import PokeEnv

        env = PokeEnv(rom_path=self.rom_path, frame_skip=24)
        try:
            env.reset(self.newgame_state)
            for a in self.actions[:d]:
                env.step(int(a))
            self._blobs[d] = env.save_state()
        finally:
            env.close()
        return self._blobs[d]

    def capture_ladder(self, depths) -> dict[int, bytes]:
        """Blobs for a set of depths (replays once per distinct depth, sorted)."""
        return {int(d): self.blob_at(int(d)) for d in sorted(set(int(x) for x in depths))}

    def restore_map(self, env_depths: dict[int, int]) -> dict[int, bytes]:
        """``{env_idx: depth}`` -> ``{env_idx: blob}`` for ``fleet.reset_all(restore=)``.

        A depth of 0 is dropped (boot from the fleet's reset_state, no restore),
        so a curriculum that has fully receded issues an empty restore = pure boot.
        """
        return {int(i): self.blob_at(int(d)) for i, d in env_depths.items() if int(d) > 0}


class BackwardCurriculum:
    """Self-paced start-depth frontier for backward-robustification.

    Holds a float ``frontier`` in ``[0, len]`` (starts near the end). Each reported
    episode outcome nudges a success EMA; the frontier RECEDES by
    ``advance_gain * success_ema`` per report — fast when the policy is winning at
    the current depth, stalled when it is not. At ``frontier == 0`` starts are cold
    boots (the gate). ``sample_depths(n)`` returns per-env start depths clustered at
    the frontier with a little spread so the policy also practises slightly earlier
    and slightly later starts (robustification, not memorization of one start).
    """

    def __init__(self, length: int, advance_gain: float = 8.0,
                 ema_decay: float = 0.9, spread_frac: float = 0.15,
                 start_frac: float = 0.97, seed: int = 0) -> None:
        self.length = int(length)
        self.advance_gain = float(advance_gain)
        self.ema_decay = float(ema_decay)
        self.spread_frac = float(spread_frac)
        self.frontier = float(start_frac) * self.length
        self.success_ema = 0.0
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def sample_depths(self, n: int) -> list[int]:
        """``n`` per-env start depths clustered at the current frontier (+/- spread)."""
        f = self.frontier
        if f <= 0.0:
            return [0] * int(n)
        spread = max(1.0, self.spread_frac * self.length)
        raw = self._rng.normal(f, spread, size=int(n))
        return [int(np.clip(round(x), 0, self.length)) for x in raw]

    def report(self, reached_goal: bool) -> None:
        """Fold one episode outcome into the success EMA and recede the frontier."""
        s = 1.0 if reached_goal else 0.0
        # EMA warm-started on the first observation so early reports have weight
        self.success_ema = (
            s if self._seen == 0
            else self.ema_decay * self.success_ema + (1.0 - self.ema_decay) * s
        )
        self._seen += 1
        self.frontier = max(0.0, self.frontier - self.advance_gain * self.success_ema)

    def report_many(self, outcomes) -> None:
        for ok in outcomes:
            self.report(bool(ok))

    @property
    def at_boot(self) -> bool:
        """True once the frontier has fully receded (all starts are cold boots)."""
        return self.frontier <= 0.0

    def state(self) -> dict:
        return {"frontier": round(self.frontier, 2), "frontier_frac": round(
            self.frontier / max(1, self.length), 4), "success_ema": round(
            self.success_ema, 4), "at_boot": self.at_boot, "seen": self._seen}


__all__ = ["DemoTrajectory", "BackwardCurriculum"]
