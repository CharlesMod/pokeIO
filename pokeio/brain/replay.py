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
        self._milestone_depth: int | None = None

    def __len__(self) -> int:
        return int(self.actions.size)

    def milestone_depth(self, spec=None) -> int:
        """First demo depth at which the milestone fires (party>0 / a badge).

        Replays the demo once (cached) and returns the step index where the
        from-boot ``started`` condition first holds — the point BEFORE which the
        curriculum has real work to do. Game-agnostic: derived from the demo, not
        hardcoded. Starting the frontier just past this avoids wasting the first
        iterations restoring into already-completed (post-milestone) states.
        """
        if self._milestone_depth is not None:
            return self._milestone_depth
        from pokeio.emu.env import PokeEnv
        from pokeio.reward.from_boot import YELLOW, decode_state

        spec = spec or YELLOW
        env = PokeEnv(rom_path=self.rom_path, frame_skip=24)
        depth = len(self)
        try:
            env.reset(self.newgame_state)
            for k, a in enumerate(self.actions):
                env.step(int(a))
                d = decode_state(env.raw_wram(), spec)
                if d["party_count"] > 0 or d["badges"] > 0:
                    depth = k + 1
                    break
        finally:
            env.close()
        self._milestone_depth = int(depth)
        return self._milestone_depth

    def _clamp(self, depth) -> int:
        return int(max(0, min(int(depth), len(self))))

    def capture_ladder(self, depths) -> dict[int, bytes]:
        """Blobs for a set of depths captured in ONE demo replay (ascending order).

        O(max_depth), not O(sum of depths): a rollout that restores 64 envs to 64
        distinct depths replays the demo once, not 64 times. Uses a RAW
        ``pyboy.save_state`` (not ``env.save_state``, whose input-flush ticks a frame
        and would corrupt the ongoing trajectory) — the mid-trajectory snapshot keeps
        the exact sticky-input state, which is the faithful restore point. Cached.
        """
        import io

        want = sorted({self._clamp(d) for d in depths})
        missing = [d for d in want if d not in self._blobs]
        if missing:
            from pokeio.emu.env import PokeEnv

            env = PokeEnv(rom_path=self.rom_path, frame_skip=24)
            try:
                env.reset(self.newgame_state)
                k = 0
                for d in sorted(missing):
                    while k < d:
                        env.step(int(self.actions[k]))
                        k += 1
                    buf = io.BytesIO()
                    env.pyboy.save_state(buf)  # raw: does NOT advance the emulator
                    self._blobs[d] = buf.getvalue()
            finally:
                env.close()
        return {d: self._blobs[d] for d in want}

    def blob_at(self, depth: int) -> bytes:
        """Emulator save-state blob after the first ``depth`` demo actions (cached)."""
        d = self._clamp(depth)
        return self.capture_ladder([d])[d]

    def restore_map(self, env_depths: dict[int, int]) -> dict[int, bytes]:
        """``{env_idx: depth}`` -> ``{env_idx: blob}`` for ``fleet.reset_all(restore=)``.

        All depths are captured in a single replay. Depth 0 is dropped (boot from
        the fleet's reset_state, no restore), so a fully-receded curriculum issues an
        empty restore = pure cold boot.
        """
        pos = {int(i): self._clamp(d) for i, d in env_depths.items() if self._clamp(d) > 0}
        ladder = self.capture_ladder(pos.values())
        return {i: ladder[d] for i, d in pos.items()}


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

    def _observe(self, ok: bool) -> None:
        """Fold one episode outcome into the success EMA (no recession)."""
        s = 1.0 if ok else 0.0
        self.success_ema = (
            s if self._seen == 0
            else self.ema_decay * self.success_ema + (1.0 - self.ema_decay) * s
        )
        self._seen += 1

    def report(self, reached_goal: bool) -> None:
        """Single-env report: observe + recede once by ``advance_gain * success_ema``."""
        self._observe(bool(reached_goal))
        self.frontier = max(0.0, self.frontier - self.advance_gain * self.success_ema)

    def report_many(self, outcomes) -> None:
        """One iteration's batch of per-env outcomes: recede ONCE by the batch
        success RATE — so the recession speed is independent of n_envs (a per-env
        recession would make 64 envs recede 64x faster and skip the whole
        curriculum). The EMA still absorbs every outcome (it drives the entropy
        neuromodulation)."""
        outs = [bool(o) for o in outcomes]
        if not outs:
            return
        for ok in outs:
            self._observe(ok)
        batch_rate = sum(outs) / len(outs)
        self.frontier = max(0.0, self.frontier - self.advance_gain * batch_rate)

    @property
    def at_boot(self) -> bool:
        """True once the frontier has fully receded (all starts are cold boots)."""
        return self.frontier <= 0.0

    def state(self) -> dict:
        return {"frontier": round(self.frontier, 2), "frontier_frac": round(
            self.frontier / max(1, self.length), 4), "success_ema": round(
            self.success_ema, 4), "at_boot": self.at_boot, "seen": self._seen}


__all__ = ["DemoTrajectory", "BackwardCurriculum"]
