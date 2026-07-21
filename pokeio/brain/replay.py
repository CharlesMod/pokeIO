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
        # Per-depth progress prefix (running maxes + maps-seen set) recorded in
        # the SAME cached replay as the blobs: a depth-k restore hands the policy
        # everything the demo did through k, so the reward baseline must be
        # seeded from this history too (teleport-decoupling for the maps term).
        self._histories: dict[int, dict] = {}
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
            from pokeio.reward.from_boot import YELLOW, decode_state

            _MAXES = ("party_count", "party_level", "badges", "events",
                      "counter_sum")
            prog: dict = {f: 0 for f in _MAXES}
            prog["maps"] = set()

            def _absorb(env):
                d = decode_state(env.raw_wram(), YELLOW)
                for f in _MAXES:
                    if d[f] > prog[f]:
                        prog[f] = d[f]
                prog["maps"].add(d["map_id"])

            env = PokeEnv(rom_path=self.rom_path, frame_skip=24)
            try:
                env.reset(self.newgame_state)
                _absorb(env)  # the newgame state itself (spawn map) is history
                k = 0
                for d in sorted(missing):
                    while k < d:
                        env.step(int(self.actions[k]))
                        _absorb(env)
                        k += 1
                    buf = io.BytesIO()
                    env.pyboy.save_state(buf)  # raw: does NOT advance the emulator
                    self._blobs[d] = buf.getvalue()
                    self._histories[d] = {**{f: prog[f] for f in _MAXES},
                                          "maps": set(prog["maps"])}
            finally:
                env.close()
        return {d: self._blobs[d] for d in want}

    def history_at(self, depth: int) -> dict | None:
        """Progress snapshot of the demo prefix through ``depth`` (running maxes
        + maps-seen set) — the ``history`` for ``ProgressReward.reset`` on a
        depth-``depth`` restore, so demo-handed progress (incl. already-traversed
        maps) is never re-payable. Depth 0 = boot = nothing handed -> ``None``.
        Returns a copy (fresh maps set) so callers may mutate freely."""
        d = self._clamp(depth)
        if d <= 0:
            return None
        if d not in self._histories:
            self.capture_ladder([d])
        h = self._histories.get(d)
        return None if h is None else {**h, "maps": set(h["maps"])}

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


class ForwardCurriculum:
    """Forward-progression spawn curriculum — a self-tuning MIX of start states.

    WHY (brain6): brain4/brain5 proved that boot-reset episodes are solvable by an
    open-loop memorized button script (mode collapse at the demo's progress
    ceiling), which starves the learned gaze of gradient — vision is unnecessary
    when every episode starts from the same frame. The fix is to start episodes
    from a DISTRIBUTION of states so no single script solves it and the policy
    must look at the screen to know where it is.

    Three spawn sources per env per episode reset:
      * ``boot``    — the canonical cold boot (the gate's own start);
      * ``demo``    — a depth along the recorded demo, sampled by an inner
        :class:`BackwardCurriculum` (the existing recede-on-success behaviour);
      * ``archive`` — a Go-Explore frontier cell restore (only once the archive
        holds cells; an empty archive degrades gracefully to boot+demo).

    Source mixing is SELF-TUNING, no hand-set ratios (no-tuned-knobs mandate):
    each source keeps its own success EMA of "crossed a spawn-relative
    MILESTONE" (``ProgressReward.earned_milestone``: a tier-0 spawn must EARN
    started(); a spawn handed that milestone must earn the next badge/event
    tier — handed progress never counts, and a mere map transition is not a
    success, else 'walk out the door' would saturate every EMA), and a
    source's sampling mass is the Bernoulli outcome VARIANCE ``p*(1-p)`` of a
    prior-smoothed estimate ``p`` of that EMA. Variance is maximal where outcomes
    are least predictable — the learning frontier — and falls toward 0 for
    sources the policy has saturated (EMA ~1, too easy) or never cracks (EMA ~0,
    too hard): exactly the BackwardCurriculum recede rule generalized from one
    scalar frontier to a set of sources. The smoothing is Laplace's rule of
    succession (uniform max-entropy prior, pseudo-count 1, capped by the EMA's
    own effective memory ``1/(1-ema_decay)``) — a statistical default, not a
    tuned threshold. It does two jobs: an unexplored source starts AT the
    frontier (max mass), and no available source's mass can reach exactly 0
    (an absorbing state would make a source's re-becoming-learnable invisible).

    A GLOBAL success EMA over all outcomes (same hard-set-on-first-sample
    mechanics as BackwardCurriculum) feeds the trainer's entropy neuromodulation
    unchanged. Constructor params are curriculum PACE (like a learning rate),
    not behaviour thresholds; they mirror BackwardCurriculum's.
    """

    SOURCES = ("boot", "demo", "archive")

    def __init__(self, length: int, advance_gain: float = 8.0,
                 ema_decay: float = 0.9, spread_frac: float = 0.15,
                 start_frac: float = 0.97, seed: int = 0) -> None:
        self.backward = BackwardCurriculum(
            length, advance_gain=advance_gain, ema_decay=ema_decay,
            spread_frac=spread_frac, start_frac=start_frac, seed=seed)
        self.ema_decay = float(ema_decay)
        self.src_ema = {s: 0.0 for s in self.SOURCES}
        self.src_seen = {s: 0 for s in self.SOURCES}
        self.success_ema = 0.0     # global (drives ent_coef neuromodulation)
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    # -- compatibility surface (checkpoint/metrics/horizon read these) ------
    @property
    def length(self) -> int:
        return self.backward.length

    @property
    def frontier(self) -> float:
        """The DEMO source's inner backward frontier (checkpoint/metrics compat)."""
        return self.backward.frontier

    @frontier.setter
    def frontier(self, v: float) -> None:
        self.backward.frontier = float(v)

    @property
    def at_boot(self) -> bool:
        return self.backward.at_boot

    # -- source mixing -------------------------------------------------------
    def _mass(self, source: str) -> float:
        """Sampling mass = Bernoulli outcome variance of the prior-smoothed EMA.

        ``n_eff`` is the EMA's effective sample count, capped at its own memory
        length ``1/(1-decay)`` — the smoothing therefore never pretends to more
        evidence than the EMA can actually hold. Strictly in (0, 0.25]."""
        n_eff = min(float(self.src_seen[source]), 1.0 / (1.0 - self.ema_decay))
        p = (self.src_ema[source] * n_eff + 0.5) / (n_eff + 1.0)
        return float(p * (1.0 - p))

    def source_masses(self, archive_size: int = 0) -> dict[str, float]:
        """Normalized sampling mass per AVAILABLE source (archive needs cells)."""
        avail = [s for s in self.SOURCES if s != "archive" or archive_size > 0]
        m = np.array([self._mass(s) for s in avail], dtype=np.float64)
        m /= m.sum()   # each mass strictly >0, so the sum is too
        return {s: float(w) for s, w in zip(avail, m)}

    def sample_spawns(self, n: int, archive_size: int = 0) -> list[tuple[str, int]]:
        """``n`` per-env spawn draws: ``(source, demo_depth)`` tuples.

        ``demo_depth`` is only meaningful for ``source == "demo"`` (the inner
        BackwardCurriculum's clustered depth; 0 = cold boot once fully receded);
        it is 0 for boot and archive (the trainer fetches archive blobs itself —
        this class never touches emulator state, mirroring BackwardCurriculum)."""
        masses = self.source_masses(archive_size)
        avail = list(masses)
        picks = self._rng.choice(len(avail), size=int(n), p=list(masses.values()))
        depths = iter(self.backward.sample_depths(int((picks == avail.index("demo")).sum())))
        return [(avail[k], next(depths) if avail[k] == "demo" else 0) for k in picks]

    # -- outcome reporting ----------------------------------------------------
    def _fold(self, ema: float, seen: int, ok: bool) -> float:
        s = 1.0 if ok else 0.0
        return s if seen == 0 else self.ema_decay * ema + (1.0 - self.ema_decay) * s

    def report_spawns(self, sources, outcomes) -> None:
        """One iteration's batch: per-env ``(source, crossed-spawn-relative-milestone)``.

        Every outcome folds into the GLOBAL EMA (entropy neuromodulation) and its
        source's EMA (mix adaptation). Demo-source outcomes also drive the inner
        backward frontier via ``report_many`` — recession stays a batch RATE, so
        it is n_envs-independent exactly like the backward path."""
        outs = [bool(o) for o in outcomes]
        srcs = list(sources)
        if not outs:
            return
        for ok in outs:
            self.success_ema = self._fold(self.success_ema, self._seen, ok)
            self._seen += 1
        for s, ok in zip(srcs, outs):
            self.src_ema[s] = self._fold(self.src_ema[s], self.src_seen[s], ok)
            self.src_seen[s] += 1
        demo_outs = [ok for s, ok in zip(srcs, outs) if s == "demo"]
        if demo_outs:
            self.backward.report_many(demo_outs)

    # -- persistence (reboot-resume safe: the box loses power constantly) -----
    def state(self) -> dict:
        """Rounded scalars for the per-iter metrics record (BackwardCurriculum
        keys preserved so logs/dashboard/print lines read identically), plus the
        per-source mix so the jsonl shows the adaptation."""
        st = {"frontier": round(self.frontier, 2),
              "frontier_frac": round(self.frontier / max(1, self.length), 4),
              "success_ema": round(self.success_ema, 4),
              "at_boot": self.at_boot, "seen": self._seen}
        masses = self.source_masses(archive_size=1)   # show all three masses
        st["sources"] = {s: {"ema": round(self.src_ema[s], 4),
                             "seen": self.src_seen[s],
                             "mass": round(masses[s], 4)} for s in self.SOURCES}
        return st

    def full_state(self) -> dict:
        """FULL-precision state for the checkpoint (the rounded :meth:`state` is
        for logs). RNG state is not persisted, matching BackwardCurriculum."""
        return {"kind": "forward",
                "success_ema": float(self.success_ema), "seen": int(self._seen),
                "src_ema": {s: float(v) for s, v in self.src_ema.items()},
                "src_seen": {s: int(v) for s, v in self.src_seen.items()},
                "backward": {"frontier": float(self.backward.frontier),
                             "success_ema": float(self.backward.success_ema),
                             "seen": int(self.backward._seen)}}

    def load_state(self, st: dict) -> None:
        """Restore from :meth:`full_state` (checkpoint resume)."""
        self.success_ema = float(st.get("success_ema", 0.0))
        self._seen = int(st.get("seen", 0))
        for s in self.SOURCES:
            self.src_ema[s] = float(st.get("src_ema", {}).get(s, 0.0))
            self.src_seen[s] = int(st.get("src_seen", {}).get(s, 0))
        bw = st.get("backward") or {}
        self.backward.frontier = float(bw.get("frontier", self.backward.frontier))
        self.backward.success_ema = float(bw.get("success_ema", 0.0))
        self.backward._seen = int(bw.get("seen", 0))


__all__ = ["DemoTrajectory", "BackwardCurriculum", "ForwardCurriculum"]
