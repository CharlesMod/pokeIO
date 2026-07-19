"""The evaluation spine — competence-vs-Goodhart measurement protocol (#29).

Fable's diagnosis (brain-architecture.md, RED-TEAM REVISION): the last run stayed
"confident and dead" for ~90 generations *because* it had no rigorous eval — we
could not tell competence from Goodhart. This module is that missing organ. It
sits on top of the standing credibility harness (:mod:`pokeio.eval.harness` /
:mod:`pokeio.eval.controls` / :mod:`pokeio.eval.metrics`) and turns a single
champion score into a defensible verdict.

Five components, exactly the spec (v2.0-integration.md §"eval seam", task #29):

1. **rliable-style aggregate stats** — the headline is reported as an
   interquartile mean (IQM) with a percentile-bootstrap confidence interval over
   seeds/episodes, never a bare mean. Reuses :func:`metrics.iqm` /
   :func:`metrics.bootstrap_ci` verbatim.

2. **Controls that catch Goodhart** — the champion must **beat** two baselines in
   the *headline currency* (from-boot progress):
     * a **best-of-K random-weight** baseline (structured search has to clear the
       random-search bar — random search famously beat DQN/A3C on Atari), and
     * a **noise-frame ablation** of itself (progress must collapse when the
       screen is scrambled — otherwise it is "playing" blind).
   It ALSO runs the reusable vision-dependence audits from
   :mod:`pokeio.eval.controls` (``noise_ablation`` screen-blind flag +
   ``random_weight_baseline`` vision margin) when the policy is a NEAT genome.
   Any failed control is FLAGGED — this is the exact check the CR harness used to
   catch the OLD champions.

3. **Frozen-core git-diff protocol** — :func:`capture_frozen_core` records
   ``git rev-parse HEAD`` (+ a content digest over the train/eval core files) at
   the start of an eval and again at the end; a mismatch (a commit *or* a dirty
   mid-eval edit to the core) is detectable and recorded in the report. A
   reproducibility guard, so a "good" number can never come from quietly-moved
   code.

4. **Pre-registered 2nd game hook** — a config slot (:class:`SecondGameSpec`) for
   a second ROM. The headline must clear chance on it (the game-agnosticity
   check). No second ROM is decoded yet, so the hook reports a clear
   ``not-run`` status; wiring a ROM in later runs it with zero changes here.

5. **Headline = from-boot competence** — the score is the from-boot progress
   metric (task #28, ``pokeio/reward/from_boot.py::measure``), reported with
   IQM+CIs next to the controls. That module is being built in PARALLEL, so the
   dependency is **injected** (``from_boot_fn``): the spine codes against the
   documented seam and never hard-imports it. :func:`resolve_from_boot_fn` binds
   the real ``measure`` when it becomes importable.

The public entry point is :func:`evaluate`; it is callable on a **checkpoint
path** or a **live policy**, and returns a :class:`SpineReport`.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from pokeio.eval import controls, harness, metrics

# The from-boot seam (task #28 <-> #29). A ``from_boot_fn`` maps a policy + a
# rollout target (a fleet or a boot gauntlet) + a seed to a metric dict that
# MUST carry a scalar ``"progress_score"`` and MAY carry a per-episode
# ``"episode_scores"`` list (used for tighter CIs when present).
FromBootFn = Callable[..., dict]
# A random-weight policy factory: seed -> a policy comparable to the champion,
# consumable by the SAME ``from_boot_fn``.
PolicyFactory = Callable[[int], Any]

PROGRESS_KEY = "progress_score"
EPISODE_KEY = "episode_scores"


class FromBootNotWired(RuntimeError):
    """Raised when no ``from_boot_fn`` is supplied and task #28's ``measure`` is
    not importable yet — the headline metric has nothing to call."""


# ==========================================================================
# component 5 (seam): resolve the from-boot metric (injected, #28 in parallel)
# ==========================================================================
def resolve_from_boot_fn(
    from_boot_fn: FromBootFn | None = None,
) -> tuple[FromBootFn | None, str]:
    """Return ``(fn, source_label)`` for the headline from-boot metric.

    Precedence:
      1. an explicitly injected ``from_boot_fn`` (tests + the training loop pass
         this) -> ``"injected"``;
      2. otherwise the real ``pokeio.reward.from_boot.measure`` if importable ->
         ``"pokeio.reward.from_boot.measure"``;
      3. otherwise ``(None, "unwired")`` — :func:`evaluate` then raises
         :class:`FromBootNotWired` with a pointer to task #28.

    ---------------------------------------------------------------------------
    INTEGRATION POINT (task #28 <-> #29). The adapter below assumes the
    documented seam::

        measure(policy, target, *, seed, episodes, **kw)
            -> {"progress_score": float, "episode_scores"?: list[float], ...}

    If the concurrent #28 agent lands ``measure`` with a different signature,
    change ONLY :func:`_adapt_measure` — the rest of the spine is signature-clean.
    ---------------------------------------------------------------------------
    """
    if from_boot_fn is not None:
        return from_boot_fn, "injected"
    try:
        from pokeio.reward.from_boot import measure  # noqa: PLC0415 (lazy: #28)

        return _adapt_measure(measure), "pokeio.reward.from_boot.measure"
    except Exception:
        return None, "unwired"


def _adapt_measure(measure: Callable) -> FromBootFn:
    """Wrap #28's ``measure`` into the spine's ``from_boot_fn`` seam."""

    def _fn(policy, target, *, seed, episodes, **kw):
        return measure(policy, target, seed=seed, episodes=episodes, **kw)

    return _fn


# ==========================================================================
# component 3: frozen-core git-diff protocol
# ==========================================================================
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_core_files(repo: Path) -> list[Path]:
    """The train/eval core whose bytes must not move during an eval."""
    rel = [
        "pokeio/eval/spine.py",
        "pokeio/eval/harness.py",
        "pokeio/eval/controls.py",
        "pokeio/eval/metrics.py",
        "pokeio/train/gauntlet.py",
        "pokeio/reward/from_boot.py",  # #28; hashed once it exists
    ]
    return [repo / r for r in rel]


def _git_sha(repo_dir: str | Path | None = None) -> str:
    """Best-effort ``<sha>[-dirty]``; ``""`` outside a git checkout.

    Mirrors ``train/loop._git_sha`` (kept local so the spine has no heavy import).
    """
    repo = Path(repo_dir) if repo_dir else _repo_root()

    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=5
        )

    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode != 0:
            return ""
        sha = head.stdout.strip()
        status = _git("status", "--porcelain")
        if status.returncode == 0 and status.stdout.strip():
            sha += "-dirty"
        return sha
    except Exception:
        return ""


@dataclass
class FrozenCore:
    """A snapshot of the code the eval ran against (git sha + core content hash).

    ``sha`` is the committed HEAD; ``dirty`` flags an uncommitted working tree.
    ``core_digest`` is a sha256 over the concatenated core source files, so a
    mid-eval edit is caught **even without a commit** (the failure mode a bare
    ``rev-parse`` would miss).
    """

    sha: str
    dirty: bool
    core_digest: str
    core_files: list[str]
    missing: list[str] = field(default_factory=list)

    def verify(self, other: "FrozenCore") -> bool:
        """True iff the git sha AND the core content digest are both unchanged."""
        return self.sha == other.sha and self.core_digest == other.core_digest

    def summary(self) -> str:
        d = " (dirty)" if self.dirty else ""
        sha = self.sha or "<no-git>"
        return f"{sha}{d}  core-digest {self.core_digest[:12]}  ({len(self.core_files)} files)"


def capture_frozen_core(
    core_files: Sequence[str | Path] | None = None,
    *,
    repo_dir: str | Path | None = None,
) -> FrozenCore:
    """Snapshot the frozen core: git HEAD (+dirty) + a digest over ``core_files``.

    ``core_files`` defaults to the train/eval core (:func:`_default_core_files`).
    Missing files are recorded (not fatal — e.g. ``from_boot.py`` before #28
    lands), so the digest is stable and re-captures are comparable.
    """
    repo = Path(repo_dir) if repo_dir else _repo_root()
    files = list(core_files) if core_files is not None else _default_core_files(repo)

    h = hashlib.sha256()
    present: list[str] = []
    missing: list[str] = []
    for f in sorted(files, key=lambda p: str(p)):
        p = Path(f)
        if not p.is_absolute():
            p = repo / p
        key = str(p)
        if p.exists() and p.is_file():
            h.update(key.encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
            present.append(key)
        else:
            missing.append(key)

    sha_full = _git_sha(repo)
    dirty = sha_full.endswith("-dirty")
    sha = sha_full[:-6] if dirty else sha_full
    return FrozenCore(
        sha=sha, dirty=dirty, core_digest=h.hexdigest(),
        core_files=present, missing=missing,
    )


# ==========================================================================
# component 1: rliable-style aggregate stat (IQM + bootstrap CI)
# ==========================================================================
@dataclass
class AggregateStat:
    """An rliable-style aggregate over run scores: IQM + bootstrap CI."""

    iqm: float
    ci: tuple[float, float]
    ci_level: float
    mean: float
    n: int
    samples: list[float] = field(default_factory=list)

    @property
    def ci_lo(self) -> float:
        return self.ci[0]

    @property
    def ci_hi(self) -> float:
        return self.ci[1]

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"IQM {self.iqm:.4f}  {int(self.ci_level * 100)}% CI "
            f"[{lo:.4f}, {hi:.4f}]  (mean {self.mean:.4f}, n={self.n})"
        )


def aggregate(
    samples: Sequence[float], *, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0
) -> AggregateStat:
    """IQM + percentile-bootstrap CI over ``samples`` (reuses :mod:`metrics`)."""
    v = np.asarray(list(samples), dtype=np.float64)
    if v.size == 0:
        nan = float("nan")
        return AggregateStat(nan, (nan, nan), ci, nan, 0, [])
    point = metrics.iqm(v)
    lo, hi = metrics.bootstrap_ci(v, metrics.iqm, n_boot=n_boot, ci=ci, seed=seed)
    return AggregateStat(
        iqm=float(point), ci=(float(lo), float(hi)), ci_level=ci,
        mean=float(np.mean(v)), n=int(v.size), samples=[float(x) for x in v],
    )


# ==========================================================================
# from-boot rollout collection (the headline currency)
# ==========================================================================
def _invoke_from_boot(
    fn: FromBootFn, policy, target, *, seed: int, episodes: int, **extra
) -> dict:
    """Call ``fn`` for one seed; tolerate a fn that omits the ``episodes`` kwarg.

    Any ``extra`` kwarg (e.g. ``noise_frames=True``) is REQUIRED to be honoured —
    if the fn rejects it we re-raise, so the caller can mark that control
    unsupported rather than silently measuring the wrong thing.
    """
    try:
        return fn(policy, target, seed=seed, episodes=episodes, **extra)
    except TypeError:
        if extra:  # a requested variant (noise) is unsupported — do NOT fake it
            raise
        # ``episodes`` unsupported: emulate it by summing per-episode calls.
        eps: list[float] = []
        for e in range(int(episodes)):
            d = fn(policy, target, seed=int(seed) * 1000 + e)
            eps.append(float(d[PROGRESS_KEY]))
        return {PROGRESS_KEY: float(np.mean(eps)), EPISODE_KEY: eps}


def _collect_scores(
    fn: FromBootFn, policy, target, seeds: Sequence[int], episodes: int, **extra
) -> tuple[list[float], list[float]]:
    """Roll ``policy`` out over ``seeds`` -> ``(sample_scores, per_seed_scores)``.

    ``sample_scores`` flattens ``episode_scores`` across seeds when the metric
    supplies them (the rliable sample unit = a seed×episode run); otherwise it
    falls back to one ``progress_score`` per seed. ``per_seed_scores`` is always
    one value per seed (for a per-seed view in the report).
    """
    samples: list[float] = []
    per_seed: list[float] = []
    for s in seeds:
        d = _invoke_from_boot(fn, policy, target, seed=int(s), episodes=episodes, **extra)
        ps = float(d[PROGRESS_KEY])
        per_seed.append(ps)
        eps = d.get(EPISODE_KEY)
        if eps:
            samples.extend(float(x) for x in eps)
        else:
            samples.append(ps)
    return samples, per_seed


# ==========================================================================
# component 2: controls in the HEADLINE currency (Goodhart catchers)
# ==========================================================================
@dataclass
class ControlComparison:
    """The champion vs one baseline, both scored by the from-boot metric."""

    name: str  # "random_weight" | "noise_frame"
    champion: AggregateStat
    control: AggregateStat  # the bar the champion must beat (best-of-K / noise self)
    margin: float  # champion.iqm - control.iqm  (>0 => champion is ahead)
    beats: bool  # margin > 0
    separated: bool  # champion.ci_lo > control.ci_hi (a *strong*, CI-clean win)
    chance_level: float
    near_chance: bool  # is the control ~= chance? (proves the control is inert)
    k: int | None = None  # K for best-of-K (None for noise_frame)
    control_distribution: list[float] = field(default_factory=list)  # per-random IQM
    status: str = "run"  # "run" | "skipped:<reason>"
    note: str = ""

    def summary(self) -> str:
        if self.status != "run":
            return f"[{self.name}] {self.status} — {self.note}"
        verdict = "champion BEATS" if self.beats else "champion FAILS TO BEAT"
        strong = " (CI-clean)" if self.separated else ""
        chance = " ~chance" if self.near_chance else ""
        extra = f" best-of-{self.k}" if self.k is not None else ""
        return (
            f"[{self.name}]{extra} champion IQM {self.champion.iqm:.4f} vs "
            f"control IQM {self.control.iqm:.4f}{chance}  margin {self.margin:+.4f} "
            f"-> {verdict}{strong}"
        )


def _near_chance(control_iqm: float, chance: float, champion_iqm: float, tol: float | None) -> bool:
    if tol is None:
        # "near chance" = far closer to chance than the champion is above it.
        gap = max(abs(champion_iqm - chance), 1e-6)
        tol = 0.15 * gap
    return abs(control_iqm - chance) <= tol


def random_weight_control(
    fn: FromBootFn,
    factory: PolicyFactory,
    target,
    *,
    seeds: Sequence[int],
    episodes: int,
    k: int,
    champion: AggregateStat,
    chance_level: float,
    chance_tol: float | None,
    n_boot: int,
    ci: float,
    factory_seed: int = 0,
) -> ControlComparison:
    """Best-of-K random-weight baseline, scored by the from-boot metric.

    Samples ``k`` random-weight policies (``factory(seed)``), scores each the same
    way as the champion, and keeps the BEST (max-IQM) as the bar. Random search
    beat DQN/A3C/ES on several Atari games — a structured champion must clear
    this. The control's IQM landing at ~chance is the proof the control is inert
    (it separates competence from spawn luck).
    """
    rng = np.random.default_rng(factory_seed)
    per_policy_iqm: list[float] = []
    best_stat: AggregateStat | None = None
    for _ in range(int(k)):
        pol = factory(int(rng.integers(0, 2**31 - 1)))
        samples, _ = _collect_scores(fn, pol, target, seeds, episodes)
        stat = aggregate(samples, n_boot=n_boot, ci=ci, seed=int(rng.integers(0, 2**31 - 1)))
        per_policy_iqm.append(stat.iqm)
        if best_stat is None or stat.iqm > best_stat.iqm:
            best_stat = stat
    assert best_stat is not None
    margin = champion.iqm - best_stat.iqm
    return ControlComparison(
        name="random_weight",
        champion=champion,
        control=best_stat,
        margin=float(margin),
        beats=bool(margin > 0.0),
        separated=bool(champion.ci_lo > best_stat.ci_hi),
        chance_level=float(chance_level),
        near_chance=_near_chance(best_stat.iqm, chance_level, champion.iqm, chance_tol),
        k=int(k),
        control_distribution=[float(x) for x in per_policy_iqm],
        status="run",
        note="best-of-K random-weight, from-boot currency",
    )


def noise_frame_control(
    fn: FromBootFn,
    policy,
    target,
    *,
    seeds: Sequence[int],
    episodes: int,
    champion: AggregateStat,
    chance_level: float,
    chance_tol: float | None,
    n_boot: int,
    ci: float,
) -> ControlComparison:
    """Noise-frame ablation of the champion, scored by the from-boot metric.

    Re-runs the champion with the OPTICAL input scrambled (via the metric's
    ``noise_frames=True`` mode). Real progress must exceed noise-frame progress,
    else the "competence" is action-timing / spawn luck, not seeing. Skipped
    cleanly (status ``skipped:...``) when ``fn`` has no ``noise_frames`` mode — the
    vision-dependence ``noise_ablation`` audit still covers the screen-blind
    verdict in that case.
    """
    try:
        samples, _ = _collect_scores(fn, policy, target, seeds, episodes, noise_frames=True)
    except TypeError:
        return ControlComparison(
            name="noise_frame",
            champion=champion,
            control=aggregate([], n_boot=n_boot, ci=ci),
            margin=float("nan"),
            beats=False,
            separated=False,
            chance_level=float(chance_level),
            near_chance=False,
            status="skipped:from_boot_fn has no noise_frames mode",
            note="see vision_controls.noise_ablation for the screen-blind verdict",
        )
    ctrl = aggregate(samples, n_boot=n_boot, ci=ci)
    margin = champion.iqm - ctrl.iqm
    return ControlComparison(
        name="noise_frame",
        champion=champion,
        control=ctrl,
        margin=float(margin),
        beats=bool(margin > 0.0),
        separated=bool(champion.ci_lo > ctrl.ci_hi),
        chance_level=float(chance_level),
        near_chance=_near_chance(ctrl.iqm, chance_level, champion.iqm, chance_tol),
        status="run",
        note="champion vs its own noise-frame rollout, from-boot currency",
    )


# ==========================================================================
# component 2b: reused vision-dependence audits (controls.py) — genome policies
# ==========================================================================
@dataclass
class VisionControls:
    """The reusable vision-dependence audits from :mod:`pokeio.eval.controls`.

    Only meaningful for a NEAT-genome policy (a compiled sparse controller); for
    an opaque/live policy this reports ``status='n/a'`` and the from-boot-currency
    controls carry the Goodhart check on their own.
    """

    status: str
    probe_source: str = ""
    noise: controls.NoiseAblationResult | None = None
    random_baseline: controls.RandomBaselineResult | None = None

    @property
    def screen_blind(self) -> bool | None:
        return None if self.noise is None else bool(self.noise.blind[0])

    @property
    def beats_random_vision(self) -> bool | None:
        return None if self.random_baseline is None else bool(self.random_baseline.beats_random)


def _vision_controls(
    champ: harness.Champion,
    *,
    config,
    device: str,
    probe_size: int,
    probe_seed: int,
    k_random: int,
) -> VisionControls:
    """Run controls.py noise_ablation + random_weight_baseline on a genome champ.

    Reuses the harness probe/layout construction so numbers are commensurate with
    the standing CR harness; ROM-guarded (synthetic probe when no ROM/config).
    """
    n_ram = 8
    connect, sparse_k = "full", 32
    npro = None
    if config is not None:
        n_ram = int(getattr(config.vision, "obs_ram_bytes", n_ram))
        connect = str(getattr(config.evo, "init_connect", connect))
        sparse_k = int(getattr(config.evo, "init_k", sparse_k))
        if getattr(config.vision, "reflex_gaze", False):
            npro = 16
    layout = controls.infer_obs_layout(champ.n_in, champ.n_out, n_ram=n_ram, n_proprio=npro)
    probe, src = harness.build_probe(
        layout, config=config, b=probe_size, seed=probe_seed, use_rom=config is not None
    )
    cp = champ.compile(device)
    noise = controls.noise_ablation(
        cp, probe, layout.optical_hi, device=device, n_buttons=min(champ.n_out, 9)
    )
    scorer = controls.vision_dependence_scorer(probe, layout.optical_hi, device=device)
    champ_score = float(scorer(cp)[0])
    rand = controls.random_weight_baseline(
        n_in=champ.n_in, n_out=champ.n_out, k=k_random, score_fn=scorer,
        champion_score=champ_score, connect=connect, sparse_k=sparse_k,
        n_ram=layout.n_ram, n_proprio=layout.n_proprio, device=device, seed=probe_seed,
    )
    return VisionControls(status="run", probe_source=src, noise=noise, random_baseline=rand)


# ==========================================================================
# component 4: pre-registered 2nd-game hook
# ==========================================================================
@dataclass
class SecondGameSpec:
    """A pre-registered slot for a second ROM (the game-agnosticity check).

    Populate ``rom_path`` (+ ``target`` = a fleet/gauntlet on that ROM) once a
    second game is decoded; until then :func:`evaluate` reports ``not-run``.
    """

    rom_path: str | None = None
    reset_state: str | None = None
    label: str = "game2"
    chance_level: float = 0.0
    target: Any = None  # a fleet/gauntlet on the 2nd ROM (None until decoded)


@dataclass
class SecondGameHook:
    """The 2nd-game result: either a not-run placeholder or a real headline."""

    label: str
    rom_path: str | None
    reset_state: str | None
    status: str  # "not-run" | "run"
    chance_level: float
    headline: AggregateStat | None = None
    clears_chance: bool | None = None
    note: str = ""

    def summary(self) -> str:
        if self.status != "run":
            return f"2nd game '{self.label}': {self.status} — {self.note}"
        assert self.headline is not None
        verdict = "CLEARS chance" if self.clears_chance else "does NOT clear chance (FLAG)"
        return f"2nd game '{self.label}': {self.headline.summary()} -> {verdict}"


def _run_second_game(
    fn: FromBootFn,
    policy,
    spec: SecondGameSpec | None,
    *,
    seeds: Sequence[int],
    episodes: int,
    n_boot: int,
    ci: float,
) -> SecondGameHook:
    """Run (or explicitly defer) the pre-registered 2nd-game check."""
    if spec is None:
        return SecondGameHook(
            label="game2", rom_path=None, reset_state=None, status="not-run",
            chance_level=0.0, note="no 2nd-game spec supplied (pre-registered hook only)",
        )
    rom_ok = bool(spec.rom_path) and Path(spec.rom_path).exists()
    if not rom_ok or spec.target is None:
        why = "no 2nd ROM decoded yet" if not rom_ok else "no rollout target for the 2nd ROM"
        return SecondGameHook(
            label=spec.label, rom_path=spec.rom_path, reset_state=spec.reset_state,
            status="not-run", chance_level=float(spec.chance_level),
            note=f"{why} — headline-clears-chance check is pre-registered, not run",
        )
    samples, _ = _collect_scores(fn, policy, spec.target, seeds, episodes)
    head = aggregate(samples, n_boot=n_boot, ci=ci)
    clears = bool(head.ci_lo > spec.chance_level)  # lower CI above chance
    return SecondGameHook(
        label=spec.label, rom_path=spec.rom_path, reset_state=spec.reset_state,
        status="run", chance_level=float(spec.chance_level), headline=head,
        clears_chance=clears, note="game-agnosticity: headline must clear chance",
    )


# ==========================================================================
# the report
# ==========================================================================
@dataclass
class SpineReport:
    """The full eval-spine verdict for one policy."""

    run_id: str
    policy_kind: str  # "checkpoint" | "champion" | "live"
    from_boot_source: str  # "injected" | "pokeio.reward.from_boot.measure"
    seeds: list[int]
    episodes: int

    headline: AggregateStat  # from-boot progress, IQM + CIs (THE number)
    headline_clears_chance: bool
    chance_level: float

    controls: list[ControlComparison]  # random_weight (+ noise_frame)
    vision_controls: VisionControls  # reused controls.py audits (genome only)
    second_game: SecondGameHook

    frozen_core: FrozenCore  # captured at eval start
    frozen_core_ok: bool  # start == end (no mid-eval core edit)

    credible: bool
    flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """A JSON-friendly snapshot (dataclasses -> dicts; numpy dropped)."""
        d = asdict(self)
        # controls.py results carry numpy arrays -> keep only the scalar verdicts.
        vc = self.vision_controls
        d["vision_controls"] = {
            "status": vc.status,
            "probe_source": vc.probe_source,
            "screen_blind": vc.screen_blind,
            "beats_random_vision": vc.beats_random_vision,
        }
        return d


def format_report(rep: SpineReport) -> str:
    """Render a :class:`SpineReport` as an aligned text block."""
    L: list[str] = []
    L.append("=" * 74)
    L.append(f"  pokeIO EVALUATION SPINE  —  '{rep.run_id}'  [{rep.policy_kind} policy]")
    L.append("=" * 74)
    L.append(
        f"seeds={rep.seeds}  episodes={rep.episodes}  "
        f"headline={rep.from_boot_source}"
    )
    L.append(f"frozen-core: {rep.frozen_core.summary()}")
    if rep.frozen_core.missing:
        L.append(f"   (core files not yet present: {', '.join(Path(m).name for m in rep.frozen_core.missing)})")
    L.append(f"   integrity during eval: {'OK (unchanged)' if rep.frozen_core_ok else '!! CHANGED MID-EVAL'}")
    L.append("")

    L.append("-- HEADLINE: from-boot competence " + "-" * 39)
    L.append("   " + rep.headline.summary())
    ch = "clears chance" if rep.headline_clears_chance else "!! DOES NOT clear chance"
    L.append(f"   vs chance={rep.chance_level:.4f}: {ch}")
    L.append("")

    L.append("-- CONTROLS (Goodhart catchers, headline currency) " + "-" * 22)
    for c in rep.controls:
        L.append("   " + c.summary())
    L.append("")

    L.append("-- vision-dependence audits (controls.py) " + "-" * 31)
    vc = rep.vision_controls
    if vc.status != "run":
        L.append(f"   {vc.status}")
    else:
        if vc.noise is not None:
            L.append("   noise_ablation : " + vc.noise.summary(0))
        if vc.random_baseline is not None:
            L.append("   random_weight  : " + vc.random_baseline.summary())
    L.append("")

    L.append("-- pre-registered 2nd game " + "-" * 46)
    L.append("   " + rep.second_game.summary())
    L.append("")

    L.append("-- VERDICT " + "-" * 62)
    L.append(f"   credible = {rep.credible}")
    if rep.flags:
        for f in rep.flags:
            L.append(f"   !! FLAG: {f}")
    else:
        L.append("   no flags — champion clears every standing control.")
    L.append("=" * 74)
    return "\n".join(L)


# ==========================================================================
# policy resolution + the public entry point
# ==========================================================================
def _resolve_policy(policy) -> tuple[Any, str, harness.Champion | None]:
    """Return ``(policy_obj, kind, genome_champion_or_None)``.

    A checkpoint path or a :class:`harness.Champion` yields a genome champion (so
    the vision-dependence audits can run + the default random-weight factory can
    mint comparable random genomes). Any other object is treated as an opaque
    live policy that only ``from_boot_fn`` knows how to roll out.
    """
    if isinstance(policy, (str, Path)):
        champ = harness.load_champion(policy)
        return champ, "checkpoint", champ
    if isinstance(policy, harness.Champion):
        return policy, "champion", policy
    return policy, "live", None


def _default_random_factory(
    champ: harness.Champion, *, config, seed_base: int = 0
) -> PolicyFactory:
    """A random-weight-policy factory that mints random *Champions* like ``champ``.

    Same I/O widths + topology budget as the real champion, so the SAME
    ``from_boot_fn`` consumes it and the comparison is apples-to-apples.
    """
    n_ram = int(getattr(getattr(config, "vision", None), "obs_ram_bytes", 8)) if config else 8
    n_proprio = 16 if (config and getattr(config.vision, "reflex_gaze", False)) else (
        14 if champ.n_out == 11 else 0
    )
    connect = str(getattr(getattr(config, "evo", None), "init_connect", "full")) if config else "full"

    def factory(seed: int):
        g = controls.sample_random_genomes(
            n_in=champ.n_in, n_out=champ.n_out, k=1, connect=connect,
            n_ram=n_ram, n_proprio=n_proprio, seed=int(seed) ^ seed_base,
        )[0]
        return harness.Champion(
            genome=g, index=0, fitness=0.0, n_in=champ.n_in, n_out=champ.n_out,
            is_retina=champ.is_retina, gen=-1, n_nodes=len(g.nodes), n_conns=len(g.conns),
        )

    return factory


def evaluate(
    policy,
    fleet_or_gauntlet,
    from_boot_fn: FromBootFn | None,
    *,
    seeds: Sequence[int],
    episodes: int,
    config=None,
    device: str = "cpu",
    run_id: str = "eval",
    random_policy_factory: PolicyFactory | None = None,
    k_random: int = 16,
    chance_level: float = 0.0,
    chance_tol: float | None = None,
    second_game: SecondGameSpec | None = None,
    run_noise_frame_control: bool = True,
    run_vision_controls: bool = True,
    vision_k_random: int = 64,
    probe_size: int = 200,
    probe_seed: int = 1,
    n_boot: int = 10_000,
    ci: float = 0.95,
    frozen_core_files: Sequence[str | Path] | None = None,
) -> SpineReport:
    """Run the evaluation spine on ``policy`` — the seam the gate (#35) calls.

    Parameters
    ----------
    policy
        A **checkpoint path** (str/Path to a run dir or ``checkpoint.pkl``), a
        :class:`harness.Champion`, or a **live policy** object that
        ``from_boot_fn`` can roll out.
    fleet_or_gauntlet
        The from-boot rollout target passed straight through to ``from_boot_fn``
        (an :class:`AsyncFleet`, a boot gauntlet, or a test stub).
    from_boot_fn
        The headline metric (task #28's ``measure``); pass ``None`` to auto-bind
        the real one via :func:`resolve_from_boot_fn` (raises
        :class:`FromBootNotWired` if #28 is not importable yet).
    seeds, episodes
        The rliable sample grid: the headline IQM+CI is taken over the flattened
        seed×episode run scores.

    Returns
    -------
    SpineReport
        Headline (from-boot IQM+CIs), the control comparisons (vs random-weights
        / noise-frames), the reused vision-dependence audits, the 2nd-game hook,
        and the frozen-core sha — with a top-level ``credible`` verdict + flags.
    """
    fb_fn, fb_source = resolve_from_boot_fn(from_boot_fn)
    if fb_fn is None:
        raise FromBootNotWired(
            "no from_boot_fn supplied and pokeio.reward.from_boot.measure is not "
            "importable yet (task #28 in parallel). Pass from_boot_fn=... to eval "
            "against the documented seam: measure(policy, target, *, seed, "
            "episodes) -> {'progress_score': float, 'episode_scores'?: [...]}."
        )

    seeds = list(int(s) for s in seeds)
    fc_begin = capture_frozen_core(frozen_core_files)

    pol_obj, kind, champ = _resolve_policy(policy)

    # --- headline: from-boot competence (IQM + CIs) ---
    head_samples, _ = _collect_scores(fb_fn, pol_obj, fleet_or_gauntlet, seeds, episodes)
    headline = aggregate(head_samples, n_boot=n_boot, ci=ci, seed=0)
    headline_clears = bool(headline.ci_lo > chance_level)

    # --- controls in the headline currency ---
    control_list: list[ControlComparison] = []
    factory = random_policy_factory
    if factory is None and champ is not None:
        factory = _default_random_factory(champ, config=config)
    if factory is not None:
        control_list.append(
            random_weight_control(
                fb_fn, factory, fleet_or_gauntlet, seeds=seeds, episodes=episodes,
                k=k_random, champion=headline, chance_level=chance_level,
                chance_tol=chance_tol, n_boot=n_boot, ci=ci,
            )
        )
    else:
        control_list.append(
            ControlComparison(
                name="random_weight", champion=headline, control=aggregate([], ci=ci),
                margin=float("nan"), beats=False, separated=False,
                chance_level=chance_level, near_chance=False,
                status="skipped:no random_policy_factory",
                note="pass random_policy_factory=... (or a genome/checkpoint policy)",
            )
        )
    if run_noise_frame_control:
        control_list.append(
            noise_frame_control(
                fb_fn, pol_obj, fleet_or_gauntlet, seeds=seeds, episodes=episodes,
                champion=headline, chance_level=chance_level, chance_tol=chance_tol,
                n_boot=n_boot, ci=ci,
            )
        )

    # --- reused vision-dependence audits (genome policies only) ---
    if run_vision_controls and champ is not None:
        try:
            vision = _vision_controls(
                champ, config=config, device=device, probe_size=probe_size,
                probe_seed=probe_seed, k_random=vision_k_random,
            )
        except Exception as e:  # never let an audit sink the whole eval
            vision = VisionControls(status=f"error:{type(e).__name__}: {e}")
    else:
        why = "policy is not a NEAT genome" if champ is None else "disabled"
        vision = VisionControls(status=f"n/a: {why}")

    # --- pre-registered 2nd game ---
    second = _run_second_game(
        fb_fn, pol_obj, second_game, seeds=seeds, episodes=episodes, n_boot=n_boot, ci=ci
    )

    # --- frozen-core integrity (start vs end) ---
    fc_end = capture_frozen_core(frozen_core_files)
    frozen_ok = fc_begin.verify(fc_end)

    # --- verdict + flags ---
    flags: list[str] = []
    if not headline_clears:
        flags.append(f"headline does not clear chance ({headline.iqm:.4f} vs {chance_level:.4f})")
    for c in control_list:
        if c.status == "run" and not c.beats:
            flags.append(f"does not beat {c.name} baseline (margin {c.margin:+.4f})")
    if vision.screen_blind:
        flags.append("vision audit: champion is SCREEN-BLIND (winning by action-timing)")
    if vision.beats_random_vision is False:
        flags.append("vision audit: does not beat random-weight vision baseline")
    if second.status == "run" and second.clears_chance is False:
        flags.append("2nd game: headline does not clear chance (game-agnosticity fail)")
    if not frozen_ok:
        flags.append("frozen-core CHANGED during eval (reproducibility guard tripped)")
    credible = len(flags) == 0

    return SpineReport(
        run_id=run_id,
        policy_kind=kind,
        from_boot_source=fb_source,
        seeds=seeds,
        episodes=int(episodes),
        headline=headline,
        headline_clears_chance=headline_clears,
        chance_level=float(chance_level),
        controls=control_list,
        vision_controls=vision,
        second_game=second,
        frozen_core=fc_begin,
        frozen_core_ok=frozen_ok,
        credible=credible,
        flags=flags,
        meta={"n_seeds": len(seeds), "device": device},
    )


__all__ = [
    "FromBootFn",
    "PolicyFactory",
    "FromBootNotWired",
    "resolve_from_boot_fn",
    "FrozenCore",
    "capture_frozen_core",
    "AggregateStat",
    "aggregate",
    "ControlComparison",
    "random_weight_control",
    "noise_frame_control",
    "VisionControls",
    "SecondGameSpec",
    "SecondGameHook",
    "SpineReport",
    "format_report",
    "evaluate",
]
