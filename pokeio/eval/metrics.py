"""Headline metric: geometric-mean-of-milestones with IQM + bootstrap CIs.

Rationale (audit 2026-07-17, STRATEGIC): *never* report a raw score as the
headline. Raw fitness here is a Goodhart trap (the mined counter is both shown
to the agent and rewarded) and a lucky-spawn artefact. The credibility number
is the **Crafter-style geometric mean of per-milestone success rates**, reported
**rliable-style** as an interquartile mean (IQM) plus a bootstrap confidence
interval so a single point can never be over-read.

Everything here is pure ``numpy`` (the bootstrap is implemented by hand — no
``rliable`` / ``scipy`` dependency) plus the repo's own telemetry reader.

Two layers, deliberately separated so the math is testable without a run:

* **The math** — :func:`geometric_mean`, :func:`iqm`, :func:`bootstrap_ci`,
  :func:`milestone_geomean_report`. These take an array / dict of rates and
  know nothing about pokeIO.
* **The extraction** — :func:`milestone_rates_from_telemetry` turns a run's
  ``telemetry.jsonl`` (+ an optional manifest) into a ``{milestone: rate}`` dict.
  With no manifest it derives generic milestones from the mined-progress / depth
  signals that already live in ``reward_terms`` (game-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from pokeio.telemetry.schema import read_generations

# --------------------------------------------------------------------------
# the math (pokeIO-agnostic)
# --------------------------------------------------------------------------


def geometric_mean(rates: Sequence[float]) -> float:
    """Geometric mean of per-milestone success rates in ``[0, 1]``.

    Crafter's headline aggregation. Computed in log-space
    (``exp(mean(log r))``) for numerical stability. Semantics at the edges:

    * an empty set -> ``nan`` (undefined; the caller decides what to show);
    * **any** rate exactly ``0`` -> ``0.0`` (you provably never cleared that
      milestone, so the geometric headline is zero — this is the point of a
      geometric mean, it refuses to average a wall away);
    * negatives are clipped to ``0`` (a rate below zero is a caller bug).

    Sanity: all ``1.0`` -> ``1.0``; ``[0.25, 1.0]`` -> ``0.5``.
    """
    r = np.asarray(list(rates), dtype=np.float64)
    if r.size == 0:
        return float("nan")
    r = np.clip(r, 0.0, None)
    if np.any(r == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(r))))


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% (drop top/bottom 25%).

    rliable's robust central-tendency estimator — insensitive to the lucky-run
    tail that makes a plain mean (or a raw ``fitness_best``) misleading. Falls
    back to the plain mean when the sample is too small to trim.
    """
    v = np.sort(np.asarray(list(values), dtype=np.float64))
    n = v.size
    if n == 0:
        return float("nan")
    k = int(n * 0.25)
    core = v[k : n - k] if (n - 2 * k) > 0 else v
    return float(np.mean(core))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = iqm,
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``values`` (numpy only).

    Resamples ``values`` with replacement ``n_boot`` times, applies
    ``statistic`` to each resample, and returns the ``(lo, hi)`` percentiles for
    a two-sided ``ci`` interval. With one run this quantifies how sensitive the
    aggregate is to *which* milestones were sampled; with many runs/seeds pass
    the per-run aggregates and it becomes the standard rliable run-level CI.
    """
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        s = float(statistic(v))
        return (s, s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    boot = np.array([statistic(v[row]) for row in idx], dtype=np.float64)
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 - (1.0 - ci) / 2.0) * 100.0
    lo, hi = np.percentile(boot, [lo_q, hi_q])
    return (float(lo), float(hi))


@dataclass
class MilestoneReport:
    """The headline milestone metric, ready to print."""

    geomean: float  # THE headline (Crafter-style geometric mean)
    iqm: float  # robust central tendency of the per-milestone rates
    ci: tuple[float, float]  # bootstrap CI on the geometric mean
    ci_level: float
    n_milestones: int
    rates: dict[str, float] = field(default_factory=dict)
    signal: str = ""  # which telemetry signal the generic milestones came from
    meta: dict[str, Any] = field(default_factory=dict)


def milestone_geomean_report(
    rates: dict[str, float] | Sequence[float],
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    signal: str = "",
    meta: dict[str, Any] | None = None,
) -> MilestoneReport:
    """Assemble the headline report from a set of per-milestone success rates.

    The point estimate is :func:`geometric_mean` (the headline). The CI is a
    bootstrap over the milestone rates with the geometric mean as the statistic;
    the IQM is a robust summary of the rates themselves.
    """
    if isinstance(rates, dict):
        labels = list(rates.keys())
        vals = np.asarray([rates[k] for k in labels], dtype=np.float64)
        rate_map = dict(rates)
    else:
        vals = np.asarray(list(rates), dtype=np.float64)
        rate_map = {f"m{i}": float(v) for i, v in enumerate(vals)}
    return MilestoneReport(
        geomean=geometric_mean(vals),
        iqm=iqm(vals),
        ci=bootstrap_ci(vals, geometric_mean_np, n_boot=n_boot, ci=ci, seed=seed),
        ci_level=ci,
        n_milestones=int(vals.size),
        rates=rate_map,
        signal=signal,
        meta=meta or {},
    )


def geometric_mean_np(v: np.ndarray) -> float:
    """``geometric_mean`` adapter with a plain-``ndarray`` signature (for the
    bootstrap ``statistic`` slot)."""
    return geometric_mean(v)


# --------------------------------------------------------------------------
# extraction: telemetry -> per-milestone success rates (game-agnostic default)
# --------------------------------------------------------------------------

# Generic progress signals to look for, in priority order. Each is a
# monotone-ish "how deep did the swarm get" measure that lives in a run's
# telemetry regardless of game. The first one present drives the generic
# milestone ladder when no manifest is supplied.
_GENERIC_SIGNALS = (
    "progress_best",  # mined-counter advancement (reward_terms)
    "goexplore_max_depth",  # Go-Explore frontier depth (reward_terms)
    "archive_cells",  # distinct cells discovered (top-level)
    "boot_depth_frac",  # boot-gauntlet depth fraction (top-level)
    "fitness_best",  # last resort (top-level)
)


def _signal_series(gens: list, signal: str) -> np.ndarray:
    """Pull one named signal's per-generation series (reward_terms or top-level)."""
    out: list[float] = []
    for g in gens:
        if signal in g.reward_terms:
            out.append(float(g.reward_terms[signal]))
        else:
            v = getattr(g, signal, None)
            # -1.0 is the "didn't run this gen" sentinel for the boot-gauntlet fields
            if v is not None and not (isinstance(v, float) and v == -1.0):
                out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def _pick_signal(gens: list, prefer: str | None) -> tuple[str, np.ndarray]:
    """Choose the progress signal to build generic milestones from."""
    if prefer:
        return prefer, _signal_series(gens, prefer)
    for sig in _GENERIC_SIGNALS:
        series = _signal_series(gens, sig)
        if series.size and float(np.max(series)) > 0.0:
            return sig, series
    return "", np.asarray([], dtype=np.float64)


def generic_milestone_rates(
    series: np.ndarray, *, signal: str, n_levels: int = 10
) -> dict[str, float]:
    """Turn a progress-signal time-series into ``n_levels`` milestone rates.

    Defines a ladder of ``n_levels`` thresholds at fractions ``i/n_levels`` of
    the run's peak signal. Each "milestone" is *reach threshold*, and its
    success rate is the fraction of generations that met it:

        rate_i = mean( series >= (i / n_levels) * peak )

    Deeper milestones have lower rates, so the geometric mean rewards
    *consistently* reaching progressively deeper progress rather than a single
    lucky spike. All-at-peak -> every rate ``1.0`` -> geomean ``1.0``.
    """
    if series.size == 0:
        return {}
    peak = float(np.max(series))
    if peak <= 0.0:
        return {f"{signal}>={i}/{n_levels}peak": 0.0 for i in range(1, n_levels + 1)}
    rates: dict[str, float] = {}
    for i in range(1, n_levels + 1):
        frac = i / n_levels
        thresh = frac * peak
        rates[f"{signal}>={int(frac * 100)}%peak"] = float(np.mean(series >= thresh))
    return rates


def _manifest_milestone_rates(gens: list, manifest) -> dict[str, float] | None:
    """Per-dimension rates from a manifest, when telemetry carries the values.

    Maps each ``progress_dimension.id`` to a ``reward_terms`` series of the same
    name (the natural place a future manifest-consuming loop would log per-
    dimension progress). A dimension improves ``up`` or ``down``; the rate is
    the fraction of generations at/above (or at/below) the dimension's peak-
    normalised midpoint. Returns ``None`` when *no* dimension has a matching
    telemetry series, so the caller can fall back to the generic ladder.

    This is the documented hook for feeding a real manifest later: once the loop
    logs ``reward_terms[<dimension id>]`` per generation, milestone rates become
    manifest-driven with zero changes here.
    """
    dims = getattr(manifest, "progress_dimensions", None) or []
    rates: dict[str, float] = {}
    matched = False
    for pd in dims:
        series = _signal_series(gens, pd.id)
        if series.size == 0:
            continue
        matched = True
        peak = float(np.max(series))
        lo = float(np.min(series))
        if peak <= lo:  # flat signal -> reached iff nonzero
            rates[pd.label or pd.id] = float(np.mean(series > 0.0))
            continue
        mid = lo + 0.5 * (peak - lo)
        if getattr(pd, "dir", "up") == "down":
            rates[pd.label or pd.id] = float(np.mean(series <= mid))
        else:
            rates[pd.label or pd.id] = float(np.mean(series >= mid))
    return rates if matched else None


def milestone_rates_from_telemetry(
    run_dir: str | Path,
    *,
    manifest=None,
    signal: str | None = None,
    n_levels: int = 10,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute ``{milestone: success_rate}`` for a run.

    Precedence:

    1. If ``manifest`` is supplied AND telemetry carries a matching per-dimension
       series, milestones are the manifest's progress dimensions.
    2. Otherwise a **generic** ladder is built from the deepest available
       mined-progress / depth signal (``signal`` overrides the auto-pick).

    Returns ``(rates, meta)``; ``meta`` records which signal/mode was used and
    the number of generations, for the report header.
    """
    gens = read_generations(run_dir)
    meta: dict[str, Any] = {"n_gens": len(gens)}
    if not gens:
        meta["mode"] = "empty"
        return {}, meta

    if manifest is not None:
        mrates = _manifest_milestone_rates(gens, manifest)
        if mrates:
            meta["mode"] = "manifest"
            meta["milestone_label"] = getattr(manifest, "milestone_label", "")
            return mrates, meta

    sig, series = _pick_signal(gens, signal)
    meta["mode"] = "generic"
    meta["signal"] = sig
    meta["peak"] = float(np.max(series)) if series.size else 0.0
    if not sig:
        return {}, meta
    return generic_milestone_rates(series, signal=sig, n_levels=n_levels), meta


__all__ = [
    "geometric_mean",
    "geometric_mean_np",
    "iqm",
    "bootstrap_ci",
    "MilestoneReport",
    "milestone_geomean_report",
    "generic_milestone_rates",
    "milestone_rates_from_telemetry",
]
