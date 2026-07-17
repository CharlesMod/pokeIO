"""Game-agnostic WRAM progress-counter miner.

Given raw Work-RAM (WRAM) snapshots recorded over one or more rollouts, this
module discovers memory addresses that *behave like progress counters* — values
that move in a consistent direction, in small regular increments, activating at
a moderate rate, and doing so consistently across independent episodes.  It bakes
in ZERO game knowledge: no hardcoded addresses, no Pokemon semantics.  The ranked
candidates it emits become the "progress dims" that a downstream LLM/manifest can
name and the reward stack can co-evolve.

Input contract
--------------
A *rollout* is a time series of WRAM snapshots.  Each snapshot is the raw byte
array returned by :meth:`pokeio.emu.env.PokeEnv.raw_wram` (an 8 KB uint8 array
covering GB addresses ``0xC000..0xDFFF``; column ``i`` -> address ``0xC000 + i``).
Accepted forms for a rollout:
  * a 2-D ndarray of shape ``(T, N)`` (T snapshots, N bytes), or
  * a sequence of 1-D snapshots that will be stacked into such a matrix.
``mine(rollouts, ...)`` takes a sequence of rollouts (possibly from different
episodes/policies) and returns a ranked list of :class:`Candidate`.

Features per candidate (computed per-rollout, then aggregated across rollouts):
  * **monotonicity**   — directional consistency of the *changes*: among steps
    where the value moves, the fraction that move the same way.
  * **change activity** — fraction of steps on which the value changes; scored
    through a band so static addresses and hyperactive noise both lose.
  * **delta entropy**   — Shannon entropy of the *nonzero* increments, normalized
    to ``[0, 1]``.  Counter-like addresses take small regular steps -> low
    entropy.  Addresses above ``entropy_mask`` are masked out entirely.
  * **cross-episode consistency** — behaves counter-like across independent
    rollouts (active in a fraction of them, with agreeing direction), not once.
  * **range/scale sanity** — increment magnitudes are regular in scale rather
    than wildly dispersed (a cheap wrap/noise guard).

16-bit little-endian pairs are considered alongside single bytes, since real
counters (money, experience, coordinates) routinely span multiple bytes; a wide
counter's low byte wraps and looks non-monotone on its own, but the pair does not.

Dependency-light: numpy + stdlib only.  Scoring weights are configurable via
:class:`MinerConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

WRAM_BASE = 0xC000  # column 0 of a snapshot maps to this GB address


# --------------------------------------------------------------------------- config
@dataclass
class MinerConfig:
    """Tunable knobs and scoring weights for the miner.

    The activity *band* rejects addresses that never move (static) and ones that
    move almost every step (hyperactive noise / free-running timers).  The
    entropy mask hard-drops addresses whose increments look random.
    """

    # -- masks / filters ---------------------------------------------------
    entropy_mask: float = 0.95      # drop candidates with delta-entropy above this
    min_activity: float = 0.005     # below this fraction of changing steps => static
    max_activity: float = 0.90      # above this => hyperactive noise
    min_changes: int = 3            # need at least this many change-events (per candidate, summed)
    consider_pairs: bool = True     # also score 16-bit little-endian byte pairs
    mono_active_thresh: float = 0.75  # per-rollout monotonicity to count as "counter-like"
    dedup_rollouts: bool = True     # drop byte-identical rollouts before scoring

    # -- activity band shape ----------------------------------------------
    activity_target: float = 0.15   # activity that scores best in the band
    activity_width: float = 0.20    # gaussian width of the activity band

    # -- scoring weights (relative; normalized internally) -----------------
    w_monotonic: float = 1.0
    w_activity: float = 0.6
    w_entropy: float = 1.0
    w_consistency: float = 1.2
    w_range: float = 0.5

    @classmethod
    def from_reward_config(cls, reward_cfg, **overrides) -> "MinerConfig":
        """Build from a :class:`pokeio.config.RewardConfig` (reads ``miner_entropy_mask``)."""
        kw = dict(entropy_mask=float(getattr(reward_cfg, "miner_entropy_mask", 0.95)))
        kw.update(overrides)
        return cls(**kw)


# --------------------------------------------------------------------------- result
@dataclass
class Candidate:
    """A single ranked progress-counter candidate."""

    address: int                 # GB address (e.g. 0xD347)
    width: int                   # 1 = byte, 2 = 16-bit little-endian pair
    score: float                 # combined score in [0, 1]
    direction: str               # "increasing" | "decreasing"
    stats: dict = field(default_factory=dict)  # per-feature values (see mine())

    @property
    def addr_hex(self) -> str:
        return f"0x{self.address:04X}"

    def __repr__(self) -> str:  # compact, useful in logs
        return (
            f"Candidate({self.addr_hex} w{self.width} "
            f"score={self.score:.3f} {self.direction} "
            f"mono={self.stats.get('monotonicity', 0):.2f} "
            f"act={self.stats.get('activity', 0):.3f} "
            f"ent={self.stats.get('delta_entropy', 0):.2f} "
            f"cons={self.stats.get('consistency', 0):.2f})"
        )


# --------------------------------------------------------------------------- helpers
def _as_matrix(rollout) -> np.ndarray:
    """Coerce a rollout into an int32 ``(T, N)`` matrix of WRAM snapshots."""
    arr = np.asarray(rollout)
    if arr.ndim == 1:
        # a single snapshot -> degenerate 1-step rollout
        arr = arr[None, :]
    elif arr.ndim != 2:
        # sequence of 1-D snapshots that didn't stack cleanly
        arr = np.stack([np.asarray(s).ravel() for s in rollout], axis=0)
    return arr.astype(np.int32, copy=False)


def _pair_values(mat: np.ndarray) -> np.ndarray:
    """16-bit little-endian pair values: ``lo + 256*hi`` for adjacent columns.

    Column ``i`` of the result is the pair at byte columns ``(i, i+1)``, i.e. GB
    address ``0xC000 + i`` as the low byte.  Shape ``(T, N-1)``.
    """
    return mat[:, :-1] + (mat[:, 1:] << 8)


def _delta_entropy(nonzero_deltas: np.ndarray) -> float:
    """Normalized Shannon entropy of a set of nonzero increments, in ``[0, 1]``.

    0.0 => perfectly regular (a single repeated increment, e.g. always +1).
    1.0 => every increment distinct (looks like noise).  Normalization is by
    ``log2(n)`` — the entropy of an all-distinct sequence of the same length.
    """
    n = nonzero_deltas.size
    if n <= 1:
        return 0.0  # a lone (or absent) increment is maximally regular
    _, counts = np.unique(nonzero_deltas, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log2(p)).sum())
    hmax = np.log2(n)
    return max(0.0, h / hmax) if hmax > 0 else 0.0


def _activity_band(activity: float, cfg: MinerConfig) -> float:
    """Gaussian bump peaking at ``activity_target``; static/hyperactive -> ~0."""
    z = (activity - cfg.activity_target) / max(cfg.activity_width, 1e-9)
    return float(np.exp(-z * z))


# --------------------------------------------------------------------------- per-level core
def _level_features(matrices: Sequence[np.ndarray], width: int, cfg: MinerConfig):
    """Compute aggregated per-address features for one width (1 = byte, 2 = pair).

    Returns a dict keyed by column index -> aggregated feature dict, only for
    columns that are active in at least one rollout (cheap pre-filter so the
    per-address entropy loop stays small).
    """
    if width == 2:
        value_mats = [_pair_values(m) for m in matrices]
    else:
        value_mats = list(matrices)

    n_cols = value_mats[0].shape[1]
    n_roll = len(value_mats)

    # ---- vectorized first pass across ALL columns & rollouts -------------
    # Accumulate, per column, the counts we need to find "interesting" columns
    # before paying for the per-address entropy computation.
    tot_changes = np.zeros(n_cols, dtype=np.int64)
    tot_steps = np.zeros(n_cols, dtype=np.int64)
    tot_up = np.zeros(n_cols, dtype=np.int64)
    tot_down = np.zeros(n_cols, dtype=np.int64)
    # per-rollout activity / monotonicity / direction, for cross-episode stats
    per_roll_active = np.zeros((n_roll, n_cols), dtype=bool)
    per_roll_counterlike = np.zeros((n_roll, n_cols), dtype=bool)
    per_roll_dir = np.zeros((n_roll, n_cols), dtype=np.int8)  # +1 up, -1 down, 0 none

    deltas_per_roll = []  # keep for the entropy loop
    for r, vals in enumerate(value_mats):
        if vals.shape[0] < 2:
            deltas_per_roll.append(np.zeros((0, n_cols), dtype=np.int64))
            continue
        d = np.diff(vals.astype(np.int64), axis=0)  # (T-1, n_cols)
        deltas_per_roll.append(d)
        changed = d != 0
        n_ch = changed.sum(axis=0)
        n_st = d.shape[0]
        up = (d > 0).sum(axis=0)
        down = (d < 0).sum(axis=0)

        tot_changes += n_ch
        tot_steps += n_st
        tot_up += up
        tot_down += down

        act = n_ch / max(n_st, 1)
        active = (act >= cfg.min_activity) & (act <= cfg.max_activity) & (n_ch > 0)
        per_roll_active[r] = active
        mono = np.maximum(up, down) / np.maximum(n_ch, 1)
        per_roll_counterlike[r] = active & (mono >= cfg.mono_active_thresh)
        per_roll_dir[r] = np.where(up >= down, 1, -1) * (n_ch > 0)

    # candidate columns: active in >=1 rollout and enough total change events
    active_any = per_roll_active.any(axis=0)
    enough = tot_changes >= cfg.min_changes
    cols = np.nonzero(active_any & enough)[0]

    out: dict[int, dict] = {}
    for c in cols:
        # ---- delta entropy over pooled nonzero increments ----------------
        pooled = []
        for d in deltas_per_roll:
            if d.shape[0] == 0:
                continue
            col = d[:, c]
            pooled.append(col[col != 0])
        pooled_nz = np.concatenate(pooled) if pooled else np.zeros(0, dtype=np.int64)
        ent = _delta_entropy(pooled_nz)

        # ---- aggregate directional / activity stats ----------------------
        overall_activity = float(tot_changes[c] / max(tot_steps[c], 1))
        up_c, down_c = int(tot_up[c]), int(tot_down[c])
        n_ch_c = up_c + down_c
        monotonicity = max(up_c, down_c) / max(n_ch_c, 1)
        modal_dir = 1 if up_c >= down_c else -1

        # cross-episode consistency: fraction of rollouts that were counter-like
        consistency = float(per_roll_counterlike[:, c].mean())
        # direction agreement among rollouts that had any movement
        moved = per_roll_dir[:, c] != 0
        if moved.any():
            dir_agree = float((per_roll_dir[moved, c] == modal_dir).mean())
        else:
            dir_agree = 0.0

        # ---- range / scale sanity: regularity of increment magnitude -----
        if pooled_nz.size >= 2:
            absd = np.abs(pooled_nz).astype(np.float64)
            disp = absd.std() / (absd.mean() + 1e-9)  # coeff. of variation
            range_sanity = 1.0 / (1.0 + disp)
        else:
            range_sanity = 1.0  # a single clean increment is perfectly regular

        out[int(c)] = dict(
            width=width,
            monotonicity=float(monotonicity),
            activity=overall_activity,
            delta_entropy=float(ent),
            consistency=consistency,
            direction_agreement=dir_agree,
            range_sanity=float(range_sanity),
            modal_dir=int(modal_dir),
            n_changes=int(tot_changes[c]),
            n_rollouts_active=int(per_roll_active[:, c].sum()),
            value_min=int(min(v[:, c].min() for v in value_mats)),
            value_max=int(max(v[:, c].max() for v in value_mats)),
        )
    return out


def _score(feat: dict, cfg: MinerConfig) -> float:
    """Combine per-feature stats into a single score in ``[0, 1]``."""
    w = np.array(
        [cfg.w_monotonic, cfg.w_activity, cfg.w_entropy, cfg.w_consistency, cfg.w_range],
        dtype=np.float64,
    )
    wsum = w.sum() or 1.0
    terms = np.array(
        [
            feat["monotonicity"],
            _activity_band(feat["activity"], cfg),
            1.0 - feat["delta_entropy"],
            feat["consistency"] * feat["direction_agreement"],
            feat["range_sanity"],
        ],
        dtype=np.float64,
    )
    return float((w * terms).sum() / wsum)


# --------------------------------------------------------------------------- public API
def mine(
    rollouts: Iterable,
    cfg: MinerConfig | None = None,
    top_k: int | None = None,
) -> list[Candidate]:
    """Discover progress-counter-like WRAM addresses.

    Parameters
    ----------
    rollouts : sequence of rollouts
        Each rollout is a time series of WRAM snapshots (see module docstring).
    cfg : MinerConfig, optional
        Filters and scoring weights.  Defaults to :class:`MinerConfig`.
    top_k : int, optional
        If given, truncate the ranked result to this many candidates.

    Returns
    -------
    list[Candidate]
        Ranked best-first.  Each carries ``score``, ``direction``, and a
        ``stats`` dict with the per-feature values used to score it.
    """
    cfg = cfg or MinerConfig()
    matrices = [_as_matrix(r) for r in rollouts]
    matrices = [m for m in matrices if m.shape[0] >= 2]
    if not matrices:
        return []

    # De-duplicate byte-identical rollouts (audit A9). The champion replay that
    # feeds this miner is a SINGLE deterministic rollout, so an unchanged
    # champion+spawn produces the same trace every generation. Feeding N copies
    # makes the cross-episode ``consistency`` feature read 1.0 for a one-time
    # ramp (a rewarded tap): identical traces are not independent episodes. Keep
    # one representative of each distinct trace so consistency reflects genuinely
    # distinct rollouts.
    if cfg.dedup_rollouts and len(matrices) > 1:
        seen: set = set()
        uniq: list[np.ndarray] = []
        for m in matrices:
            key = (m.shape, m.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(m)
        matrices = uniq

    # sanity: all rollouts must share the WRAM width
    widths = {m.shape[1] for m in matrices}
    if len(widths) != 1:
        raise ValueError(f"rollouts have differing WRAM widths: {sorted(widths)}")

    levels = [1, 2] if cfg.consider_pairs else [1]
    candidates: list[Candidate] = []
    for width in levels:
        feats = _level_features(matrices, width, cfg)
        for col, feat in feats.items():
            # hard masks
            if feat["delta_entropy"] > cfg.entropy_mask:
                continue
            if not (cfg.min_activity <= feat["activity"] <= cfg.max_activity):
                continue
            if feat["consistency"] <= 0.0:
                continue
            score = _score(feat, cfg)
            direction = "increasing" if feat["modal_dir"] > 0 else "decreasing"
            candidates.append(
                Candidate(
                    address=WRAM_BASE + col,
                    width=width,
                    score=score,
                    direction=direction,
                    stats=feat,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)

    # de-emphasize redundancy: a 2-byte pair and its low byte often both fire.
    # We keep both (the LLM/manifest disambiguates) but stable-sort already puts
    # the higher-scoring representation first.
    if top_k is not None:
        candidates = candidates[:top_k]
    return candidates


__all__ = ["MinerConfig", "Candidate", "mine"]
