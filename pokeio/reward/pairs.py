r"""[MF] Clip extraction + pair sampling for the preference potential (spec §2b).

Turns the data the loop already produces every generation — between-gen
``_retina_collect`` segments, the champion replay trace, and Go-Explore archive
cells — into short K-frame CLIPS, then selects informative CLIP PAIRS for the
offline LLM to judge. Five game-agnostic strategies (spec §2b), a reserve-uniform
guard, and a manifest weak-label cross-check that only reweights label confidence
(never a model input, preserving the input wall).

Everything here is offline and imports nothing from :mod:`pokeio.train.loop`. All
stochastic draws use a DEDICATED numpy ``SeedSequence`` child derived from
``config.run.seed`` (:func:`pref_rng`), entirely separate from the
mutation/crossover reproduce stream, so ``fast_reproduce`` stays bit-identical
(spec §11). Dependency-light: numpy + stdlib.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# WRAM base kept in sync with the miner (column 0 -> 0xC000). Local copy avoids a
# hard import while documenting the single source of truth.
WRAM_BASE = 0xC000

# The pref stochastic stream: a fixed spawn key separates it from the reproduce
# stream (which draws directly off config.run.seed). Bump only to reshuffle all
# pref sampling without touching evolution's rng.
PREF_STREAM_KEY = 0xB7

# Engine-agnostic screen-digest descriptor geometry (spec §3a foveal fallback):
# a 16×14 quantized screen ⊕ a cell-depth scalar. 16/14 ≈ the GB 160/144 aspect.
DIGEST_GW = 16
DIGEST_GH = 14
DIGEST_SHADES = 4
DIGEST_DIM = DIGEST_GH * DIGEST_GW + 1  # 224 + 1 = 225


def pref_rng(seed: int, salt: int = 0) -> np.random.Generator:
    """Dedicated pref-sampling rng, a ``SeedSequence`` child of ``config.run.seed``.

    Separate spawn key from the reproduce stream (spec §11) so pair sampling draws
    never perturb the mutation/crossover rng — ``fast_reproduce`` stays
    bit-identical whether MF is on or off. ``salt`` lets a caller derive
    independent sub-streams (e.g. per generation) deterministically.
    """
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(PREF_STREAM_KEY, int(salt)))
    return np.random.default_rng(ss)


# --------------------------------------------------------------- screen digest
def _area_resample(img: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Area-average downscale of a 2-D image to ``(gh, gw)`` (numpy-only, cheap)."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim != 2:
        a = a.reshape(a.shape[-2], a.shape[-1]) if a.ndim >= 2 else a.reshape(1, -1)
    h, w = a.shape
    ys = np.linspace(0, h, gh + 1).astype(int)
    xs = np.linspace(0, w, gw + 1).astype(int)
    out = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            block = a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out


def screen_digest(frame: np.ndarray, cell_depth: float = 0.0, *,
                  gh: int = DIGEST_GH, gw: int = DIGEST_GW,
                  shades: int = DIGEST_SHADES) -> np.ndarray:
    """Engine-agnostic screen-digest descriptor: ``gh×gw`` quantized screen ⊕ depth.

    Works on any Game Boy screen (no game constants): area-average to a tiny grid,
    quantize to ``shades`` levels in ``[0,1]``, append the Go-Explore cell-depth
    scalar. This is the foveal-mode fallback input so MF is not retina-locked
    (spec §3a). Returns a ``(gh*gw + 1,)`` float32 vector.
    """
    a = np.asarray(frame, dtype=np.float32)
    if a.size and a.max() > 1.0:  # uint8-ish -> normalize
        a = a / 255.0
    small = _area_resample(a, gh, gw)
    q = np.round(small * (shades - 1)) / max(shades - 1, 1)
    return np.concatenate([q.ravel(), [float(cell_depth)]]).astype(np.float32)


# --------------------------------------------------------------------- clips
def clip_hash(frames: np.ndarray) -> str:
    """Stable content hash of a clip's raw frames (shape + bytes)."""
    a = np.ascontiguousarray(np.asarray(frames))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class Clip:
    """A short K-frame clip — game-agnostic pixels (or latents) + offline meta.

    ``frames`` is ``(K, H, W)`` raw screens (or ``(K, D)`` latents); ``meta`` holds
    the OFFLINE-ONLY selection signals: ``depth`` (archive depth quantile),
    ``traj`` (trajectory id), ``t0`` (window start), ``counters`` (manifest
    progress-dim values), ``milestone`` (anonymized DAG order index), ``surprise``
    / ``eventful`` (active-learning triggers), ``source``. None of this enters Φ.
    """

    frames: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return clip_hash(self.frames)


def extract_clips(traces, *, k: int = 8, stride: int | None = None,
                  max_clips: int | None = None, rng: np.random.Generator | None = None):
    """Slice length-``k`` clips from a list of traces.

    Each trace is ``{"frames": (L, ...), "meta": {...}}`` — a retina-collect
    segment, the champion trace, or an archive-cell rollout. Windows are
    ``frames[t:t+k]``; ``stride`` defaults to ``k`` (non-overlapping). Per-window
    meta inherits the trace meta and adds ``t0``. When ``max_clips`` is given the
    clips are uniformly subsampled with ``rng`` (dedicated pref stream).
    """
    stride = int(stride) if stride else int(k)
    clips: list[Clip] = []
    for tr in traces or []:
        frames = np.asarray(tr["frames"])
        base_meta = dict(tr.get("meta", {}))
        L = int(frames.shape[0])
        if L < k:
            continue
        for t in range(0, L - k + 1, stride):
            meta = dict(base_meta)
            meta["t0"] = int(t)
            clips.append(Clip(frames=frames[t:t + k], meta=meta))
    if max_clips is not None and len(clips) > max_clips:
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.choice(len(clips), size=int(max_clips), replace=False)
        clips = [clips[i] for i in sorted(idx.tolist())]
    return clips


# ------------------------------------------------------------- pair proposals
@dataclass
class PairProposal:
    """A candidate clip pair for the LLM, with a weak-label prior it may overturn.

    ``weak_dir`` is the strategy's cheap prior on which clip shows more progress
    ("A", "B", or "tie") — a starting hypothesis, not a training target; the LLM
    verdict is authoritative and the manifest cross-check reweights confidence.
    """

    a: Clip
    b: Clip
    strategy: str
    weak_dir: str = "tie"

    @property
    def a_hash(self) -> str:
        return self.a.key

    @property
    def b_hash(self) -> str:
        return self.b.key

    @property
    def pair_hash(self) -> str:
        """Order-SENSITIVE hash (the prompt lists A then B) for the cache key."""
        h = hashlib.sha256()
        h.update(self.a_hash.encode())
        h.update(b"|")
        h.update(self.b_hash.encode())
        return h.hexdigest()

    @property
    def unordered_key(self) -> tuple:
        """Order-INSENSITIVE key for dedup ({A,B} == {B,A})."""
        return tuple(sorted((self.a_hash, self.b_hash)))


# -- strategy 1: temporal (later >= earlier prior) --------------------------
def temporal_pairs(clips, *, dt: int = 1, rng=None, max_pairs: int | None = None):
    by_traj: dict = {}
    for c in clips:
        by_traj.setdefault(c.meta.get("traj", 0), []).append(c)
    out: list[PairProposal] = []
    for _tid, cs in by_traj.items():
        cs = sorted(cs, key=lambda c: c.meta.get("t0", 0))
        for i in range(len(cs) - dt):
            a, b = cs[i], cs[i + dt]  # earlier, later
            if a.key == b.key:
                continue
            out.append(PairProposal(a, b, "temporal", "B"))
    return _maybe_cap(out, max_pairs, rng)


# -- strategy 2: depth-straddle (deep cell >= shallow cell) -----------------
def depth_straddle_pairs(clips, *, rng=None, max_pairs: int | None = None):
    depths = np.array([float(c.meta.get("depth", np.nan)) for c in clips])
    valid = [i for i in range(len(clips)) if np.isfinite(depths[i])]
    if len(valid) < 2:
        return []
    med = float(np.nanmedian(depths[valid]))
    shallow = [i for i in valid if depths[i] <= med]
    deep = [i for i in valid if depths[i] > med]
    out: list[PairProposal] = []
    for i in shallow:
        for j in deep:
            if clips[i].key == clips[j].key:
                continue
            out.append(PairProposal(clips[i], clips[j], "depth", "B"))  # b deeper
    return _maybe_cap(out, max_pairs, rng)


# -- strategy 3: manifest-straddle (a manifest counter changed) -------------
def _counter_progress(a: Clip, b: Clip, manifest) -> float:
    """Signed manifest progress from ``a`` to ``b`` (+ => b advanced). Offline only."""
    ca = a.meta.get("counters", {}) or {}
    cb = b.meta.get("counters", {}) or {}
    if not ca and not cb:
        return 0.0
    total = 0.0
    dims = getattr(manifest, "progress_dimensions", None)
    if dims:
        for pd in dims:
            did = getattr(pd, "id", None) or (pd.get("id") if isinstance(pd, dict) else None)
            ddir = getattr(pd, "dir", None) or (pd.get("dir") if isinstance(pd, dict) else "up")
            if did is None:
                continue
            sign = 1.0 if str(ddir).lower().startswith("u") else -1.0
            total += sign * (float(cb.get(did, 0.0)) - float(ca.get(did, 0.0)))
    else:  # no manifest dirs: assume "up is progress" over shared keys
        for kk in set(ca) | set(cb):
            total += float(cb.get(kk, 0.0)) - float(ca.get(kk, 0.0))
    return total


def manifest_straddle_pairs(clips, *, manifest=None, rng=None, max_pairs: int | None = None):
    if manifest is None:
        return []
    out: list[PairProposal] = []
    n = len(clips)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = clips[i], clips[j]
            if a.key == b.key:
                continue
            prog = _counter_progress(a, b, manifest)
            if prog == 0.0:
                continue
            weak = "B" if prog > 0 else "A"
            out.append(PairProposal(a, b, "manifest", weak))
    return _maybe_cap(out, max_pairs, rng)


# -- strategy 4: milestone-order (anonymized DAG boundary) ------------------
def milestone_order_pairs(clips, *, rng=None, max_pairs: int | None = None):
    tagged = [c for c in clips if c.meta.get("milestone") is not None]
    out: list[PairProposal] = []
    for i in range(len(tagged)):
        for j in range(i + 1, len(tagged)):
            a, b = tagged[i], tagged[j]
            if a.key == b.key:
                continue
            ma = float(a.meta["milestone"])
            mb = float(b.meta["milestone"])
            if ma == mb:
                continue
            weak = "B" if mb > ma else "A"
            out.append(PairProposal(a, b, "milestone", weak))
    return _maybe_cap(out, max_pairs, rng)


# -- strategy 5: uncertainty / active learning (Φ near 0.5, eventful) -------
def _is_eventful(clip: Clip) -> bool:
    if clip.meta.get("eventful"):
        return True
    thr = clip.meta.get("surprise_thresh")
    if thr is not None:
        return float(clip.meta.get("surprise", 0.0)) >= float(thr)
    return False


def uncertainty_pairs(clips, scorer, encode_fn, *, rng=None, max_pairs: int = 32,
                      eventful_only: bool = True):
    """Pairs the CURRENT frozen Φ is most unsure about (|Φ_a − Φ_b| smallest),
    restricted to eventful clips (a mined counter moved / EMA-z surprise fired).

    ``scorer`` exposes ``score_batch(feats)`` (a :class:`PrefScorer`); ``encode_fn``
    maps a clip to per-frame model features. Spends the small LLM budget on
    near-decision-boundary transitions (spec §2b.5)."""
    idx = [i for i, c in enumerate(clips) if _is_eventful(c)] if eventful_only else list(range(len(clips)))
    if len(idx) < 2:
        idx = list(range(len(clips)))
    if len(idx) < 2:
        return []
    phi: dict = {}
    for i in idx:
        feats = np.atleast_2d(np.asarray(encode_fn(clips[i]), dtype=np.float32))
        phi[i] = float(np.mean(scorer.score_batch(feats)))
    ranked = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if clips[i].key == clips[j].key:
                continue
            ranked.append((abs(phi[i] - phi[j]), i, j))
    ranked.sort(key=lambda x: x[0])  # nearest the decision boundary first
    return [PairProposal(clips[i], clips[j], "uncertainty", "tie")
            for _d, i, j in ranked[:int(max_pairs)]]


def _maybe_cap(pairs, max_pairs, rng):
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    rng = rng if rng is not None else np.random.default_rng()
    idx = rng.choice(len(pairs), size=int(max_pairs), replace=False)
    return [pairs[i] for i in idx.tolist()]


# -- reserve-uniform guard ---------------------------------------------------
def uniform_pairs(clips, n: int, *, rng):
    """``n`` uniformly-random clip pairs (the reserve guard so active learning
    never starves Φ of idle/negative examples; spec §2b reserve guard)."""
    n = int(n)
    if n <= 0 or len(clips) < 2:
        return []
    out: list[PairProposal] = []
    seen: set = set()
    tries = 0
    limit = n * 20 + 50
    while len(out) < n and tries < limit:
        tries += 1
        i, j = rng.integers(0, len(clips), size=2).tolist()
        if i == j or clips[i].key == clips[j].key:
            continue
        key = tuple(sorted((clips[i].key, clips[j].key)))
        if key in seen:
            continue
        seen.add(key)
        out.append(PairProposal(clips[i], clips[j], "uniform", "tie"))
    return out


def dedup_pairs(proposals):
    """Drop duplicate unordered clip pairs, keeping first occurrence (strategy order)."""
    seen: set = set()
    out: list[PairProposal] = []
    for p in proposals:
        key = p.unordered_key
        if key in seen or p.a_hash == p.b_hash:
            continue
        seen.add(key)
        out.append(p)
    return out


# --------------------------------------------------------------- orchestration
def sample_pairs(clips, *, n: int, seed: int = 0, gen: int = 0,
                 uniform_frac: float = 0.25, dt: int = 1,
                 scorer=None, encode_fn=None, manifest=None,
                 strategies: list[str] | None = None):
    """Select up to ``n`` pairs for one round across the five strategies + guard.

    ``uniform_frac`` of the budget is RESERVED for uniform-random pairs (the
    reserve guard); the rest is filled from the enabled strategies, shuffled and
    deduped. Uses :func:`pref_rng` seeded from ``config.run.seed`` (+ ``gen`` salt)
    — a dedicated stream that never touches the reproduce rng. Returns a deduped,
    ``n``-capped list of :class:`PairProposal`.
    """
    n = int(n)
    if n <= 0 or len(clips) < 2:
        return []
    rng = pref_rng(seed, salt=gen)
    strategies = strategies or ["temporal", "depth", "manifest", "milestone", "uncertainty"]

    proposals: list[PairProposal] = []
    if "temporal" in strategies:
        proposals += temporal_pairs(clips, dt=dt, rng=rng)
    if "depth" in strategies:
        proposals += depth_straddle_pairs(clips, rng=rng)
    if "manifest" in strategies:
        proposals += manifest_straddle_pairs(clips, manifest=manifest, rng=rng)
    if "milestone" in strategies:
        proposals += milestone_order_pairs(clips, rng=rng)
    if "uncertainty" in strategies and scorer is not None and encode_fn is not None:
        proposals += uncertainty_pairs(clips, scorer, encode_fn, rng=rng)

    proposals = dedup_pairs(proposals)
    # shuffle strategy proposals deterministically (dedicated stream)
    if proposals:
        order = rng.permutation(len(proposals))
        proposals = [proposals[i] for i in order.tolist()]

    n_uniform = int(round(float(uniform_frac) * n))
    n_uniform = min(max(n_uniform, 0), n)
    n_strategy = n - n_uniform

    chosen = list(proposals[:n_strategy])
    # reserve-uniform guard: draw fresh uniform pairs to fill the reserved slots
    reserved = uniform_pairs(clips, n - len(chosen), rng=rng)
    combined = dedup_pairs(chosen + reserved)

    # top up from any leftover strategy proposals if dedup shrank the set
    if len(combined) < n:
        have = {p.unordered_key for p in combined}
        for p in proposals[n_strategy:]:
            if len(combined) >= n:
                break
            if p.unordered_key not in have:
                combined.append(p)
                have.add(p.unordered_key)
    return combined[:n]


# ---------------------------------------------------- manifest weak-label graft
def crosscheck_confidence(winner: str, confidence: float, a: Clip, b: Clip,
                          manifest=None, *, boost: float = 1.25,
                          penalty: float = 0.6) -> float:
    """Manifest weak-label cross-check: reweight LLM confidence, never a model input.

    When the manifest independently confirms a counter moved across the pair in the
    direction the LLM chose, raise the training confidence; on disagreement, lower
    it (spec §2b). A cheap offline de-noiser that tightens the accuracy gate. This
    NEVER becomes a model feature — it only scales the label's confidence.
    """
    if manifest is None:
        return float(np.clip(confidence, 0.0, 1.0))
    prog = _counter_progress(a, b, manifest)  # + => b advanced
    if prog == 0.0:
        return float(np.clip(confidence, 0.0, 1.0))
    llm_dir = 1 if str(winner).upper() == "B" else (-1 if str(winner).upper() == "A" else 0)
    if llm_dir == 0:
        return float(np.clip(confidence, 0.0, 1.0))
    if np.sign(prog) == np.sign(llm_dir):
        return float(np.clip(confidence * boost, 0.0, 1.0))
    return float(np.clip(confidence * penalty, 0.0, 1.0))


# ----------------------------------------------- clip -> anonymized descriptor
def describe_clip(clip: Clip, *, manifest=None, milestone_digest=None) -> str:
    """A GAME-AGNOSTIC text description of a clip for the LLM prompt (label wall).

    Uses only generic scene stats (motion energy, cell-depth/novelty, ‖Δz‖
    surprise), an anonymized milestone-order id, and anonymized counter deltas.
    Contains NO game name, address, or mechanic — every game-specific fact stays
    offline (spec §2a). Reads ``milestone_digest`` (from
    :func:`pokeio.manifest.generate.milestone_ordering_digest`) only for anonymized
    node ids.
    """
    m = clip.meta
    parts: list[str] = []
    frames = np.asarray(clip.frames)
    if frames.ndim == 3 and frames.shape[0] >= 2:  # image frames: motion energy
        motion = float(np.mean(np.abs(np.diff(frames.astype(np.float32), axis=0))))
        parts.append(f"motion={motion:.3f}")
    if "depth" in m and np.isfinite(float(m.get("depth", np.nan))):
        parts.append(f"novelty_depth={float(m['depth']):.3f}")
    if "surprise" in m:
        parts.append(f"latent_surprise={float(m['surprise']):.3f}")
    if m.get("eventful"):
        parts.append("eventful=1")
    ms = m.get("milestone")
    if ms is not None:
        node = f"node_{int(ms)}"
        if milestone_digest:
            nodes = milestone_digest.get("nodes", [])
            if 0 <= int(ms) < len(nodes):
                node = str(nodes[int(ms)].get("id", node))
        parts.append(f"milestone={node}")
    counters = m.get("counters")
    if counters:
        # anonymize counter ids to c0,c1,... in sorted order (no game labels)
        anon = {f"c{i}": counters[k] for i, k in enumerate(sorted(counters))}
        parts.append("counters=" + ",".join(f"{k}:{v}" for k, v in anon.items()))
    if not parts:
        parts.append("features=none")
    return "[" + "; ".join(parts) + "]"


def build_pair_item(proposal: PairProposal, *, manifest=None, milestone_digest=None) -> dict:
    """Assemble the enqueue item for :class:`pokeio.llm.annotate.AnnotationWorker`.

    A plain dict (no loop/annotate import): anonymized descriptions + stable hashes
    + the strategy/weak-label metadata the label record carries.
    """
    return {
        "desc_a": describe_clip(proposal.a, manifest=manifest, milestone_digest=milestone_digest),
        "desc_b": describe_clip(proposal.b, manifest=manifest, milestone_digest=milestone_digest),
        "a_hash": proposal.a_hash,
        "b_hash": proposal.b_hash,
        "pair_hash": proposal.pair_hash,
        "strategy": proposal.strategy,
        "weak_dir": proposal.weak_dir,
    }


__all__ = [
    "WRAM_BASE",
    "PREF_STREAM_KEY",
    "DIGEST_DIM",
    "DIGEST_GH",
    "DIGEST_GW",
    "pref_rng",
    "screen_digest",
    "clip_hash",
    "Clip",
    "extract_clips",
    "PairProposal",
    "temporal_pairs",
    "depth_straddle_pairs",
    "manifest_straddle_pairs",
    "milestone_order_pairs",
    "uncertainty_pairs",
    "uniform_pairs",
    "dedup_pairs",
    "sample_pairs",
    "crosscheck_confidence",
    "describe_clip",
    "build_pair_item",
]
