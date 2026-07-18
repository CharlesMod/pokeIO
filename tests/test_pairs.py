r"""[MF] Unit tests for clip extraction + pair sampling (docs/specs/manifest-reward.md §2b/§10).

Isolated — imports only :mod:`pokeio.reward.pairs`, no loop, no GPU, no LLM. All
five pair strategies are game-agnostic; the module draws stochastic choices from a
DEDICATED :func:`pref_rng` stream so ``fast_reproduce`` stays bit-identical (§11).

Covers: strategy proposals stay within budget and game-agnostic; dedup of
unordered pairs; stable order-sensitive pair-hash for the cache key; the
reserve-uniform guard so active learning never starves Φ of idle examples; the
engine-agnostic screen-digest descriptor shape/determinism; and the offline
manifest weak-label cross-check (confidence reweight only, never a model input).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pokeio.reward.pairs import (
    DIGEST_DIM,
    DIGEST_GH,
    DIGEST_GW,
    PREF_STREAM_KEY,
    Clip,
    PairProposal,
    clip_hash,
    crosscheck_confidence,
    dedup_pairs,
    depth_straddle_pairs,
    describe_clip,
    extract_clips,
    manifest_straddle_pairs,
    milestone_order_pairs,
    pref_rng,
    sample_pairs,
    screen_digest,
    temporal_pairs,
    uniform_pairs,
)


def _clip(seed: int, **meta) -> Clip:
    frames = np.random.default_rng(seed).integers(0, 255, size=(4, 8, 8), dtype=np.uint8)
    return Clip(frames=frames, meta=meta)


def _manifest(*dims):
    """A duck-typed manifest exposing progress_dimensions (id, dir) — game-agnostic."""
    return SimpleNamespace(progress_dimensions=[SimpleNamespace(**d) for d in dims])


# --------------------------------------------------------------------------- #
# clip extraction
# --------------------------------------------------------------------------- #
def test_extract_clips_windows_and_meta():
    frames = np.arange(20 * 4).reshape(20, 4).astype(np.float32)
    traces = [{"frames": frames, "meta": {"traj": 7}}]
    clips = extract_clips(traces, k=8, stride=8)
    assert len(clips) == 2                              # floor((20-8)/8)+1 = 2
    assert all(c.frames.shape[0] == 8 for c in clips)
    assert clips[0].meta["traj"] == 7
    assert clips[0].meta["t0"] == 0 and clips[1].meta["t0"] == 8


def test_extract_clips_skips_short_and_subsamples():
    traces = [{"frames": np.zeros((3, 4)), "meta": {}}]   # shorter than k
    assert extract_clips(traces, k=8) == []
    long = [{"frames": np.arange(40 * 2).reshape(40, 2), "meta": {}}]
    capped = extract_clips(long, k=4, stride=1, max_clips=5,
                           rng=np.random.default_rng(0))
    assert len(capped) == 5


# --------------------------------------------------------------------------- #
# clip / pair hashing — stable + order semantics
# --------------------------------------------------------------------------- #
def test_clip_hash_is_content_stable():
    a = _clip(1)
    same = Clip(frames=a.frames.copy(), meta={"unused": 1})  # meta not hashed
    assert clip_hash(a.frames) == clip_hash(same.frames) == a.key == same.key
    assert clip_hash(a.frames) != clip_hash(_clip(2).frames)


def test_pair_hash_stable_and_order_sensitive():
    a, b = _clip(1), _clip(2)
    p1 = PairProposal(a, b, "temporal")
    p2 = PairProposal(a, b, "uniform")                  # same clips, other strategy
    assert p1.pair_hash == p2.pair_hash                 # cache key = clip content only
    rev = PairProposal(b, a, "temporal")
    assert rev.pair_hash != p1.pair_hash                # order-SENSITIVE (A then B)
    assert rev.unordered_key == p1.unordered_key        # order-INSENSITIVE dedup key


def test_dedup_drops_unordered_duplicates_and_self_pairs():
    a, b = _clip(1), _clip(2)
    props = [
        PairProposal(a, b, "temporal"),
        PairProposal(b, a, "depth"),      # same unordered pair -> dropped
        PairProposal(a, a, "uniform"),    # self-pair -> dropped
        PairProposal(a, _clip(3), "milestone"),
    ]
    out = dedup_pairs(props)
    assert len(out) == 2
    keys = {p.unordered_key for p in out}
    assert len(keys) == 2


# --------------------------------------------------------------------------- #
# strategies are game-agnostic + correctly directed
# --------------------------------------------------------------------------- #
def test_temporal_pairs_order_within_trajectory():
    clips = [_clip(i, traj=0, t0=i) for i in range(4)]
    pairs = temporal_pairs(clips, dt=1)
    assert len(pairs) == 3
    assert all(p.strategy == "temporal" and p.weak_dir == "B" for p in pairs)  # later=B


def test_depth_straddle_pairs_deep_is_b():
    clips = [_clip(0, depth=0.1), _clip(1, depth=0.2),
             _clip(2, depth=0.8), _clip(3, depth=0.9)]
    pairs = depth_straddle_pairs(clips)
    assert pairs and all(p.strategy == "depth" and p.weak_dir == "B" for p in pairs)


def test_milestone_order_pairs_directed_by_dag_index():
    clips = [_clip(0, milestone=0), _clip(1, milestone=5)]
    pairs = milestone_order_pairs(clips)
    assert len(pairs) == 1
    assert pairs[0].weak_dir == "B"                     # higher milestone index = B


def test_manifest_straddle_uses_counters_not_labels():
    a = _clip(0, counters={"x": 1.0})
    b = _clip(1, counters={"x": 3.0})                   # x advanced -> b more progress
    manifest = _manifest({"id": "x", "dir": "up"})
    pairs = manifest_straddle_pairs([a, b], manifest=manifest)
    assert len(pairs) == 1 and pairs[0].weak_dir == "B"
    assert manifest_straddle_pairs([a, b], manifest=None) == []   # offline-only signal


# --------------------------------------------------------------------------- #
# reserve-uniform guard + orchestration budget
# --------------------------------------------------------------------------- #
def test_uniform_pairs_are_distinct_and_no_self():
    clips = [_clip(i) for i in range(6)]
    rng = np.random.default_rng(0)
    pairs = uniform_pairs(clips, 5, rng=rng)
    assert len(pairs) == 5
    keys = {p.unordered_key for p in pairs}
    assert len(keys) == 5                               # all distinct unordered pairs
    assert all(p.a_hash != p.b_hash for p in pairs)     # no self-pair


def test_sample_pairs_within_budget_and_deduped():
    clips = [_clip(i, traj=0, t0=i, depth=i / 8.0, milestone=i) for i in range(8)]
    manifest = _manifest({"id": "c", "dir": "up"})
    out = sample_pairs(clips, n=6, seed=42, gen=0, manifest=manifest)
    assert 0 < len(out) <= 6                            # capped to the budget
    keys = {p.unordered_key for p in out}
    assert len(keys) == len(out)                        # deduped
    assert all(isinstance(p, PairProposal) for p in out)
    allowed = {"temporal", "depth", "manifest", "milestone", "uncertainty", "uniform"}
    assert {p.strategy for p in out} <= allowed


def test_sample_pairs_reserve_guard_fires_without_strategy_signal():
    # each clip in its OWN trajectory with no depth/milestone/counter meta -> every
    # stratified strategy yields nothing; the reserve-uniform guard must still fire.
    clips = [_clip(i, traj=i) for i in range(6)]
    out = sample_pairs(clips, n=4, seed=1, uniform_frac=0.25)
    assert out, "reserve-uniform guard starved with no strategy signal"
    assert all(p.strategy == "uniform" for p in out)


def test_sample_pairs_game_agnostic_no_manifest():
    # runs with manifest=None (no game contract) and still proposes pairs.
    clips = [_clip(i, traj=0, t0=i) for i in range(5)]
    out = sample_pairs(clips, n=3, seed=3, manifest=None)
    assert 0 < len(out) <= 3


def test_sample_pairs_deterministic_for_fixed_seed():
    clips = [_clip(i, traj=0, t0=i, depth=i / 5.0) for i in range(5)]
    a = sample_pairs(clips, n=4, seed=9, gen=2)
    b = sample_pairs(clips, n=4, seed=9, gen=2)
    assert [p.pair_hash for p in a] == [p.pair_hash for p in b]


# --------------------------------------------------------------------------- #
# dedicated pref rng stream (fast_reproduce isolation, §11)
# --------------------------------------------------------------------------- #
def test_pref_rng_deterministic_and_salt_separated():
    assert PREF_STREAM_KEY == 0xB7
    a = pref_rng(1234, salt=0).integers(0, 1_000_000, size=8)
    b = pref_rng(1234, salt=0).integers(0, 1_000_000, size=8)
    c = pref_rng(1234, salt=1).integers(0, 1_000_000, size=8)
    assert np.array_equal(a, b)                         # reproducible per (seed,salt)
    assert not np.array_equal(a, c)                     # salt derives an independent sub-stream


# --------------------------------------------------------------------------- #
# engine-agnostic screen digest
# --------------------------------------------------------------------------- #
def test_screen_digest_shape_and_determinism():
    frame = np.random.default_rng(0).integers(0, 255, size=(144, 160), dtype=np.uint8)
    d = screen_digest(frame, cell_depth=0.3)
    assert d.shape == (DIGEST_DIM,) == (DIGEST_GH * DIGEST_GW + 1,)
    assert np.array_equal(d, screen_digest(frame, cell_depth=0.3))   # deterministic
    assert d[-1] == np.float32(0.3)                                   # depth scalar tail


def test_screen_digest_quantized_and_normalized():
    # uint8 input is normalized to [0,1] and quantized to DIGEST_SHADES levels.
    frame = np.full((144, 160), 255, dtype=np.uint8)
    d = screen_digest(frame)[:-1]
    assert d.max() <= 1.0 + 1e-6 and d.min() >= 0.0
    levels = np.unique(np.round(d * 3) / 3)             # DIGEST_SHADES-1 == 3
    assert set(np.round(levels, 6)).issubset({0.0, 1 / 3, 2 / 3, 1.0})


# --------------------------------------------------------------------------- #
# offline manifest cross-check (confidence reweight ONLY, never a model input)
# --------------------------------------------------------------------------- #
def test_crosscheck_confidence_boosts_and_penalizes():
    a = Clip(np.zeros((2, 4)), {"counters": {"x": 1.0}})
    b = Clip(np.zeros((2, 4)), {"counters": {"x": 3.0}})   # x advanced a->b
    manifest = _manifest({"id": "x", "dir": "up"})
    # LLM agrees with the manifest (B advanced) -> confidence boosted (clipped <=1).
    assert crosscheck_confidence("B", 0.6, a, b, manifest, boost=1.25) > 0.6
    # LLM disagrees -> confidence penalized.
    assert crosscheck_confidence("A", 0.8, a, b, manifest, penalty=0.5) < 0.8
    # no manifest -> passthrough (clamped only).
    assert crosscheck_confidence("B", 0.7, a, b, manifest=None) == 0.7


def test_describe_clip_anonymizes_counters():
    # counter ids are anonymized to c0,c1,... — no game label reaches the prompt.
    clip = Clip(np.zeros((2, 4, 4)), {"counters": {"badge_count": 2, "map_id": 5}})
    text = describe_clip(clip)
    assert "c0" in text and "c1" in text
    assert "badge_count" not in text and "map_id" not in text
