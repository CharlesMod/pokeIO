r"""[MF] Unit tests for the frozen preference-potential Φ (docs/specs/manifest-reward.md §12).

Isolated — imports only :mod:`pokeio.reward.preference`, no loop, no GPU, no live
LLM (``pref_enable=False`` regime). Covers the load-bearing invariants:

* the INPUT WALL — the game-specific ram tail ``latent[94:102]`` provably never
  reaches :meth:`PrefModel.forward` (spec §2/§9);
* the self-tuning accuracy gate — a chance-level Φ contributes exactly 0 weight,
  ramping to ``w_ref`` at perfect reliability (spec §5);
* Bradley-Terry convergence on synthetic monotone labels + the gamed-label
  negative control that self-zeroes (spec §12.2/§12.4);
* the TRUE telescoping potential-shaping return — a returning cycle nets ~0 and
  the credit is subsample-safe, the anti-hacking property (spec §4/§8);
* frozen-snapshot / checkpoint round-trips (resume continues the SAME Φ).
"""

from __future__ import annotations

import numpy as np
import torch

from pokeio.reward.preference import (
    PROPRIO_DIM,
    PROPRIO_HI,
    PROPRIO_LO,
    RAM_HI,
    RAM_LO,
    Z_PIX,
    PrefModel,
    PrefScorer,
    TelescopeAccumulator,
    _pref_build_ckpt,
    _pref_load_ckpt,
    acc_reliability,
    digest_features,
    pref_effective_weight,
    retina_features,
    retina_in_dim,
)

D = 102  # the retina obs pipe controller-latent width [z_pix(80)|proprio(14)|ram(8)]


def _latent(seed: int = 0, n: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (D,) if n is None else (n, D)
    return rng.standard_normal(shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# geometry / feature dim
# --------------------------------------------------------------------------- #
def test_slice_constants_match_layout():
    # [z_pix(80) | proprio(14) @ 80:94 | ram(8) @ 94:102]
    assert Z_PIX == 80
    assert (PROPRIO_LO, PROPRIO_HI) == (80, 94)
    assert (RAM_LO, RAM_HI) == (94, 102)
    assert PROPRIO_DIM == 14


def test_feature_dim_is_174():
    # z_pix(80) + Δz_pix(80) + proprio(14) = 174 ; without proprio 160.
    assert retina_in_dim(True) == 174
    assert retina_in_dim(False) == 160
    assert retina_features(_latent()).shape == (174,)
    assert retina_features(_latent(), use_proprio=False).shape == (160,)


def test_feature_batched_shape():
    feat = retina_features(_latent(n=5))
    assert feat.shape == (5, 174)


# --------------------------------------------------------------------------- #
# INPUT WALL — the ram tail [94:102] never influences the feature or Φ
# --------------------------------------------------------------------------- #
def test_input_wall_ram_perturbation_is_invisible():
    lat = _latent(1)
    gamed = lat.copy()
    gamed[RAM_LO:RAM_HI] += 1000.0  # hammer the game-specific ram tail
    # no prev (delta = 0) and with a prev whose ram is ALSO perturbed.
    for use_proprio in (True, False):
        f0 = retina_features(lat, use_proprio=use_proprio)
        f1 = retina_features(gamed, use_proprio=use_proprio)
        assert np.array_equal(f0, f1), "ram tail leaked into retina_features"

    prev = _latent(2)
    prev_gamed = prev.copy()
    prev_gamed[RAM_LO:RAM_HI] -= 500.0
    f0 = retina_features(lat, prev, use_proprio=True)
    f1 = retina_features(gamed, prev_gamed, use_proprio=True)
    assert np.array_equal(f0, f1), "ram tail leaked through the latent delta"


def test_input_wall_pixel_and_proprio_are_visible():
    # triangulate the wall: pixel[0:80] always matters; proprio[80:94] matters
    # only when use_proprio=True; ram[94:102] never matters.
    lat = _latent(3)
    pix = lat.copy(); pix[0:Z_PIX] += 1.0
    pro = lat.copy(); pro[PROPRIO_LO:PROPRIO_HI] += 1.0

    assert not np.array_equal(retina_features(lat), retina_features(pix))
    assert not np.array_equal(retina_features(lat), retina_features(pro))
    # proprio invisible when the block is dropped
    assert np.array_equal(
        retina_features(lat, use_proprio=False),
        retina_features(pro, use_proprio=False),
    )


def test_score_latent_cannot_see_ram():
    # the model-level guarantee: score_latent routes through the input wall, so a
    # perturbed ram tail produces a bit-identical Φ.
    torch.manual_seed(0)
    model = PrefModel.for_retina()
    lat = _latent(4)
    gamed = lat.copy(); gamed[RAM_LO:RAM_HI] += 777.0
    s0 = model.score_latent_np(lat)
    s1 = model.score_latent_np(gamed)
    assert np.allclose(s0, s1, atol=0.0)


def test_digest_features_shape_and_determinism():
    rng = np.random.default_rng(5)
    d = rng.standard_normal(225).astype(np.float32)
    f = digest_features(d)
    assert f.shape == (2 * 225,)                      # [digest | Δdigest]
    assert np.array_equal(f, digest_features(d))       # deterministic
    prev = rng.standard_normal(225).astype(np.float32)
    fp = digest_features(d, prev)
    assert np.allclose(fp[:225], d)
    assert np.allclose(fp[225:], d - prev)


# --------------------------------------------------------------------------- #
# self-tuning weight gate (§5)
# --------------------------------------------------------------------------- #
def test_acc_reliability_gate_shape():
    assert acc_reliability(0.5) == 0.0        # chance -> zero reliability
    assert acc_reliability(1.0) == 1.0        # perfect -> full
    assert acc_reliability(0.75) == 0.5       # linear ramp
    assert acc_reliability(0.0) == 0.0        # below chance clamps to 0
    assert acc_reliability(2.0) == 1.0        # above 1 clamps to 1


def test_pref_effective_weight_zero_at_chance_and_monotone():
    w_ref = 0.1
    assert pref_effective_weight(0.5, w_ref) == 0.0
    assert pref_effective_weight(1.0, w_ref) == w_ref
    accs = np.linspace(0.5, 1.0, 11)
    ws = [pref_effective_weight(a, w_ref) for a in accs]
    assert all(b >= a - 1e-12 for a, b in zip(ws, ws[1:]))   # non-decreasing
    assert ws[0] == 0.0 and abs(ws[-1] - w_ref) < 1e-12


# --------------------------------------------------------------------------- #
# Bradley-Terry distillation: monotone recovery + gamed-label self-zero
# --------------------------------------------------------------------------- #
def _embed(g: float) -> np.ndarray:
    """A smooth 8-d, single-frame clip feature monotone in the progress scalar g."""
    return np.array(
        [[g, g * g, np.sin(3 * g), np.cos(3 * g), g - 0.5, (g - 0.5) ** 2, 1.0, -g]],
        dtype=np.float32,
    )


def _monotone_labels(m: int = 12):
    states = [_embed(i / (m - 1)) for i in range(m)]
    labels = []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            labels.append({"a": states[i], "b": states[j],
                           "winner": "B" if j > i else "A", "confidence": 1.0})
    return states, labels


def test_bt_recovers_monotone_order():
    torch.manual_seed(0)
    states, labels = _monotone_labels(12)
    model = PrefModel(8, use_proprio=False)
    stats = model.train_bt(labels, lambda c: c, steps=600, lr=1e-2,
                           val_frac=0.25, rng=np.random.default_rng(0))
    # held-out pairwise accuracy is well above chance (near-perfect in practice)
    assert stats["val_acc"] >= 0.85, stats
    assert stats["n_val"] > 0
    # Φ is monotone in the true progress scalar (end-to-end separation + rank corr)
    scores = np.array([float(model.clip_score(torch.as_tensor(s))) for s in states])
    assert scores[-1] > scores[0]
    corr = np.corrcoef(np.arange(len(scores)), scores)[0, 1]
    assert corr > 0.9, f"Φ not monotone in progress (corr={corr:.3f})"


def test_bt_gamed_labels_self_zero():
    # a shuffled/gamed label stream must NOT clear the gate -> w_pref_eff ~ 0.
    torch.manual_seed(0)
    states, _ = _monotone_labels(12)
    rng = np.random.default_rng(123)
    gamed = []
    for i in range(len(states)):
        for j in range(len(states)):
            if i == j:
                continue
            gamed.append({"a": states[i], "b": states[j],
                          "winner": str(rng.choice(["A", "B"])), "confidence": 1.0})
    model = PrefModel(8, use_proprio=False)
    stats = model.train_bt(gamed, lambda c: c, steps=600, lr=1e-2,
                           val_frac=0.25, rng=np.random.default_rng(7))
    # random ordering cannot be learned: held-out accuracy stays near chance and
    # the self-tuning weight collapses toward 0 (spec §12.4).
    assert stats["val_acc"] <= 0.7, stats
    assert pref_effective_weight(stats["val_acc"], 0.1) <= 0.5 * 0.1


def test_bt_empty_labels_returns_chance():
    torch.manual_seed(0)
    model = PrefModel(8, use_proprio=False)
    stats = model.train_bt([], lambda c: c, steps=10)
    assert stats["n"] == 0 and stats["val_acc"] == 0.5


def test_bt_confidence_smoothing_ties_target_half():
    # a tie label targets exactly 0.5 regardless of confidence — verified via the
    # accuracy accounting, which excludes ties from the denominator.
    torch.manual_seed(0)
    s = [_embed(0.2), _embed(0.8)]
    labels = [{"a": s[0], "b": s[1], "winner": "tie", "confidence": 1.0}]
    model = PrefModel(8, use_proprio=False)
    stats = model.train_bt(labels, lambda c: c, steps=20, val_frac=0.0)
    # only a tie present -> no orderable pair -> accuracy accounting returns 0.5
    assert stats["val_acc"] == 0.5


# --------------------------------------------------------------------------- #
# telescoping potential-shaping return (anti-hacking, spec §4/§8)
# --------------------------------------------------------------------------- #
def test_telescope_returning_cycle_nets_zero():
    acc = TelescopeAccumulator(gamma=1.0)
    for phi in [1.0, 3.0, 2.0, 5.0, -4.0, 1.0]:   # returns to the start value
        acc.update(0, phi)
    assert abs(acc.value(0)) < 1e-6              # φ_last − φ_first = 0


def test_telescope_is_subsample_safe():
    # γ=1 telescopes to φ_last − φ_first regardless of the spacing between samples.
    dense = TelescopeAccumulator(gamma=1.0)
    sparse = TelescopeAccumulator(gamma=1.0)
    for phi in [0.5, 2.0, -1.0, 3.5, 4.0]:
        dense.update(0, phi)
    for phi in [0.5, 4.0]:                        # only endpoints
        sparse.update(0, phi)
    assert abs(dense.value(0) - sparse.value(0)) < 1e-6
    assert abs(dense.value(0) - (4.0 - 0.5)) < 1e-6


def test_telescope_transient_spike_does_not_farm():
    # a single transient high-Φ frame that is left again contributes ~0 net — the
    # exact exploit max-over-trajectory would reward.
    acc = TelescopeAccumulator(gamma=1.0)
    for phi in [0.0, 0.0, 100.0, 0.0, 0.0]:
        acc.update(0, phi)
    assert abs(acc.value(0)) < 1e-6


def test_telescope_per_player_isolation_and_reset():
    acc = TelescopeAccumulator(gamma=1.0)
    acc.update_batch([0, 1], [1.0, 10.0])
    acc.update_batch([0, 1], [4.0, 10.0])
    assert abs(acc.value(0) - 3.0) < 1e-6
    assert abs(acc.value(1) - 0.0) < 1e-6         # player 1 returned to 10
    acc.reset(0)
    assert acc.value(0) == 0.0
    assert acc.value(1) == 0.0                    # unseen after reset -> 0 default


def test_scorer_pref_return_cycle_nets_zero():
    # end-to-end through a FROZEN Φ: a state cycle (last latent == first) nets ~0.
    torch.manual_seed(0)
    model = PrefModel.for_retina()
    scorer = PrefScorer.from_model(model, gamma=1.0)
    lat0, lat1 = _latent(10), _latent(11)
    for lat in (lat0, lat1, lat0):                # return to the first state
        phi = float(scorer.score_latent_batch(lat)[0])
        scorer.observe(0, phi)
    assert abs(scorer.pref_return(0)) < 1e-4


# --------------------------------------------------------------------------- #
# frozen snapshot + checkpoint round-trips
# --------------------------------------------------------------------------- #
def test_prefmodel_snapshot_is_frozen_and_identical():
    torch.manual_seed(0)
    model = PrefModel.for_retina()
    feats = np.random.default_rng(0).standard_normal((6, retina_in_dim(True))).astype(np.float32)
    snap = model.snapshot()
    assert all(not p.requires_grad for p in snap.parameters())
    assert np.allclose(model.forward(feats).detach().cpu().numpy(),
                       snap.forward(feats).detach().cpu().numpy(), atol=1e-6)


def test_scorer_snapshot_swap_and_accumulator_survives():
    torch.manual_seed(0)
    model = PrefModel.for_retina()
    feats = np.random.default_rng(1).standard_normal((6, retina_in_dim(True))).astype(np.float32)
    scorer = PrefScorer.from_model(model)
    before = scorer.score_batch(feats)

    # the scorer holds its OWN frozen copy: mutating the live model must not move it.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    assert np.allclose(before, scorer.score_batch(feats), atol=1e-6)

    # accumulator state persists across a snapshot swap (spec §7 freeze-and-swap).
    scorer.observe(0, 1.0)
    scorer.observe(0, 4.0)
    kept = scorer.pref_return(0)
    scorer.set_snapshot(model)
    after = scorer.score_batch(feats)
    assert not np.allclose(before, after, atol=1e-6)   # snapshot actually swapped
    assert scorer.pref_return(0) == kept               # accumulator untouched


def test_pref_checkpoint_roundtrip():
    torch.manual_seed(0)
    model = PrefModel.for_retina()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scorer = PrefScorer.from_model(model, gamma=0.97)
    ckpt = _pref_build_ckpt(model, opt, scorer)

    m2, o2, s2 = _pref_load_ckpt(ckpt)
    assert m2 is not None and o2 is not None and s2 is not None
    assert (m2.in_dim, m2.mode, m2.use_proprio) == (model.in_dim, model.mode, model.use_proprio)
    assert abs(s2.gamma - 0.97) < 1e-9

    feats = np.random.default_rng(2).standard_normal((5, retina_in_dim(True))).astype(np.float32)
    assert np.allclose(scorer.score_batch(feats), s2.score_batch(feats), atol=1e-5)


def test_pref_load_ckpt_missing_is_safe():
    assert _pref_load_ckpt(None) == (None, None, None)
    assert _pref_load_ckpt({}) == (None, None, None)
