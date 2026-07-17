"""E2 per-cell baseline + coupled selection test
(active-vision spine §6.1/§6.5; unit-test plan §11 test 6).

The gen-100 pathology: a screen-blind constant at a lucky Go-Explore restore
out-scored a seeing policy at a bad restore, because selection ranked raw
novelty (spawn luck) instead of policy. The cure is two-fold and tested here:

* E2 ``_cell_baseline_advantage`` — per-cell leave-one-out advantage: two
  genomes restored to the SAME cell with identical policy get equal ``A_i``;
  a genome alone in a restore cell gets ``A_i = 0`` (its spawn value cancels);
  newgame genomes keep ``A_i = f_i``.
* ``cohort_rank_normalize`` with the E2 advantage + E3 gate makes a screen-blind
  genome at a lucky restore rank BELOW a seeing genome at a bad restore — the
  exact inversion of the diagnosis.
"""

from __future__ import annotations

import numpy as np

from pokeio.train.loop import _cell_baseline_advantage, cohort_rank_normalize


class _G:
    """Minimal genome stub: cohort_rank_normalize only reads .fitness/.progress."""

    def __init__(self, fitness: float, resp: float = 0.0, progress: float = 0.0):
        self.fitness = float(fitness)
        self._resp = float(resp)
        self.progress = float(progress)


# --------------------------------------------------------------------------
# E2 advantage semantics
# --------------------------------------------------------------------------
def test_same_cell_identical_policy_equal_advantage():
    # idx 0,1 restored to the same cell with identical policy -> identical f.
    raw = np.array([5.0, 5.0])
    A = _cell_baseline_advantage(raw, {0, 1}, {0: b"cellA", 1: b"cellA"})
    assert A[0] == A[1] == 0.0  # leave-one-out of equal peers cancels exactly


def test_same_cell_leave_one_out_difference_reward():
    # three peers in one cell -> A_i = f_i - mean(others)
    raw = np.array([3.0, 6.0, 9.0])
    A = _cell_baseline_advantage(raw, {0, 1, 2}, {i: b"c" for i in range(3)})
    assert np.allclose(A, [3.0 - 7.5, 6.0 - 6.0, 9.0 - 4.5])


def test_singleton_restore_cancels_spawn_value():
    # a lucky restore (high f) alone in its cell earns NO counterfactual credit.
    raw = np.array([2.0, 9.0])
    A = _cell_baseline_advantage(raw, {0, 1}, {0: b"poor", 1: b"rich"})
    assert A[0] == 0.0 and A[1] == 0.0  # spawn luck fully cancelled


def test_newgame_cohort_keeps_raw_fitness():
    # genomes NOT in the restored set keep A_i = f_i (shared fixed spawn).
    raw = np.array([2.0, 9.0, 4.0])
    A = _cell_baseline_advantage(raw, restored=set(), spawn_keys={})
    assert np.allclose(A, raw)


# --------------------------------------------------------------------------
# coupled selection: blind-at-lucky no longer beats seeing-at-bad
# --------------------------------------------------------------------------
def test_blind_lucky_no_longer_outranks_seeing_bad():
    # g0 = screen-blind at a lucky restore (raw f = 9), g1 = seeing at a bad
    # restore (raw f = 2). DIFFERENT singleton cells. Raw fitness ranks blind
    # ABOVE seeing; E2 (singleton cancel) + E3 gate must invert it.
    g_blind = _G(9.0)
    g_see = _G(2.0)
    genomes = [g_blind, g_see]
    assert g_blind.fitness > g_see.fitness  # raw ranking favours spawn luck

    gate = np.array([0.40, 1.00])  # blind crushed, seeing saturated (E3)
    cohort_rank_normalize(
        genomes, {0, 1}, progress_weight=0.0,
        spawn_keys={0: b"lucky", 1: b"bad"},
        gate=gate, w_resp=0.0,
        restore_baseline=True, blind_gate=True,
    )
    assert g_see.fitness > g_blind.fitness, (
        f"seeing({g_see.fitness:.3f}) failed to out-rank blind({g_blind.fitness:.3f})"
    )


def test_same_cell_identical_policy_equal_final_fitness():
    # identical policy + same cell + equal gate -> equal selection fitness.
    g0, g1 = _G(5.0, resp=1.0), _G(5.0, resp=1.0)
    cohort_rank_normalize(
        [g0, g1], {0, 1}, progress_weight=0.0,
        spawn_keys={0: b"c", 1: b"c"},
        gate=np.array([1.0, 1.0]), w_resp=0.1,
        restore_baseline=True, blind_gate=True,
    )
    assert abs(g0.fitness - g1.fitness) < 1e-9


def test_backward_compatible_default_is_plain_rank_normalize():
    # With no keyword extras, the function reduces to the previous within-cohort
    # rank-normalize (idempotent re-ranking), so legacy callers are unchanged.
    genomes = [_G(f) for f in (1.0, 4.0, 2.0, 9.0)]
    cohort_rank_normalize(genomes, restored=set(), progress_weight=0.0)
    ranks = [g.fitness for g in genomes]
    # strictly increasing with the input order of the sorted fitnesses
    assert ranks[3] > ranks[1] > ranks[2] > ranks[0]
    assert min(ranks) > 0.0 and max(ranks) < 1.0  # quantiles in (0,1)


def test_gate_is_dominant_over_responsiveness():
    # A blind genome with maximal responsiveness must still lose to a seeing
    # genome: gate multiplies the task quantile, w_resp only shapes.
    g_blind = _G(9.0, resp=10.0)
    g_see = _G(9.0, resp=0.0)
    cohort_rank_normalize(
        [g_blind, g_see], {0, 1}, progress_weight=0.0,
        spawn_keys={0: b"x", 1: b"y"},  # singletons -> A tie
        gate=np.array([0.40, 1.0]), w_resp=0.1,
        restore_baseline=True, blind_gate=True,
    )
    assert g_see.fitness > g_blind.fitness
