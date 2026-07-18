"""Neural-native timing: evolvable per-neuron time constant (CTRNN leaky
integration) — spec ``docs/specs/execution-plan.md`` [TC].

Each node carries an evolvable update rate ``alpha`` in ``[tau_min, 1]`` and the
node update becomes ``a_t = (1-alpha)*a_{t-1} + alpha*f(net)``. ``alpha == 1`` is
the legacy full-overwrite reflex; small ``alpha`` is a slow integrator (a native
dwell-timer / "how long since X").

Covered here:
* **alpha == 1 is bit-identical** to the pre-time-constants forward (the
  load-bearing backward-compat guarantee) — sparse, dense, and the cross-agent
  ``state`` recurrence, on legacy AND recurrent topologies.
* **leaky integration is correct** — a slow-alpha node ramps toward its target
  over ~1/alpha steps (matching ``1-(1-alpha)^t``); a fast (alpha==1) node jumps.
* **alpha is evolvable** — ``make_genome`` seeds a spread, ``mutate_tau`` moves it
  (log-scale, bounded), and it survives the padded-tensor pack + ``copy()``.
* **XOR still solves** with time constants active (alpha doesn't break search).
* **timing benefit** — a temporal-integration task ("output high iff a cue fired
  within the last K steps") is solvable by an evolvable-alpha CTRNN but not by a
  fixed alpha==1 (memoryless) net under the same budget.

CPU-safe (matches the rest of the evo suite; uses cuda:1 when present).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from pokeio.config import Config
from pokeio.evo import ops
from pokeio.evo.demo_xor import evaluate_xor
from pokeio.evo.forward import (
    IDENTITY,
    SIGMOID,
    population_forward,
    population_forward_sparse,
)
from pokeio.evo.genome import (
    HIDDEN,
    OUTPUT,
    InnovationTracker,
    Population,
    make_genome,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


# ==========================================================================
# 1. alpha == 1 is bit-identical to the pre-change forward (backward-compat)
# ==========================================================================
def _mutated_pop(recurrent: bool, seed: int = 5, n: int = 10):
    """A legacy population (time_constants off -> every alpha == 1.0), grown a
    few structural mutations so hidden nodes and (optionally) recurrent edges
    exercise both the within-step and cross-step update paths."""
    rng = np.random.default_rng(seed)
    t = InnovationTracker(3, 2)
    gs = [make_genome(3, 2, t, rng, connect="full", weight_scale=1.0) for _ in range(n)]
    rates = ops.MutationRates(add_node=1.0, add_conn=1.0, feedforward=not recurrent)
    for g in gs:
        for _ in range(4):
            ops.mutate_genome(g, t, rng, rates)
    return Population.from_genomes(gs, max_nodes=64, max_conns=256)


def test_legacy_population_compiles_to_none_tau():
    """A population with no time constants (all alpha == 1) hands the forward
    ``node_tau=None`` — the safety valve that keeps the legacy code path (and the
    trainer's dataclasses.replace row-slicer) untouched."""
    cp = _mutated_pop(recurrent=False).compile(_device())
    assert cp.node_tau is None


@pytest.mark.parametrize("recurrent", [False, True])
def test_alpha_one_is_bit_identical_sparse(recurrent):
    cp = _mutated_pop(recurrent=recurrent).compile(_device())
    assert cp.node_tau is None  # legacy path
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((4, 3)).astype(np.float32)).to(_device())

    legacy = population_forward_sparse(cp, X, steps=10)
    # force an explicit all-ones alpha tensor: the leaky blend at alpha==1 must
    # reproduce the legacy overwrite BIT-FOR-BIT (not merely allclose).
    cp_ones = dataclasses.replace(cp, node_tau=torch.ones_like(cp.node_bias))
    blended = population_forward_sparse(cp_ones, X, steps=10)
    assert torch.equal(legacy, blended)


@pytest.mark.parametrize("recurrent", [False, True])
def test_alpha_one_is_bit_identical_state_recurrence(recurrent):
    """The cross-agent-step ``state`` recurrence is also bit-identical at alpha==1."""
    cp = _mutated_pop(recurrent=recurrent).compile(_device())
    cp_ones = dataclasses.replace(cp, node_tau=torch.ones_like(cp.node_bias))
    rng = np.random.default_rng(1)
    state_a = state_b = None
    for _ in range(5):  # thread several agent-steps
        X = torch.from_numpy(rng.standard_normal((3, 3)).astype(np.float32)).to(_device())
        oa, state_a = population_forward_sparse(
            cp, X, steps=6, state=state_a, return_state=True
        )
        ob, state_b = population_forward_sparse(
            cp_ones, X, steps=6, state=state_b, return_state=True
        )
        assert torch.equal(oa, ob)
        assert torch.equal(state_a, state_b)


def test_alpha_one_is_bit_identical_dense():
    cp = _mutated_pop(recurrent=False).compile(_device())
    cp_ones = dataclasses.replace(cp, node_tau=torch.ones_like(cp.node_bias))
    rng = np.random.default_rng(2)
    X = torch.from_numpy(rng.standard_normal((4, 3)).astype(np.float32)).to(_device())
    legacy = population_forward(cp, X, steps=10)
    blended = population_forward(cp_ones, X, steps=10)
    assert torch.equal(legacy, blended)


def test_time_constants_off_forces_alpha_one():
    """config semantics: time_constants=False => alpha == 1 everywhere (legacy)."""
    rng = np.random.default_rng(0)
    t = InnovationTracker(3, 4)
    g = make_genome(3, 4, t, rng, connect="full", time_constants=False)
    assert all(n.alpha == 1.0 for n in g.nodes.values())
    cp = Population.from_genomes([g]).compile(_device())
    assert cp.node_tau is None


# ==========================================================================
# 2. leaky integration is correct (the time constant is what it claims)
# ==========================================================================
def _single_node_cp(alpha: float):
    """input -> output (identity, w=1); bias edge zeroed; output alpha = alpha."""
    t = InnovationTracker(1, 1)
    rng = np.random.default_rng(0)
    g = make_genome(1, 1, t, rng, connect="full", weight_scale=0.0, output_act=IDENTITY)
    out_id = list(g.output_ids())[0]
    for c in g.conns.values():
        c.weight = 1.0 if c.in_id == 0 else 0.0  # input->out = 1, bias->out = 0
    g.nodes[out_id].alpha = alpha
    return Population.from_genomes([g]).compile(_device())


def _ramp(alpha: float, T: int = 20) -> list[float]:
    """Feed a constant step input=1 across T agent-steps (one hop each), threading
    state, and record the output — the impulse/step response of the leaky node."""
    cp = _single_node_cp(alpha)
    X = torch.ones(1, 1, 1, device=_device())
    state = None
    out = []
    for _ in range(T):
        o, state = population_forward_sparse(cp, X, steps=1, state=state, return_state=True)
        out.append(float(o[0, 0, 0]))
    return out


def test_slow_node_ramps_at_its_time_constant():
    alpha = 0.1
    obs = _ramp(alpha, T=20)
    ref = [1.0 - (1.0 - alpha) ** (k + 1) for k in range(20)]  # closed-form step response
    assert np.allclose(obs, ref, atol=1e-5), f"ramp deviates: {obs[:6]} vs {ref[:6]}"
    # at t = 1/alpha the leaky node has risen ~1 - 1/e ~ 0.63 of the way.
    assert 0.60 < obs[int(round(1 / alpha)) - 1] < 0.68
    assert all(b > a for a, b in zip(obs, obs[1:]))  # monotone rise toward the target


def test_fast_node_jumps_immediately():
    obs = _ramp(1.0, T=8)
    assert obs[0] == pytest.approx(1.0, abs=1e-6)  # reaches target in ONE step
    assert all(v == pytest.approx(1.0, abs=1e-6) for v in obs)  # and holds it


def test_slower_alpha_integrates_more_slowly():
    """Monotonicity in alpha: a smaller alpha rises strictly slower at t=1/e-ish."""
    fast = _ramp(0.5, T=10)
    slow = _ramp(0.1, T=10)
    assert slow[3] < fast[3]  # after a few steps the slow node lags the fast one


# ==========================================================================
# 3. alpha is evolvable (seeded spread, mutable, survives pack + copy)
# ==========================================================================
def test_make_genome_seeds_a_timescale_spread():
    rng = np.random.default_rng(0)
    t = InnovationTracker(4, 12)  # 12 outputs -> a decent sample of seeds
    alphas = []
    for _ in range(20):
        g = make_genome(4, 12, t, rng, connect="full", time_constants=True,
                        tau_min=0.05, tau_init_fast_frac=0.5,
                        tau_init_slow_lo=0.1, tau_init_slow_hi=0.5)
        alphas += [n.alpha for n in g.nodes.values() if n.type == OUTPUT]
    a = np.array(alphas)
    assert (a >= 0.05).all() and (a <= 1.0).all()  # bounded [tau_min, 1]
    assert (a >= 0.999).mean() > 0.2  # a real fraction of fast (alpha ~ 1) reflex nodes
    assert (a < 0.9).mean() > 0.2  # and a real fraction of slow integrators
    assert a.std() > 0.05  # genuine diversity, not a single timescale


def test_mutate_tau_moves_alpha_within_bounds():
    rng = np.random.default_rng(1)
    t = InnovationTracker(2, 2)
    g = make_genome(2, 2, t, rng, connect="full", time_constants=True)
    for _ in range(3):
        ops.mutate_add_node(g, t, rng)  # grow some hidden nodes too
    rates = ops.MutationRates(mutate_tau=1.0, tau_min=0.05, tau_perturb_sigma=0.3)
    before = {nid: n.alpha for nid, n in g.nodes.items()}
    for _ in range(200):
        ops.mutate_tau(g, rng, rates)
        for n in g.nodes.values():
            if n.type in (HIDDEN, OUTPUT):
                assert 0.05 - 1e-9 <= n.alpha <= 1.0 + 1e-9  # never escapes bounds
    moved = any(g.nodes[nid].alpha != before[nid] for nid in before
                if g.nodes[nid].type in (HIDDEN, OUTPUT))
    assert moved  # mutation actually changed some time constants


def test_mutate_tau_is_log_scale_symmetric():
    """A log-scale step is multiplicatively symmetric: from alpha=0.5, up and down
    moves are geometric, so many steps keep alpha spread across decades, not
    collapsed to a bound."""
    rng = np.random.default_rng(3)
    t = InnovationTracker(1, 1)
    g = make_genome(1, 1, t, rng, connect="full", time_constants=False)
    out_id = list(g.output_ids())[0]
    g.nodes[out_id].alpha = 0.5
    rates = ops.MutationRates(mutate_tau=1.0, tau_min=0.01, tau_perturb_sigma=0.2)
    logs = []
    for _ in range(500):
        ops.mutate_tau(g, rng, rates)
        logs.append(math.log10(g.nodes[out_id].alpha))
    assert min(logs) < -0.5 and max(logs) > -0.2  # explores a wide multiplicative range


def test_alpha_survives_pack_roundtrip():
    rng = np.random.default_rng(2)
    t = InnovationTracker(3, 5)
    gs = [make_genome(3, 5, t, rng, connect="full", time_constants=True) for _ in range(6)]
    for g in gs:
        ops.mutate_add_node(g, t, rng)
    pop = Population.from_genomes(gs, max_nodes=64, max_conns=256)
    for i, g in enumerate(gs):
        ids = sorted(g.nodes)
        expected = torch.tensor([g.nodes[nid].alpha for nid in ids], dtype=torch.float32)
        assert torch.allclose(pop.node_tau[i, : len(ids)], expected)
        # padding slots are 1.0 (legacy full-overwrite)
        assert torch.all(pop.node_tau[i, len(ids):] == 1.0)


def test_copy_preserves_alpha():
    rng = np.random.default_rng(4)
    t = InnovationTracker(2, 3)
    g = make_genome(2, 3, t, rng, connect="full", time_constants=True)
    h = g.copy()
    assert {nid: n.alpha for nid, n in g.nodes.items()} == {
        nid: n.alpha for nid, n in h.nodes.items()
    }
    # and it's a deep copy — mutating the clone doesn't touch the original
    for n in h.nodes.values():
        n.alpha = 0.123
    assert any(g.nodes[nid].alpha != 0.123 for nid in g.nodes)


def test_config_exposes_time_constant_knobs():
    c = Config()
    assert c.evo.time_constants is True
    assert 0.0 < c.evo.tau_min < 1.0
    assert 0.0 <= c.evo.tau_init_fast_frac <= 1.0
    assert c.evo.mutate_tau >= 0.0


def test_crossover_carries_alpha():
    rng = np.random.default_rng(6)
    t = InnovationTracker(2, 2)
    p1 = make_genome(2, 2, t, rng, connect="full", time_constants=True)
    p2 = p1.copy()
    # give the two parents distinct output alphas; the matching-node child should
    # inherit the AVERAGE (deterministic, like a matching-gene weight blend).
    out_ids = list(p1.output_ids())
    for k, oid in enumerate(out_ids):
        p1.nodes[oid].alpha = 0.2
        p2.nodes[oid].alpha = 0.8
    p1.fitness, p2.fitness = 2.0, 1.0
    child = ops.crossover(p1, p2, rng)
    for oid in out_ids:
        assert child.nodes[oid].alpha == pytest.approx(0.5)


# ==========================================================================
# 4. XOR still solves with time constants active (search isn't broken)
# ==========================================================================
def test_xor_still_solves_with_time_constants():
    rng = np.random.default_rng(7)
    t = InnovationTracker(2, 1)
    pop_size = 150
    genomes = [
        make_genome(2, 1, t, rng, connect="full", weight_scale=1.0, time_constants=True)
        for _ in range(pop_size)
    ]
    rates = ops.MutationRates(
        add_node=0.03, add_conn=0.08, weight=0.9, weight_perturb_sigma=0.6,
        weight_reset_prob=0.1, weight_reset_scale=1.5, toggle=0.01, feedforward=True,
        mutate_tau=0.3, tau_min=0.05, tau_perturb_sigma=0.25,
    )
    spec = ops.Speciation(threshold=3.0, c1=1.0, c2=1.0, c3=0.5)
    dev = _device()
    solved = False
    for _ in range(150):
        pop = Population.from_genomes(genomes, max_nodes=64, max_conns=512)
        fit, _err, correct = evaluate_xor(pop, dev, steps=12)
        for i, g in enumerate(genomes):
            g.fitness = float(fit[i])
        if bool(correct.any()):
            solved = True
            break
        species = spec.assign(genomes, rng)
        genomes = ops.reproduce(
            genomes, species, t, rng, rates, pop_size=pop_size,
            tournament_size=3, elitism=1, crossover_rate=0.75, c3=0.5,
        )
    assert solved, "XOR failed to solve with time constants active"


# ==========================================================================
# 5. timing benefit — a temporal-integration task a CTRNN solves and a
#    memoryless (alpha == 1) net cannot
# ==========================================================================
# Task: a cue channel fires at a few steps; the target is high for K steps AFTER
# each cue ("cue fired within the last K steps"). With enough propagation steps a
# feedforward alpha==1 net fully settles each agent-step -> its output is a pure
# function of the CURRENT cue (memoryless) and it cannot cover the K-1 trailing
# steps where cue==0. An evolvable-alpha CTRNN integrates the cue in-place and can.
_T, _K = 20, 5
_CUES = (2, 11)
_CUE = np.zeros(_T, np.float32)
for _p in _CUES:
    _CUE[_p] = 1.0
_TARGET = np.array(
    [1.0 if any(0 <= t - p < _K for p in _CUES) else 0.0 for t in range(_T)],
    np.float32,
)


def _eval_temporal(genomes, steps: int) -> torch.Tensor:
    pop = Population.from_genomes(genomes, max_nodes=64, max_conns=256)
    cp = pop.compile(_device())
    state = None
    outs = []
    for t in range(_T):
        X = torch.full((cp.n, 1, 1), float(_CUE[t]), device=_device())
        o, state = population_forward_sparse(cp, X, steps=steps, state=state, return_state=True)
        outs.append(o[:, 0, 0])
    O = torch.stack(outs, dim=1)  # (N, T)
    tgt = torch.tensor(_TARGET, device=_device()).unsqueeze(0)
    return (_T - (O - tgt).abs().sum(dim=1)).cpu()  # higher is better


def _evolve_temporal(time_constants: bool, mutate_tau: float, seed: int,
                     pop_size: int = 48, gens: int = 30, steps: int = 8) -> float:
    rng = np.random.default_rng(seed)
    t = InnovationTracker(1, 1)
    genomes = [
        make_genome(1, 1, t, rng, connect="full", weight_scale=1.5,
                    output_act=SIGMOID, time_constants=time_constants)
        for _ in range(pop_size)
    ]
    rates = ops.MutationRates(
        add_node=0.1, add_conn=0.25, weight=0.9, weight_perturb_sigma=0.5,
        weight_reset_prob=0.1, weight_reset_scale=1.5, toggle=0.02, feedforward=True,
        mutate_tau=mutate_tau, tau_min=0.05, tau_perturb_sigma=0.3,
    )
    spec = ops.Speciation(threshold=3.0, c3=0.5)
    best = -1e9
    for _ in range(gens):
        fit = _eval_temporal(genomes, steps)
        for i, g in enumerate(genomes):
            g.fitness = float(fit[i])
        best = max(best, float(fit.max()))
        species = spec.assign(genomes, rng)
        genomes = ops.reproduce(
            genomes, species, t, rng, rates, pop_size=pop_size,
            tournament_size=3, elitism=2, crossover_rate=0.75, c3=0.5,
        )
    return best


def test_ctrnn_beats_memoryless_on_temporal_integration():
    # identical everything (feedforward, state-threaded, same budget) except alpha:
    # fixed at 1 (memoryless once settled) vs evolvable (CTRNN).
    memoryless = _evolve_temporal(time_constants=False, mutate_tau=0.0, seed=0)
    ctrnn = _evolve_temporal(time_constants=True, mutate_tau=0.4, seed=0)
    # the settled alpha==1 net is hard-capped (it cannot represent post-cue memory);
    # the CTRNN clears that ceiling by a wide margin.
    assert ctrnn > memoryless + 2.0, f"ctrnn={ctrnn:.2f} memoryless={memoryless:.2f}"
