"""Tests for the tensorized NEAT evolution core (``pokeio.evo``).

Covers: innovation bookkeeping, batched forward correctness (dense & sparse),
mutation/crossover/speciation invariants, and an end-to-end XOR evolve.

GPU tests use ``cuda:1`` when available (card 1 = training; card 0 = GLM), and
fall back to CPU otherwise so the suite runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokeio.evo import ops
from pokeio.evo.forward import (
    SIGMOID,
    apply_activation,
    population_forward,
    population_forward_sparse,
)
from pokeio.evo.genome import (
    INPUT,
    OUTPUT,
    InnovationTracker,
    Population,
    make_genome,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# innovation tracking
# --------------------------------------------------------------------------
def test_innovation_is_deterministic():
    t = InnovationTracker(2, 1)
    a = t.conn_innov(0, 3)
    b = t.conn_innov(0, 3)
    c = t.conn_innov(1, 3)
    assert a == b
    assert a != c


def test_split_node_shares_id():
    t = InnovationTracker(2, 1)
    innov = t.conn_innov(0, 3)
    n1 = t.split_node(innov)
    n2 = t.split_node(innov)
    assert n1 == n2
    assert n1 >= t.n_io  # hidden ids live past the I/O block


def test_make_genome_structure():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    g = make_genome(2, 1, t, rng, connect="full")
    assert len(g.nodes) == 4  # 2 in + bias + 1 out
    assert len(g.conns) == 3  # (in0,in1,bias) -> out
    assert g.nodes[0].type == INPUT
    assert g.nodes[3].type == OUTPUT


# --------------------------------------------------------------------------
# forward pass
# --------------------------------------------------------------------------
def test_forward_computes_or_gate():
    """A hand-set net (large positive weights, negative bias, sigmoid) is an OR."""
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    g = make_genome(2, 1, t, rng, connect="full", weight_scale=0.0)
    for c in g.conns.values():
        c.weight = 10.0 if c.in_id in (0, 1) else -5.0  # bias id = 2
    g.nodes[3].act = SIGMOID
    pop = Population.from_genomes([g])
    cp = pop.compile(_device())
    X = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]])
    out = population_forward(cp, X, steps=5)[0, :, 0].cpu()
    assert out[0] < 0.5  # 0 OR 0 = 0
    assert out[1] > 0.5 and out[2] > 0.5 and out[3] > 0.5


def test_forward_runs_on_gpu_when_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(0)
    t = InnovationTracker(3, 2)
    gs = [make_genome(3, 2, t, rng, connect="full") for _ in range(16)]
    pop = Population.from_genomes(gs)
    cp = pop.compile("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    X = torch.randn(16, 4, 3)
    out = population_forward(cp, X, steps=6)
    assert out.device.type == "cuda"
    assert out.shape == (16, 4, 2)


def test_dense_and_sparse_agree():
    rng = np.random.default_rng(4)
    t = InnovationTracker(3, 2)
    gs = [make_genome(3, 2, t, rng, connect="full", weight_scale=1.0) for _ in range(12)]
    rates = ops.MutationRates(add_node=1.0, add_conn=1.0)
    for g in gs:
        for _ in range(4):
            ops.mutate_genome(g, t, rng, rates)
    pop = Population.from_genomes(gs, max_nodes=64, max_conns=256)
    cp = pop.compile(_device())
    X = torch.randn(12, 3, 3)
    d = population_forward(cp, X, steps=10)
    s = population_forward_sparse(cp, X, steps=10)
    assert torch.allclose(d, s, atol=1e-4)


def test_activations_supported():
    z = torch.tensor([[[0.5, -0.5, 0.5, 0.5, 0.5]]])  # (1,1,5)
    act = torch.tensor([[0, 2, 1, 3, 4]])  # identity relu tanh sigmoid sin
    out = apply_activation(z, act)[0, 0]
    assert torch.isclose(out[0], torch.tensor(0.5))  # identity
    assert torch.isclose(out[1], torch.tensor(0.0))  # relu(-0.5)
    assert torch.isclose(out[2], torch.tanh(torch.tensor(0.5)))
    assert torch.isclose(out[3], torch.sigmoid(torch.tensor(0.5)))
    assert torch.isclose(out[4], torch.sin(torch.tensor(0.5)))


def test_population_masks_and_padding():
    """Genomes of differing sizes pad to a common budget with valid masks."""
    rng = np.random.default_rng(1)
    t = InnovationTracker(2, 1)
    g_small = make_genome(2, 1, t, rng, connect="full")
    g_big = make_genome(2, 1, t, rng, connect="full")
    for _ in range(3):
        ops.mutate_add_node(g_big, t, rng)
    pop = Population.from_genomes([g_small, g_big], max_nodes=32, max_conns=64)
    assert pop.node_mask[0].sum() == len(g_small.nodes)
    assert pop.node_mask[1].sum() == len(g_big.nodes)
    assert pop.M == 32 and pop.C == 64


# --------------------------------------------------------------------------
# mutation
# --------------------------------------------------------------------------
def test_add_node_grows_topology():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    g = make_genome(2, 1, t, rng, connect="full")
    n_nodes, n_conns = len(g.nodes), len(g.conns)
    assert ops.mutate_add_node(g, t, rng)
    assert len(g.nodes) == n_nodes + 1  # one new hidden node
    assert len(g.conns) == n_conns + 2  # split into two edges
    assert len(g.hidden_ids()) == 1


def test_add_connection_respects_feedforward():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    g = make_genome(2, 1, t, rng, connect="full")
    ops.mutate_add_node(g, t, rng)  # now there is a hidden node to wire
    rates = ops.MutationRates(feedforward=True)
    for _ in range(50):
        ops.mutate_add_connection(g, t, rng, rates)
    # feedforward: no cycles -> no node reaches itself over enabled edges.
    for nid in g.nodes:
        assert not ops._reachable(g, nid, nid)


def test_weight_perturbation_changes_weights():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    g = make_genome(2, 1, t, rng, connect="full")
    before = [c.weight for c in g.conns.values()]
    ops.perturb_weights(g, rng, ops.MutationRates(weight_reset_prob=0.0))
    after = [c.weight for c in g.conns.values()]
    assert before != after


# --------------------------------------------------------------------------
# crossover
# --------------------------------------------------------------------------
def test_crossover_inherits_from_fitter():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    p1 = make_genome(2, 1, t, rng, connect="full")
    p2 = p1.copy()
    ops.mutate_add_node(p1, t, rng)  # p1 has an extra node/edges (disjoint)
    p1.fitness = 10.0
    p2.fitness = 1.0
    child = ops.crossover(p1, p2, rng)
    # child must contain p1's disjoint genes and be structurally valid.
    assert set(p1.conns).issubset(set(child.conns))
    for c in child.conns.values():
        assert c.in_id in child.nodes and c.out_id in child.nodes


def test_crossover_child_is_evaluable():
    rng = np.random.default_rng(2)
    t = InnovationTracker(2, 1)
    p1 = make_genome(2, 1, t, rng, connect="full")
    p2 = make_genome(2, 1, t, rng, connect="full")
    for _ in range(4):
        ops.mutate_genome(p1, t, rng, ops.MutationRates(add_node=1.0))
        ops.mutate_genome(p2, t, rng, ops.MutationRates(add_node=1.0))
    child = ops.crossover(p1, p2, rng)
    pop = Population.from_genomes([child], max_nodes=64, max_conns=256)
    out = population_forward(pop.compile(_device()), torch.randn(1, 2, 2), steps=8)
    assert out.shape == (1, 2, 1)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# speciation / selection
# --------------------------------------------------------------------------
def test_compatibility_distance_zero_for_identical():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    g = make_genome(2, 1, t, rng, connect="full")
    assert ops.compatibility_distance(g, g.copy()) == 0.0


def test_compatibility_distance_grows_with_divergence():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    g1 = make_genome(2, 1, t, rng, connect="full")
    g2 = g1.copy()
    d_before = ops.compatibility_distance(g1, g2)
    for _ in range(4):
        ops.mutate_add_node(g2, t, rng)
    d_after = ops.compatibility_distance(g1, g2)
    assert d_after > d_before


def test_speciation_splits_divergent_genomes():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    a = make_genome(2, 1, t, rng, connect="full")
    b = a.copy()
    for _ in range(6):
        ops.mutate_add_node(b, t, rng)  # push b far from a
    spec = ops.Speciation(threshold=1.0, c1=1.0, c2=1.0, c3=0.5)
    species = spec.assign([a, b], rng)
    assert len(species) == 2


def test_reproduce_preserves_population_size():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    genomes = [make_genome(2, 1, t, rng, connect="full") for _ in range(20)]
    for g in genomes:
        g.fitness = float(rng.random())
    spec = ops.Speciation(threshold=3.0)
    species = spec.assign(genomes, rng)
    kids = ops.reproduce(
        genomes, species, t, rng, ops.MutationRates(), pop_size=20
    )
    assert len(kids) == 20
    for g in kids:  # all evaluable
        assert all(c.in_id in g.nodes and c.out_id in g.nodes for c in g.conns.values())


def test_tournament_prefers_fitter():
    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    pool = [make_genome(2, 1, t, rng, connect="full") for _ in range(8)]
    for i, g in enumerate(pool):
        g.fitness = float(i)
    winners = [ops.tournament_select(pool, 8, rng).fitness for _ in range(10)]
    assert all(w == 7.0 for w in winners)  # k==len -> always the best


# --------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------
def test_xor_evolves_to_solution():
    from pokeio.evo.demo_xor import run_xor

    dev = "cuda:1" if torch.cuda.device_count() > 1 else (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    res = run_xor(seed=7, pop_size=150, max_gens=300, device_str=dev, verbose=False)
    assert res.solved  # all four XOR patterns on the correct side of 0.5
    assert res.best_error < 2.0


def test_xor_pipeline_improves_quickly():
    """A short evolve must at least improve fitness (fast, non-slow test)."""
    from pokeio.evo.demo_xor import evaluate_xor

    rng = np.random.default_rng(0)
    t = InnovationTracker(2, 1)
    genomes = [make_genome(2, 1, t, rng, connect="full") for _ in range(60)]
    spec = ops.Speciation(threshold=3.0, c3=0.5)
    dev = _device()

    def best_fitness(gs):
        pop = Population.from_genomes(gs, max_nodes=32, max_conns=256)
        fit, _, _ = evaluate_xor(pop, dev)
        for i, g in enumerate(gs):
            g.fitness = float(fit[i])
        return float(fit.max())

    f0 = best_fitness(genomes)
    for _ in range(15):
        species = spec.assign(genomes, rng)
        genomes = ops.reproduce(
            genomes, species, t, rng, ops.MutationRates(), pop_size=60
        )
        f1 = best_fitness(genomes)
    assert f1 >= f0  # evolution does not regress the champion
