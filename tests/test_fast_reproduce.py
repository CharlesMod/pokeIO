"""``loop.fast_reproduce`` must be an rng-stream-identical drop-in for
``ops.reproduce`` (audit A10 / findings #6, #26).

The equivalence used to be asserted only in a code comment
(``fast_reproduce`` docstring: *"same rng stream + results"*). This is a
load-bearing determinism guarantee — the live trainer swaps the fast path in
for throughput, and a silent divergence would make runs unreproducible while
looking fine. These tests pin it:

  * given identical inputs and two rngs seeded identically, both functions must
    emit **structurally identical** offspring AND leave the rng in the **same
    bit-generator state** (proof they consumed the identical number of draws in
    the identical order);
  * this holds for a single species and for balanced multi-species cohorts.

``fast_reproduce`` additionally carries deliberate diversity guards (a ~40%
per-species allocation cap, audit EVO#2/#6) that ``ops.reproduce`` lacks; when
that cap actually fires the two intentionally diverge. We do NOT assert
byte-identity there (that would false-RED a future fix that ports the guard to
``ops.reproduce``); instead we pin the weaker contract that survives either
way: ``fast_reproduce`` still returns exactly ``pop_size`` genomes and is
internally deterministic under a fixed seed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from pokeio.evo import ops
from pokeio.evo.genome import InnovationTracker, make_genome
from pokeio.train.loop import fast_reproduce


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _genome_sig(g):
    """A hashable structural fingerprint: nodes + connections + bookkeeping.

    Weights rounded to 12 places so we compare the actual float payload without
    being hostage to the last ULP (both paths are pure-python float math, so in
    practice they match exactly — the rounding is belt-and-suspenders)."""
    nodes = tuple(
        sorted(
            (n.id, n.type, n.act, round(float(n.bias), 12), round(float(n.alpha), 12))
            for n in g.nodes.values()
        )
    )
    conns = tuple(
        sorted(
            (c.in_id, c.out_id, round(float(c.weight), 12), bool(c.enabled), c.innov)
            for c in g.conns.values()
        )
    )
    return (g.n_in, g.n_out, g.age, nodes, conns)


def _sigs(genomes):
    return [_genome_sig(g) for g in genomes]


def _build(seed, npop, *, n_in=3, n_out=2, diversify=0, threshold=100.0, time_constants=False):
    """A small, evaluable population + its species assignment + tracker."""
    rng = np.random.default_rng(seed)
    tracker = InnovationTracker(n_in, n_out)
    genomes = [
        make_genome(n_in, n_out, tracker, rng, connect="full", time_constants=time_constants)
        for _ in range(npop)
    ]
    for i, g in enumerate(genomes):
        for _ in range(i % (diversify + 1) if diversify else 0):
            ops.mutate_add_node(g, tracker, rng)
        g.fitness = float(rng.random())
    spec = ops.Speciation(threshold=threshold)
    species = spec.assign(genomes, rng)
    return genomes, species, tracker


def _run_both(genomes, species, tracker, *, pop_size, seed=1234, rates=None):
    """Run both reproduce implementations from an identical starting rng state.

    Returns ``(kids_ops, rng_ops, kids_fast, rng_fast)``. The trackers are
    deep-copied so each path grows innovations independently; the source
    genomes/species are only READ (offspring are fresh copies), so sharing them
    is safe."""
    rates = rates or ops.MutationRates()
    rng_ops = np.random.default_rng(seed)
    rng_fast = np.random.default_rng(seed)
    t_ops = copy.deepcopy(tracker)
    t_fast = copy.deepcopy(tracker)
    sp = {k: list(v) for k, v in species.items()}
    kids_ops = ops.reproduce(genomes, sp, t_ops, rng_ops, rates, pop_size=pop_size)
    kids_fast = fast_reproduce(genomes, sp, t_fast, rng_fast, rates, pop_size=pop_size)
    return kids_ops, rng_ops, kids_fast, rng_fast


# --------------------------------------------------------------------------
# rng-stream identity (the load-bearing invariant)
# --------------------------------------------------------------------------
def test_single_species_bit_identical():
    genomes, species, tracker = _build(1, 16, threshold=1e9)  # one species
    assert len(species) == 1
    ko, ro, kf, rf = _run_both(genomes, species, tracker, pop_size=16)

    assert len(ko) == len(kf) == 16
    assert _sigs(ko) == _sigs(kf), "offspring diverged despite identical inputs"
    # Same final rng state => identical number of draws in identical order.
    assert ro.bit_generator.state == rf.bit_generator.state


def test_balanced_multi_species_bit_identical():
    """Several roughly-equal species, none large enough to trip the 40% cap.

    Species are assigned by hand so the cohort is guaranteed balanced (3 groups
    of 10 at pop 30 -> alloc ~10 each, cap = ceil(0.4*30) = 12): the diversity
    guards stay inert and the two paths must agree bit-for-bit."""
    rng = np.random.default_rng(7)
    tracker = InnovationTracker(3, 2)
    genomes = [make_genome(3, 2, tracker, rng, connect="full") for _ in range(30)]
    for i, g in enumerate(genomes):
        for _ in range(i // 10):  # groups 0/1/2 differ slightly in topology
            ops.mutate_add_node(g, tracker, rng)
        g.fitness = 1.0  # equal fitness -> equal allocation
    species = {0: genomes[0:10], 1: genomes[10:20], 2: genomes[20:30]}

    ko, ro, kf, rf = _run_both(genomes, species, tracker, pop_size=30)
    assert len(ko) == len(kf) == 30
    assert _sigs(ko) == _sigs(kf)
    assert ro.bit_generator.state == rf.bit_generator.state


def test_identity_holds_with_nondefault_rates():
    """The equivalence is not an artifact of the default mutation rates."""
    genomes, species, tracker = _build(3, 20, threshold=1e9)
    rates = ops.MutationRates(
        add_node=0.5, add_conn=0.5, weight=1.0, toggle=0.2, weight_perturb_sigma=0.3
    )
    ko, ro, kf, rf = _run_both(genomes, species, tracker, pop_size=20, rates=rates)
    assert _sigs(ko) == _sigs(kf)
    assert ro.bit_generator.state == rf.bit_generator.state


def test_identity_holds_with_tau_active():
    """The rng-identity contract must survive the time-constants gene (spec [TC]).

    The default rates leave ``mutate_tau=0`` (tau perturbation drawn zero times),
    so the other cases above never exercise the α path — the determinism proof
    with α *live* previously lived only in a throwaway scratch script. This pins
    it in CI: seed a genuine timescale spread (``time_constants=True`` seeds
    α<1 slow integrators at gen 0), reproduce with a heavy ``mutate_tau=0.5``,
    and require both the offspring fingerprint (now including α) AND the final
    rng bit-generator state to match between the fast path and ``ops.reproduce``.
    If ``_fast_mutate`` drew tau at a different stream position, or ``_fast_copy``
    /``_fast_crossover`` dropped/averaged α differently, this goes RED."""
    genomes, species, tracker = _build(11, 20, threshold=1e9, time_constants=True)
    # Sanity: the seed actually produced slow integrators, else the test is vacuous.
    assert any(
        float(n.alpha) < 1.0 for g in genomes for n in g.nodes.values()
    ), "time_constants=True seeded no α<1 node — test would be vacuous"
    rates = ops.MutationRates(mutate_tau=0.5, tau_perturb_sigma=0.2)
    ko, ro, kf, rf = _run_both(genomes, species, tracker, pop_size=20, rates=rates)
    assert len(ko) == len(kf) == 20
    # At least one child's α must have moved, else mutate_tau never fired.
    assert any(
        float(n.alpha) < 1.0 for g in kf for n in g.nodes.values()
    ), "no α<1 survived reproduction — mutate_tau path not exercised"
    assert _sigs(ko) == _sigs(kf), "offspring (incl. α) diverged with tau active"
    assert ro.bit_generator.state == rf.bit_generator.state


def test_produced_offspring_are_evaluable():
    """Whatever the path, every child references only nodes it defines."""
    genomes, species, tracker = _build(9, 18, diversify=3, threshold=3.0)
    _, _, kf, _ = _run_both(genomes, species, tracker, pop_size=18)
    for g in kf:
        for c in g.conns.values():
            assert c.in_id in g.nodes and c.out_id in g.nodes


# --------------------------------------------------------------------------
# cap-triggered divergence: only the weaker, fix-proof contract is pinned
# --------------------------------------------------------------------------
def test_cap_case_preserves_pop_size_and_is_deterministic():
    """When a single species would dominate, fast_reproduce's diversity cap may
    fire (and it then diverges from ops.reproduce by design). We do not assert
    identity here — only that fast_reproduce keeps its own guarantees: exact
    population size and per-seed determinism."""
    genomes, species, tracker = _build(2, 24, diversify=6, threshold=0.8)
    # Skewed fitness so one species' adjusted share is huge -> cap engages.
    for i, g in enumerate(genomes):
        g.fitness = 100.0 if i < 4 else 0.01

    def once(seed):
        rng = np.random.default_rng(seed)
        t = copy.deepcopy(tracker)
        sp = {k: list(v) for k, v in species.items()}
        return fast_reproduce(genomes, sp, t, rng, ops.MutationRates(), pop_size=24)

    a = once(555)
    b = once(555)
    assert len(a) == 24
    assert _sigs(a) == _sigs(b)  # deterministic under a fixed seed
