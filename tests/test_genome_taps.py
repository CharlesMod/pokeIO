"""Tests for the RAM-tap / proprioception connect+protect redesign
(spec ``docs/specs/active-vision-spine.md`` §7/§8, unit-test plan §11 test 3).

Covers:
* every genome (``full`` and ``sparse``) has all tap->output and proprio->output
  edges live from gen 0;
* a forwarded genome shows ``Δoutput > 0`` for 100% of genomes when the tap
  inputs are zeroed (the exact inverse of the gen-100 diagnosis's Δ = 0);
* ``mutate_toggle`` / ``mutate_add_node`` never sever a ``protected`` edge over
  10k mutations;
* the prefer-unconnected source sampler measurably raises the rate at which
  unwired inputs get connected versus uniform sampling.
"""

from __future__ import annotations

import numpy as np
import torch

from pokeio.evo import ops
from pokeio.evo.forward import TANH, population_forward_sparse
from pokeio.evo.genome import (
    INPUT,
    ConnGene,
    InnovationTracker,
    Population,
    make_genome,
)

# spec §2.1 block sizes
N_RAM = 8
N_PROPRIO = 14
N_OUT = 11  # 9 buttons + 2 saccade


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def _tap_ids(n_in: int) -> range:
    return range(n_in - N_RAM, n_in)


def _proprio_ids(n_in: int) -> range:
    return range(n_in - N_RAM - N_PROPRIO, n_in - N_RAM)


def _make(n_in: int, connect: str, seed: int, output_act: int = TANH):
    tracker = InnovationTracker(n_in, N_OUT)
    rng = np.random.default_rng(seed)
    g = make_genome(
        n_in,
        N_OUT,
        tracker,
        rng,
        connect=connect,
        output_act=output_act,
        n_ram=N_RAM,
        n_proprio=N_PROPRIO,
    )
    return g


# --------------------------------------------------------------------------
# 1. every tap/proprio edge is live at init (full AND sparse)
# --------------------------------------------------------------------------
def _assert_block_live(g, n_in: int):
    """Every id in the protected trailing block has a live, protected edge to
    every output."""
    edge = {(c.in_id, c.out_id): c for c in g.conns.values()}
    out_ids = list(g.output_ids())
    for s in list(_tap_ids(n_in)) + list(_proprio_ids(n_in)):
        for o in out_ids:
            c = edge.get((s, o))
            assert c is not None, f"missing edge {s}->{o}"
            assert c.enabled, f"edge {s}->{o} not enabled"
            assert c.protected, f"edge {s}->{o} not protected"


def test_full_taps_and_proprio_live():
    n_in = 60
    for seed in range(16):
        g = _make(n_in, "full", seed)
        _assert_block_live(g, n_in)


def test_sparse_taps_and_proprio_live():
    n_in = 60
    for seed in range(16):
        g = _make(n_in, "sparse", seed)
        _assert_block_live(g, n_in)


def test_safe_defaults_no_protection_when_no_taps():
    """With n_ram = n_proprio = 0 (every legacy caller) nothing is protected."""
    tracker = InnovationTracker(6, N_OUT)
    rng = np.random.default_rng(0)
    for connect in ("full", "sparse"):
        g = make_genome(6, N_OUT, tracker, rng, connect=connect)
        assert not any(c.protected for c in g.conns.values())


# --------------------------------------------------------------------------
# 2. zeroing the tap inputs changes the output for 100% of genomes
#    (the exact inverse of the diagnosis's Δ = 0 for all 224 genomes)
# --------------------------------------------------------------------------
def _delta_when_taps_zeroed(connect: str, n_in: int = 60, n_pop: int = 32):
    genomes = [_make(n_in, connect, seed) for seed in range(n_pop)]
    pop = Population.from_genomes(genomes, max_nodes=n_in + N_OUT + 8,
                                  max_conns=(n_in + 1) * N_OUT + 8)
    cp = pop.compile(_device())

    rng = np.random.default_rng(1234)
    B = 6  # probe screens
    X = torch.from_numpy(rng.standard_normal((B, n_in)).astype(np.float32))
    X_blind = X.clone()
    for s in _tap_ids(n_in):
        X_blind[:, s] = 0.0

    out_real = population_forward_sparse(cp, X, steps=8)
    out_blind = population_forward_sparse(cp, X_blind, steps=8)
    # per-genome L1 change of the 11-d output, summed over probe screens
    delta = (out_real - out_blind).abs().sum(dim=(1, 2))
    return delta


def test_full_delta_positive_for_all_genomes():
    delta = _delta_when_taps_zeroed("full")
    assert (delta > 0).all(), f"{int((delta == 0).sum())} genomes were tap-blind"


def test_sparse_delta_positive_for_all_genomes():
    delta = _delta_when_taps_zeroed("sparse")
    assert (delta > 0).all(), f"{int((delta == 0).sum())} genomes were tap-blind"


def test_proprio_delta_positive_for_all_genomes():
    """Same causal check, isolating the proprio block."""
    n_in = 60
    genomes = [_make(n_in, "full", seed) for seed in range(32)]
    pop = Population.from_genomes(genomes)
    cp = pop.compile(_device())
    rng = np.random.default_rng(7)
    X = torch.from_numpy(rng.standard_normal((6, n_in)).astype(np.float32))
    X_blind = X.clone()
    for s in _proprio_ids(n_in):
        X_blind[:, s] = 0.0
    d = (population_forward_sparse(cp, X, steps=8)
         - population_forward_sparse(cp, X_blind, steps=8)).abs().sum(dim=(1, 2))
    assert (d > 0).all()


# --------------------------------------------------------------------------
# 3. protected edges are never severed over 10k mutations
# --------------------------------------------------------------------------
def test_protected_edges_never_severed():
    n_in = 40
    tracker = InnovationTracker(n_in, N_OUT)
    rng = np.random.default_rng(0)
    g = make_genome(
        n_in, N_OUT, tracker, rng, connect="full",
        output_act=TANH, n_ram=N_RAM, n_proprio=N_PROPRIO,
    )
    protected_innovs = [c.innov for c in g.conns.values() if c.protected]
    assert protected_innovs, "expected some protected edges"

    rates = ops.MutationRates()
    for _ in range(10_000):
        # bias toward toggle (the direct disable risk); periodically grow the
        # topology via add_node (the split/disable risk).
        if rng.random() < 0.85:
            ops.mutate_toggle(g, rng)
        else:
            ops.mutate_add_node(g, tracker, rng)
        # a protected edge disabled on *this* iteration is caught immediately,
        # even if a later mutation would re-enable it.
        for iv in protected_innovs:
            c = g.conns[iv]
            assert c.enabled and c.protected, f"protected edge {iv} severed"


def test_add_connection_never_overwrites_protected_flag():
    """add_connection only creates fresh (unprotected) edges; it must not affect
    the protected set."""
    n_in = 40
    tracker = InnovationTracker(n_in, N_OUT)
    rng = np.random.default_rng(3)
    g = make_genome(
        n_in, N_OUT, tracker, rng, connect="sparse",
        output_act=TANH, n_ram=N_RAM, n_proprio=N_PROPRIO,
    )
    protected = {c.innov for c in g.conns.values() if c.protected}
    rates = ops.MutationRates(feedforward=True)
    for _ in range(2000):
        ops.mutate_add_connection(g, tracker, rng, rates)
    still = {c.innov for c in g.conns.values() if c.protected}
    assert protected == still
    assert all(g.conns[iv].enabled for iv in protected)


# --------------------------------------------------------------------------
# 4. prefer-unconnected sampler raises the rate unwired inputs get connected
# --------------------------------------------------------------------------
def _build_half_wired(n_in: int, n_out: int, tracker: InnovationTracker, rng):
    """Genome whose first half of inputs are wired (to output 0) and second half
    are unconnected — so the prefer-unconnected boost has room to show."""
    g = make_genome(n_in, n_out, tracker, rng, connect="none")
    out0 = list(g.output_ids())[0]
    half = n_in // 2
    for s in range(half):
        innov = tracker.conn_innov(s, out0)
        g.conns[innov] = ConnGene(s, out0, 0.1, True, innov)
    return g, range(half, n_in)  # (genome, initially-unconnected input ids)


def _count_newly_wired(prefer: bool, seeds: int = 40) -> float:
    n_in, n_out, k_adds = 40, 2, 15
    total = 0
    for seed in range(seeds):
        tracker = InnovationTracker(n_in, n_out)
        rng = np.random.default_rng(1000 + seed)
        g, unconnected = _build_half_wired(n_in, n_out, tracker, rng)
        rates = ops.MutationRates(feedforward=True)
        for _ in range(k_adds):
            ops.mutate_add_connection(
                g, tracker, rng, rates,
                prefer_unconnected=prefer,
                prefer_unconnected_weight=4.0,
            )
        wired = {c.in_id for c in g.conns.values()}
        total += sum(1 for s in unconnected if s in wired)
    return total / seeds


def test_prefer_unconnected_beats_uniform():
    prefer_mean = _count_newly_wired(prefer=True)
    uniform_mean = _count_newly_wired(prefer=False)
    # the boost must measurably raise how many previously-unwired inputs get a
    # fresh edge within a fixed budget of additions.
    assert prefer_mean > uniform_mean + 1.0, (
        f"prefer={prefer_mean:.2f} uniform={uniform_mean:.2f}"
    )


def test_prefer_unconnected_weight_one_is_uniform():
    """weight == 1.0 collapses to the uniform sampler (back-compat guarantee)."""
    n_in = 30
    tracker = InnovationTracker(n_in, 2)
    rng_a = np.random.default_rng(5)
    g_a, _ = _build_half_wired(n_in, 2, tracker, rng_a)

    tracker_b = InnovationTracker(n_in, 2)
    rng_b = np.random.default_rng(5)
    g_b, _ = _build_half_wired(n_in, 2, tracker_b, rng_b)

    rates = ops.MutationRates(feedforward=True)
    # same seed stream -> weight-1.0 prefer path and uniform path must agree.
    r1 = np.random.default_rng(99)
    r2 = np.random.default_rng(99)
    for _ in range(30):
        ops.mutate_add_connection(g_a, tracker, r1, rates,
                                  prefer_unconnected=True, prefer_unconnected_weight=1.0)
        ops.mutate_add_connection(g_b, tracker_b, r2, rates,
                                  prefer_unconnected=False)
    edges_a = sorted((c.in_id, c.out_id) for c in g_a.conns.values())
    edges_b = sorted((c.in_id, c.out_id) for c in g_b.conns.values())
    assert edges_a == edges_b
