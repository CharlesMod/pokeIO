"""E3 blind-ablation gate test (active-vision spine §6.2; unit-test plan §11 test 5).

The dominant selection multiplier. Two batched forwards over a probe buffer —
real obs vs the optical blocks [0:432] zeroed (proprio+ram preserved) — give
``Δ_i = mean_P || a_real − a_blind ||_1`` and ``gate_i = sigmoid(β(Δ_i − Δ_min))``.

* a hand-built constant-button genome (no edges from the optical block) scores
  Δ ≈ 0 and is crushed toward the gate floor (sigmoid(−β·Δ_min) ≈ 0.40 at the
  committed β=8, Δ_min=0.05);
* a fovea-wired (full-connect) genome scores Δ ≫ Δ_min and gates to ≈ 1;
* the seeing genome's gate dominates the blind one's by a wide margin (the crush).
"""

from __future__ import annotations

import numpy as np
import torch

from pokeio.evo.forward import TANH, population_forward_sparse
from pokeio.evo.genome import ConnGene, InnovationTracker, Population, make_genome
from pokeio.train.loop import _blind_ablation_gate

N_IN = 454
N_OUT = 11
N_RAM = 8
N_PROPRIO = 14
BETA = 8.0
DMIN = 0.05
OPTICAL_HI = 432  # periphery + fovea + motion; proprio+ram start at 432


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def _seeing_and_blind():
    """A full-connect seeing genome and a screen-blind constant-button genome
    (bias-only edges, so zeroing the optical block cannot change its output)."""
    tracker = InnovationTracker(N_IN, N_OUT)
    rng = np.random.default_rng(0)
    g_see = make_genome(
        N_IN, N_OUT, tracker, rng, connect="full",
        output_act=TANH, n_ram=N_RAM, n_proprio=N_PROPRIO,
    )
    g_blind = make_genome(
        N_IN, N_OUT, tracker, rng, connect="none",
        output_act=TANH, n_ram=N_RAM, n_proprio=N_PROPRIO,
    )
    for o in g_blind.output_ids():  # constant drive from bias only
        innov = tracker.conn_innov(g_blind.bias_id, o)
        g_blind.conns[innov] = ConnGene(g_blind.bias_id, o, 0.7, True, innov)
    return g_see, g_blind


def _probe(B: int = 200, seed: int = 1) -> list:
    rng = np.random.default_rng(seed)
    X = np.empty((B, N_IN), dtype=np.float32)
    X[:, :432] = rng.random((B, 432))
    X[:, 432:446] = rng.uniform(-1.0, 1.0, (B, 14))
    X[:, 446:454] = rng.random((B, 8))
    return list(X)


def _delta_gate():
    g_see, g_blind = _seeing_and_blind()
    pop = Population.from_genomes(
        [g_see, g_blind], max_nodes=N_IN + N_OUT + 16,
        max_conns=(N_IN + 1) * N_OUT + 16,
    )
    cp = pop.compile(_device())
    gate, median_delta = _blind_ablation_gate(
        cp, _probe(), _device(), beta=BETA, dmin=DMIN, optical_hi=OPTICAL_HI
    )
    return gate, median_delta


def test_constant_genome_delta_is_zero_and_gate_floored():
    gate, _ = _delta_gate()
    gate_see, gate_blind = float(gate[0]), float(gate[1])
    # Δ_blind = 0 exactly -> gate = sigmoid(-β·Δ_min) ≈ 0.401: the crush floor.
    floor = 1.0 / (1.0 + np.exp(BETA * DMIN))
    assert abs(gate_blind - floor) < 1e-3, f"blind gate {gate_blind:.4f} != floor {floor:.4f}"


def test_seeing_genome_gate_saturates_high():
    gate, _ = _delta_gate()
    assert float(gate[0]) > 0.9, f"seeing gate {float(gate[0]):.4f} did not saturate"


def test_gate_crushes_blind_relative_to_seeing():
    gate, _ = _delta_gate()
    gate_see, gate_blind = float(gate[0]), float(gate[1])
    assert gate_see - gate_blind > 0.4, (
        f"gate did not separate seeing ({gate_see:.3f}) from blind ({gate_blind:.3f})"
    )


def test_median_delta_clears_dmin_when_population_sees():
    # A seeing genome present -> the population-median Δ must clear Δ_min, the
    # smoke's input-sensitivity gate.
    _, median_delta = _delta_gate()
    assert median_delta > DMIN, f"median Δ {median_delta:.4f} did not clear Δ_min={DMIN}"


def test_empty_probe_is_safe():
    _, g_blind = _seeing_and_blind()
    pop = Population.from_genomes([g_blind], max_nodes=N_IN + N_OUT + 16,
                                 max_conns=(N_IN + 1) * N_OUT + 16)
    cp = pop.compile(_device())
    gate, med = _blind_ablation_gate(cp, [], _device(), beta=BETA, dmin=DMIN)
    assert gate.shape == (1,) and gate[0] == 1.0 and med == 0.0
