"""Latent-injection transparency test (active-vision spine §2.2/§4.6; §11 test 7).

The Phase-1 retina replaces the pixel obs blocks with a learned latent, changing
``n_in`` from 454 (foveal) to 102 (retina). The invariant that makes this a
zero-forward-change swap: ``population_forward_sparse`` re-clamps the input slots
and bias every propagation hop, so an arbitrary-width input vector drives the
population unchanged — the genome graph is the sole evolved object.

Asserted here:
* an arbitrary 102-d AND 454-d input vector produce finite outputs of shape
  (N, B, n_out) with no code change;
* the input slots of the returned node-state equal the injected vector exactly
  (re-clamped every hop — no drift, no leak from recurrence);
* distinct inputs drive distinct outputs (the injection actually reaches the head).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokeio.evo.forward import TANH, population_forward_sparse
from pokeio.evo.genome import InnovationTracker, Population, make_genome

N_OUT = 11


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def _compiled(n_in: int, n_pop: int = 8, seed: int = 0):
    tracker = InnovationTracker(n_in, N_OUT)
    rng = np.random.default_rng(seed)
    genomes = [
        make_genome(
            n_in, N_OUT, tracker, rng, connect="full",
            output_act=TANH, n_ram=8, n_proprio=14,
        )
        for _ in range(n_pop)
    ]
    pop = Population.from_genomes(
        genomes, max_nodes=n_in + N_OUT + 64, max_conns=(n_in + 1) * N_OUT + 64
    )
    return pop.compile(_device())


@pytest.mark.parametrize("n_in", [102, 454])
def test_arbitrary_width_input_drives_forward(n_in: int):
    cp = _compiled(n_in)
    rng = np.random.default_rng(3)
    X = torch.from_numpy(rng.standard_normal((5, n_in)).astype(np.float32)).to(_device())
    out = population_forward_sparse(cp, X, steps=8)
    assert out.shape == (cp.n, 5, N_OUT)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n_in", [102, 454])
def test_inputs_reclamped_every_hop(n_in: int):
    cp = _compiled(n_in)
    rng = np.random.default_rng(7)
    X = torch.from_numpy(rng.standard_normal((5, n_in)).astype(np.float32)).to(_device())
    _out, state = population_forward_sparse(cp, X, steps=8, return_state=True)
    # node-state input slots [0:n_in] must equal the injected vector for every
    # genome after 8 hops — re-clamped each step, unaffected by recurrence.
    in_slots = state[:, :, :n_in]
    assert torch.allclose(in_slots, X.unsqueeze(0).expand(cp.n, -1, -1))


@pytest.mark.parametrize("n_in", [102, 454])
def test_distinct_inputs_distinct_outputs(n_in: int):
    cp = _compiled(n_in)
    rng = np.random.default_rng(11)
    X = torch.from_numpy(rng.standard_normal((6, n_in)).astype(np.float32)).to(_device())
    out = population_forward_sparse(cp, X, steps=8)
    # at least one genome maps two different probes to different outputs
    spread = (out[:, 0, :] - out[:, 1, :]).abs().max().item()
    assert spread > 1e-3


def test_same_function_handles_both_widths_without_change():
    """The exact same call site handles 102-d and 454-d — no branching on n_in."""
    for n_in in (102, 454):
        cp = _compiled(n_in)
        X = torch.zeros((2, n_in), dtype=torch.float32, device=_device())
        out = population_forward_sparse(cp, X, steps=6)
        assert out.shape[-1] == N_OUT
