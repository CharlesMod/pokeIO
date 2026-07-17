"""Decisive-head init test (active-vision spine §8; unit-test plan §11 test 4).

The gen-100 diagnosis: 9 sigmoid button outputs bunched in [0.3, 0.7],
output-std across 400 screens ~0.005, argmax ties broken by index -> a constant
action regardless of the screen. The fix is a TANH head with fan-in-scaled
``full`` init. This test asserts the cure at gen 0:

* on ~400 probe screens the top-2 button gap exceeds 0.05 for >> half the
  population (vs the ~half coin-flips of the bunched baseline);
* no output saturates to exact +-1 at init (fan-in scaling keeps the head in its
  responsive region);
* TANH is measurably more decisive than the SIGMOID head under the same init.
"""

from __future__ import annotations

import numpy as np
import torch

from pokeio.evo.forward import SIGMOID, TANH, population_forward_sparse
from pokeio.evo.genome import InnovationTracker, Population, make_genome

N_IN = 454  # foveal obs dim (3*12^2 + 14 proprio + 8 ram)
N_OUT = 11  # 9 buttons + 2 saccade
N_BUTTONS = 9
N_RAM = 8
N_PROPRIO = 14


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def _population(output_act: int, connect: str = "full", n: int = 64, seed: int = 0):
    tracker = InnovationTracker(N_IN, N_OUT)
    rng = np.random.default_rng(seed)
    genomes = [
        make_genome(
            N_IN, N_OUT, tracker, rng, connect=connect,
            output_act=output_act, n_ram=N_RAM, n_proprio=N_PROPRIO,
        )
        for _ in range(n)
    ]
    pop = Population.from_genomes(
        genomes, max_nodes=N_IN + N_OUT + 16, max_conns=(N_IN + 1) * N_OUT + 16
    )
    return pop.compile(_device())


def _probe_obs(B: int = 400, seed: int = 1) -> torch.Tensor:
    """~B obs vectors in the real block ranges: periphery/fovea/motion in [0,1],
    proprio in [-1,1], ram in [0,1]."""
    rng = np.random.default_rng(seed)
    X = np.empty((B, N_IN), dtype=np.float32)
    X[:, :432] = rng.random((B, 432))
    X[:, 432:446] = rng.uniform(-1.0, 1.0, (B, 14))
    X[:, 446:454] = rng.random((B, 8))
    return torch.from_numpy(X).to(_device())


def _top2_gap(cp, X) -> torch.Tensor:
    out = population_forward_sparse(cp, X, steps=8)  # (N, B, N_OUT)
    btn = out[..., :N_BUTTONS]
    top2 = btn.topk(2, dim=2).values
    return (top2[..., 0] - top2[..., 1]).mean(dim=1)  # per-genome mean over screens


def test_top2_gap_exceeds_threshold_for_most_of_population():
    cp = _population(TANH)
    gap = _top2_gap(cp, _probe_obs())
    frac = float((gap > 0.05).float().mean().item())
    # >> half the population is decisive (the bunched baseline was ~coin-flip).
    assert frac > 0.75, f"only {frac:.2%} of the population clears a 0.05 top-2 gap"


def test_no_output_saturates_at_init():
    cp = _population(TANH)
    out = population_forward_sparse(cp, _probe_obs(), steps=8)
    m = float(out.abs().max().item())
    assert m < 0.999, f"an output saturated to +-1 at init (max|out|={m:.4f})"


def test_tanh_head_more_decisive_than_sigmoid():
    tanh_gap = _top2_gap(_population(TANH), _probe_obs()).mean().item()
    sig_gap = _top2_gap(_population(SIGMOID), _probe_obs()).mean().item()
    assert tanh_gap > sig_gap, f"tanh={tanh_gap:.4f} !> sigmoid={sig_gap:.4f}"


def test_output_std_across_screens_beats_baseline():
    """Population-wide button-output std across the probe set clears the gen-100
    baseline (~0.005) by well over an order of magnitude."""
    cp = _population(TANH)
    out = population_forward_sparse(cp, _probe_obs(), steps=8)
    std = out[..., :N_BUTTONS].std(dim=1).mean().item()  # mean over genomes/buttons
    assert std > 0.05, f"output std across screens {std:.4f} <= 0.05 (bunched)"
