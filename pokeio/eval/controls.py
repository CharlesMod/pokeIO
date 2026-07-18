"""Standing credibility controls (audit 2026-07-17, STRATEGIC).

Three research-motivated audits you run *against a champion* so a headline
number can be trusted:

1. :func:`noise_ablation` — **noise/blank-frame ablation.** Score the champion on
   real observations vs observations whose OPTICAL block is blanked (zeros) and
   scrambled (noise), holding proprioception + RAM intact. A policy whose action
   barely changes is winning by action-timing, not vision — it gets FLAGGED.
   This generalises the training loop's blind-ablation *gate*
   (``train/loop.py:_blind_ablation_gate``) into a reusable, ROM-free audit.

2. :func:`random_weight_baseline` — **random-weight-search baseline.** Sample K
   random genomes with the *same topology budget / init* as the real population,
   score them the SAME way as the champion, and report the best random score and
   the champion's margin over it. Random search famously beat DQN/A3C/ES on
   several Atari games; structured search has to prove it clears this bar.

Both reuse the batched sparse forward (:func:`population_forward_sparse`) and the
same ``vision_dependence`` behavioural score, so the numbers are commensurate.

Everything is ``numpy`` + ``torch`` + repo modules; nothing here needs a ROM.
The observation layout is inferred so the same code covers foveal (454-d),
retina-latent (102-d) and the legacy flat (res^2 + ram) controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from pokeio.evo.forward import TANH, CompiledPopulation, population_forward_sparse
from pokeio.evo.genome import InnovationTracker, Population, make_genome

# Propagation hops per inference — matches train/loop.FORWARD_STEPS so the
# behaviour we measure is the behaviour the loop selected on.
FORWARD_STEPS = 4

# A CompiledPopulation-producing scorer: cp -> per-genome score (N,).
Scorer = Callable[[CompiledPopulation], np.ndarray]


# --------------------------------------------------------------------------
# observation layout + probe synthesis
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ObsLayout:
    """Where the optical / proprio / ram blocks live in a controller obs vector.

    * ``optical_hi`` — end (exclusive) of the OPTICAL block ``[0:optical_hi]``;
      blanking/scrambling this range is the vision ablation.
    * ``n_proprio`` — 14 for the active-vision spine (n_out==11), 0 for the
      legacy flat controller (n_out==9, no efference copy).
    * ``n_ram`` — trailing mined-tap bytes.
    """

    n_in: int
    optical_hi: int
    n_proprio: int
    n_ram: int


def infer_obs_layout(
    n_in: int, n_out: int, n_ram: int = 8, n_proprio: int | None = None
) -> ObsLayout:
    """Infer the obs block boundaries from the genome's I/O widths.

    The active-vision controllers (``n_out == 11``) carry a 14-d proprio block
    before the RAM tail; the legacy flat controller (``n_out == 9``) has none.
    Verified against real checkpoints: foveal 454 (opt 432), retina-latent 102
    (opt 80), legacy flat 584 @ n_out 9 (opt 576).

    ``n_proprio`` overrides the heuristic — needed for reflex gaze
    (optical-frontend-v2 §4), which grows proprio 14->16 WITHOUT changing
    ``n_out`` (it re-purposes the saccade outputs), so the width can't be read off
    ``n_out`` and the caller passes the config-derived value.
    """
    if n_proprio is None:
        n_proprio = 14 if n_out == 11 else 0
    optical_hi = max(0, n_in - n_proprio - n_ram)
    return ObsLayout(n_in=n_in, optical_hi=optical_hi, n_proprio=n_proprio, n_ram=n_ram)


def synthetic_probe(layout: ObsLayout, b: int = 200, seed: int = 1) -> np.ndarray:
    """A ``(b, n_in)`` stand-in observation distribution (no ROM needed).

    Optical + RAM blocks are drawn ``U[0,1]`` (the encoders' output range),
    proprio ``U[-1,1]`` (its efference-copy range) — the same construction the
    blind-gate smoke uses. Real emulator obs give sharper numbers, but this is a
    faithful probe for *vision dependence* (how much the action moves when the
    screen is removed) and keeps the harness runnable anywhere.
    """
    rng = np.random.default_rng(seed)
    n_in, opt, npr, nram = (
        layout.n_in,
        layout.optical_hi,
        layout.n_proprio,
        layout.n_ram,
    )
    X = np.empty((b, n_in), dtype=np.float32)
    if opt > 0:
        X[:, :opt] = rng.random((b, opt), dtype=np.float32)
    if npr > 0:
        X[:, opt : opt + npr] = rng.uniform(-1.0, 1.0, (b, npr)).astype(np.float32)
    if nram > 0:
        X[:, n_in - nram :] = rng.random((b, nram), dtype=np.float32)
    return X


def _forward_actions(cp: CompiledPopulation, X: np.ndarray, device) -> torch.Tensor:
    """Batched forward over a probe; returns ``(N, B, n_out)`` raw outputs."""
    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
        return population_forward_sparse(cp, xt, steps=FORWARD_STEPS)


# --------------------------------------------------------------------------
# behavioural score (shared by both controls, so numbers are commensurate)
# --------------------------------------------------------------------------
def vision_dependence(
    cp: CompiledPopulation, probe: np.ndarray, optical_hi: int, device=None
) -> np.ndarray:
    """Per-genome ``Δ = mean_P || a_real − a_blank ||_1`` over the output.

    Identical formula to the loop's blind-ablation gate: how much the raw action
    output moves when the optical block ``[0:optical_hi]`` is zeroed (proprio +
    RAM held). ``~0`` means the policy ignores the screen. This is the score
    both controls rank on, so "beats random" and "survives ablation" are the
    same currency.
    """
    device = device or cp.device
    if probe is None or len(probe) == 0:
        return np.zeros(int(cp.n), dtype=np.float64)
    P = np.ascontiguousarray(probe, dtype=np.float32)
    Pb = P.copy()
    Pb[:, :optical_hi] = 0.0
    real = _forward_actions(cp, P, device)
    blank = _forward_actions(cp, Pb, device)
    d = (real - blank).abs().sum(dim=2).mean(dim=1)
    return d.detach().cpu().numpy().astype(np.float64)


def vision_dependence_scorer(probe: np.ndarray, optical_hi: int, device=None) -> Scorer:
    """Bind a probe + optical boundary into a ``cp -> scores`` scorer."""

    def _score(cp: CompiledPopulation) -> np.ndarray:
        return vision_dependence(cp, probe, optical_hi, device=device)

    return _score


# --------------------------------------------------------------------------
# CONTROL 1 — noise / blank-frame ablation
# --------------------------------------------------------------------------
@dataclass
class NoiseAblationResult:
    """Per-genome noise/blank ablation outcome (one genome = index 0 for a champ)."""

    delta_blank: np.ndarray  # (N,) |a_real - a_blank|_1, mean over probe
    delta_noise: np.ndarray  # (N,) |a_real - a_noise|_1, mean over probe
    action_change_blank: np.ndarray  # (N,) frac of probe where argmax action flips
    action_change_noise: np.ndarray  # (N,)
    gate: np.ndarray  # (N,) sigmoid(beta*(delta_blank - dmin))  in [0,1]
    blind: np.ndarray  # (N,) bool: flagged screen-blind (winning by timing)
    dmin: float
    beta: float

    def summary(self, i: int = 0) -> str:
        v = "SCREEN-BLIND (winning by action-timing, not vision)" if self.blind[i] else "vision-dependent"
        return (
            f"Δ_blank={self.delta_blank[i]:.4f}  Δ_noise={self.delta_noise[i]:.4f}  "
            f"action-flip(blank)={self.action_change_blank[i] * 100:.1f}%  "
            f"gate={self.gate[i]:.3f}  ->  {v}"
        )


def noise_ablation(
    cp: CompiledPopulation,
    probe: np.ndarray,
    optical_hi: int,
    *,
    device=None,
    beta: float = 8.0,
    dmin: float = 0.05,
    n_buttons: int = 9,
    noise_seed: int = 7,
) -> NoiseAblationResult:
    """Control 1: score real vs blanked vs noise optical obs; flag screen-blind.

    Blank = optical block zeroed. Noise = optical block replaced by ``U[0,1]``
    scramble (proprio + RAM intact in both). Reports both the raw output drift
    (``Δ``) and the *behavioural* drift (fraction of the probe on which the
    argmax button flips). A genome is FLAGGED ``blind`` when ``Δ_blank < dmin``
    (equivalently ``gate < 0.5``) — its action does not depend on the screen, so
    any fitness it earns is action-timing / spawn luck, not seeing.
    """
    device = device or cp.device
    n = int(cp.n)
    if probe is None or len(probe) == 0:
        z = np.zeros(n)
        return NoiseAblationResult(z, z, z.copy(), z.copy(), np.ones(n), np.zeros(n, bool), dmin, beta)

    P = np.ascontiguousarray(probe, dtype=np.float32)
    Pb = P.copy()
    Pb[:, :optical_hi] = 0.0
    Pn = P.copy()
    rng = np.random.default_rng(noise_seed)
    if optical_hi > 0:
        Pn[:, :optical_hi] = rng.random((P.shape[0], optical_hi), dtype=np.float32)

    a_real = _forward_actions(cp, P, device)
    a_blank = _forward_actions(cp, Pb, device)
    a_noise = _forward_actions(cp, Pn, device)

    d_blank = (a_real - a_blank).abs().sum(dim=2).mean(dim=1).cpu().numpy().astype(np.float64)
    d_noise = (a_real - a_noise).abs().sum(dim=2).mean(dim=1).cpu().numpy().astype(np.float64)

    nb = min(int(n_buttons), a_real.shape[2])
    arg_real = a_real[:, :, :nb].argmax(dim=2)
    arg_blank = a_blank[:, :, :nb].argmax(dim=2)
    arg_noise = a_noise[:, :, :nb].argmax(dim=2)
    ac_blank = (arg_real != arg_blank).float().mean(dim=1).cpu().numpy().astype(np.float64)
    ac_noise = (arg_real != arg_noise).float().mean(dim=1).cpu().numpy().astype(np.float64)

    gate = 1.0 / (1.0 + np.exp(-float(beta) * (d_blank - float(dmin))))
    blind = d_blank < float(dmin)
    return NoiseAblationResult(d_blank, d_noise, ac_blank, ac_noise, gate, blind, dmin, beta)


# --------------------------------------------------------------------------
# CONTROL 2 — random-weight-search baseline
# --------------------------------------------------------------------------
@dataclass
class RandomBaselineResult:
    """Best-of-K random-weight-search outcome vs the champion."""

    scores: np.ndarray  # (K,) random-genome scores
    best: float
    mean: float
    best_index: int
    champion_score: float
    margin: float  # champion_score - best (must be > 0 to claim we beat random)
    k: int
    connect: str
    beats_random: bool = field(init=False)

    def __post_init__(self) -> None:
        self.beats_random = bool(self.margin > 0.0)

    def summary(self) -> str:
        verdict = "champion BEATS best random" if self.beats_random else "champion does NOT beat random (FLAG)"
        return (
            f"best-of-{self.k} random={self.best:.4f} (mean {self.mean:.4f})  "
            f"champion={self.champion_score:.4f}  margin={self.margin:+.4f}  ->  {verdict}"
        )


def sample_random_genomes(
    *,
    n_in: int,
    n_out: int,
    k: int,
    connect: str = "full",
    sparse_k: int = 32,
    n_ram: int = 8,
    n_proprio: int = 14,
    weight_scale: float = 1.0,
    output_act: int = TANH,
    seed: int = 0,
):
    """Sample ``k`` random genomes with the population's topology budget / init.

    Mirrors ``train/loop`` population seeding: same ``connect`` mode, fan-in-
    scaled init, connect-protected RAM/proprio taps, decisive ``tanh`` head. The
    result is exactly the FS-NEAT starting distribution — a fair random-search
    baseline.
    """
    tracker = InnovationTracker(n_in, n_out)
    rng = np.random.default_rng(seed)
    return [
        make_genome(
            n_in, n_out, tracker, rng,
            connect=connect, weight_scale=weight_scale, output_act=output_act,
            sparse_k=sparse_k, n_ram=n_ram, n_proprio=n_proprio,
        )
        for _ in range(int(k))
    ]


def random_weight_baseline(
    *,
    n_in: int,
    n_out: int,
    k: int,
    score_fn: Scorer,
    champion_score: float,
    connect: str = "full",
    sparse_k: int = 32,
    n_ram: int = 8,
    n_proprio: int = 14,
    weight_scale: float = 1.0,
    max_nodes: int | None = None,
    max_conns: int | None = None,
    device=None,
    seed: int = 0,
) -> RandomBaselineResult:
    """Control 2: best-of-K random-weight search vs the champion, same scorer.

    ``score_fn`` maps a compiled population to per-genome scores (typically
    :func:`vision_dependence_scorer` bound to the champion's probe, so champion
    and randoms are scored identically). Returns the best random score and the
    champion's ``margin`` over it — positive means structured search cleared the
    random-search bar.
    """
    genomes = sample_random_genomes(
        n_in=n_in, n_out=n_out, k=k, connect=connect, sparse_k=sparse_k,
        n_ram=n_ram, n_proprio=n_proprio, weight_scale=weight_scale, seed=seed,
    )
    mn = max_nodes or (n_in + n_out + 16)
    mc = max_conns or ((n_in + 1) * n_out + 16)
    pop = Population.from_genomes(genomes, max_nodes=mn, max_conns=mc)
    cp = pop.compile(device or "cpu")
    scores = np.asarray(score_fn(cp), dtype=np.float64)
    best_i = int(np.argmax(scores))
    best = float(scores[best_i])
    return RandomBaselineResult(
        scores=scores,
        best=best,
        mean=float(np.mean(scores)),
        best_index=best_i,
        champion_score=float(champion_score),
        margin=float(champion_score) - best,
        k=int(k),
        connect=connect,
    )


__all__ = [
    "FORWARD_STEPS",
    "Scorer",
    "ObsLayout",
    "infer_obs_layout",
    "synthetic_probe",
    "vision_dependence",
    "vision_dependence_scorer",
    "NoiseAblationResult",
    "noise_ablation",
    "RandomBaselineResult",
    "sample_random_genomes",
    "random_weight_baseline",
]
