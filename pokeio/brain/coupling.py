"""Permanent coupling instrumentation (#30) — 'the failure is silent'.

The RED-TEAM REVISION mandates instrumenting the reward<->perception<->action
coupling PERMANENTLY, because the way this project died is invisible on the loss
curve: the 2026-07-17 audit found policies that were **screen-blind constant
perceptrons** (output std ~0.005 across 400 real screens) with **dead RAM taps**
(zeroing the 8 tap inputs changed the action by 0.000000 for all 224 genomes),
while fitness kept rising off Go-Explore restores. Nothing in the training signal
flagged it.

These are cheap, framework-agnostic probes the RL loop runs periodically. They
operationalize "obs->action mutual information" as concrete SENSITIVITY deltas —
the thing that actually caught the failure:

  * :func:`action_entropy` / :func:`action_diversity` — is the policy decisive and
    varied, or collapsed onto one button?
  * :func:`blind_delta` — how much does the action distribution change when the obs
    is BLANKED? ~0 => the policy ignores what it sees (the audit failure).
  * :func:`ram_ablation_delta` — how much does it change when only the RAM-tap slice
    of the obs is zeroed? ~0 => the RAM taps are dead weight.

A ``policy_fn`` is any callable ``obs_batch (B, dim) float -> logits (B, A) float``
(wrap the torch actor with a to-numpy shim). All distances are total variation in
[0, 1], averaged over the batch, so thresholds are interpretable and game-agnostic.
"""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def _tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total-variation distance between two categorical batches: 0.5*sum|p-q|."""
    return 0.5 * np.abs(p - q).sum(axis=-1)


def action_entropy(logits: np.ndarray) -> float:
    """Mean Shannon entropy (nats) of the per-state action distribution.

    Near 0 => the policy is (near-)deterministic on each state; that is fine for a
    trained policy but a WHOLE BATCH near 0 combined with low :func:`action_diversity`
    is the constant-action collapse. Reported alongside diversity, not alone.
    """
    p = _softmax(logits)
    h = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)
    return float(h.mean())


def action_diversity(logits: np.ndarray) -> float:
    """Entropy (nats) of the MARGINAL argmax-action histogram over the batch.

    This is the collapse detector: if every state maps to the same button the
    marginal is a point mass (0), regardless of per-state confidence. A healthy
    policy visiting varied states spreads its chosen actions.
    """
    a = _softmax(logits).argmax(axis=-1)
    n_actions = logits.shape[-1]
    counts = np.bincount(a, minlength=n_actions).astype(np.float64)
    p = counts / max(1.0, counts.sum())
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())


def blind_delta(policy_fn, obs_batch: np.ndarray, blank_value: float = 0.0) -> float:
    """Mean TV distance between the policy on the real obs vs a BLANKED obs.

    ~0 => the action distribution is (near) invariant to what the agent sees — a
    screen-blind policy (the audit's core failure). A healthy policy's actions
    depend on the observation, so blanking it moves the distribution.
    """
    obs = np.asarray(obs_batch, dtype=np.float32)
    real = _softmax(np.asarray(policy_fn(obs), dtype=np.float64))
    blank = np.full_like(obs, float(blank_value))
    off = _softmax(np.asarray(policy_fn(blank), dtype=np.float64))
    return float(_tv(real, off).mean())


def ram_ablation_delta(policy_fn, obs_batch: np.ndarray, ram_lo: int, ram_hi: int) -> float:
    """Mean TV distance when only the obs RAM-tap slice ``[ram_lo:ram_hi]`` is zeroed.

    ~0 => the RAM taps are dead inputs (audit: Delta=0.000000 for all 224 genomes,
    despite the "the agent SEES the mined counters" claim). Any real coupling makes
    this nonzero. ``ram_lo:ram_hi`` are the flat-obs indices of the ram block.
    """
    obs = np.asarray(obs_batch, dtype=np.float32)
    real = _softmax(np.asarray(policy_fn(obs), dtype=np.float64))
    abl = obs.copy()
    abl[:, ram_lo:ram_hi] = 0.0
    off = _softmax(np.asarray(policy_fn(abl), dtype=np.float64))
    return float(_tv(real, off).mean())


def coupling_report(policy_fn, obs_batch: np.ndarray, ram_slice=None) -> dict:
    """Run all coupling probes; return a dict of scalars for the eval/telemetry log.

    ``ram_slice`` = ``(lo, hi)`` flat-obs indices of the RAM block, or None to skip
    the RAM-ablation probe. The RL loop logs this each eval; a persistently ~0
    ``blind_delta`` or ``ram_ablation_delta`` is the silent-failure alarm.
    """
    obs = np.asarray(obs_batch, dtype=np.float32)
    logits = np.asarray(policy_fn(obs), dtype=np.float64)
    out = {
        "action_entropy": action_entropy(logits),
        "action_diversity": action_diversity(logits),
        "blind_delta": blind_delta(policy_fn, obs),
    }
    if ram_slice is not None:
        out["ram_ablation_delta"] = ram_ablation_delta(policy_fn, obs, ram_slice[0], ram_slice[1])
    return out


__all__ = [
    "action_entropy", "action_diversity", "blind_delta",
    "ram_ablation_delta", "coupling_report",
]
