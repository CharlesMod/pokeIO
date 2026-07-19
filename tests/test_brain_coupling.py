"""Coupling-instrumentation tests (pokeio.brain.coupling, task #30).

These probes are the silent-failure alarm: they must READ ~0 on exactly the
degenerate policies the 2026-07-17 audit found (screen-blind constant perceptrons,
dead RAM taps) and clearly nonzero on a policy that actually uses its input.
"""

from __future__ import annotations

import numpy as np

from pokeio.brain.coupling import (
    action_diversity,
    action_entropy,
    blind_delta,
    coupling_report,
    ram_ablation_delta,
)

DIM = 454
RAM_LO, RAM_HI = 446, 454  # the foveal-obs RAM block
N_ACT = 9


def _obs_batch(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, DIM), dtype=np.float32)


# --- degenerate policies (the audit failures) --------------------------------
def _constant_policy(obs):
    # ignores obs entirely: same logits for every row (screen-blind)
    return np.tile(np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0]), (len(obs), 1))


def _obs_blind_but_ram_using(obs):
    # uses ONLY the RAM slice, ignores the visual blocks
    ram = np.asarray(obs)[:, RAM_LO:RAM_HI]
    logits = np.zeros((len(obs), N_ACT))
    logits[:, : ram.shape[1]] = ram * 5.0
    return logits


def _seeing_policy(obs):
    # uses the visual blocks: coarse features drive the button logits
    o = np.asarray(obs)
    feats = np.stack([o[:, :144].mean(1), o[:, 144:288].mean(1),
                      o[:, 288:432].mean(1)], axis=1)
    logits = np.zeros((len(o), N_ACT))
    logits[:, :3] = feats * 8.0
    logits[:, 3:] = -o[:, 400:400 + (N_ACT - 3)] * 8.0
    return logits


# --- action collapse ---------------------------------------------------------
def test_constant_policy_has_zero_action_diversity():
    obs = _obs_batch()
    logits = _constant_policy(obs)
    assert action_diversity(logits) == 0.0          # all argmax -> same button
    assert action_entropy(logits) >= 0.0


def test_seeing_policy_has_positive_action_diversity():
    obs = _obs_batch()
    logits = _seeing_policy(obs)
    assert action_diversity(logits) > 0.3           # actions vary across states


# --- screen-blind detection --------------------------------------------------
def test_blind_delta_zero_for_constant_policy():
    obs = _obs_batch()
    assert blind_delta(_constant_policy, obs) == 0.0   # invariant to blanking


def test_blind_delta_positive_for_seeing_policy():
    obs = _obs_batch()
    assert blind_delta(_seeing_policy, obs) > 0.05     # blanking moves the actions


# --- dead-RAM-tap detection --------------------------------------------------
def test_ram_ablation_zero_when_policy_ignores_ram():
    obs = _obs_batch()
    # the seeing policy reads visual blocks + [400:], NOT the [446:454] ram slice
    assert ram_ablation_delta(_seeing_policy, obs, RAM_LO, RAM_HI) == 0.0


def test_ram_ablation_positive_when_policy_uses_ram():
    obs = _obs_batch()
    assert ram_ablation_delta(_obs_blind_but_ram_using, obs, RAM_LO, RAM_HI) > 0.05


# --- the combined report -----------------------------------------------------
def test_coupling_report_flags_the_degenerate_policy():
    obs = _obs_batch()
    rep = coupling_report(_constant_policy, obs, ram_slice=(RAM_LO, RAM_HI))
    assert rep["action_diversity"] == 0.0
    assert rep["blind_delta"] == 0.0
    assert rep["ram_ablation_delta"] == 0.0          # every alarm trips

    rep2 = coupling_report(_seeing_policy, obs, ram_slice=(RAM_LO, RAM_HI))
    assert rep2["action_diversity"] > 0.3
    assert rep2["blind_delta"] > 0.05                # healthy: uses what it sees
