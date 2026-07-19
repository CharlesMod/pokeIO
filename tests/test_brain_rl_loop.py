"""RL loop logic tests (pokeio.brain.rl_loop, task #32/#34).

Hermetic: a FakeFleet (no furnace/ROM/GPU) drives the full rollout -> GAE -> PPO ->
gate-eval path on CPU, with a scripted WRAM progression so the reward actually flows.
The real end-to-end smoke on BarrierFleet lives in a scratchpad script.
"""

from __future__ import annotations

import numpy as np
import torch

from pokeio.brain.rl_loop import BrainConfig, BrainTrainer, _gae

WRAM_BASE = 0xC000


class FakeDemo:
    """No-ROM stand-in: fixed length, and every start is a cold boot (no restore)."""

    def __len__(self):
        return 657

    def restore_map(self, env_depths):
        return {}


class FakeFleet:
    """Minimal fleet stub: scripts a WRAM progression so reward + reached fire.

    t==1 -> entered a new map (37); t>=milestone -> party=1 / level=5 / Oak's lab.
    """

    def __init__(self, n_envs=4, obs_dim=454, obs_ram=8, milestone=3, seed=0):
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.obs_ram = obs_ram
        self.milestone = milestone
        self.arr = {"wram": np.zeros((n_envs, 8192), np.uint8)}
        self._t = 0
        self._rng = np.random.default_rng(seed)

    def _newgame(self):
        self.arr["wram"][:] = 0
        self.arr["wram"][:, 0xD35D - WRAM_BASE] = 38  # bedroom

    def _obs(self):
        return self._rng.random((self.n_envs, self.obs_dim), dtype=np.float32)

    def reset_all(self, restore=None):
        self._t = 0
        self._newgame()
        return self._obs()

    def step_all(self, buttons, gaze_dx=None, gaze_dy=None):
        self._t += 1
        w = self.arr["wram"]
        if self._t == 1:
            w[:, 0xD35D - WRAM_BASE] = 37             # entered house 1F (+1 map)
        if self._t >= self.milestone:
            w[:, 0xD162 - WRAM_BASE] = 1              # party count
            w[:, 0xD163 - WRAM_BASE] = 84             # Pikachu species
            w[:, 0xD18B - WRAM_BASE] = 5              # level 5
            w[:, 0xD35D - WRAM_BASE] = 40             # Oak's lab
        return self._obs(), None, np.zeros(self.n_envs, bool), {}


def _trainer(milestone=3, horizon_max=8, n_envs=4, eval_every=0):
    fleet = FakeFleet(n_envs=n_envs, milestone=milestone)
    cfg = BrainConfig(horizon_min=4, horizon_max=horizon_max, minibatches=2,
                      epochs=2, eval_every=eval_every, seed=1)
    return BrainTrainer(fleet, device="cpu", demo=FakeDemo(), cfg=cfg)


# --------------------------------------------------------------------- GAE
def test_gae_shapes_and_terminal_reward():
    # a single terminal +1 with zero values and gamma=lam=1 -> advantage 1 everywhere
    T, N = 5, 3
    rew = np.zeros((T, N), np.float32)
    rew[-1] = 1.0
    val = np.zeros((T, N), np.float32)
    adv, ret = _gae(rew, val, np.zeros(N, np.float32), gamma=1.0, lam=1.0)
    assert adv.shape == (T, N) and ret.shape == (T, N)
    assert np.allclose(adv, 1.0)


# --------------------------------------------------------------- rollout
def test_rollout_flows_reward_and_reaches_milestone():
    tr = _trainer(milestone=3, horizon_max=8)
    batch = tr.rollout()
    H, N = batch["H"], tr.n_envs
    assert batch["obs"].shape == (H * N, tr.obs_dim)
    assert batch["act"].shape == (H * N,)
    assert np.isfinite(batch["adv"]).all() and np.isfinite(batch["ret"]).all()
    assert batch["reached"] == 1.0            # every env got the starter
    assert batch["rew_sum_mean"] > 0.0        # real progress paid out


def test_curriculum_recedes_after_successful_rollout():
    tr = _trainer(milestone=2, horizon_max=8)
    f0 = tr.curriculum.frontier
    tr.rollout()
    assert tr.curriculum.frontier < f0        # all envs reached -> frontier recedes


# ------------------------------------------------------------------ PPO
def test_update_runs_gradient_step_and_changes_params():
    tr = _trainer()
    before = torch.cat([p.detach().flatten() for p in tr.policy.parameters()]).clone()
    stats = tr.update(tr.rollout())
    after = torch.cat([p.detach().flatten() for p in tr.policy.parameters()])
    assert not torch.allclose(before, after)  # a real optimization step happened
    for k in ("pi_loss", "vf_loss", "entropy", "clipfrac", "kl", "ent_coef"):
        assert np.isfinite(stats[k])


def test_entropy_coef_neuromodulation_scales_with_success():
    tr = _trainer(milestone=2)
    # after successful rollouts the success EMA rises -> ent_coef shrinks (exploit)
    tr.rollout(); tr.rollout()
    s = tr.update(tr.rollout())
    assert 0.0 <= s["ent_coef"] <= tr.cfg.ent_coef_base
    assert tr.curriculum.success_ema > 0.0


# ---------------------------------------------------------------- gate eval
def test_evaluate_gate_returns_report():
    tr = _trainer(milestone=3)
    rep = tr.evaluate_gate(horizon=6)
    assert rep["gate_reached_frac"] == 1.0     # scripted to reach the milestone
    assert rep["progress_mean"] > 0.0
    for k in ("action_entropy", "action_diversity", "blind_delta", "ram_ablation_delta"):
        assert k in rep and np.isfinite(rep[k])


def test_train_runs_multiple_iterations():
    tr = _trainer(milestone=3, eval_every=2)
    hist = tr.train(3)
    assert len(hist) == 3
    assert hist[-1]["iter"] == 3
    assert "eval" in hist[1]                    # eval_every=2 -> logged at iter 2
