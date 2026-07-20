"""Path-B learned-gaze head tests (ActorCritic.learned_gaze, Step 4a).

The gaze head is a diagonal-Gaussian saccade-delta action (RAM-style hard gaze),
opt-in.  OFF => byte-identical button-only policy.  ON => a joint (button, gaze)
action whose log-prob/entropy add, gradients flow to the gaze parameters, and greedy
gaze is the distribution mean.  Hermetic (flat-MLP actor, no fleet/torch-gpu).
"""

from __future__ import annotations

import torch

from pokeio.brain.actor_critic import ActorCritic


def _actor(learned_gaze):
    # periph_grid=0 -> flat-MLP fallback (no conv layout to satisfy); isolates the head
    return ActorCritic(64, periph_grid=0, learned_gaze=learned_gaze)


def test_off_is_button_only():
    ac = _actor(False)
    obs = torch.randn(5, 64)
    out = ac.act(obs)
    assert "gaze" not in out
    assert set(out) == {"buttons", "value", "logp", "entropy"}
    logp, ent, val = ac.evaluate_actions(obs, out["buttons"])
    assert logp.shape == (5,) and ent.shape == (5,) and val.shape == (5,)
    # a stray gaze arg is ignored when the head is off
    logp2, _, _ = ac.evaluate_actions(obs, out["buttons"], gaze=torch.randn(5, 2))
    assert torch.equal(logp, logp2)


def test_on_emits_joint_action_consistent_with_evaluate():
    torch.manual_seed(0)
    ac = _actor(True)
    obs = torch.randn(5, 64)
    out = ac.act(obs)
    assert out["gaze"].shape == (5, 2)
    assert torch.isfinite(out["logp"]).all() and torch.isfinite(out["entropy"]).all()
    # re-evaluating the sampled (button, gaze) reproduces the joint log-prob
    logp, ent, val = ac.evaluate_actions(obs, out["buttons"], out["gaze"])
    assert torch.allclose(logp, out["logp"], atol=1e-5)
    assert torch.allclose(ent, out["entropy"], atol=1e-5)


def test_joint_logp_is_button_plus_gaze():
    torch.manual_seed(1)
    ac = _actor(True)
    obs = torch.randn(3, 64)
    out = ac.act(obs)
    h = ac._features(obs)
    from torch.distributions import Categorical
    button_logp = Categorical(logits=ac.pi(h)).log_prob(out["buttons"])
    gaze_logp = ac._gaze_dist(h).log_prob(out["gaze"]).sum(-1)
    assert torch.allclose(out["logp"], button_logp + gaze_logp, atol=1e-5)


def test_gradients_flow_to_gaze_params():
    ac = _actor(True)
    obs = torch.randn(6, 64)
    out = ac.act(obs)
    logp, ent, val = ac.evaluate_actions(obs, out["buttons"], out["gaze"])
    (-(logp.mean()) - 0.01 * ent.mean() + val.pow(2).mean()).backward()
    assert ac.gaze_mu.weight.grad is not None and ac.gaze_mu.weight.grad.abs().sum() > 0
    assert ac.gaze_log_std.grad is not None and torch.isfinite(ac.gaze_log_std.grad).all()


def test_greedy_gaze_is_the_mean():
    ac = _actor(True)
    obs = torch.randn(4, 64)
    out = ac.act(obs, greedy=True)
    mu = ac.gaze_mu(ac._features(obs))
    assert torch.allclose(out["gaze"], mu, atol=1e-6)
