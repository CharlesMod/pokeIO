"""ActorCritic tests (pokeio.brain.actor_critic, task #32/#33)."""

from __future__ import annotations

import numpy as np
import torch

from pokeio.brain.actor_critic import ActorCritic, N_BUTTONS

OBS_DIM = 454  # foveal default (periph144+fovea144+motion144+proprio14+ram8)


def test_conv_trunk_used_for_default_foveal():
    ac = ActorCritic(OBS_DIM, grid=12)
    assert ac.use_conv is True
    assert ac.n_spatial == 3 * 12 * 12  # 432
    assert ac.n_extra == OBS_DIM - 432   # 22


def test_forward_shapes():
    ac = ActorCritic(OBS_DIM, grid=12)
    obs = torch.rand(7, OBS_DIM)
    logits, value = ac(obs)
    assert logits.shape == (7, N_BUTTONS)
    assert value.shape == (7,)


def test_act_and_evaluate_shapes():
    ac = ActorCritic(OBS_DIM, grid=12)
    obs = torch.rand(5, OBS_DIM)
    out = ac.act(obs)
    assert out["buttons"].shape == (5,)
    assert out["logp"].shape == (5,) and out["value"].shape == (5,)
    assert out["buttons"].max().item() < N_BUTTONS
    logp, ent, val = ac.evaluate_actions(obs, out["buttons"])
    assert logp.shape == (5,) and ent.shape == (5,) and val.shape == (5,)


def test_greedy_is_argmax():
    ac = ActorCritic(OBS_DIM, grid=12)
    obs = torch.rand(4, OBS_DIM)
    logits, _ = ac(obs)
    assert torch.equal(ac.act(obs, greedy=True)["buttons"], logits.argmax(-1))


def test_mlp_fallback_when_not_three_equal_sheets():
    # a dim that can't hold 3 equal 12x12 sheets at the front still works (MLP path)
    ac = ActorCritic(100, grid=12)   # 3*144=432 > 100 -> fallback
    assert ac.use_conv is False
    logits, value = ac(torch.rand(3, 100))
    assert logits.shape == (3, N_BUTTONS) and value.shape == (3,)


def test_numpy_policy_fn_adapter_for_coupling():
    ac = ActorCritic(OBS_DIM, grid=12).eval()
    fn = ac.numpy_policy_fn(device=torch.device("cpu"))
    logits = fn(np.random.rand(6, OBS_DIM).astype(np.float32))
    assert isinstance(logits, np.ndarray) and logits.shape == (6, N_BUTTONS)


def test_init_policy_is_near_uniform():
    # small pi-head gain => actions near uniform at init (no constant-action collapse)
    ac = ActorCritic(OBS_DIM, grid=12)
    obs = torch.rand(256, OBS_DIM)
    logits, _ = ac(obs)
    probs = torch.softmax(logits, -1)
    # every action retains meaningful mass at init (max prob well below 1)
    assert probs.max().item() < 0.5
    # and gradients flow to the conv trunk (percept is in the graph, #33)
    (logits.sum() + 0.0).backward()
    assert ac.conv[0].weight.grad is not None
    assert ac.conv[0].weight.grad.abs().sum().item() > 0
