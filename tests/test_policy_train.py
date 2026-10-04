import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokeio.config import PolicyConfig, load_config  # noqa: E402
from pokeio.env import GameEnv  # noqa: E402
from pokeio.policy import Policy  # noqa: E402


def _obs_batch(env, n):
    obs = [env.reset() for _ in range(n)]
    return {k: torch.from_numpy(np.stack([o[k] for o in obs])) for k in obs[0]}


def test_policy_step_and_sequence_agree(grid_spec):
    env = GameEnv(grid_spec)
    pol = Policy(env.obs_spec, PolicyConfig(conv=[[8, 4, 2]], hidden=32, bits_hidden=8))
    B, T = 3, 5
    seq_obs = [_obs_batch(env, B) for _ in range(T)]
    starts = torch.zeros(T, B)
    starts[0] = 1
    starts[3, 1] = 1  # an episode boundary mid-sequence for env 1
    h = pol.initial_state(B, "cpu")
    step_logits = []
    state = h
    for t in range(T):
        lg, v, state = pol.step(seq_obs[t], state, starts[t])
        step_logits.append(lg)
    stacked = {k: torch.stack([o[k] for o in seq_obs]) for k in seq_obs[0]}
    seq_logits, seq_v = pol.sequence(stacked, h, starts)
    assert seq_logits.shape == (T, B, env.obs_spec.n_actions) and seq_v.shape == (T, B)
    assert torch.allclose(torch.stack(step_logits), seq_logits, atol=1e-5)


def test_unpack_matches_numpy(grid_spec):
    from pokeio.screen import unpack

    env = GameEnv(grid_spec)
    obs = env.reset()
    pol = Policy(env.obs_spec, PolicyConfig(conv=[[8, 4, 2]], hidden=16, bits_hidden=8))
    t = pol.unpack_pixels(torch.from_numpy(obs["pixels"][None]))
    ref = np.stack([unpack(obs["pixels"][c], env.obs_spec.pixel_bpp) for c in range(2)])
    assert np.allclose(t[0].numpy(), ref / 3.0)


def test_trainer_runs_two_updates_on_cpu(grid_spec, tmp_path):
    from pokeio.train import Trainer, load_policy

    cfg = load_config(
        overrides=[
            "vec.num_workers=0", "vec.envs_per_worker=4",
            "policy.conv=[[8, 4, 2]]", "policy.hidden=32", "policy.bits_hidden=8",
            "train.rollout_len=32", "train.bptt=8", "train.minibatch_seqs=4",
            "train.total_steps=256", "train.device=cpu", f"train.run_dir={tmp_path}",
            "train.run_name=t", "train.checkpoint_every=1",
            "dash.port=0",  # ephemeral port: exercises the hub + server without clashing
        ]
    )
    tr = Trainer(cfg, grid_spec)
    tr.train()
    assert tr.global_step >= 256
    ck = tmp_path / "t" / "ckpt" / "latest.pt"
    assert ck.exists()
    pol, meta = load_policy(ck)
    assert meta["global_step"] == tr.global_step
    # the wall persisted a snapshot with real numbers in it
    import json

    snap = json.loads((tmp_path / "t" / "dash_state.json").read_text())
    assert snap["global_step"] == tr.global_step and snap["history"]
    assert snap["meta"]["buttons"] == list(grid_spec.controls.buttons)


def test_hub_live_snapshot(grid_spec, tmp_path):
    from pokeio.dash import LiveHub
    from pokeio.vec import SerialVec

    v = SerialVec(grid_spec, 4)
    hub = LiveHub(grid_spec, v.obs_spec, 4, tmp_path, wall_size=2)
    v.reset()
    for step in range(30):
        b = v.recv()
        a = np.random.default_rng(step).integers(len(grid_spec.controls.buttons), size=4)
        probs = np.full((4, len(grid_spec.controls.buttons)), 1 / len(grid_spec.controls.buttons))
        hub.on_infos(b.env_ids, b.obs, b.infos)
        hub.on_step(b.env_ids, b.obs, a, probs, np.zeros(4), b.rewards, b.dones, step * 4)
        v.send(a.astype(np.int32))
    live = hub.snapshot_live(0)
    assert set(live["wall"]) == set(hub.wall_ids.tolist()) and live["hero_frames"]
    state = hub.snapshot_state()
    assert state["heat"] and state["total_cells"] > 0
    v.close()
