import numpy as np

from pokeio.env import GameEnv
from pokeio.vec import ProcVec, SerialVec


def _check_obs(env_or_vec, obs, batch=None):
    for k, (shape, dt) in env_or_vec.obs_spec.arrays().items():
        want = shape if batch is None else (batch, *shape)
        assert obs[k].shape == want, k
        assert obs[k].dtype == dt, k


def test_env_reset_step_shapes(grid_spec):
    env = GameEnv(grid_spec)
    obs = env.reset()
    _check_obs(env, obs)
    assert env.has_position and env.obs_spec.pixels[0] == 2
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, r, d, info = env.step(int(rng.integers(len(env.buttons))))
        _check_obs(env, obs)
        assert np.isfinite(r)
    assert env.cells.unique >= 1


def test_exploration_reward_and_interaction(grid_spec):
    env = GameEnv(grid_spec)
    env.reset()
    room = env.p.rooms[0]
    # walk next to the NPC (west of it), face east, press A
    nx, ny = room["npc"]
    env.p.ram[0x01], env.p.ram[0x02] = nx - 1, ny
    env.p.ram[0x03] = 3
    env.mem.invalidate()
    env._pos = env._position()
    obs, r, d, info = env.step(env.buttons.index("a"))
    # the dialog was auto-advanced (wait_while busy) and the event paid out
    assert info.get("waits", 0) > 0
    assert r >= grid_spec.terms[0].weight
    assert env.p.ram[0x10] & 1
    # the frontier score went up by >= min_delta -> swarm report with a save state
    assert "frontier" in info and isinstance(info["frontier"]["state"], bytes)


def test_load_state_rebases_without_reward(grid_spec):
    a, b = GameEnv(grid_spec, env_id=0), GameEnv(grid_spec, env_id=1)
    a.reset()
    b.reset()
    a.p._set_event(0)
    st = a.save_state()
    b.load_state(st)
    obs, r, d, info = b.step(b.buttons.index("b"))
    assert r < grid_spec.terms[0].weight  # no reward for inherited progress
    assert b.frontier_state == st


def test_serial_vec_swarm_load_marks_done2(grid_spec):
    v = SerialVec(grid_spec, 3)
    v.reset()
    b = v.recv()
    st = v.envs[0].save_state()
    v.load_states({1: st})
    v.send(np.zeros(3, np.int32))
    b = v.recv()
    assert b.dones.tolist()[1] == 2 and b.dones[0] != 2
    v.close()


def test_proc_vec_async_roundtrip(grid_spec):
    v = ProcVec(grid_spec, num_workers=2, envs_per_worker=2, batch_workers=1)
    try:
        v.reset()
        seen = set()
        for _ in range(20):
            b = v.recv()
            assert len(b.env_ids) == 2
            seen.update(b.env_ids.tolist())
            for k, (shape, dt) in v.obs_spec.arrays().items():
                assert b.obs[k].shape == (2, *shape)
            v.send(np.zeros(len(b.env_ids), np.int32))
        assert seen == {0, 1, 2, 3}
        # holding one worker back must not deadlock recv()
        b = v.recv()
        held = b.env_ids
        b2 = v.recv()
        assert set(b2.env_ids.tolist()).isdisjoint(held.tolist())
        v.send(np.zeros(len(b2.env_ids), np.int32), b2.env_ids)
        v.send(np.zeros(len(held), np.int32), held)
    finally:
        v.close()
