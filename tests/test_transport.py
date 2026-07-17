"""Correctness of the default 'furnace' (AsyncFleet) shared-memory transport
(audit A10 / findings #6, #29).

The live trainer runs on ``AsyncFleet``: free-running worker processes step
their emulators and publish observations into shared memory, coordinated with
the parent purely through two monotonic counters per env — ``obs_seq[i]`` (obs
the worker has published) and ``act_seq[i]`` (action the parent has answered
with). A worker steps env ``i`` exactly when ``act_seq[i] == obs_seq[i] <
target``. This lock-free handshake carries every training step, yet had no
test.

We drive a tiny fleet (4 single-env workers, a handful of steps) through the
same handshake ``evaluate_wave_async`` uses and assert:

  * **round-trip fidelity** — the obs a fleet worker returns for a known action
    sequence is *bit-identical* to a single-process reference emulator driven
    with the same actions (so the transport delivers exactly the action given
    and returns exactly the obs the emulator produced — no drop, no stale row,
    no misroute);
  * **per-env isolation + determinism** — two envs fed identical action streams
    end bit-identical; an env fed a different stream ends different (actions
    reach the right worker and actually drive it);
  * **no deadlock** — every active env advances exactly ``target`` steps within
    a wall-clock budget;
  * **graceful shutdown** — ``close()`` joins every worker (no orphans / leaked
    shared memory).

Kept small and fast (~1-2s): 4 workers, obs_res 8, 16 steps.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

ROM = "roms/pokemon_yellow.gb"
STATE = "roms/yellow_newgame.state"
RES, RAM, STRIDE = 8, 4, 64
FRAME_SKIP, HOLD = 1, 0
N_ENVS, STEPS = 4, 16

pytestmark = pytest.mark.skipif(
    not (Path(ROM).exists() and Path(STATE).exists()),
    reason="ROM / reset-state not available",
)

_ARCHIVE_KW = dict(
    screen_cells=(16, 14), screen_levels=4,
    wram_stride=STRIDE, wram_levels=16, wram_mask=None,
)


def _encoder():
    from pokeio.emu.fleet import ObsEncoder

    return ObsEncoder(RES, RAM)


def _reference(actions):
    """Single-process ground truth: final (obs, cell_key) for an action list.

    Mirrors exactly what an async worker does per step (see
    ``_async_worker_main._emit``): ``step_fast`` -> ``encode_compact`` +
    ``cell_key_compact`` over the same strided WRAM."""
    from pokeio.emu.env import PokeEnv
    from pokeio.reward.archive import NoveltyArchive

    enc = _encoder()
    arch = NoveltyArchive(**_ARCHIVE_KW)
    env = PokeEnv(ROM, frame_skip=FRAME_SKIP, hold_frames=HOLD)
    try:
        screen = env.reset(STATE)
        w = env.wram_strided(STRIDE)
        obs = enc.encode_compact(screen, w)
        for a in actions:
            screen, w, _ = env.step_fast(int(a), STRIDE)
            obs = enc.encode_compact(screen, w)
        key = arch.cell_key_compact(screen, w)
    finally:
        env.close()
    return obs, key


def _new_fleet():
    from pokeio.emu.fleet import AsyncFleet

    enc = _encoder()
    return AsyncFleet(
        n_envs=N_ENVS, obs_dim=enc.dim, obs_res=RES, obs_ram=RAM, rom_path=ROM,
        frame_skip=FRAME_SKIP, hold_frames=HOLD, reset_state=STATE,
        archive_kwargs=dict(_ARCHIVE_KW), wram_stride=STRIDE,
        goexplore=False, envs_per_worker=1,
    )


def _drive(fleet, schedules, target, timeout=45.0):
    """Run one wave via the obs_seq/act_seq handshake (as evaluate_wave_async).

    ``schedules[i][k]`` is the action to apply to env ``i``'s obs ``k``. Raises
    ``TimeoutError`` if the wave stalls (the no-deadlock guard)."""
    n = len(schedules)
    obs_seq = fleet.arr["obs_seq"]
    act_seq = fleet.arr["act_seq"]
    actions = fleet.arr["actions"]
    acted = np.full(n, -1, dtype=np.int64)
    fleet.begin_wave(n, target)
    deadline = time.time() + timeout
    while True:
        snap = obs_seq[:n].copy()
        ready = (snap > acted) & (snap < target)
        if ready.any():
            idx = np.nonzero(ready)[0]
            for i in idx:
                ii = int(i)
                actions[ii] = schedules[ii][int(snap[ii])]
            act_seq[idx] = snap[idx]  # publish AFTER the action rows
            acted[idx] = snap[idx]
        elif bool((snap >= target).all()):
            break
        else:
            time.sleep(2e-5)
        if time.time() > deadline:
            raise TimeoutError(
                f"AsyncFleet wave deadlocked at obs_seq={obs_seq[:n].tolist()}"
            )
    fleet.end_wave()


def test_async_transport_roundtrip_and_shutdown():
    # env 0,1 -> "down"x16 (this newgame state visibly changes under 'down');
    # env 2,3 -> "noop"x16 (stays put). So A advances, B doesn't, A != B.
    down, noop = 1, 8
    A = [down] * STEPS
    B = [noop] * STEPS
    schedules = {0: A, 1: A, 2: B, 3: B}

    fleet = _new_fleet()
    try:
        assert 2 <= fleet.n_workers <= N_ENVS
        obs0 = fleet.reset_all()
        assert obs0.shape == (N_ENVS, _encoder().dim)
        assert fleet.arr["obs_seq"][:N_ENVS].tolist() == [0] * N_ENVS

        _drive(fleet, schedules, STEPS)

        # no deadlock: every active env ran exactly `STEPS` steps.
        assert fleet.arr["obs_seq"][:N_ENVS].tolist() == [STEPS] * N_ENVS

        final = fleet.arr["obs"].copy()
        ref_A, key_A = _reference(A)
        ref_B, _ = _reference(B)

        # round-trip fidelity vs single-process reference
        assert np.array_equal(final[0], ref_A)
        assert np.array_equal(final[2], ref_B)
        # per-env isolation + determinism
        assert np.array_equal(final[0], final[1])
        assert np.array_equal(final[2], final[3])
        # actions actually drive the worker (A moved; A differs from B)
        assert not np.array_equal(final[0], obs0[0])
        assert not np.array_equal(final[0], final[2])
        # cell-key transport round-trips too
        assert fleet.key_bytes(0) == key_A
    finally:
        fleet.close()

    # graceful shutdown: no worker left alive.
    assert all(not p.is_alive() for p in fleet._procs)


def test_close_is_graceful_without_a_wave():
    """Construct + reset + close (no wave) must still shut down cleanly, and a
    second close must be a harmless no-op."""
    fleet = _new_fleet()
    try:
        fleet.reset_all()
    finally:
        fleet.close()
    assert all(not p.is_alive() for p in fleet._procs)
    fleet.close()  # idempotent
