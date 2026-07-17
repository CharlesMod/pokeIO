"""Saccade action plumbing through the shared-memory fleet (spec §3.4, test #2).

Drives a real ``AsyncFleet`` (the default 'furnace' engine) with a scripted
``(dx, dy)`` sweep and asserts the saccade seam end-to-end:

  * the ``gaze_dx`` / ``gaze_dy`` sibling shm arrays round-trip the saccade
    command **exactly** (float32, no quantization);
  * each worker integrates the command into its per-env gaze BEFORE building the
    obs, so the fovea crop **walks across the screen** (many distinct centres);
  * the ``proprio`` block records the applied ``(dx, dy)`` efference copy and the
    new ``(gx, gy)`` gaze;
  * the worker-produced optical + proprio vector ``[0:446]`` is **byte-identical**
    to a parent-side ``FovealEncoder`` driven with the same screens + the same
    submitted ``(dx, dy, button)`` sequence — the cross-PROCESS half of the
    "three build sites must agree" invariant (§2.1).

Spawns real PokeEnv workers, so it needs the Yellow ROM; skips cleanly if absent.
Kept tiny: 2 single-env workers, 10 steps.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from pokeio.emu.fleet import AsyncFleet, FovealEncoder

ROM = "roms/pokemon_yellow.gb"
STATE = "roms/yellow_newgame.state"
STRIDE = 64
FRAME_SKIP, HOLD = 1, 0
N_ENVS, STEPS = 2, 10

# Foveal geometry (committed defaults; small n_ram keeps the shm blocks tiny).
G, F, N_RAM = 12, 48, 8
GAIN, EVERY_K = 32.0, 1

pytestmark = pytest.mark.skipif(
    not (Path(ROM).exists() and Path(STATE).exists()),
    reason="ROM / reset-state not available",
)

_ARCHIVE_KW = dict(
    screen_cells=(16, 14), screen_levels=4,
    wram_stride=STRIDE, wram_levels=16, wram_mask=None,
)

# ---- optical + proprio block (everything but the ram tail, which needs wram)
OPTICAL = slice(0, 446)
PROPRIO = slice(432, 446)
FOVEA = slice(144, 288)


def _mk_encoder(n_envs: int, episode_steps: int) -> FovealEncoder:
    return FovealEncoder(
        n_envs, periph_grid=G, fovea_native_px=F, n_ram=N_RAM,
        saccade_gain=GAIN, saccade_every_k=EVERY_K, episode_steps=episode_steps,
    )


def _new_fleet(dim: int) -> AsyncFleet:
    return AsyncFleet(
        n_envs=N_ENVS, obs_dim=dim, obs_res=G, obs_ram=N_RAM, rom_path=ROM,
        frame_skip=FRAME_SKIP, hold_frames=HOLD, reset_state=STATE,
        archive_kwargs=dict(_ARCHIVE_KW), wram_stride=STRIDE,
        goexplore=False, envs_per_worker=1,
        periph_grid=G, fovea_native_px=F, saccade_gain=GAIN,
        saccade_every_k=EVERY_K, episode_steps=STEPS,
    )


# A deterministic 2-D sweep per env (all values float32-exact so proprio can be
# checked for an exact round-trip). Env 0 sweeps right/down then left/up; env 1
# is the mirror image, so the two envs' gazes diverge (per-env isolation).
def _cmd(env: int, k: int) -> tuple[float, float]:
    sign = 1.0 if k < STEPS // 2 else -1.0
    if env == 1:
        sign = -sign
    return (0.75 * sign, 0.75 * sign)


def _button(env: int, k: int) -> int:
    return 8  # NOOP: keep the game quiet so the sweep is about gaze, not play


def _drive_and_capture(fleet: AsyncFleet, timeout: float = 60.0):
    """Run one wave; return per-env {k: (obs_row_copy, screen_copy)} for k=0..STEPS.

    A worker parks env i at obs k until the parent publishes act_seq[i]=k, so the
    obs/screen rows are stable to snapshot before we submit the action for obs k.
    """
    n = N_ENVS
    obs_seq = fleet.arr["obs_seq"]
    obs_shm = fleet.arr["obs"]
    gaze_dx = fleet.arr["gaze_dx"]
    gaze_dy = fleet.arr["gaze_dy"]
    screens = fleet.screens
    recorded: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {i: {} for i in range(n)}
    seen_gaze: dict[int, dict[int, tuple[float, float]]] = {i: {} for i in range(n)}
    acted = np.full(n, -1, dtype=np.int64)

    fleet.begin_wave(n, STEPS)
    deadline = time.time() + timeout
    while True:
        snap = obs_seq[:n].copy()
        for i in range(n):
            k = int(snap[i])
            if k not in recorded[i]:
                recorded[i][k] = (obs_shm[i].copy(), screens[i].copy())
        ready = (snap > acted) & (snap < STEPS)
        if ready.any():
            idx = np.nonzero(ready)[0]
            gdx = np.zeros(n, np.float32)
            gdy = np.zeros(n, np.float32)
            btn = np.zeros(n, np.int32)
            for i in idx:
                ii = int(i)
                k = int(snap[ii])
                dx, dy = _cmd(ii, k)
                gdx[ii], gdy[ii], btn[ii] = dx, dy, _button(ii, k)
                seen_gaze[ii][k] = (dx, dy)
            # The extended per-step action-submission API (button + exact saccade).
            fleet.submit_actions(idx, btn[idx], gdx[idx], gdy[idx], snap[idx])
            # gaze_dx/gaze_dy round-trip through shm EXACTLY (float32, no quant).
            for i in idx:
                ii = int(i)
                assert gaze_dx[ii] == np.float32(seen_gaze[ii][int(snap[ii])][0])
                assert gaze_dy[ii] == np.float32(seen_gaze[ii][int(snap[ii])][1])
            acted[idx] = snap[idx]
        elif bool((snap >= STEPS).all()):
            break
        else:
            time.sleep(2e-5)
        if time.time() > deadline:
            raise TimeoutError(f"saccade wave stalled at obs_seq={obs_seq[:n].tolist()}")
    fleet.end_wave()
    return recorded


def test_saccade_plumbing_end_to_end() -> None:
    ref = _mk_encoder(1, STEPS)  # parent-side reference (single env, reused)
    fleet = _new_fleet(ref.dim)
    try:
        assert ref.dim == 454
        obs0 = fleet.reset_all()
        assert obs0.shape == (N_ENVS, 454)

        recorded = _drive_and_capture(fleet)

        for env in range(N_ENVS):
            steps = recorded[env]
            assert set(steps) == set(range(STEPS + 1)), f"env {env} missing steps"

            # ---- parent-side byte-identity of the optical + proprio blocks ----
            # Reproduce each worker obs from the raw screen it saw + the same
            # submitted (dx, dy, button); the parent uses the identical encode().
            ref.reset(0)
            # reset obs (k=0): worker built it with last-button = NOOP, no saccade.
            obs_k0, scr_k0 = steps[0]
            rep0 = ref.encode(0, scr_k0, None, button=8)
            assert np.array_equal(rep0[OPTICAL], obs_k0[OPTICAL]), f"env {env} reset obs"

            gx_seen = set()
            fovea_first = None
            fovea_last = None
            for k in range(1, STEPS + 1):
                dx, dy = _cmd(env, k - 1)          # command submitted for obs k-1
                btn = _button(env, k - 1)
                ref.update_gaze(0, dx, dy)          # §3.3 integration (as the worker)
                obs_k, scr_k = steps[k]
                rep = ref.encode(0, scr_k, None, button=btn)
                assert np.array_equal(rep[OPTICAL], obs_k[OPTICAL]), (
                    f"env {env} step {k}: worker obs != parent reproduction"
                )

                # ---- proprio records the APPLIED (dx, dy) and the new (gx, gy) ----
                gy, gx = ref.gaze(0)
                p = obs_k[PROPRIO]
                assert p[2] == np.float32(dx), f"env {env} step {k} dx_prev"
                assert p[3] == np.float32(dy), f"env {env} step {k} dy_prev"
                assert p[0] == np.float32(gx / 160.0 * 2.0 - 1.0), "gx proprio"
                assert p[1] == np.float32(gy / 144.0 * 2.0 - 1.0), "gy proprio"

                gx_seen.add(round(gx))
                if k == 1:
                    fovea_first = obs_k[FOVEA].copy()
                if k == STEPS:
                    fovea_last = obs_k[FOVEA].copy()

            # ---- the fovea crop actually WALKED across the screen ----
            assert len(gx_seen) >= 4, f"env {env}: gaze visited too few columns {gx_seen}"
            assert max(gx_seen) - min(gx_seen) > 50, f"env {env}: gaze span too small"
            assert not np.array_equal(fovea_first, fovea_last), (
                f"env {env}: fovea did not change as the gaze swept"
            )

        # ---- per-env isolation: the two envs' gazes diverged (mirror sweeps) ----
        # Env 0 ends with gx pushed one way, env 1 the other; their final proprio
        # gaze coords must differ.
        assert recorded[0][STEPS][0][PROPRIO][0] != recorded[1][STEPS][0][PROPRIO][0]
    finally:
        fleet.close()
