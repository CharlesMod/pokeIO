"""RETINA-MODE obs tensor (Phase-1 retina-in-loop; spec docs/specs/retina-in-loop.md).

Covers the retina observation mode added to ``FovealEncoder`` — the workers ship
the retina encoder's PIXEL input (periph84 + fovea84) instead of the Phase-0
12x12 vectors, so the parent can batch-encode through the frozen retina:

  * fixed **14134-dim** float32 output with the exact contiguous block layout
    (periph84 / fovea84 / proprio / ram);
  * the SAME ``encode`` logic drives the parent and both worker paths, so three
    independently-driven encoders are **byte-identical** (checksum) — the
    cross-build-site invariant Phase-0 relies on, now for the pixel obs;
  * ``periph84`` / ``fovea84`` are byte-exactly ``retina.downscale`` /
    ``retina.crop_fovea`` on the same raw screen + the current (rounded) gaze;
  * moving the gaze moves ONLY the fovea block (periphery is whole-screen);
  * the ``proprio`` / ``ram`` blocks are IDENTICAL in value + semantics to the
    Phase-0 foveal path (efference copy + connect-protected tap overlay);
  * the Phase-0 foveal 454 obs still works unchanged (safe default).

The final test spawns a real ``AsyncFleet`` at ``obs_dim=14134`` to prove the
cross-PROCESS byte-identity (worker auto-selects retina mode from obs_dim). It
needs the Yellow ROM; skips cleanly if absent.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest

from pokeio.emu.fleet import AsyncFleet, FovealEncoder
from pokeio.evo import retina as R

# Committed foveal defaults (spec §2.1 / config.vision).
G = 12          # periph_grid
F = 48          # fovea_native_px
N_RAM = 8       # obs_ram_bytes
SIDE = 84       # retina.IN_SIDE

# Retina obs geometry.
N_PIX = SIDE * SIDE                       # 7056 per stream
RETINA_DIM = 2 * N_PIX + 14 + N_RAM       # 7056 + 7056 + 14 + 8 = 14134
PERIPH84 = slice(0, N_PIX)                # [0:7056]
FOVEA84 = slice(N_PIX, 2 * N_PIX)         # [7056:14112]
R_PROPRIO = slice(2 * N_PIX, 2 * N_PIX + 14)   # [14112:14126]
R_RAM = slice(2 * N_PIX + 14, RETINA_DIM)      # [14126:14134]

# Foveal (Phase-0) block boundaries, for the "identical semantics" cross-checks.
FOVEAL_DIM = 3 * G * G + 14 + N_RAM       # 454
F_PROPRIO = slice(432, 446)
F_RAM = slice(446, 454)


def _screen(seed: int) -> np.ndarray:
    """A DMG-like 4-shade (144,160) uint8 frame."""
    rs = np.random.RandomState(seed)
    return (rs.randint(0, 4, size=(144, 160)) * 85).astype(np.uint8)


def _wram(seed: int) -> np.ndarray:
    return np.random.RandomState(seed + 7).randint(0, 256, size=8192).astype(np.uint8)


def _mk_retina(**kw) -> FovealEncoder:
    base = dict(periph_grid=G, fovea_native_px=F, n_ram=N_RAM,
                saccade_gain=32.0, saccade_every_k=1, episode_steps=100,
                mode="retina")
    base.update(kw)
    return FovealEncoder(1, **base)


def _mk_foveal(**kw) -> FovealEncoder:
    base = dict(periph_grid=G, fovea_native_px=F, n_ram=N_RAM,
                saccade_gain=32.0, saccade_every_k=1, episode_steps=100)
    base.update(kw)
    return FovealEncoder(1, **base)


def _gaze_px(enc: FovealEncoder) -> tuple[int, int]:
    """Reproduce the encoder's gaze->pixel rounding (int(floor(g+0.5)))."""
    gy, gx = enc.gaze(0)
    return int(np.floor(gy + 0.5)), int(np.floor(gx + 0.5))


# ---------------------------------------------------------------- dims + layout
def test_retina_dim_is_14134_and_blocks_are_contiguous() -> None:
    enc = _mk_retina()
    assert enc.dim == 14134 == RETINA_DIM
    assert enc.mode == "retina"
    enc.reset()
    v = enc.encode(0, _screen(0), _wram(0), button=4)
    assert v.shape == (14134,)
    assert v.dtype == np.float32
    # exact block boundaries
    assert PERIPH84.stop == FOVEA84.start == 7056
    assert FOVEA84.stop == R_PROPRIO.start == 14112
    assert R_PROPRIO.stop == R_RAM.start == 14126
    assert R_RAM.stop == 14134
    # pixel blocks in [0,1]; proprio in [-1,1]; ram in [0,1].
    assert 0.0 <= v[PERIPH84].min() and v[PERIPH84].max() <= 1.0
    assert 0.0 <= v[FOVEA84].min() and v[FOVEA84].max() <= 1.0
    assert -1.0 <= v[R_PROPRIO].min() and v[R_PROPRIO].max() <= 1.0
    assert 0.0 <= v[R_RAM].min() and v[R_RAM].max() <= 1.0


# ---------------------------------------------------------------- byte-identity
def _checksum(enc: FovealEncoder, screens, wrams, cmds, buttons) -> str:
    """Drive an encoder through a scripted trajectory; hash every vector."""
    h = hashlib.md5()
    enc.reset()
    h.update(enc.encode(0, screens[0], wrams[0], button=8).tobytes())  # reset obs
    for k in range(len(cmds)):
        enc.update_gaze(0, cmds[k][0], cmds[k][1])
        v = enc.encode(0, screens[k + 1], wrams[k + 1], button=buttons[k])
        h.update(v.tobytes())
    return h.hexdigest()


def test_parent_and_both_workers_byte_identical() -> None:
    """The parent and BOTH worker _emit paths call the SAME retina-mode encode
    with the same per-env state, so three independently-constructed encoders
    driven through an identical trajectory emit byte-identical vectors."""
    n = 12
    screens = [_screen(i) for i in range(n + 1)]
    wrams = [_wram(i) for i in range(n + 1)]
    rs = np.random.RandomState(99)
    cmds = [(float(rs.uniform(-1.5, 1.5)), float(rs.uniform(-1.5, 1.5))) for _ in range(n)]
    buttons = [int(rs.randint(0, 9)) for _ in range(n)]

    parent = _mk_retina()
    worker_a = _mk_retina()   # stands in for the barrier worker's encoder
    worker_b = _mk_retina()   # stands in for the async worker's encoder
    cs_p = _checksum(parent, screens, wrams, cmds, buttons)
    cs_a = _checksum(worker_a, screens, wrams, cmds, buttons)
    cs_b = _checksum(worker_b, screens, wrams, cmds, buttons)
    assert cs_p == cs_a == cs_b


def test_encode_is_a_pure_function_of_state() -> None:
    """Re-driving from a fresh reset reproduces the vector exactly (no drift)."""
    enc = _mk_retina()
    s, w = _screen(3), _wram(3)
    enc.reset()
    v1 = enc.encode(0, s, w, button=2)
    enc.reset()
    v2 = enc.encode(0, s, w, button=2)
    assert np.array_equal(v1, v2)


# ------------------------------------------------ pixels match retina helpers
def test_periph84_matches_retina_downscale() -> None:
    """periph84 is byte-exactly retina.downscale(raw_screen) (gaze-invariant)."""
    enc = _mk_retina()
    s = _screen(11)
    enc.reset()
    # Move the gaze; the periphery must NOT depend on it.
    for _ in range(5):
        enc.update_gaze(0, 1.0, -1.0)
    v = enc.encode(0, s, _wram(11), button=8)
    periph = v[PERIPH84].reshape(SIDE, SIDE)
    assert np.array_equal(periph, R.downscale(s, SIDE))


def test_fovea84_matches_crop_fovea_at_current_gaze() -> None:
    """fovea84 is byte-exactly retina.crop_fovea(raw_screen, *rounded_gaze)."""
    enc = _mk_retina()
    s = _screen(12)
    enc.reset()
    # centre gaze first.
    v0 = enc.encode(0, s, None, button=8)
    gy0, gx0 = _gaze_px(enc)
    assert np.array_equal(
        v0[FOVEA84].reshape(SIDE, SIDE), R.crop_fovea(s, gy0, gx0, F, SIDE))

    # then a real saccade, and re-check against the crop at the NEW gaze.
    enc.update_gaze(0, 1.2, -0.8)
    v1 = enc.encode(0, s, None, button=8)
    gy1, gx1 = _gaze_px(enc)
    assert np.array_equal(
        v1[FOVEA84].reshape(SIDE, SIDE), R.crop_fovea(s, gy1, gx1, F, SIDE))


def test_moving_gaze_moves_only_the_fovea_block() -> None:
    """Moving the fovea centre changes the fovea block but NOT the periphery."""
    s, w = _screen(8), _wram(8)
    still = _mk_retina(); still.reset(); still.encode(0, s, w, button=8)
    moved = _mk_retina(); moved.reset(); moved.encode(0, s, w, button=8)

    still.update_gaze(0, 0.0, 0.0)     # tanh(0)=0 -> gaze stays centred
    moved.update_gaze(0, 1.5, -1.5)    # a large real saccade

    v_still = still.encode(0, s, w, button=8)
    v_moved = moved.encode(0, s, w, button=8)

    assert np.array_equal(v_still[PERIPH84], v_moved[PERIPH84]), "periphery moved"
    assert not np.array_equal(v_still[FOVEA84], v_moved[FOVEA84]), "fovea did NOT move"
    # the proprio efference copy DID change (intended).
    assert v_still[R_PROPRIO][0] != v_moved[R_PROPRIO][0]


def test_gaze_clamps_to_fovea_safe_window() -> None:
    """Retina keeps the IDENTICAL gaze integration: gx in [24,136], gy in [24,120]."""
    enc = _mk_retina(saccade_gain=32.0)
    enc.reset()
    rs = np.random.RandomState(4)
    lo = np.array([1e9, 1e9]); hi = np.array([-1e9, -1e9])
    for _ in range(600):
        gy, gx = enc.update_gaze(0, float(rs.uniform(-4, 4)), float(rs.uniform(-4, 4)))
        lo = np.minimum(lo, [gy, gx]); hi = np.maximum(hi, [gy, gx])
    assert lo[1] == 24.0 and hi[1] == 136.0   # gx rails
    assert lo[0] == 24.0 and hi[0] == 120.0   # gy rails


# -------------------------------------- proprio/ram identical to foveal path
def test_proprio_and_ram_match_foveal_semantics() -> None:
    """The 14-d proprio + 8-d ram blocks are VALUE-identical to the Phase-0
    foveal path when both encoders are driven through the same state."""
    ret = _mk_retina()
    fov = _mk_foveal()
    assert fov.dim == 454
    n = 8
    screens = [_screen(i) for i in range(n + 1)]
    wrams = [_wram(i) for i in range(n + 1)]
    rs = np.random.RandomState(5)
    cmds = [(float(rs.uniform(-1.2, 1.2)), float(rs.uniform(-1.2, 1.2))) for _ in range(n)]
    buttons = [int(rs.randint(0, 9)) for _ in range(n)]

    ret.reset(); fov.reset()
    vr0 = ret.encode(0, screens[0], wrams[0], button=8)
    vf0 = fov.encode(0, screens[0], wrams[0], button=8)
    assert np.array_equal(vr0[R_PROPRIO], vf0[F_PROPRIO])
    assert np.array_equal(vr0[R_RAM], vf0[F_RAM])
    for k in range(n):
        ret.update_gaze(0, *cmds[k]); fov.update_gaze(0, *cmds[k])
        vr = ret.encode(0, screens[k + 1], wrams[k + 1], button=buttons[k])
        vf = fov.encode(0, screens[k + 1], wrams[k + 1], button=buttons[k])
        assert np.array_equal(vr[R_PROPRIO], vf[F_PROPRIO]), f"proprio step {k}"
        assert np.array_equal(vr[R_RAM], vf[F_RAM]), f"ram step {k}"


def test_ram_tap_overlay_matches_foveal() -> None:
    """set_taps overlays the SAME connect-protected bytes into the ram tail."""
    addrs = [0xC000, 0xC010, 0xD000, 0xDABC]
    ret = _mk_retina(); ret.set_taps(addrs)
    fov = _mk_foveal(); fov.set_taps(addrs)
    s, w = _screen(21), _wram(21)
    ret.reset(); fov.reset()
    vr = ret.encode(0, s, w, button=3)
    vf = fov.encode(0, s, w, button=3)
    assert np.array_equal(vr[R_RAM], vf[F_RAM])
    # and the overlay is really the mined bytes, not the blind stride sample.
    for j, a in enumerate(addrs):
        assert vr[R_RAM][j] == np.float32(w[a - 0xC000] / 255.0)


# ---------------------------------------------------------- foveal unchanged
def test_foveal_mode_454_still_works_unchanged() -> None:
    """Safe default: the Phase-0 foveal encoder is untouched (dim, motion=0.5)."""
    enc = _mk_foveal()
    assert enc.dim == 454 and enc.mode == "foveal"
    enc.reset()
    v = enc.encode(0, _screen(5), _wram(5), button=8)
    assert v.shape == (454,)
    # motion block resets to 0.5 (no previous frame) — the Phase-0 invariant.
    assert np.allclose(v[288:432], 0.5)


def test_default_mode_is_foveal() -> None:
    """No mode kwarg -> foveal 454 (the committed safe default)."""
    enc = FovealEncoder(1, periph_grid=G, fovea_native_px=F, n_ram=N_RAM)
    assert enc.mode == "foveal" and enc.dim == 454


# =====================================================================
# Cross-PROCESS byte-identity through a real AsyncFleet at obs_dim=14134.
# =====================================================================
ROM = "roms/pokemon_yellow.gb"
STATE = "roms/yellow_newgame.state"
STRIDE = 64
N_ENVS, STEPS = 2, 8

_ARCHIVE_KW = dict(
    screen_cells=(16, 14), screen_levels=4,
    wram_stride=STRIDE, wram_levels=16, wram_mask=None,
)
# optical+proprio block (everything but the ram tail, which needs live wram).
OPTICAL = slice(0, 2 * N_PIX + 14)   # [0:14126]


@pytest.mark.skipif(
    not (Path(ROM).exists() and Path(STATE).exists()),
    reason="ROM / reset-state not available",
)
def test_retina_fleet_cross_process_byte_identical() -> None:
    """A real AsyncFleet with obs_dim=14134 auto-selects retina mode in-worker;
    the worker obs [0:14126] reproduces byte-for-byte from a parent-side
    retina encoder driven with the same screens + submitted (dx,dy,button)."""
    ref = FovealEncoder(
        1, periph_grid=G, fovea_native_px=F, n_ram=N_RAM, saccade_gain=32.0,
        saccade_every_k=1, episode_steps=STEPS, mode="retina",
    )
    assert ref.dim == 14134
    fleet = AsyncFleet(
        n_envs=N_ENVS, obs_dim=14134, obs_res=G, obs_ram=N_RAM, rom_path=ROM,
        frame_skip=1, hold_frames=0, reset_state=STATE,
        archive_kwargs=dict(_ARCHIVE_KW), wram_stride=STRIDE, goexplore=False,
        envs_per_worker=1, periph_grid=G, fovea_native_px=F, saccade_gain=32.0,
        saccade_every_k=1, episode_steps=STEPS,
    )
    try:
        obs0 = fleet.reset_all()
        assert obs0.shape == (N_ENVS, 14134)

        def cmd(k):  # a deterministic sweep so the fovea walks
            return (0.8, 0.8) if k < STEPS // 2 else (-0.8, -0.8)

        obs_seq = fleet.arr["obs_seq"]
        obs_shm = fleet.arr["obs"]
        screens = fleet.screens
        rec: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # env0 only: k -> (obs,screen)
        seen: dict[int, tuple[float, float]] = {}
        acted = np.full(N_ENVS, -1, dtype=np.int64)

        fleet.begin_wave(N_ENVS, STEPS)
        deadline = time.time() + 120.0
        while True:
            snap = obs_seq[:N_ENVS].copy()
            k0 = int(snap[0])
            if k0 not in rec:
                rec[k0] = (obs_shm[0].copy(), screens[0].copy())
            ready = (snap > acted) & (snap < STEPS)
            if ready.any():
                idx = np.nonzero(ready)[0]
                gdx = np.zeros(N_ENVS, np.float32)
                gdy = np.zeros(N_ENVS, np.float32)
                btn = np.full(N_ENVS, 8, np.int32)   # NOOP: quiet game
                for i in idx:
                    ii = int(i); k = int(snap[ii])
                    gdx[ii], gdy[ii] = cmd(k)
                    if ii == 0:
                        seen[k] = (float(gdx[ii]), float(gdy[ii]))
                fleet.submit_actions(idx, btn[idx], gdx[idx], gdy[idx], snap[idx])
                acted[idx] = snap[idx]
            elif bool((snap >= STEPS).all()):
                break
            else:
                time.sleep(2e-5)
            if time.time() > deadline:
                raise TimeoutError(f"retina wave stalled at {obs_seq[:N_ENVS].tolist()}")
        fleet.end_wave()

        assert set(rec) == set(range(STEPS + 1)), "missing steps"
        # reproduce env0's optical+proprio obs from the raw screens it saw.
        ref.reset(0)
        obs_k0, scr_k0 = rec[0]
        rep0 = ref.encode(0, scr_k0, None, button=8)
        assert np.array_equal(rep0[OPTICAL], obs_k0[OPTICAL]), "reset obs mismatch"
        foveas = []
        for k in range(1, STEPS + 1):
            dx, dy = seen[k - 1]
            ref.update_gaze(0, dx, dy)
            obs_k, scr_k = rec[k]
            rep = ref.encode(0, scr_k, None, button=8)
            assert np.array_equal(rep[OPTICAL], obs_k[OPTICAL]), f"step {k} mismatch"
            foveas.append(obs_k[FOVEA84].copy())
        # sanity: the fovea actually walked as the gaze swept.
        assert not np.array_equal(foveas[0], foveas[-1]), "fovea never moved"
    finally:
        fleet.close()
