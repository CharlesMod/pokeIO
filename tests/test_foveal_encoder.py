"""FovealEncoder — active-vision obs tensor (spec §2, unit test #1).

Covers the observation half of the active-vision spine
(``docs/specs/active-vision-spine.md`` §2/§3.3), all without an emulator:

  * fixed **454-dim** float32 output with the exact contiguous block layout
    (periphery / fovea / motion / proprio / ram);
  * the SAME ``encode`` logic drives the parent and both worker paths, so two
    independently-driven encoders are **byte-identical** (checksum) — the
    invariant that keeps novelty/tap keys from desyncing across the 3 build
    sites;
  * motion resets to **0.5** after ``.reset()`` (no previous frame);
  * gaze clamps to ``[24,136] x [24,120]`` under an adversarial saccade sweep;
  * moving the gaze changes ONLY the fovea block among the optical blocks
    (periphery + motion are gaze-invariant whole-screen streams).

The cross-PROCESS byte-identity of these vectors (parent vs a real fleet worker)
is exercised in ``test_saccade_plumbing.py``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from pokeio.emu.fleet import FovealEncoder

# Committed defaults (spec §2.1 / config.vision).
G = 12          # periph_grid
F = 48          # fovea_native_px
N_RAM = 8       # obs_ram_bytes
DIM = 3 * G * G + 14 + N_RAM  # 432 + 14 + 8 = 454

# Named block boundaries (spec §2.1 table).
PERIPH = slice(0, 144)
FOVEA = slice(144, 288)
MOTION = slice(288, 432)
PROPRIO = slice(432, 446)
RAM = slice(446, 454)


def _screen(seed: int) -> np.ndarray:
    """A DMG-like 4-shade (144,160) uint8 frame."""
    rs = np.random.RandomState(seed)
    return (rs.randint(0, 4, size=(144, 160)) * 85).astype(np.uint8)


def _wram(seed: int) -> np.ndarray:
    return np.random.RandomState(seed + 7).randint(0, 256, size=8192).astype(np.uint8)


def _mk(**kw) -> FovealEncoder:
    base = dict(periph_grid=G, fovea_native_px=F, n_ram=N_RAM,
                saccade_gain=32.0, saccade_every_k=1, episode_steps=100)
    base.update(kw)
    return FovealEncoder(1, **base)


# ---------------------------------------------------------------- dims + layout
def test_dim_is_454_and_blocks_are_contiguous() -> None:
    enc = _mk()
    assert enc.dim == 454 == DIM
    enc.reset()
    v = enc.encode(0, _screen(0), _wram(0), button=4)
    assert v.shape == (454,)
    assert v.dtype == np.float32
    # exact block boundaries
    assert PERIPH.stop == FOVEA.start == 144
    assert FOVEA.stop == MOTION.start == 288
    assert MOTION.stop == PROPRIO.start == 432
    assert PROPRIO.stop == RAM.start == 446
    assert RAM.stop == 454
    # optical + proprio blocks are all bounded (grayscale in [0,1]; proprio in
    # [-1,1]); ram in [0,1].
    assert 0.0 <= v[PERIPH].min() and v[PERIPH].max() <= 1.0
    assert 0.0 <= v[FOVEA].min() and v[FOVEA].max() <= 1.0
    assert 0.0 <= v[MOTION].min() and v[MOTION].max() <= 1.0
    assert -1.0 <= v[PROPRIO].min() and v[PROPRIO].max() <= 1.0
    assert 0.0 <= v[RAM].min() and v[RAM].max() <= 1.0


def test_proprio_layout_gaze_and_button_onehot() -> None:
    enc = _mk()
    enc.reset()
    # reset gaze is screen centre (72,80) -> gx*2/W-1 = 0, gy*2/H-1 = 0.
    v = enc.encode(0, _screen(1), _wram(1), button=8)  # NOOP
    p = v[PROPRIO]
    assert p[0] == 0.0  # gx = 80 -> 80*2/160 - 1 = 0
    assert p[1] == 0.0  # gy = 72 -> 72*2/144 - 1 = 0
    assert p[2] == 0.0 and p[3] == 0.0  # no saccade yet (dx_prev, dy_prev)
    # NOOP is button id 8 -> proprio one-hot index 4+8 = 12.
    onehot = p[4:13]
    assert onehot.sum() == 1.0 and onehot[8] == 1.0  # NOOP slot
    assert p[13] == 0.0  # step_frac 0 at the reset obs

    # A directional button (up = id 0) lights proprio index 4.
    enc.reset()
    v2 = enc.encode(0, _screen(1), _wram(1), button=0)
    assert v2[PROPRIO][4] == 1.0 and v2[PROPRIO][4:13].sum() == 1.0


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


def test_parent_and_worker_encoders_are_byte_identical() -> None:
    """The parent and BOTH worker _emit paths call the SAME FovealEncoder.encode
    with the same per-env state, so two independently-constructed encoders driven
    through an identical trajectory emit byte-identical vectors (checksum)."""
    n = 12
    screens = [_screen(i) for i in range(n + 1)]
    wrams = [_wram(i) for i in range(n + 1)]
    rs = np.random.RandomState(99)
    cmds = [(float(rs.uniform(-1.5, 1.5)), float(rs.uniform(-1.5, 1.5))) for _ in range(n)]
    buttons = [int(rs.randint(0, 9)) for _ in range(n)]

    parent = _mk()
    worker_a = _mk()   # stands in for the barrier worker's encoder
    worker_b = _mk()   # stands in for the async worker's encoder
    cs_p = _checksum(parent, screens, wrams, cmds, buttons)
    cs_a = _checksum(worker_a, screens, wrams, cmds, buttons)
    cs_b = _checksum(worker_b, screens, wrams, cmds, buttons)
    assert cs_p == cs_a == cs_b


def test_encode_is_a_pure_function_of_state() -> None:
    """Re-driving the same encoder from a fresh reset reproduces the vector
    exactly (no hidden drift) — the property the shm build sites rely on."""
    enc = _mk()
    s, w = _screen(3), _wram(3)
    enc.reset()
    v1 = enc.encode(0, s, w, button=2)
    enc.reset()
    v2 = enc.encode(0, s, w, button=2)
    assert np.array_equal(v1, v2)


# ---------------------------------------------------------------- motion + reset
def test_motion_is_half_after_reset() -> None:
    enc = _mk()
    enc.reset()
    v = enc.encode(0, _screen(5), _wram(5), button=8)
    assert np.allclose(v[MOTION], 0.5)  # no previous periphery -> 0.5 everywhere
    # A second, DIFFERENT frame yields a real (non-constant) motion sheet.
    enc.update_gaze(0, 0.0, 0.0)
    v2 = enc.encode(0, _screen(6), _wram(6), button=8)
    assert not np.allclose(v2[MOTION], 0.5)
    # And reset() clears it back to 0.5.
    enc.reset()
    v3 = enc.encode(0, _screen(7), _wram(7), button=8)
    assert np.allclose(v3[MOTION], 0.5)


def test_reset_returns_gaze_to_screen_centre() -> None:
    enc = _mk()
    enc.reset()
    for _ in range(20):
        enc.update_gaze(0, 1.0, -1.0)  # drive the gaze far off-centre
    gy, gx = enc.gaze(0)
    assert (gy, gx) != (72.0, 80.0)
    enc.reset()
    assert enc.gaze(0) == (72.0, 80.0)  # spec: reset gaze to (gy,gx)=(72,80)


# ---------------------------------------------------------------- gaze clamp
def test_gaze_clamps_to_fovea_safe_window() -> None:
    """gx in [F/2, W-F/2] = [24,136]; gy in [F/2, H-F/2] = [24,120] (spec §3.3),
    so the FxF fovea window never overhangs the screen."""
    enc = _mk(saccade_gain=32.0)
    enc.reset()
    rs = np.random.RandomState(4)
    gx_lo = gy_lo = 1e9
    gx_hi = gy_hi = -1e9
    for _ in range(600):
        dx, dy = float(rs.uniform(-4, 4)), float(rs.uniform(-4, 4))
        gy, gx = enc.update_gaze(0, dx, dy)
        gx_lo, gx_hi = min(gx_lo, gx), max(gx_hi, gx)
        gy_lo, gy_hi = min(gy_lo, gy), max(gy_hi, gy)
    assert 24.0 <= gx_lo and gx_hi <= 136.0
    assert 24.0 <= gy_lo and gy_hi <= 120.0
    # The adversarial sweep actually reached both clamp rails.
    assert gx_hi == 136.0 and gx_lo == 24.0
    assert gy_hi == 120.0 and gy_lo == 24.0


# ---------------------------------------------------------------- fovea isolation
def test_moving_gaze_changes_only_the_fovea_block() -> None:
    """Among the OPTICAL blocks, moving the fovea centre changes only the fovea
    sub-vector: periphery + motion are whole-screen and gaze-invariant.

    (The proprio block necessarily reflects the new gaze/saccade, so it is not
    part of the "optical" comparison — that is the efference copy, by design.)
    """
    s, w = _screen(8), _wram(8)

    still = _mk()
    still.reset()
    still.encode(0, s, w, button=8)          # prime previous periphery
    moved = _mk()
    moved.reset()
    moved.encode(0, s, w, button=8)          # identical priming

    # Advance both one step so step parity holds; only `moved` actually shifts.
    still.update_gaze(0, 0.0, 0.0)           # tanh(0)=0 -> gaze stays centred
    moved.update_gaze(0, 1.5, -1.5)          # a large real saccade

    v_still = still.encode(0, s, w, button=8)
    v_moved = moved.encode(0, s, w, button=8)

    assert np.array_equal(v_still[PERIPH], v_moved[PERIPH]), "periphery moved"
    assert np.array_equal(v_still[MOTION], v_moved[MOTION]), "motion moved"
    assert not np.array_equal(v_still[FOVEA], v_moved[FOVEA]), "fovea did NOT move"
    # Sanity: proprio gaze coords DID change (that is the intended efference copy).
    assert v_still[PROPRIO][0] != v_moved[PROPRIO][0]
