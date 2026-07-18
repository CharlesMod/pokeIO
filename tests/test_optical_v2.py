"""Optical front-end v2 — Increment A: sharp fovea + sensor seam.

Covers task #5 (SHARP FOVEA) + task #15 (sensor seam) of
``docs/specs/optical-frontend-v2.md`` §2/§9a, entirely without an emulator:

  * **v2-off == legacy:** ``fovea_grid`` of 0 OR ``periph_grid`` reproduces the
    pre-change encoder byte-for-byte (same ``dim`` and same obs values as the old
    F->G downsample fovea) — the determinism invariant (obs change only; zero rng);
  * **sharpness:** the fovea now resamples the FxF crop to FG x FG (px/cell =
    F/FG), decoupled from the coarse GxG periphery; ``dim`` follows the
    ``2*G^2 + FG^2 + 14 + n_ram`` formula for several (G, FG) pairs;
  * **actually sharper:** on a high-frequency crop a native (FG=F) fovea keeps far
    more variance/detail than the legacy FG=G downsample (the quantitative fix for
    the "4 px/cell smudge" defect);
  * **sensor seam (#15):** ``screen_h/screen_w`` and ``channels`` are params (an
    (H,W,C) seam), C=1 grayscale now.

The legacy default (G=FG=12, F=48, n_ram=8) is the 454-d obs the ``live1``
baseline trains on; the sharp fovea grows ``n_in`` (=> a fresh run).
"""

from __future__ import annotations

import hashlib

import numpy as np

from pokeio.emu.fleet import FovealEncoder, _area_matrix

# Committed foveal defaults (config.vision / spec §2.1).
G = 12          # periph_grid
F = 48          # fovea_native_px
N_RAM = 8       # obs_ram_bytes
LEGACY_DIM = 2 * G * G + G * G + 14 + N_RAM  # == 3*G^2+14+n_ram == 454


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


def _fovea_dim(g: int, fg: int, n_ram: int = N_RAM) -> int:
    """periph(G^2) + fovea(FG^2) + motion(G^2) + proprio(14) + ram(n_ram)."""
    return 2 * g * g + fg * fg + 14 + n_ram


# --------------------------------------------------------------- v2-off == legacy
def _checksum(enc: FovealEncoder, screens, wrams, cmds, buttons) -> str:
    """Drive an encoder through a scripted trajectory; hash every vector."""
    h = hashlib.md5()
    enc.reset()
    h.update(enc.encode(0, screens[0], wrams[0], button=8).tobytes())  # reset obs
    for k in range(len(cmds)):
        enc.update_gaze(0, cmds[k][0], cmds[k][1])
        h.update(enc.encode(0, screens[k + 1], wrams[k + 1], button=buttons[k]).tobytes())
    return h.hexdigest()


def test_v2_off_dim_is_legacy_454() -> None:
    """fovea_grid 0 (default) and == periph_grid both give the legacy 454-d obs."""
    assert LEGACY_DIM == 454
    assert _mk().dim == 454                      # fovea_grid defaults to 0
    assert _mk(fovea_grid=0).dim == 454
    assert _mk(fovea_grid=G).dim == 454          # explicit FG==G is the same obs
    assert _mk().FG == G                         # 0 resolves to periph_grid
    assert _mk(fovea_grid=G).FG == G


def test_v2_off_is_byte_identical_across_a_trajectory() -> None:
    """fovea_grid in {0, periph_grid} emit byte-identical vectors to each other
    over a scripted saccade/motion/proprio/ram trajectory (obs change only, zero
    rng — the determinism invariant that keeps v2-off == the live1 baseline)."""
    n = 12
    screens = [_screen(i) for i in range(n + 1)]
    wrams = [_wram(i) for i in range(n + 1)]
    rs = np.random.RandomState(99)
    cmds = [(float(rs.uniform(-1.5, 1.5)), float(rs.uniform(-1.5, 1.5))) for _ in range(n)]
    buttons = [int(rs.randint(0, 9)) for _ in range(n)]

    cs_default = _checksum(_mk(), screens, wrams, cmds, buttons)
    cs_zero = _checksum(_mk(fovea_grid=0), screens, wrams, cmds, buttons)
    cs_eqG = _checksum(_mk(fovea_grid=G), screens, wrams, cmds, buttons)
    assert cs_default == cs_zero == cs_eqG


def test_v2_off_fovea_block_equals_manual_legacy_resample() -> None:
    """The strongest v2-off proof: with FG==G the fovea block is bit-for-bit the
    OLD F->G area-resample of the same native crop (`_area_matrix(F, G)`)."""
    enc = _mk(fovea_grid=0)
    enc.reset()
    s, w = _screen(5), _wram(5)
    v = enc.encode(0, s, w, button=4)
    fovea_block = v[enc._o_fovea:enc._o_motion].reshape(G, G)

    # Recompute the fovea the pre-change way, straight from the encoder crop.
    normd = enc._shade.normalize_shades(s).astype(np.float64)
    crop = enc._crop(normd, *enc.gaze(0))          # native FxF crop @ gaze
    row, col = _area_matrix(F, G), _area_matrix(F, G).T
    legacy_fovea = (row @ crop @ col).astype(np.float32)
    assert np.array_equal(fovea_block, legacy_fovea)


# ------------------------------------------------------------------- sharpness
def test_fovea_px_per_cell_and_dim_formula() -> None:
    """px/cell = F/FG; obs dim = 2*G^2 + FG^2 + 14 + n_ram for several pairs."""
    # Legacy: 48px crop -> 12 cells = 4 px/cell (the unreadable smudge).
    e_legacy = _mk(fovea_native_px=48, fovea_grid=12)
    assert e_legacy.F / e_legacy.FG == 4.0
    assert e_legacy.dim == _fovea_dim(12, 12) == 454

    # Recommended successor: 32px crop -> 32 cells = 1 px/cell (a GB tile legible).
    e_sharp = _mk(fovea_native_px=32, fovea_grid=32)
    assert e_sharp.F / e_sharp.FG == 1.0
    assert e_sharp.dim == _fovea_dim(12, 32) == 1334

    # Wider window at 2 px/cell.
    e_mid = _mk(fovea_native_px=48, fovea_grid=24)
    assert e_mid.F / e_mid.FG == 2.0
    assert e_mid.dim == _fovea_dim(12, 24) == 886

    # The fovea block is exactly FG^2 wide and the offsets are contiguous.
    for g, fg in ((12, 12), (12, 24), (12, 32), (16, 40)):
        e = _mk(periph_grid=g, fovea_native_px=48, fovea_grid=fg)
        assert e.n_periph == g * g
        assert e.n_fovea == fg * fg
        assert e.n_motion == g * g          # periphery/motion stay coarse at GxG
        assert e.dim == _fovea_dim(g, fg)
        assert (e._o_periph, e._o_fovea) == (0, g * g)
        assert e._o_motion == g * g + fg * fg
        assert e._o_proprio == 2 * g * g + fg * fg
        assert e._o_ram == 2 * g * g + fg * fg + 14
        # Public [AC] salience slices track the motion block.
        assert e.o_motion == e._o_motion and e.o_proprio == e._o_proprio


def test_encode_emits_the_declared_dim_for_a_sharp_fovea() -> None:
    """A sharp-fovea encoder actually produces .dim floats, all blocks bounded."""
    e = _mk(fovea_native_px=32, fovea_grid=32)
    e.reset()
    v = e.encode(0, _screen(3), _wram(3), button=4)
    assert v.shape == (e.dim,) == (1334,)
    assert v.dtype == np.float32
    # optical + ram blocks in [0,1]; proprio in [-1,1].
    assert 0.0 <= v[:e._o_proprio].min() and v[:e._o_proprio].max() <= 1.0
    assert -1.0 <= v[e._o_proprio:e._o_ram].min()
    assert v[e._o_proprio:e._o_ram].max() <= 1.0


# --------------------------------------------------------------- actually sharper
def _hi_freq_screen() -> np.ndarray:
    """A 2px checkerboard (the highest spatial frequency a GB screen carries):
    downsampling it to a coarse grid averages toward mid-gray (variance -> 0),
    while a native fovea preserves the full black/white contrast."""
    r = np.arange(144)[:, None]
    c = np.arange(160)[None, :]
    board = (((r // 2) + (c // 2)) % 2).astype(np.uint8) * 255  # 0 / 255 checker
    return board


def test_sharp_fovea_preserves_more_variance_than_the_legacy_downsample() -> None:
    """On a high-frequency crop the native (FG=F) fovea keeps far more detail
    (variance) than the legacy FG=G downsample — the quantitative fix for the
    4 px/cell smudge. Periphery is identical (both coarse at GxG)."""
    screen = _hi_freq_screen()

    coarse = _mk(fovea_native_px=48, fovea_grid=12)   # legacy 4 px/cell
    sharp = _mk(fovea_native_px=48, fovea_grid=48)    # native 1 px/cell
    coarse.reset()
    sharp.reset()
    vc = coarse.encode(0, screen, None, button=8)
    vs = sharp.encode(0, screen, None, button=8)

    fov_coarse = vc[coarse._o_fovea:coarse._o_motion]
    fov_sharp = vs[sharp._o_fovea:sharp._o_motion]

    # The 12x12 fovea averages ~16 checker pixels/cell -> nearly flat mid-gray.
    assert fov_coarse.var() < 1e-3
    # The native 48x48 fovea keeps the full checker contrast -> large variance.
    assert fov_sharp.var() > 0.2
    # ...orders of magnitude more detail retained.
    assert fov_sharp.var() > 100.0 * fov_coarse.var()

    # The periphery block (coarse for BOTH) is byte-identical: only the fovea
    # sharpened, the low-acuity periphery is unchanged (biomimetic).
    assert np.array_equal(vc[coarse._o_periph:coarse._o_fovea],
                          vs[sharp._o_periph:sharp._o_fovea])


# ------------------------------------------------------------------- sensor seam
def test_sensor_seam_channels_and_resolution_are_params() -> None:
    """#15/§9a: the encoder abstracts over an (H,W,C) sensor — screen_h/screen_w
    and channels are constructor params (C=1 grayscale now; the seam is present,
    not the color machinery)."""
    e = _mk(channels=1)
    assert e.C == 1
    assert e.H == 144 and e.W == 160          # GB defaults, but overridable
    # A different-resolution "sensor" resamples through the SAME grids (no GB
    # hardcode beyond the defaults): periph/fovea grids are resolution-agnostic.
    e2 = FovealEncoder(1, periph_grid=G, fovea_native_px=F, fovea_grid=24,
                       n_ram=N_RAM, screen_h=120, screen_w=200, episode_steps=50)
    assert (e2.H, e2.W) == (120, 200)
    assert e2.dim == _fovea_dim(G, 24)         # dim independent of input resolution
    e2.reset()
    v = e2.encode(0, np.zeros((120, 200), np.uint8), None, button=8)
    assert v.shape == (e2.dim,)
