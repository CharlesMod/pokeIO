"""Unit tests for the Phase 1 vision pipeline: ObsBuilder + VecFleet obs spec.

These are pure-CPU and do not require the emulator (they feed synthetic frames),
except the fleet smoke test which boots a couple of real VisionEnvs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeio.config import Config
from pokeio.vision.preprocess import ObsBuilder

ROM = Config().emu.rom_path
STATE = Config().emu.reset_state


def _dmg_frame(seed: int = 0) -> np.ndarray:
    """A synthetic 144x160 uint8 frame using the 4 canonical DMG shades."""
    rng = np.random.default_rng(seed)
    shades = np.array([0, 85, 170, 255], dtype=np.uint8)
    return rng.choice(shades, size=(144, 160)).astype(np.uint8)


# --------------------------------------------------------------------------
# ObsBuilder — shapes / dtypes / ranges
# --------------------------------------------------------------------------
def test_obs_shapes_and_dtypes():
    b = ObsBuilder(Config())
    obs = b.build(_dmg_frame())
    assert set(obs.keys()) == {"coarse", "fovea", "motion", "ram_aux"}
    assert obs["coarse"].shape == (32, 32)
    assert obs["fovea"].shape == (64, 64)
    assert obs["motion"].shape == (32, 32)
    assert obs["ram_aux"].shape == (32,)
    for v in obs.values():
        assert v.dtype == np.float32


def test_obs_ranges_in_unit_interval():
    b = ObsBuilder(Config())
    obs = b.build(_dmg_frame(1))
    for k, v in obs.items():
        assert v.min() >= 0.0 - 1e-6, k
        assert v.max() <= 1.0 + 1e-6, k


def test_shapes_property_matches_build():
    b = ObsBuilder(Config())
    obs = b.build(_dmg_frame())
    for k, shp in b.shapes.items():
        assert obs[k].shape == shp


# --------------------------------------------------------------------------
# Grayscale normalization
# --------------------------------------------------------------------------
def test_four_shades_map_to_quarters():
    b = ObsBuilder(Config())
    norm = b.normalize_shades(_dmg_frame())
    assert np.allclose(np.unique(norm), [0.0, 1 / 3, 2 / 3, 1.0], atol=1e-6)


def test_more_than_four_values_fall_back_to_255():
    b = ObsBuilder(Config())
    # 8 distinct values -> not a clean DMG buffer -> /255 fallback.
    frame = np.tile(np.arange(0, 160, 20, dtype=np.uint8), (144, 1))
    norm = b.normalize_shades(frame)
    assert np.allclose(norm, frame.astype(np.float32) / 255.0)


# --------------------------------------------------------------------------
# Motion channel
# --------------------------------------------------------------------------
def test_motion_is_half_when_no_previous_frame():
    b = ObsBuilder(Config())
    obs = b.build(_dmg_frame())
    # No prior frame -> motion is the neutral 0.5 everywhere.
    assert np.allclose(obs["motion"], 0.5)


def test_motion_near_zero_diff_for_identical_frames():
    b = ObsBuilder(Config())
    frame = _dmg_frame(3)
    b.build(frame)  # prime the previous-frame store
    obs = b.build(frame)  # identical frame -> diff 0 -> (0+1)/2 = 0.5
    assert np.allclose(obs["motion"], 0.5, atol=1e-6)
    assert float(obs["motion"].std()) < 1e-6


def test_motion_changes_for_different_frames():
    b = ObsBuilder(Config())
    b.build(_dmg_frame(4))
    obs = b.build(_dmg_frame(5))
    # Different frames -> motion departs from the neutral 0.5.
    assert float(obs["motion"].std()) > 1e-3


# --------------------------------------------------------------------------
# ram_aux — separate field, never mixed into sheets
# --------------------------------------------------------------------------
def test_ram_aux_zeros_by_default():
    b = ObsBuilder(Config())
    obs = b.build(_dmg_frame())
    assert obs["ram_aux"].shape == (32,)
    assert np.all(obs["ram_aux"] == 0.0)


def test_ram_aux_normalizes_raw_bytes():
    b = ObsBuilder(Config())
    ram = np.arange(8192, dtype=np.uint8)  # raw WRAM-like read
    obs = b.build(_dmg_frame(), ram_vector=ram)
    assert obs["ram_aux"].shape == (32,)
    assert obs["ram_aux"].max() <= 1.0 + 1e-6
    assert obs["ram_aux"].min() >= 0.0


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_build_is_deterministic():
    frame = _dmg_frame(7)
    a = ObsBuilder(Config())
    c = ObsBuilder(Config())
    oa = a.build(frame)
    oc = c.build(frame)
    for k in oa:
        assert np.array_equal(oa[k], oc[k]), k


# --------------------------------------------------------------------------
# flatten
# --------------------------------------------------------------------------
def test_flatten_length_and_dtype():
    b = ObsBuilder(Config())
    obs = b.build(_dmg_frame())
    flat = b.flatten(obs)
    assert flat.dtype == np.float32
    assert flat.ndim == 1
    # coarse 32*32 + fovea 64*64 + motion 32*32 + ram_aux 32
    assert flat.shape[0] == 32 * 32 + 64 * 64 + 32 * 32 + 32
    assert flat.shape[0] == b.flat_dim
    assert b.optical_dim == 32 * 32 + 64 * 64 + 32 * 32


def test_flatten_order_matches_fields():
    b = ObsBuilder(Config())
    obs = b.build(_dmg_frame())
    flat = b.flatten(obs)
    n_coarse = 32 * 32
    assert np.array_equal(flat[:n_coarse], obs["coarse"].ravel())


# --------------------------------------------------------------------------
# Config-driven sizing
# --------------------------------------------------------------------------
def test_sizes_are_config_driven():
    cfg = Config()
    cfg.vision.coarse_size = 16
    cfg.vision.fovea_size = 48
    cfg.vision.ram_aux_dim = 8
    b = ObsBuilder(cfg)
    obs = b.build(_dmg_frame())
    assert obs["coarse"].shape == (16, 16)
    assert obs["fovea"].shape == (48, 48)
    assert obs["ram_aux"].shape == (8,)


# --------------------------------------------------------------------------
# Fleet smoke test (boots real emulators; skipped if ROM/state missing)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (Config().emu.rom_path and Config().emu.reset_state),
    reason="rom/state paths not configured",
)
def test_vecfleet_batched_obs_smoke():
    import os

    if not (os.path.exists(ROM) and os.path.exists(STATE)):
        pytest.skip("rom or reset state file missing")

    from pokeio.emu.fleet import VecFleet

    n = 2
    with VecFleet(n, config=Config(), state_path=STATE) as fleet:
        obs = fleet.reset_all()
        assert obs["coarse"].shape == (n, 32, 32)
        assert obs["fovea"].shape == (n, 64, 64)
        assert obs["ram_aux"].shape == (n, 32)
        obs, dones, infos = fleet.step_all([0, 1])
        assert obs["motion"].shape == (n, 32, 32)
        assert dones.shape == (n,)
        assert len(infos) == n
        for v in obs.values():
            assert v.dtype == np.float32
