"""Optical front-end v2 — rendering decode (task #12, spec §8.7 / §1a / §2).

Unit-tests the ``pokeio.train.live`` obs decoders that feed the dashboard "what the
AI sees" panel, covering the ONE thing that can silently break when a run turns the
v2 knobs on: the flat foveal obs is decoded at the ENCODER'S real offsets, not the
legacy uniform grid.

  * **legacy == pre-change:** for a legacy encoder (``fovea_grid=0``, memory off) the
    generalized :func:`_optical_blocks` / :func:`_gaze_from_obs` return blocks + gaze
    byte-identical to the legacy uniform-grid :func:`_block2d` / :func:`_gaze_from_proprio`
    reference (the decode-identity invariant that keeps live1 rendering unchanged);
  * **v2 shapes:** for a sharp-fovea + memory + reflex encoder (FG=32, M=24) the decode
    returns correctly-shaped periph(G) / fovea(FG) / motion(G) / buffer(M) / staleness(M)
    blocks and the gaze/reflex payloads WITHOUT index errors — the legacy 3*G^2 proprio
    offset would land inside the buffer instead;
  * **round-trips:** decoded buffer/staleness equal the encoder's own maps, and the
    recovered gaze/reflex target match the encoder state that produced the obs;
  * **payload wiring:** ``_memplus`` / ``_memplus_from_obs`` attach buffer/staleness +
    reflex only when the run has them (legacy payloads gain no new keys).

Torch-free encoder built by hand; the ``live`` module itself pulls torch (CPU only).
"""

from __future__ import annotations

import numpy as np

from pokeio.emu.fleet import FovealEncoder
from pokeio.train import live

G = 12          # periph_grid
F = 48          # fovea_native_px
N_RAM = 8       # obs_ram_bytes


def _screen(seed: int) -> np.ndarray:
    """A DMG-like 4-shade (144,160) uint8 frame."""
    return (np.random.RandomState(seed).randint(0, 4, (144, 160)) * 85).astype(np.uint8)


def _wram(seed: int) -> np.ndarray:
    return np.random.RandomState(seed + 7).randint(0, 256, 8192).astype(np.uint8)


def _legacy(**kw) -> FovealEncoder:
    base = dict(periph_grid=G, fovea_native_px=F, n_ram=N_RAM, saccade_gain=32.0,
                saccade_every_k=1, episode_steps=200)
    base.update(kw)
    return FovealEncoder(1, **base)


def _v2(**kw) -> FovealEncoder:
    """Sharp fovea (FG=32) + trans-saccadic memory (M=24) + reflex gaze — the
    successor-run obs the wall must decode without hardcoded G/3*G^2 offsets."""
    base = dict(periph_grid=G, fovea_native_px=F, fovea_grid=32, n_ram=N_RAM,
                saccade_gain=32.0, saccade_every_k=1, episode_steps=200,
                foveal_memory=True, mem_grid=24, mem_stale_warmup=6, reflex_gaze=True)
    base.update(kw)
    return FovealEncoder(1, **base)


def _drive(enc: FovealEncoder, steps: int = 6, seed0: int = 0):
    """Run a scripted saccade/motion trajectory; return the LAST obs vector."""
    enc.reset()
    v = enc.encode(0, _screen(seed0), _wram(seed0), button=8)
    for k in range(steps):
        enc.update_gaze(0, 0.5, 0.3)
        v = enc.encode(0, _screen(seed0 + k + 1), _wram(seed0 + k + 1), button=2)
    return v


# --------------------------------------------------------- legacy == pre-change
def test_legacy_optical_blocks_byte_identical_to_block2d() -> None:
    """A legacy encoder decodes periph/fovea/motion bit-for-bit like the pre-change
    uniform-grid ``_block2d`` — the decode-identity invariant for the live1 baseline."""
    enc = _legacy()
    v = _drive(enc)
    b = live._optical_blocks(v, enc)
    assert np.array_equal(b["periph"], live._block2d(v, G, "periph"))
    assert np.array_equal(b["fovea"], live._block2d(v, G, "fovea"))
    assert np.array_equal(b["motion"], live._block2d(v, G, "motion"))
    # legacy has no memory blocks and no reflex target.
    assert "buffer" not in b and "stale" not in b
    assert live._reflex_from_obs(v, enc) is None


def test_legacy_gaze_recovery_matches_legacy_offset() -> None:
    """``_gaze_from_obs`` == the legacy ``_gaze_from_proprio`` (3*G^2 offset) for a
    legacy encoder, so where-it-looked is decoded byte-identically to today."""
    enc = _legacy()
    v = _drive(enc)
    gy, gx = live._gaze_from_obs(v, enc)
    gy0, gx0 = live._gaze_from_proprio(v, G, enc.H, enc.W)
    assert gy == gy0 and gx == gx0
    # and it is the actual gaze the fovea was cropped at (up to float32 storage).
    assert np.allclose((gy, gx), enc.gaze(0), atol=1e-3)


def test_legacy_block_shapes() -> None:
    """Legacy blocks are GxG (fovea included, since FG==G)."""
    enc = _legacy()
    b = live._optical_blocks(_drive(enc), enc)
    assert b["periph"].shape == (G, G)
    assert b["fovea"].shape == (G, G)
    assert b["motion"].shape == (G, G)


# ----------------------------------------------------------------- v2 shapes
def test_v2_optical_blocks_shapes_no_index_error() -> None:
    """A sharp-fovea + memory encoder decodes to periph(G) / fovea(FG) / motion(G) /
    buffer(M) / staleness(M) — the wider fovea + inserted memory blocks are read at
    the encoder's real offsets, no index error."""
    enc = _v2()
    v = _drive(enc)
    b = live._optical_blocks(v, enc)
    assert b["periph"].shape == (G, G)
    assert b["fovea"].shape == (32, 32)             # sharp fovea, NOT GxG
    assert b["motion"].shape == (G, G)
    assert b["buffer"].shape == (24, 24)
    assert b["stale"].shape == (24, 24)


def test_v2_buffer_and_staleness_round_trip_to_encoder_maps() -> None:
    """The decoded buffer/staleness blocks are exactly the encoder's own per-env
    maps (the accessors the showcase reads live)."""
    enc = _v2()
    v = _drive(enc)
    b = live._optical_blocks(v, enc)
    assert np.array_equal(b["buffer"], enc.mem_buffer(0))
    assert np.array_equal(b["stale"], enc.mem_staleness(0))


def test_v2_gaze_and_reflex_recovery() -> None:
    """Gaze + reflex target survive the wider-fovea/memory offset shift: the legacy
    3*G^2 proprio offset would read inside the buffer, so this is the load-bearing fix."""
    enc = _v2()
    v = _drive(enc)
    gy, gx = live._gaze_from_obs(v, enc)
    assert np.allclose((gy, gx), enc.gaze(0), atol=1e-3)
    rt = live._reflex_from_obs(v, enc)
    assert rt is not None
    assert -1.0 <= rt[0] <= 1.0 and -1.0 <= rt[1] <= 1.0
    assert np.allclose(rt, enc.reflex_target(0), atol=1e-4)   # == proprio[14:16]


def test_reflex_off_returns_none() -> None:
    """Memory on but reflex off: buffer/staleness decode, reflex target is None."""
    enc = _v2(reflex_gaze=False)
    v = _drive(enc)
    b = live._optical_blocks(v, enc)
    assert "buffer" in b and "stale" in b
    assert live._reflex_from_obs(v, enc) is None


# ------------------------------------------------------------- payload helpers
def test_reflex_payload_shape_and_none() -> None:
    """``_reflex_payload`` maps [-1,1] -> normalised [0,1] frame fraction; None-safe."""
    p = live._reflex_payload(-1.0, 1.0)
    assert p == {"tx": -1.0, "ty": 1.0, "x01": 0.0, "y01": 1.0}
    assert live._reflex_payload(None, None) is None


def test_memplus_from_obs_attaches_blocks_only_when_present() -> None:
    """``_memplus_from_obs`` attaches buffer_res/buffer_b64/stale_b64 + reflex on a v2
    obs, and adds NOTHING on a legacy obs (legacy payloads gain no keys)."""
    v2, leg = _v2(), _legacy()
    vv, vl = _drive(v2), _drive(leg)

    out_v2: dict = {}
    live._memplus_from_obs(out_v2, vv, v2, v2.M)
    assert {"buffer_res", "buffer_b64", "stale_b64", "reflex"} <= set(out_v2)
    assert out_v2["buffer_res"] == 24
    # buffer_b64 decodes to M*M grayscale bytes.
    import base64
    raw = base64.b64decode(out_v2["buffer_b64"])
    assert len(raw) == 24 * 24

    out_leg: dict = {}
    live._memplus_from_obs(out_leg, vl, leg, leg.M)
    assert out_leg == {}


def test_memplus_live_accessors_match_obs_decode() -> None:
    """The showcase ``_memplus`` (live accessors) and ``_memplus_from_obs`` (flat obs)
    produce the SAME buffer/staleness base64 for a matched encoder+obs."""
    enc = _v2()
    v = _drive(enc)
    a: dict = {}
    b: dict = {}
    live._memplus(a, enc, enc.M)                 # from encoder accessors
    live._memplus_from_obs(b, v, enc, enc.M)     # from the flat obs
    assert a["buffer_b64"] == b["buffer_b64"]
    assert a["stale_b64"] == b["stale_b64"]
