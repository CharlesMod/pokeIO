"""Optical front-end v2 — Increment B: trans-saccadic foveal memory (the keystone).

Covers task #13 (persistent stamp buffer) + task #14 (peripheral-change
invalidation & staleness) of ``docs/specs/optical-frontend-v2.md`` §1a, entirely
without an emulator:

  * **v2-B-off == Increment A:** ``foveal_memory=False`` (default) reproduces the
    sharp-fovea obs byte-for-byte — same ``dim`` and same vectors over a scripted
    trajectory (the off-switch invariant); the motion-block salience slice is
    unchanged (``o_motion_hi == o_proprio`` when off);
  * **buffer/staleness blocks:** ON adds two M^2 blocks after motion, so
    ``dim == 2*G^2 + FG^2 + 2*M^2 + 14 + n_ram``, offsets contiguous;
  * **stamp persists:** a foveated region keeps its hi-res content when the gaze
    moves away, on a static scene, with zero invalidations;
  * **change-blindness:** a foveated region under a static periphery stays fresh
    (never re-invalidated); a scene change invalidates the changed region and marks
    it stale, while a control region is untouched;
  * **self-calibration (the [AC] proof shape):** a constant-high change never fires
    once the per-region EMA warms; a RELATIVE spike above the env's own baseline
    fires; two envs at DIFFERENT absolute baselines both fire on the same relative
    spike — which NO fixed magnitude threshold can do (the §11b mandate);
  * **zero-rng / per-env / engine-parity:** the buffer + EMA update is
    deterministic and depends ONLY on the env's own frame sequence (identical
    serial-vs-furnace).

The buffer/staleness grow ``n_in`` (=> a fresh run); ``foveal_memory=False`` keeps
the obs byte-identical to Increment A.
"""

from __future__ import annotations

import hashlib

import numpy as np

from pokeio.emu.fleet import FovealEncoder

# Committed foveal defaults (config.vision / spec §2.1) + a modest memory grid.
G = 12          # periph_grid
F = 48          # fovea_native_px
N_RAM = 8       # obs_ram_bytes
M = 24          # mem_grid (buffer + staleness are M*M each)


def _screen(seed: int) -> np.ndarray:
    """A DMG-like 4-shade (144,160) uint8 frame."""
    return (np.random.RandomState(seed).randint(0, 4, (144, 160)) * 85).astype(np.uint8)


def _wram(seed: int) -> np.ndarray:
    return np.random.RandomState(seed + 7).randint(0, 256, 8192).astype(np.uint8)


def _mk(**kw) -> FovealEncoder:
    base = dict(periph_grid=G, fovea_native_px=F, n_ram=N_RAM, saccade_gain=32.0,
                saccade_every_k=1, episode_steps=200, foveal_memory=True, mem_grid=M,
                mem_ema_decay=0.99, mem_stale_z=1.5, mem_stale_warmup=6)
    base.update(kw)
    return FovealEncoder(1, **base)


def _fovea_dim(fg: int) -> int:
    """Increment A obs dim: periph(G^2) + fovea(FG^2) + motion(G^2) + 14 + n_ram."""
    return 2 * G * G + fg * fg + 14 + N_RAM


def _mem_dim(fg: int, m: int = M) -> int:
    """Increment B obs dim: Increment A + buffer(M^2) + staleness(M^2)."""
    return _fovea_dim(fg) + 2 * m * m


# ------------------------------------------------------------ v2-B-off == Inc A
def _checksum(enc: FovealEncoder, screens, wrams, cmds, buttons) -> str:
    """Drive an encoder through a scripted trajectory; hash every vector."""
    h = hashlib.md5()
    enc.reset()
    h.update(enc.encode(0, screens[0], wrams[0], button=8).tobytes())  # reset obs
    for k in range(len(cmds)):
        enc.update_gaze(0, cmds[k][0], cmds[k][1])
        h.update(enc.encode(0, screens[k + 1], wrams[k + 1], button=buttons[k]).tobytes())
    return h.hexdigest()


def test_off_dim_is_increment_a() -> None:
    """foveal_memory defaults OFF and reproduces the Increment A dim exactly."""
    assert _mk(foveal_memory=False).dim == _fovea_dim(G) == 454
    assert _mk(foveal_memory=False, fovea_grid=32).dim == _fovea_dim(32) == 1334
    # Default constructor (no foveal_memory kwarg) is OFF (Increment A).
    assert FovealEncoder(1, periph_grid=G, fovea_native_px=F, n_ram=N_RAM).dim == 454


def test_off_is_byte_identical_to_increment_a() -> None:
    """foveal_memory=False emits byte-identical vectors to an encoder built without
    the flag at all, over a scripted saccade/motion/proprio/ram trajectory — the
    off-switch invariant (obs change only, zero rng)."""
    n = 12
    screens = [_screen(i) for i in range(n + 1)]
    wrams = [_wram(i) for i in range(n + 1)]
    rs = np.random.RandomState(99)
    cmds = [(float(rs.uniform(-1.5, 1.5)), float(rs.uniform(-1.5, 1.5))) for _ in range(n)]
    buttons = [int(rs.randint(0, 9)) for _ in range(n)]
    for fg in (0, 32):
        a = _mk(foveal_memory=False, fovea_grid=fg)
        b = FovealEncoder(1, periph_grid=G, fovea_native_px=F, fovea_grid=fg,
                          n_ram=N_RAM, saccade_gain=32.0, saccade_every_k=1,
                          episode_steps=200)  # no foveal_memory kwarg at all
        assert a.dim == b.dim == _fovea_dim(fg or G)
        assert _checksum(a, screens, wrams, cmds, buttons) == \
            _checksum(b, screens, wrams, cmds, buttons)


def test_off_motion_slice_unchanged() -> None:
    """With memory OFF the [AC] salience slice is unchanged: the motion block still
    immediately precedes proprio, so o_motion_hi == o_proprio (byte-identical AC)."""
    a = _mk(foveal_memory=False)
    assert a.o_motion_hi == a._o_proprio == a.o_proprio
    assert a._o_buffer is None and a._o_stale is None


# ------------------------------------------------- buffer/staleness obs blocks
def test_memory_on_dim_formula_and_offsets() -> None:
    """ON: dim = 2*G^2 + FG^2 + 2*M^2 + 14 + n_ram; blocks contiguous; the memory
    blocks sit BETWEEN motion and proprio (visual channels the blind gate covers),
    and the [AC] salience slice is the motion block ALONE (o_motion:o_motion_hi)."""
    for fg, m in ((0, 24), (32, 24), (32, 32), (24, 16)):
        e = _mk(fovea_grid=fg, mem_grid=m)
        eff_fg = fg or G
        assert e.M == m and e.n_mem == m * m
        assert e.dim == _mem_dim(eff_fg, m)
        # contiguous: periph | fovea | motion | buffer | staleness | proprio | ram
        assert e._o_periph == 0
        assert e._o_fovea == G * G
        assert e._o_motion == G * G + eff_fg * eff_fg
        assert e.o_motion_hi == e._o_motion + G * G           # motion block end
        assert e._o_buffer == e.o_motion_hi
        assert e._o_stale == e._o_buffer + m * m
        assert e._o_proprio == e._o_stale + m * m
        assert e._o_ram == e._o_proprio + 14
        assert e.dim == e._o_ram + N_RAM
        # salience slice is motion only (NOT o_proprio, which is now past memory).
        assert e.o_motion_hi - e.o_motion == G * G
        assert e.o_motion_hi < e._o_proprio                   # memory sits between


def test_encode_emits_declared_dim_with_bounded_blocks() -> None:
    """A memory encoder produces .dim floats; buffer + staleness are in [0,1]."""
    e = _mk(fovea_grid=32, mem_grid=24)
    e.reset()
    v = e.encode(0, _screen(3), _wram(3), button=4)
    assert v.shape == (e.dim,) == (_mem_dim(32, 24),)
    assert v.dtype == np.float32
    buf = v[e._o_buffer:e._o_stale]
    stale = v[e._o_stale:e._o_proprio]
    assert buf.size == stale.size == 24 * 24
    assert 0.0 <= buf.min() and buf.max() <= 1.0
    assert 0.0 <= stale.min() and stale.max() <= 1.0
    # the whole optical span [0:o_proprio] (incl. buffer+staleness) is in [0,1] so
    # the E3 blind-ablation gate can zero it and leave proprio(±1)+ram untouched.
    assert 0.0 <= v[:e._o_proprio].min() and v[:e._o_proprio].max() <= 1.0


def test_buffer_block_in_obs_matches_internal_buffer() -> None:
    """The obs buffer/staleness blocks are exactly the internal per-env maps."""
    e = _mk(fovea_grid=32)
    e.reset()
    e.update_gaze(0, 0.4, -0.3)
    v = e.encode(0, _screen(11), None, button=2)
    assert np.array_equal(v[e._o_buffer:e._o_stale], e.mem_buffer(0).ravel())
    assert np.array_equal(v[e._o_stale:e._o_proprio], e.mem_staleness(0).ravel())


# ------------------------------------------------------------------ episode reset
def test_reset_clears_buffer_and_staleness() -> None:
    """reset(i) clears the buffer (-> 0.5 neutral), staleness (-> 1 fully stale),
    and the per-region change EMA + step counter (§1a: per-episode percept)."""
    e = _mk()
    e.reset()
    for k in range(6):
        e.update_gaze(0, 0.5, 0.5)
        e.encode(0, _screen(k), None, button=4)
    assert not np.all(e.mem_buffer(0) == 0.5)          # buffer got stamped
    assert int(e._mem_steps[0]) > 0
    e.reset(0)
    assert np.all(e.mem_buffer(0) == 0.5)
    assert np.all(e.mem_staleness(0) == 1.0)
    assert int(e._mem_steps[0]) == 0
    assert np.all(e._chg_mu[0] == 0.0) and np.all(e._chg_var[0] == 0.0)


# ------------------------------------------------------------------ stamp persists
def test_stamp_persists_when_gaze_moves_away() -> None:
    """A foveated region keeps its hi-res content across subsequent steps once the
    gaze saccades away, on a STATIC scene — until invalidated (§1a persistence).
    A high saccade gain teleports the fovea off the region in one step (no transit
    sweep), so the region is cleanly abandoned; the static scene never invalidates."""
    e = _mk(fovea_grid=48, saccade_gain=400.0)   # native fovea + one-step teleport
    scr = _screen(1)
    e.reset()
    e.encode(0, scr, None, button=8)             # gaze centred -> stamp centre
    r0, c0 = e._stamp_origin(0)
    mr, mc = e._mf_rows, e._mf_cols
    stamped = e.mem_buffer(0)[r0:r0 + mr, c0:c0 + mc].copy()
    assert np.any(np.abs(stamped - 0.5) > 1e-6)   # it holds real hi-res content
    e.update_gaze(0, 5.0, 5.0)                     # teleport to the far corner
    assert e._stamp_origin(0) != (r0, c0)         # fovea no longer over the region
    fires = 0
    for _ in range(15):                            # hold on the static scene
        e.encode(0, scr, None, button=8)
        fires += int(e.mem_last_invalidated(0).sum())
    assert fires == 0                              # static scene -> no invalidations
    now = e.mem_buffer(0)[r0:r0 + mr, c0:c0 + mc]
    assert np.array_equal(stamped, now)           # the stamp persisted


# ---------------------------------------------------- change-blindness / staleness
def test_static_foveated_region_stays_fresh() -> None:
    """A region held under the fovea on a static periphery stays FRESH (staleness 0,
    re-stamped every step) and is NEVER re-invalidated — adaptive change-blindness
    when nothing changes (the flip-side of invalidating what did change)."""
    e = _mk()
    scr = _screen(2)
    e.reset()
    fires = 0
    for _ in range(20):                            # gaze fixed at centre, static scene
        e.encode(0, scr, None, button=8)
        fires += int(e.mem_last_invalidated(0).sum())
    r0, c0 = e._stamp_origin(0)
    centre_stale = e.mem_staleness(0)[r0:r0 + e._mf_rows, c0:c0 + e._mf_cols]
    assert fires == 0
    assert np.all(centre_stale == 0.0)             # continuously refreshed -> fresh


def _wobble(base: np.ndarray, seed: int, k: int = 150) -> np.ndarray:
    """Deterministic mild whole-screen perturbation of ``base`` (gives every region
    a non-zero change variance so the per-region EMA has a scale)."""
    n = base.astype(int)
    r = np.random.RandomState(seed)
    ys, xs = r.randint(0, 144, k), r.randint(0, 160, k)
    n[ys, xs] = (n[ys, xs] + 85) % (4 * 85)        # cycle within the 4 DMG shades
    return n.astype(np.uint8)


def test_scene_change_invalidates_and_goes_stale() -> None:
    """A CHANGED periphery region invalidates (enters the fire mask) and its buffer
    cells go stale (staleness -> 1); a control region that did NOT change is not
    re-invalidated.  The gaze stays centred, so both test regions are peripheral."""
    base = _screen(2)
    e = _mk(mem_stale_z=4.0, mem_stale_warmup=6)
    e.reset()
    for t in range(30):                            # warm the per-region EMA (var>0)
        e.encode(0, _wobble(base, 400 + t), None, button=8)
    # big change confined to the TOP-LEFT screen corner (G-regions [0:3,0:3]).
    changed = base.copy()
    changed[0:36, 0:40] = 255 - changed[0:36, 0:40]
    e.encode(0, changed, None, button=8)
    fire = e.mem_last_invalidated(0)
    stale = e.mem_staleness(0)
    assert fire[0:3, 0:3].all()                    # every changed region invalidated
    assert fire[9:12, 9:12].sum() == 0             # far control corner untouched
    # the changed corner's buffer cells (G [0:3] -> M [0:6]) are now fully stale.
    assert np.isclose(stale[0:6, 0:6].max(), 1.0)


# ------------------------------------------------ self-calibration ([AC] proof)
def _calib(n_envs: int = 1, **kw) -> FovealEncoder:
    base = dict(periph_grid=G, fovea_native_px=F, n_ram=N_RAM, foveal_memory=True,
                mem_grid=M, episode_steps=200, mem_ema_decay=0.99,
                mem_stale_warmup=4, mem_stale_z=2.0)
    base.update(kw)
    return FovealEncoder(n_envs, **base)


def test_selfcal_constant_high_never_fires() -> None:
    """A constant-high change NEVER fires once the EMA warms: with no observed
    variance there is no surprise scale (the [AC] 'stops firing' half — an absolute
    threshold could not resist a constant high level)."""
    e = _calib(mem_stale_z=1.5)
    e.reset()
    fired = [int(e._calibrate_invalidation(0, np.full((G, G), 0.8)).sum())
             for _ in range(25)]
    assert set(fired) == {0}


def test_selfcal_relative_spike_fires() -> None:
    """A RELATIVE spike above the env's own warmed baseline fires exactly where it
    spiked (the '[AC] fires on a relative surprise' half)."""
    e = _calib(mem_stale_z=3.0)
    e.reset()
    for t in range(50):                             # warm a uniform low baseline (var>0)
        e._calibrate_invalidation(0, np.full((G, G), 0.10 + 0.01 * (t % 2)))
    mu = e._chg_mu[0].mean()
    spike = np.full((G, G), 0.105)
    spike[5, 5] = mu + 0.3                           # one region jumps
    fire = e._calibrate_invalidation(0, spike)
    assert fire[5, 5]
    assert fire.sum() == 1                           # ONLY the spiked region


def test_selfcal_two_baselines_same_relative_spike() -> None:
    """The crux no absolute threshold can pass: two envs at DIFFERENT absolute change
    baselines both fire on the SAME relative spike.  Env A's spike level sits BELOW
    env B's quiescent baseline, so any fixed magnitude that catches A would false-fire
    on B's calm — only a per-env EMA-z separates them (§11b)."""
    e = _calib(2, mem_stale_z=3.0)
    e.reset()
    for t in range(60):                             # A calm ~0.10, B calm ~0.60
        wob = 0.01 * (t % 2)
        e._calibrate_invalidation(0, np.full((G, G), 0.10 + wob))
        e._calibrate_invalidation(1, np.full((G, G), 0.60 + wob))
    mu_a, mu_b = e._chg_mu[0].mean(), e._chg_mu[1].mean()
    spike_a = np.full((G, G), 0.105); spike_a[5, 5] = mu_a + 0.3
    spike_b = np.full((G, G), 0.605); spike_b[5, 5] = mu_b + 0.3
    fire_a = e._calibrate_invalidation(0, spike_a)
    fire_b = e._calibrate_invalidation(1, spike_b)
    assert fire_a[5, 5] and fire_b[5, 5]            # both fire on their own spike
    assert fire_a.sum() == 1 and fire_b.sum() == 1  # and only there
    # the impossibility proof: A's spike magnitude is BELOW B's calm baseline, so no
    # single fixed threshold can fire A's spike without also firing B's quiescence.
    assert (mu_a + 0.3) < mu_b


# ------------------------------------------------ zero-rng / per-env / parity
def test_zero_rng_deterministic() -> None:
    """The full encode+memory path is deterministic: the identical scripted frame
    sequence yields byte-identical obs on two independent encoders (zero rng)."""
    screens = [_screen(1000 + k) for k in range(11)]
    cmds = [(0.4, -0.3), (-0.6, 0.2), (0.1, 0.5), (0.9, -0.9), (-0.2, -0.1),
            (0.3, 0.3), (-0.5, 0.7), (0.8, 0.1), (-0.1, -0.6), (0.2, 0.4)]

    def run() -> bytes:
        e = _mk(fovea_grid=32)
        e.reset()
        out = [e.encode(0, screens[0], None, button=8).tobytes()]
        for k, (dx, dy) in enumerate(cmds):
            e.update_gaze(0, dx, dy)
            out.append(e.encode(0, screens[k + 1], None, button=4).tobytes())
        return b"".join(out)

    assert run() == run()


def test_per_env_state_is_isolated_engine_parity() -> None:
    """Per-env buffer/EMA depend ONLY on that env's own frame sequence: driving one
    env SOLO vs INTERLEAVED with unrelated activity on other envs yields the same
    buffer — the property that makes serial and furnace bit-identical (§1a)."""
    seq = [_screen(700 + t) for t in range(8)]
    cmds = [(0.3, -0.2)] * 8

    solo = FovealEncoder(3, periph_grid=G, fovea_native_px=F, n_ram=N_RAM,
                         foveal_memory=True, mem_grid=M, episode_steps=200)
    solo.reset()
    for s, (dx, dy) in zip(seq, cmds):
        solo.update_gaze(1, dx, dy)
        solo.encode(1, s, None, button=3)
    solo_buf, solo_stale = solo.mem_buffer(1), solo.mem_staleness(1)

    inter = FovealEncoder(3, periph_grid=G, fovea_native_px=F, n_ram=N_RAM,
                          foveal_memory=True, mem_grid=M, episode_steps=200)
    inter.reset()
    junk = _screen(1)
    for s, (dx, dy) in zip(seq, cmds):
        inter.update_gaze(0, 1.0, 1.0); inter.encode(0, junk, None, button=1)
        inter.update_gaze(1, dx, dy); inter.encode(1, s, None, button=3)   # same as solo
        inter.update_gaze(2, -0.5, 0.5); inter.encode(2, junk, None, button=2)
    assert np.array_equal(solo_buf, inter.mem_buffer(1))
    assert np.array_equal(solo_stale, inter.mem_staleness(1))
