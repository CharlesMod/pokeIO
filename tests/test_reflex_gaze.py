"""Optical front-end v2 — Increment C: reflex gaze (motion × staleness) + evolved
top-down modulation.

Covers task #25 (dedicated tests for the v2-C reflex gaze) of
``docs/specs/optical-frontend-v2.md`` §3/§4, entirely without an emulator:

  * **off-switch invariant:** ``reflex_gaze=False`` (default) keeps ``n_proprio``
    at 14, the obs dim byte-identical to Increment B (no reflex kwarg at all),
    ``ReflexGaze.maybe`` returns ``None`` (gaze == legacy), and the telemetry
    accessor ``reflex_target()`` reads ``(0, 0)``;
  * **input-side only:** ON grows PROPRIO 14->16 (the reflex TARGET rides in the
    trailing 2 efference dims) and adds NOTHING on the output side — it re-purposes
    the existing saccade outputs, so ``fast_reproduce`` / N_OUT are untouched (§4);
  * **soft-argmax aims (§3):** a strong-motion region (with the region STALE) pulls
    the target TOWARD it (bottom-right -> both coords >0, top-left -> both <0), and
    the two are mirror images; a flat motion field -> zero pull (current gaze);
  * **staleness gates it (the headline, §1a):** a strong-motion but FRESH region
    does NOT attract — ``motion × staleness`` means "orient to what changed OR
    hasn't been refreshed lately", and a just-foveated region is fresh => no pull;
  * **proprio handshake (§4):** ON, after an encode the reflex target is written
    into ``proprio[o_proprio+14:+16]`` and equals ``reflex_target()`` (single
    source of truth the controller conditions its top-down correction on);
  * **blend (reflex + learned, §4):** ``ReflexGaze.blend`` reads target+gaze from
    the obs matrix and adds the reflex delta to the learned saccade — pure reflex
    when learned==0 (pull toward target), learned OVERRIDES when it is large;
  * **self-calibration (the §11b [AC] proof shape):** the reflex delta is
    normalized by a per-env running EMA of its OWN pull magnitude — NO fixed
    magnitude, so two envs at DIFFERENT pull baselines respond differently to the
    SAME pull (a fixed pixel step could not); the EMA adapts across calls;
  * **zero-rng / per-env / engine-parity:** ``_reflex_target`` and ``blend`` are
    deterministic given inputs, the ``ReflexGaze`` EMA is per-env (a non-ready env
    does not tick), and a solo-vs-interleaved drive yields the identical reflex
    target — the property that makes serial and furnace bit-identical.

Reflex gaze grows ``n_in`` by 2 (=> a fresh run); ``reflex_gaze=False`` keeps the
obs byte-identical to Increment B (retina is always reflex-off).
"""

from __future__ import annotations

import numpy as np

from pokeio.emu.fleet import FovealEncoder, ReflexGaze

# Committed foveal defaults + the sharp-fovea / memory grid the v2 run uses
# (config.vision / spec §2.1); reflex gaze is foveal-only and rides on top.
G = 12          # periph_grid
F = 32          # fovea_native_px
FG = 32         # fovea_grid (1 px/cell sharp fovea)
N_RAM = 8       # obs_ram_bytes
M = 24          # mem_grid (trans-saccadic memory; the reflex weights motion by it)


def _screen(seed: int) -> np.ndarray:
    """A DMG-like 4-shade (144,160) uint8 frame."""
    return (np.random.RandomState(seed).randint(0, 4, (144, 160)) * 85).astype(np.uint8)


def _mk(reflex_gaze: bool = True, n: int = 1, **kw) -> FovealEncoder:
    base = dict(periph_grid=G, fovea_native_px=F, fovea_grid=FG, n_ram=N_RAM,
                saccade_gain=32.0, saccade_every_k=1, episode_steps=200,
                foveal_memory=True, mem_grid=M, reflex_gaze=reflex_gaze)
    base.update(kw)
    return FovealEncoder(n, **base)


def _mem_dim(n_proprio: int) -> int:
    """Increment C obs dim: 2*G^2 + FG^2 + 2*M^2 + n_proprio + n_ram."""
    return 2 * G * G + FG * FG + 2 * M * M + n_proprio + N_RAM


def _hot_motion(r: int, c: int) -> np.ndarray:
    """A G×G motion sheet: 0.5 (no motion) everywhere except one strong-motion cell."""
    m = np.full((G, G), 0.5, np.float32)
    m[r, c] = 1.0
    return m


# ------------------------------------------------------------ off-switch invariant
def test_off_keeps_proprio_14_and_increment_b_dim() -> None:
    """reflex_gaze defaults OFF: n_proprio stays 14 and the obs dim is byte-identical
    to an Increment B encoder built without the reflex kwarg at all (off => no obs
    change; the successor run turns it on for a fresh n_in)."""
    off = _mk(reflex_gaze=False)
    inc_b = FovealEncoder(1, periph_grid=G, fovea_native_px=F, fovea_grid=FG,
                          n_ram=N_RAM, foveal_memory=True, mem_grid=M)  # no reflex kwarg
    assert off.n_proprio == 14
    assert off.dim == inc_b.dim == _mem_dim(14)
    # The default constructor is reflex-off (Increment B).
    assert FovealEncoder(1, periph_grid=G, fovea_native_px=F, fovea_grid=FG,
                         n_ram=N_RAM, foveal_memory=True, mem_grid=M).reflex_gaze is False


def test_off_maybe_is_none_and_reflex_target_is_zero() -> None:
    """OFF: ReflexGaze.maybe(encoder, n) is None (the blend is skipped, gaze ==
    legacy) and the telemetry accessor reflex_target() reads screen-centre (0,0)."""
    off = _mk(reflex_gaze=False)
    off.reset()
    assert ReflexGaze.maybe(off, 8) is None
    assert off.reflex_target(0) == (0.0, 0.0)
    # maybe(None, ...) is also None (defensive: no encoder => no reflex).
    assert ReflexGaze.maybe(None, 8) is None


def test_on_is_input_side_only_no_output_change() -> None:
    """ON grows PROPRIO 14->16 (the reflex TARGET is the only new obs) and the dim
    grows by exactly 2 vs OFF — the reflex adds INPUT dims only.  It re-purposes the
    existing saccade outputs for the top-down term, so there is no new gene/output
    (fast_reproduce / N_OUT untouched, §4)."""
    on = _mk(reflex_gaze=True)
    off = _mk(reflex_gaze=False)
    assert on.n_proprio == 16
    assert on.reflex_gaze is True
    assert on.dim == _mem_dim(16)
    assert on.dim - off.dim == 2               # +2 proprio, nothing else
    assert on.n_proprio - off.n_proprio == 2   # the whole delta is on the input side


# ---------------------------------------------------- soft-argmax target (§3)
def test_reflex_target_aims_toward_motion_peak() -> None:
    """[§3] With the region stale, a strong-motion cell pulls the soft-argmax target
    TOWARD it: a bottom-right peak -> both coords >0, a top-left peak -> both <0, and
    the two are mirror images (the salience center-of-mass is symmetric)."""
    e = _mk(reflex_gaze=True)
    e.reset()
    e._stale[0][:] = 1.0                       # all stale => motion alone drives salience
    br = e._reflex_target(0, _hot_motion(G - 1, G - 1))
    tl = e._reflex_target(0, _hot_motion(0, 0))
    assert br[0] > 0 and br[1] > 0             # bottom-right pull
    assert tl[0] < 0 and tl[1] < 0             # top-left pull
    assert np.allclose(br, np.negative(tl))    # symmetric about screen centre


def test_flat_field_gives_zero_pull() -> None:
    """[§3] A flat motion field (no salient change) has an all-zero salience map =>
    the soft-argmax is undefined, so the target falls back to the CURRENT gaze — a
    harmless zero pull.  After reset the gaze is screen-centre, so target == (0,0)."""
    e = _mk(reflex_gaze=True)
    e.reset()
    e._stale[0][:] = 1.0
    tgt = e._reflex_target(0, np.full((G, G), 0.5, np.float32))
    assert tgt == (0.0, 0.0)


def test_staleness_gates_the_reflex() -> None:
    """THE HEADLINE (§1a/§3): strong motion but a FRESH region does NOT attract.
    motion × staleness means "orient to what CHANGED **or** hasn't been refreshed
    lately"; a just-foveated (fresh) region has staleness 0, so its motion is gated
    out and the target is zero pull — despite an identical motion sheet that pulls
    hard when the region is stale."""
    e = _mk(reflex_gaze=True)
    e.reset()
    motion = _hot_motion(G - 1, G - 1)
    e._stale[0][:] = 1.0                       # stale: motion pulls
    assert e._reflex_target(0, motion) != (0.0, 0.0)
    e._stale[0][:] = 0.0                       # fresh: staleness gates the same motion
    assert e._reflex_target(0, motion) == (0.0, 0.0)


# ---------------------------------------------------- proprio handshake (§4)
def test_reflex_target_surfaced_in_proprio() -> None:
    """[§4] ON, after an encode the reflex target is written into
    proprio[o_proprio+14:+16] and equals reflex_target() — the single source of
    truth the controller conditions its top-down correction on."""
    e = _mk(reflex_gaze=True)
    e.reset()
    v = None
    for k in range(6):                         # a few steps so motion/staleness are non-trivial
        e.update_gaze(0, 0.3, -0.2)
        v = e.encode(0, _screen(k), None, button=4)
    op = e.o_proprio
    tx, ty = e.reflex_target(0)
    assert v[op + 14] == np.float32(tx)
    assert v[op + 15] == np.float32(ty)
    assert -1.0 <= v[op + 14] <= 1.0 and -1.0 <= v[op + 15] <= 1.0


# -------------------------------------------------- blend: reflex + learned (§4)
def test_blend_pure_reflex_pulls_toward_target() -> None:
    """[§4] With the learned saccade zero, blend() reads target+gaze from the obs
    matrix and returns the pure reflex delta pointing TOWARD (target - gaze); a row
    whose target equals its gaze gets a zero pull (no salient direction)."""
    e = _mk(reflex_gaze=True)
    op, dim = e.o_proprio, e.dim
    n = 3
    X = np.zeros((n, dim), np.float32)
    X[:, op + 0] = 0.0                         # current gaze x
    X[:, op + 1] = 0.0                         # current gaze y
    # per-row reflex targets in proprio[14:16]
    X[0, op + 14], X[0, op + 15] = 0.5, -0.5   # -> +dx, -dy
    X[1, op + 14], X[1, op + 15] = -0.6, 0.2   # -> -dx, +dy
    X[2, op + 14], X[2, op + 15] = 0.0, 0.0    # target == gaze -> zero pull

    rg = ReflexGaze.maybe(e, n)
    assert rg is not None
    gdx = np.zeros(n, np.float32)
    gdy = np.zeros(n, np.float32)
    dx, dy = rg.blend(X, op, gdx, gdy)
    assert dx[0] > 0 and dy[0] < 0             # toward (0.5, -0.5)
    assert dx[1] < 0 and dy[1] > 0             # toward (-0.6, 0.2)
    assert dx[2] == 0.0 and dy[2] == 0.0       # target == gaze -> no pull


def test_blend_large_learned_overrides_reflex() -> None:
    """[§4] The learned saccade is ADDITIVE on top of the reflex (superior-colliculus
    reflex + cortical override): a large learned term dominates and can flip the net
    direction against the reflex.  The reflex pulls +dx (toward the target), yet a
    large NEGATIVE learned dx overrides it, and the reflex contribution stays bounded
    by the (self-calibrated) gain."""
    e = _mk(reflex_gaze=True)
    op, dim = e.o_proprio, e.dim
    n = 2
    X = np.zeros((n, dim), np.float32)
    X[:, op + 14] = 0.5                        # target x -> reflex pulls +dx
    X[:, op + 15] = 0.0

    reflex_only = ReflexGaze.maybe(e, n)
    r_dx, _ = reflex_only.blend(X, op, np.zeros(n, np.float32), np.zeros(n, np.float32))
    assert r_dx[0] > 0                         # pure reflex would push +dx

    override = ReflexGaze.maybe(e, n)
    learned = np.full(n, -100.0, np.float32)
    dx, _ = override.blend(X, op, learned, np.zeros(n, np.float32))
    assert dx[0] < 0                           # learned wins => net flips negative
    # the reflex delta it added is bounded (a self-calibrated unit-ish step, not a
    # fixed pixel jump): net == learned + reflex, |reflex| <= ~gain.
    assert abs(dx[0] - learned[0]) <= 1.5


# -------------------------------------------- self-calibration ([AC] proof, §11b)
def test_reflex_gain_self_calibrates_no_fixed_magnitude() -> None:
    """[§11b] The crux no fixed magnitude can pass (the [AC] two-baselines shape):
    two envs warmed at DIFFERENT pull baselines respond DIFFERENTLY to the SAME pull.
    Env 0 (big-pull history) barely moves; env 1 (small-pull history) jerks hard —
    the delta = gain × pull / (per-env EMA of its own pull magnitude), so it is a
    dimensionless velocity, not a fixed pixel step."""
    rg = ReflexGaze(2, gain=1.0, ema_decay=0.9)
    for _ in range(80):                        # env0 baseline ~10, env1 baseline ~0.1
        rg.command(np.array([10.0, 0.1]), np.array([0.0, 0.0]))
    assert not np.isclose(rg._ema[0], rg._ema[1])   # each adapted to its own history
    dx, _ = rg.command(np.array([1.0, 1.0]), np.array([0.0, 0.0]))  # identical pull
    assert dx[0] < 0.5                         # big-baseline env: gentle
    assert dx[1] > 2.0                         # small-baseline env: strong orienting jerk
    # the impossibility proof: env0's response is >10x smaller for the SAME input pull,
    # which no single fixed magnitude could produce.
    assert dx[1] > 10.0 * dx[0]


def test_reflex_gain_ema_adapts_across_calls() -> None:
    """[§11b] The reflex_gain EMA is not a fixed constant — it tracks the env's own
    pull magnitude: seeded on the first sample, then it climbs toward a new,
    larger pull level over subsequent calls (the running-EMA calibration)."""
    rg = ReflexGaze(1, gain=1.0, ema_decay=0.5)
    rg.command(np.array([2.0]), np.array([0.0]))   # first sample seeds the EMA
    assert np.isclose(rg._ema[0], 2.0)
    trace = []
    for _ in range(20):                            # switch to a larger pull baseline
        rg.command(np.array([8.0]), np.array([0.0]))
        trace.append(float(rg._ema[0]))
    assert trace[0] > 2.0                           # started climbing off the seed
    assert all(b >= a - 1e-9 for a, b in zip(trace, trace[1:]))  # monotone up
    assert np.isclose(trace[-1], 8.0, atol=1e-3)    # converged to the new magnitude


# ------------------------------------------------ zero-rng / per-env / parity
def test_reflex_target_and_blend_are_deterministic() -> None:
    """Zero rng: _reflex_target and blend are pure arithmetic over resident floats,
    so the identical inputs give byte-identical outputs on repeated calls (the
    property that keeps fast_reproduce untouched and engine-parity intact)."""
    e = _mk(reflex_gaze=True)
    e.reset()
    e._stale[0][:] = 1.0
    motion = _hot_motion(8, 3)
    assert e._reflex_target(0, motion) == e._reflex_target(0, motion)

    op, dim = e.o_proprio, e.dim
    X = np.zeros((4, dim), np.float32)
    X[:, op + 14] = 0.4
    X[:, op + 15] = -0.7
    a = ReflexGaze.maybe(e, 4).blend(X, op, np.zeros(4, np.float32), np.zeros(4, np.float32))
    b = ReflexGaze.maybe(e, 4).blend(X, op, np.zeros(4, np.float32), np.zeros(4, np.float32))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_reflex_gain_ema_is_per_env_isolated() -> None:
    """Per-env EMA: only the ready envs tick their calibration (a non-ready env did
    not step, so it must not advance).  Driving env 0 alone (ready_idx=[0]) leaves
    env 1's EMA and step counter untouched — the isolation that makes serial and
    furnace bit-identical."""
    rg = ReflexGaze(2, gain=1.0)
    rg.command(np.array([3.0, 3.0]), np.array([0.0, 0.0]), ready_idx=[0])
    assert rg._ema[0] == 3.0 and rg._steps[0] == 1   # env 0 ticked
    assert rg._ema[1] == 0.0 and rg._steps[1] == 0   # env 1 untouched
    # an empty ready set is a no-op that ticks nobody.
    rdx, rdy = rg.command(np.array([9.0, 9.0]), np.array([9.0, 9.0]), ready_idx=[])
    assert np.all(rdx == 0) and np.all(rdy == 0)
    assert rg._steps[0] == 1 and rg._steps[1] == 0


def test_reflex_target_engine_parity_solo_vs_interleaved() -> None:
    """The per-env reflex TARGET depends ONLY on that env's own frame sequence:
    driving env 1 SOLO vs INTERLEAVED with unrelated activity on other envs yields
    the identical reflex target — the structural property behind serial-vs-furnace
    bit-identity (the encoder reflex is per-env, zero rng)."""
    seq = [_screen(700 + t) for t in range(8)]

    solo = _mk(reflex_gaze=True, n=3)
    solo.reset()
    for s in seq:
        solo.update_gaze(1, 0.3, -0.2)
        solo.encode(1, s, None, button=3)
    solo_tgt = solo.reflex_target(1)

    inter = _mk(reflex_gaze=True, n=3)
    inter.reset()
    junk = _screen(1)
    for s in seq:
        inter.update_gaze(0, 1.0, 1.0); inter.encode(0, junk, None, button=1)
        inter.update_gaze(1, 0.3, -0.2); inter.encode(1, s, None, button=3)  # same as solo
        inter.update_gaze(2, -0.5, 0.5); inter.encode(2, junk, None, button=2)
    assert inter.reflex_target(1) == solo_tgt
