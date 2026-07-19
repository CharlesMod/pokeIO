"""Priority-map scaffold tests (FovealEncoder._priority_map, Step 2).

The reflex target now reads a PRIORITY MAP (weighted channel sum · recency gate)
instead of a hard-coded motion×staleness product.  At the warm-start (only the
motion channel, weight 1.0, no top-down) it must be BYTE-IDENTICAL to the prior
reflex — this is the A/B baseline and the safe no-op refactor that unlocks the
learned channels (Step 3/4).
"""

from __future__ import annotations

import numpy as np

from pokeio.emu.fleet import FovealEncoder


def _mk(foveal=True):
    return FovealEncoder(1, periph_grid=12, fovea_native_px=48, fovea_grid=0, n_ram=8,
                         reflex_gaze=True, foveal_memory=foveal, mem_grid=96)


def _reference_target(e, i, motion):
    """The pre-Step-2 reflex: soft-argmax of |motion-0.5| (× staleness if foveal)."""
    sal = np.abs(motion.astype(np.float64) - 0.5)
    if e.foveal_memory:
        sal = np.take(sal, e._g2m_flat).reshape(e.M, e.M) * e._stale[i]
        P = e.M
    else:
        P = e.G
    flat = sal.ravel()
    smax = float(flat.max())
    if smax <= 1e-9:
        return (e._gx[i] / e.W * 2.0 - 1.0, e._gy[i] / e.H * 2.0 - 1.0)
    w = np.exp(e.reflex_beta * (flat / smax - 1.0))
    w /= w.sum()
    wm = w.reshape(P, P)
    idx = e._reflex_idx
    r = float((wm.sum(axis=1) * idx).sum())
    c = float((wm.sum(axis=0) * idx).sum())
    return ((c + 0.5) / P * 2.0 - 1.0, (r + 0.5) / P * 2.0 - 1.0)


def test_priority_map_is_reflex_equivalent_foveal():
    e = _mk(True)
    rng = np.random.default_rng(0)
    e._stale[0] = rng.random((e.M, e.M)).astype(np.float32)   # non-trivial recency gate
    for t in range(30):
        motion = rng.random((e.G, e.G)).astype(np.float32)
        got = e._reflex_target(0, motion)
        ref = _reference_target(e, 0, motion)
        assert got == ref, f"t={t}: {got} != {ref}"           # bit-identical


def test_priority_map_is_reflex_equivalent_no_canvas():
    e = _mk(False)
    rng = np.random.default_rng(1)
    for t in range(30):
        motion = rng.random((e.G, e.G)).astype(np.float32)
        assert e._reflex_target(0, motion) == _reference_target(e, 0, motion)


def test_flat_priority_returns_current_gaze():
    e = _mk(True)
    e._stale[0] = np.zeros((e.M, e.M), np.float32)            # gate zero -> flat priority
    motion = np.full((e.G, e.G), 0.5, np.float32)             # no transient
    gx, gy = e._gx[0], e._gy[0]
    assert e._reflex_target(0, motion) == (gx / e.W * 2.0 - 1.0, gy / e.H * 2.0 - 1.0)


def test_single_channel_weight_is_scale_free_through_softargmax():
    # with ONE channel the weight normalizes out of the soft-argmax: w[0]=1 and w[0]=5
    # give the identical target.  (Weights only bite once >1 channel exists, Step 3.)
    e = _mk(True)
    rng = np.random.default_rng(2)
    e._stale[0] = rng.random((e.M, e.M)).astype(np.float32)
    motion = rng.random((e.G, e.G)).astype(np.float32)
    t1 = e._reflex_target(0, motion)
    e._pw[0] = 5.0
    t5 = e._reflex_target(0, motion)
    assert t1 == t5
