"""Intra-action sub-saccade tests (FovealEncoder.substep / _reflex_command).

Hermetic (no fleet/ROM): drive the encoder directly. Locks the cadence-phasing
invariants — a sub-saccade STAMPS the canvas + re-aims but must NOT advance the
action counter (_nstep), the self-calibrated command holds on a zero pull, and
more sub-saccades fill more of the persisted-vision canvas. The fleet-level 2x
cadence (2.49 -> 4.98 Hz) is exercised by the scratchpad smoke.
"""

from __future__ import annotations

import numpy as np

from pokeio.emu.fleet import FovealEncoder


def _mk():
    return FovealEncoder(1, periph_grid=12, fovea_native_px=48, fovea_grid=0, n_ram=8,
                         reflex_gaze=True, foveal_memory=True, mem_grid=96)


def _moving_frames(n):
    """Frames with a bright patch that translates each frame, so motion drives the
    reflex toward a shifting target (a non-trivial saccade)."""
    out = []
    for t in range(n):
        f = np.zeros((144, 160), np.uint8)
        y = 20 + 7 * t
        x = 20 + 9 * t
        f[y:y + 22, x:x + 22] = 3
        out.append(f)
    return out


def test_substep_does_not_advance_the_action_counter():
    e = _mk()
    w = np.zeros(8192, np.uint8)
    fr = _moving_frames(4)
    e.encode(0, fr[0], w)                       # one action step
    n0 = int(e._nstep[0])
    e.substep(0, fr[1])
    e.substep(0, fr[2])                         # two sub-saccades within a step
    assert int(e._nstep[0]) == n0               # proprio step-fraction stays honest


def test_substep_stamps_canvas_and_reaims():
    e = _mk()
    w = np.zeros(8192, np.uint8)
    # frame 0: uniform; frame 1: a bright block INSIDE the central fovea (so the
    # stamp content actually changes) plus an off-centre patch to pull the reflex.
    f0 = np.ones((144, 160), np.uint8)
    f1 = np.ones((144, 160), np.uint8)
    f1[60:84, 70:94] = 3        # inside the ~48px central fovea -> stamp differs
    f1[20:40, 20:40] = 3        # off-centre motion -> non-zero reflex pull
    e.encode(0, f0, w)
    buf0 = e.mem_buffer(0).copy()
    gaze0 = e.gaze(0)
    e.substep(0, f1)
    assert not np.array_equal(buf0, e.mem_buffer(0))   # a fresh (different) stamp landed
    assert e.gaze(0) != gaze0                           # the fovea re-aimed (motion pull)


def test_reflex_command_self_calibrates_and_holds_on_zero_pull():
    e = _mk()
    # a zero pull must NOT seed the EMA (else the next real pull divides by ~0 and
    # saturates) and emits no command
    dx, dy = e._reflex_command(0, 0.0, 0.0)
    assert (dx, dy) == (0.0, 0.0)
    assert int(e._sub_seen[0]) == 0
    # first meaningful pull seeds + points along the pull direction
    dx, dy = e._reflex_command(0, 0.5, -0.3)
    assert int(e._sub_seen[0]) == 1
    assert np.isfinite([dx, dy]).all()
    assert dx > 0 and dy < 0


def test_more_substeps_fill_more_canvas():
    w = np.zeros(8192, np.uint8)
    fr = _moving_frames(7)
    # S=1: 3 action steps, one glimpse each (encode only)
    e1 = _mk()
    for f in fr[:3]:
        e1.encode(0, f, w)
    cov1 = float(np.mean(np.abs(e1.mem_buffer(0) - 0.5) > 1e-3))
    # S=2: same 3 action steps, but each does a mid-step sub-saccade then the final
    # encode -> twice the glimpses, at re-aimed locations
    e2 = _mk()
    e2.encode(0, fr[0], w)                       # seed step
    e2.substep(0, fr[1]); e2.encode(0, fr[2], w)  # action step w/ sub-saccade
    e2.substep(0, fr[3]); e2.encode(0, fr[4], w)  # another
    cov2 = float(np.mean(np.abs(e2.mem_buffer(0) - 0.5) > 1e-3))
    assert cov2 >= cov1                          # more saccades -> more of the scene painted
