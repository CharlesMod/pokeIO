"""Frozen-gaze generalization battery tests (pokeio.train.gaze_battery).

Hermetic: NO ROM, NO PyBoy — synthetic (144,160) uint8 screens are fed straight into
run_stream/Lane.step (the per-frame lane update).  Locks the battery's load-bearing
invariants: determinism, the canvas/priority/change metrics actually discriminate,
buttons are open-loop (lane-independent), and the 'learned' lane's applied delta is
bit-for-bit the deployed rl_loop._gaze blend (the regression that matters most —
a silent divergence there would invalidate every 'learned' number the battery emits).
"""

from __future__ import annotations

import numpy as np
import torch

from pokeio.emu.fleet import ReflexGaze
from pokeio.train.gaze_battery import (
    Lane, LearnedGaze, RandomGaze, ReflexOnlyGaze, _priority_capture_at,
    build_trace, center_fixation, run_stream,
)


# ------------------------------------------------------------------ synthetic streams
def _wram():
    return np.zeros(8192, np.uint8)


def _frames(n, draw):
    """Frames whose byte set is ALWAYS {0,1,3} (static rank anchors), so the encoder's
    per-frame shade ranking is stable: background 1 -> 0.5 (== the canvas' neutral
    init, so unstamped cells are error-free and the metric isolates the CHANGE)."""
    out = []
    for t in range(n):
        f = np.ones((144, 160), np.uint8)
        f[68:72, 76:80] = 0          # anchors: keep 0 and 3 present in every frame
        f[68:72, 80:84] = 3
        draw(f, t)
        out.append(f)
    return out


def _steps_from_frames(fr, trace=None):
    """S=2 action steps from a frame sequence: begin on fr[0]; step t perceives
    mid=fr[2t+1] (substep) and final=fr[2t+2] (encode) — the trained cadence."""
    n = (len(fr) - 1) // 2
    trace = list(trace) if trace is not None else [8] * n
    steps = [(trace[t], [fr[2 * t + 1]], fr[2 * t + 2], _wram()) for t in range(n)]
    return (fr[0], _wram()), steps


def _corner_patch(f, t):     # bright patch crawling inside the top-left 48x48 fovea
    y = 6 + 2 * (t % 12)
    x = 6 + 2 * (t % 12)
    f[y:y + 14, x:x + 14] = 3


def _flash_patch(f, t):      # patch toggling once per ACTION STEP (finals alternate)
    if (t // 2) % 2 == 0:
        f[14:34, 14:34] = 3


def _fix_at(gy, gx):
    return lambda enc, prio, P: (float(gy), float(gx))


def _center_lane():
    return Lane("center", fixation=center_fixation)


# ------------------------------------------------------------------ t1: determinism
def test_center_lane_deterministic():
    fr = _frames(13, _corner_patch)
    first, steps = _steps_from_frames(fr)

    def run():
        lane = _center_lane()
        run_stream([lane], first, steps)
        return lane.report()

    assert run() == run()      # identical stream + controller -> identical metrics


# ---------------------------------------------- t2: canvas_err discriminates looking
def test_canvas_err_foveating_change_beats_fixed_away():
    # change confined to the top-left corner; lane A foveates it, lane B fixates the
    # opposite corner.  6 steps (13 perceives) stays under mem_stale_warmup=16 so
    # invalidation never bails lane B out.
    fr = _frames(13, _corner_patch)
    first, steps = _steps_from_frames(fr)
    a = Lane("on_change", fixation=_fix_at(24.0, 24.0))
    b = Lane("away", fixation=_fix_at(120.0, 136.0))
    run_stream([a, b], first, steps)
    assert a.report()["canvas_err"] < b.report()["canvas_err"]


# ------------------------------------------------------- t3: priority_capture readout
def test_priority_capture_exact_argmax_and_center():
    pr = np.zeros((96, 96), np.float64)
    pr[7, 13] = 2.0
    # fixation at the argmax cell's centre pixel -> exact capture (any half)
    gy = (7 + 0.5) * 144 / 96
    gx = (13 + 0.5) * 160 / 96
    ratio, hit = _priority_capture_at(pr, 96, gy, gx, half=0.0)
    assert ratio == 1.0 and hit
    ratio, hit = _priority_capture_at(pr, 96, gy, gx, half=24.0)
    assert ratio == 1.0 and hit
    # off the argmax cell but the FOOTPRINT covers it -> capture credits the window
    ratio_w, hit_w = _priority_capture_at(pr, 96, gy + 20.0, gx + 20.0, half=24.0)
    assert ratio_w == 1.0 and hit_w
    # centre fixation while the argmax is far outside the window -> strictly below
    ratio_c, hit_c = _priority_capture_at(pr, 96, 72.0, 80.0, half=24.0)
    assert ratio_c < 1.0 and not hit_c


def test_priority_capture_footprint_reaches_clamped_border():
    # The deployed gaze clamp (gy in [24,120], gx in [24,136]) confines the fixated
    # CELL to r in [16,80], c in [14,81] — a single-cell readout can NEVER credit an
    # argmax in the border band (e.g. the Gen-1 dialog-box rows 81-95).  The fovea
    # WINDOW at the clamp edge does cover it — the footprint metric must say hit.
    pr = np.zeros((96, 96), np.float64)
    pr[90, 48] = 2.0                      # M-row 90 centre = 135.75px; clamp pins gy=120
    ratio, hit = _priority_capture_at(pr, 96, 120.0, (48 + 0.5) * 160 / 96, half=24.0)
    assert ratio == 1.0 and hit


def test_priority_capture_lane_level():
    def argmax_fix(enc, prio, P):
        if prio is None:
            return enc.gaze(0)
        r, c = np.unravel_index(int(np.asarray(prio).argmax()), (P, P))
        return ((float(r) + 0.5) * enc.H / P, (float(c) + 0.5) * enc.W / P)

    fr = _frames(13, _corner_patch)
    first, steps = _steps_from_frames(fr)
    seeker = Lane("argmax_seeker", fixation=argmax_fix)
    center = _center_lane()
    run_stream([seeker, center], first, steps)
    rs, rc = seeker.report(), center.report()
    assert rs["priority_capture"] == 1.0 and rs["argmax_hit_frac"] == 1.0
    assert rc["priority_capture"] is not None and rc["priority_capture"] < 1.0


def test_salience_capture_ungated_credits_refreshed_hotspot():
    # A lane parked ON the changing patch keeps stamping it, so the staleness gate
    # (IOR) zeroes the GATED priority right where it is correctly looking.  The
    # UNGATED salience_capture must still credit it, and must stay discriminative
    # (a lane fixating the far corner scores strictly lower).
    fr = _frames(13, _corner_patch)
    first, steps = _steps_from_frames(fr)
    on = Lane("on_change", fixation=_fix_at(24.0, 24.0))
    far = Lane("far_away", fixation=_fix_at(120.0, 136.0))
    run_stream([on, far], first, steps)
    ro, rf = on.report(), far.report()
    assert ro["salience_capture"] == 1.0
    assert rf["salience_capture"] is not None
    assert rf["salience_capture"] < ro["salience_capture"]


# ------------------------------------------------------------- t4: change_capture
def test_change_capture_cover_vs_far():
    fr = _frames(13, _flash_patch)
    first, steps = _steps_from_frames(fr)
    on = Lane("over_change", fixation=_fix_at(24.0, 24.0))      # window covers patch
    far = Lane("far_away", fixation=_fix_at(120.0, 136.0))      # opposite corner
    run_stream([on, far], first, steps)
    assert on.report()["change_capture"] == 1.0
    assert far.report()["change_capture"] == 0.0


# ------------------------------------------------------- t5: the open-loop invariant
def test_open_loop_trace_is_lane_independent():
    fr = _frames(13, _corner_patch)
    trace = [0, 3, 4, 8, 6, 1]
    first, steps = _steps_from_frames(fr, trace)
    t_one = run_stream([_center_lane()], first, steps)
    t_many = run_stream(
        [_center_lane(), Lane("fix", fixation=_fix_at(24.0, 24.0)),
         Lane("random", controller=RandomGaze(np.random.default_rng(0)))],
        first, steps)
    assert t_one == t_many == trace        # buttons never depend on which lanes run
    # and trace builders are pure/seed-deterministic (built BEFORE any lane exists)
    assert build_trace("s1", 40, 0) == build_trace("s1", 40, 0)
    assert build_trace("s3", 40, 0) == build_trace("s3", 40, 0)


# ------------------------------- t6: 'learned' lane == the deployed rl_loop._gaze blend
def test_learned_lane_matches_deployed_blend():
    mu = np.array([[0.3, -0.2]], np.float32)

    class _MockPolicy:
        learned_gaze = True

        def act(self, obs, greedy=False):
            assert greedy
            return {"gaze": torch.tensor(mu)}

    lane = Lane("learned", keep_obs_stride=1)
    lane.controller = LearnedGaze(_MockPolicy(), lane.enc.o_proprio,
                                  lane.enc.reflex_gain, lane.enc.reflex_ema_decay)
    fr = _frames(13, _corner_patch)
    first, steps = _steps_from_frames(fr)
    run_stream([lane], first, steps)
    assert len(lane.obs_kept) == len(lane.cmds) > 0

    # independent bit-for-bit reimplementation of rl_loop.BrainTrainer._gaze
    # (learned mu seeds gdx/gdy, ReflexGaze.blend ADDS the reflex) + the worker's
    # float() cast — on the exact decision-obs sequence this lane generated.
    rg = ReflexGaze(1, lane.enc.reflex_gain, lane.enc.reflex_ema_decay)
    o = lane.enc.o_proprio
    for t, obs in enumerate(lane.obs_kept):
        ldx = np.ascontiguousarray(mu[:, 0], np.float32)
        ldy = np.ascontiguousarray(mu[:, 1], np.float32)
        gdx, gdy = rg.blend(np.asarray(obs[None, :], np.float32), o,
                            ldx.copy(), ldy.copy())
        exp = (float(gdx.astype(np.float32)[0]), float(gdy.astype(np.float32)[0]))
        assert lane.cmds[t] == exp, f"step {t}: {lane.cmds[t]} != {exp}"


def test_reflex_lane_matches_deployed_zero_seed_blend():
    # the 'reflex' lane must equal rl_loop._gaze(obs, 1, learned=None): zero seed +
    # reflex added — the brain4 gaze, byte-identical.
    lane = Lane("reflex", keep_obs_stride=1)
    lane.controller = ReflexOnlyGaze(lane.enc.o_proprio, lane.enc.reflex_gain,
                                     lane.enc.reflex_ema_decay)
    fr = _frames(13, _corner_patch)
    first, steps = _steps_from_frames(fr)
    run_stream([lane], first, steps)
    rg = ReflexGaze(1, lane.enc.reflex_gain, lane.enc.reflex_ema_decay)
    o = lane.enc.o_proprio
    for t, obs in enumerate(lane.obs_kept):
        z = np.zeros(1, np.float32)
        gdx, gdy = rg.blend(np.asarray(obs[None, :], np.float32), o, z.copy(), z.copy())
        exp = (float(gdx.astype(np.float32)[0]), float(gdy.astype(np.float32)[0]))
        assert lane.cmds[t] == exp
