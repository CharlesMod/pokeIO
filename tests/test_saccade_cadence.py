"""Saccade-cadence derivation tests (pokeio.emu.saccade_cadence).

The cadence must be DERIVED from (domain_fps, action_frames, actuator ceiling) with
no hand-tuned frame count, must match-or-beat human on capable hardware, and must
NEVER exceed the actuator's settled ceiling (sim2real cap).
"""

from __future__ import annotations

from pokeio.emu.saccade_cadence import (
    DEFAULT_ACTUATOR_CEILING_HZ,
    HUMAN_SACCADE_HZ,
    chunk_frames,
    derive_cadence,
)

GB_FPS = 59.7275  # Game Boy DMG refresh
GB_ACTION_FRAMES = 24  # frame_skip


def test_gameboy_defaults_reach_just_above_human():
    c = derive_cadence(GB_FPS, GB_ACTION_FRAMES)
    # current single-saccade-per-step rate is BELOW human
    assert abs(c.action_hz - GB_FPS / GB_ACTION_FRAMES) < 1e-6
    assert c.action_hz < HUMAN_SACCADE_HZ            # 2.49 Hz < 4 Hz
    # sub-stepping lifts it into the human band
    assert c.sub_steps == 2
    assert abs(c.achieved_hz - 2 * GB_FPS / GB_ACTION_FRAMES) < 1e-6  # ~4.98 Hz
    assert c.at_or_above_human
    assert c.within_actuator
    # frames_per_saccade chunks the 24-frame advance into 2x12
    assert chunk_frames(GB_ACTION_FRAMES, c.sub_steps) == [12, 12]


def test_action_rate_is_untouched_by_substepping():
    c = derive_cadence(GB_FPS, GB_ACTION_FRAMES)
    # the policy still acts once per action_frames — only the gaze loop sped up
    assert abs(c.action_hz - GB_FPS / GB_ACTION_FRAMES) < 1e-6
    assert c.achieved_hz > c.action_hz


def test_slow_geared_ptz_collapses_to_one_saccade_per_step():
    # a ~2.5 Hz geared consumer PTZ can't settle faster than the action rate:
    c = derive_cadence(GB_FPS, GB_ACTION_FRAMES, actuator_ceiling_hz=2.5)
    assert c.sub_steps == 1                          # no super-hardware gaze
    assert abs(c.achieved_hz - GB_FPS / GB_ACTION_FRAMES) < 1e-6
    assert c.within_actuator                         # never exceeds the gimbal ceiling
    # hardware-limited below human — correctly refuses to fake a faster eye
    assert not c.at_or_above_human


def test_never_exceeds_actuator_ceiling_across_a_sweep():
    for ceiling in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0):
        c = derive_cadence(GB_FPS, GB_ACTION_FRAMES, actuator_ceiling_hz=ceiling)
        # sim2real invariant: achieved rate is at or below the ceiling, unless even
        # ONE re-aim per action step already exceeds it (then S=1 is the floor).
        assert c.sub_steps >= 1
        if c.action_hz <= ceiling + 1e-9:
            assert c.achieved_hz <= ceiling + 1e-9


def test_faster_domain_rederives_frames():
    # a higher-fps rung (e.g. N64 ~60 fps, shorter action window) re-derives per rung
    c = derive_cadence(60.0, 12, actuator_ceiling_hz=DEFAULT_ACTUATOR_CEILING_HZ)
    assert c.sub_steps >= 1
    assert c.within_actuator
    assert sum(chunk_frames(12, c.sub_steps)) == 12


def test_chunk_frames_partitions_exactly():
    for af in (24, 12, 20, 17, 1, 60):
        for S in range(1, af + 1):
            chunks = chunk_frames(af, S)
            assert len(chunks) == S
            assert sum(chunks) == af
            assert all(x >= 1 for x in chunks)
            # near-equal: max and min differ by at most 1
            assert max(chunks) - min(chunks) <= 1
