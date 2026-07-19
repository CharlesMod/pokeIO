"""Saccade-cadence derivation — domain-agnostic, no hand-tuned constants.

The artificial fovea must saccade at >= human rate per second of the DOMAIN's OWN
time base (game-time for emulators, real-time for a robot), but CAPPED at the rate a
real gaze actuator (servo / brushless PTZ gimbal) can actually SETTLE at — so the
learned gaze transfers to hardware (sim2real). Everything here is DERIVED from
``(domain_fps, action_frames, actuator_ceiling)``; nothing is a hand-picked frame
count (respects the no-hand-tuned-behavior-knobs mandate — the reference rates below
are cited physiological / hardware quantities, not tunable thresholds).

Grounding (saccade-attention research synthesis, 2026-07-19):
  * Human saccade rate ~3-4/s (scene viewing ~3, reading ~4); overt-attention
    benefit saturates ~5 Hz. Eye movements are far more frequent than deliberate
    limb actions, so the gaze sub-loop is DECOUPLED from (and faster than) the
    action step.
  * Real actuator SETTLED re-aim rate is settle-limited (small-move + settle time
    ~50-250 ms), NOT peak slew: ~2-6 Hz overall — direct-drive brushless / tendon
    eye ~3-5 Hz (matches human by construction), geared consumer PTZ ~1-3 Hz,
    exotic piezo settle-capped ~5 Hz. Hardware BRACKETS human; it does not
    dramatically beat it on the transfer-relevant quantity.

This module is intentionally pure arithmetic (float + int), so the eventual C port
is a one-to-one translation.
"""

from __future__ import annotations

from dataclasses import dataclass

# Cited reference rates (Hz) — physiological / hardware constants, NOT behavior knobs.
HUMAN_SACCADE_HZ = 4.0          # the rate we match-or-beat (scene ~3, reading ~4)
USEFUL_MAX_HZ = 5.0             # gaze benefit saturates here AND top of the achievable
#                                actuator bracket — no point (and no hardware) beyond.
# Default actuator ceiling: a good direct-drive brushless / tendon eye settles ~5 Hz.
# Override per TARGET actuator (geared consumer PTZ ~2-3; exotic piezo ~5).
DEFAULT_ACTUATOR_CEILING_HZ = 5.0

_EPS = 1e-9


@dataclass(frozen=True)
class SaccadeCadence:
    """Derived saccade sub-stepping for one domain/embodiment."""

    domain_fps: float           # frames per second of the domain's own clock
    action_frames: int          # frames advanced per policy action (e.g. frame_skip)
    target_hz: float            # derived target saccade rate (capped at actuator ceiling)
    frames_per_saccade: int     # domain frames between re-aims
    sub_steps: int              # S: re-aims per action step (>= 1)
    achieved_hz: float          # actual saccade rate given integer S
    actuator_ceiling_hz: float  # the hardware settled re-aim ceiling used

    @property
    def action_hz(self) -> float:
        """Policy action rate (UNCHANGED by sub-stepping — the gaze loop is decoupled)."""
        return self.domain_fps / self.action_frames

    @property
    def at_or_above_human(self) -> bool:
        """Does the achieved gaze rate match-or-beat the human saccade rate?"""
        return self.achieved_hz >= HUMAN_SACCADE_HZ - _EPS

    @property
    def within_actuator(self) -> bool:
        """sim2real guard: never command re-aims faster than the actuator can settle."""
        return self.achieved_hz <= self.actuator_ceiling_hz + _EPS


def derive_cadence(
    domain_fps: float,
    action_frames: int,
    actuator_ceiling_hz: float = DEFAULT_ACTUATOR_CEILING_HZ,
    useful_max_hz: float = USEFUL_MAX_HZ,
) -> SaccadeCadence:
    """Derive the saccade sub-stepping for a domain from first principles.

    ``target_hz`` = the fastest rate that both HELPS (<= useful_max) and a real gimbal
    can SETTLE at (<= actuator ceiling), floored at the action rate (never slower than
    one re-aim per action step). ``frames_per_saccade`` = round(fps / target_hz);
    ``sub_steps`` S = round(action_frames / frames_per_saccade), then reduced if the
    integer S would exceed the actuator ceiling (sim2real cap). The ACTION rate
    (fps / action_frames) is untouched — only the gaze sub-loop speeds up.

    On a slower rung/actuator the same call yields fewer (or 1) sub-steps: e.g. a
    ~2-3 Hz geared PTZ collapses to S=1 (no super-hardware gaze), while a ~5 Hz
    brushless/tendon eye gives S=2 on Game Boy (~5 Hz, just above human).
    """
    fps = float(domain_fps)
    af = max(1, int(action_frames))
    ceiling = float(actuator_ceiling_hz)
    action_rate = fps / af

    # fastest rate that helps AND the gimbal can settle; never below the action rate.
    target = min(ceiling, float(useful_max_hz))
    target = max(target, action_rate)

    frames_per_saccade = max(1, round(fps / target))
    S = max(1, round(af / frames_per_saccade))
    # sim2real cap: shed sub-steps until the achieved rate fits under the ceiling.
    while S > 1 and (S * action_rate) > ceiling + _EPS:
        S -= 1
    achieved = S * action_rate
    # frames_per_saccade reflects the realized S (so callers can chunk the advance).
    fps_per_sacc = max(1, round(af / S))
    return SaccadeCadence(
        domain_fps=fps,
        action_frames=af,
        target_hz=target,
        frames_per_saccade=fps_per_sacc,
        sub_steps=S,
        achieved_hz=achieved,
        actuator_ceiling_hz=ceiling,
    )


def chunk_frames(action_frames: int, sub_steps: int) -> list[int]:
    """Split ``action_frames`` into ``sub_steps`` near-equal chunks (sum == total).

    The worker holds the button for the whole action and advances the emulator one
    chunk at a time, re-aiming the fovea (+ stamping the canvas) at each boundary.
    Any remainder is spread over the LAST chunks so the biggest gaps come first
    (earliest re-aim). ``sum(chunk_frames(af, S)) == af`` for all S in [1, af]."""
    af = max(1, int(action_frames))
    S = max(1, min(int(sub_steps), af))
    base, rem = divmod(af, S)
    # front chunks get `base`, the last `rem` chunks get one extra -> sums to af.
    return [base + (1 if i >= S - rem else 0) for i in range(S)]


__all__ = [
    "SaccadeCadence",
    "derive_cadence",
    "chunk_frames",
    "HUMAN_SACCADE_HZ",
    "USEFUL_MAX_HZ",
    "DEFAULT_ACTUATOR_CEILING_HZ",
]
