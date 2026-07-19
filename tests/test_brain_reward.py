"""Dense from-boot progress reward tests (pokeio.brain.reward, task #30).

The four requirements from the RED-TEAM REVISION, each a test:
  * dense       — a positive reward fires the step real progress happens;
  * teleport-decoupled — a reset re-baselines from the (progressed) state, so handed
                  progress pays nothing; only progress BEYOND the restore earns;
  * monotone    — a faint (party drops) does NOT claw back already-earned reward;
  * coupled     — reward reads the CORRECTED from-boot RAM milestones; an end-to-end
                  demo replay telescopes to the true total progress.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokeio.brain.reward import ProgressReward
from pokeio.reward.from_boot import YELLOW

WRAM_BASE = 0xC000
WRAM_SIZE = 0x2000
A_PARTY = 0xD162
A_LEVEL = 0xD18B
A_STRIDE = 0x2C
A_BADGE = 0xD355
A_EVENT = 0xD746
A_MAP = 0xD35D


def _blank(map_id: int = 38) -> np.ndarray:
    w = np.zeros(WRAM_SIZE, dtype=np.uint8)
    w[A_MAP - WRAM_BASE] = map_id  # default spawn = bedroom
    return w


def _set(w, addr, val):
    w[addr - WRAM_BASE] = val & 0xFF


# -------------------------------------------------------------------- dense
def test_new_map_pays_w_map_each():
    r = ProgressReward(1)
    r.reset(0, _blank(38))                 # spawn bedroom
    assert r.step(0, _blank(38)) == 0.0    # same map, no progress
    assert r.step(0, _blank(37)) == pytest.approx(YELLOW.w_map)   # entered house 1F
    assert r.step(0, _blank(0)) == pytest.approx(YELLOW.w_map)    # entered Pallet
    assert r.step(0, _blank(38)) == 0.0    # revisiting bedroom is not new


def test_starter_pays_w_party_plus_levels():
    r = ProgressReward(1)
    r.reset(0, _blank(40))                  # in Oak's lab, empty
    w = _blank(40)
    _set(w, A_PARTY, 1)
    _set(w, A_LEVEL, 5)                     # Pikachu L5
    got = r.step(0, w)
    assert got == pytest.approx(YELLOW.w_party * 1 + YELLOW.w_level * 5)  # 5 + 5


def test_event_flags_pay_w_event():
    r = ProgressReward(1)
    r.reset(0, _blank(38))
    w = _blank(38)
    _set(w, A_EVENT, 0b0000_0111)          # 3 event flags
    assert r.step(0, w) == pytest.approx(YELLOW.w_event * 3)


# ----------------------------------------------------------- teleport-decoupled
def test_reset_rebaselines_handed_progress_pays_nothing():
    """A Go-Explore restore into a progressed state must pay 0 for what it was
    handed; only progress BEYOND the restore earns reward."""
    r = ProgressReward(1)
    progressed = _blank(40)
    _set(progressed, A_PARTY, 1)
    _set(progressed, A_LEVEL, 5)
    _set(progressed, A_EVENT, 0b0001_1111)  # 5 flags
    r.reset(0, progressed)                  # teleport here — emits nothing
    # a step at the SAME progressed state earns nothing (no new progress)
    assert r.step(0, progressed) == 0.0
    # now make progress beyond the restore: a 2nd Pokemon
    w = progressed.copy()
    _set(w, A_PARTY, 2)
    _set(w, A_LEVEL + A_STRIDE, 3)          # mon2 L3
    assert r.step(0, w) == pytest.approx(YELLOW.w_party * 1 + YELLOW.w_level * 3)


def test_restore_does_not_credit_the_jump():
    """Two envs: one plays from boot, one is restored to a deep state. The restored
    env's baseline reflects its depth, so its next-step reward is only its OWN
    forward progress — not the depth it was teleported into."""
    r = ProgressReward(2)
    r.reset(0, _blank(38))                  # env0 boots at bedroom
    deep = _blank(2)                        # env1 restored to Pewter with 2 badges
    _set(deep, A_BADGE, 0b0000_0011)
    r.reset(1, deep)
    # env1 sitting still earns nothing despite being deep
    assert r.step(1, deep.copy()) == 0.0
    # env0 entering a new map earns the map point
    assert r.step(0, _blank(37)) == pytest.approx(YELLOW.w_map)


# ------------------------------------------------------------------- monotone
def test_faint_does_not_claw_back_reward():
    r = ProgressReward(1)
    r.reset(0, _blank(40))
    w = _blank(40)
    _set(w, A_PARTY, 1)
    _set(w, A_LEVEL, 5)
    assert r.step(0, w) > 0                 # earned the Pokemon
    fainted = _blank(40)                    # party wiped to 0 (a faint / uninit read)
    assert r.step(0, fainted) == 0.0        # NOT negative — reward never clawed back
    # regaining does not double-pay (running max already at 1/5)
    assert r.step(0, w) == 0.0


def test_transient_uninit_read_does_not_zero_progress():
    r = ProgressReward(1)
    r.reset(0, _blank(40))
    w = _blank(40); _set(w, A_PARTY, 1); _set(w, A_LEVEL, 5)
    r.step(0, w)
    glitch = _blank(40); _set(glitch, A_PARTY, 0xFF)  # transient uninit
    assert r.step(0, glitch) == 0.0
    assert r.progress(0)["party_count"] == 1          # max held


# -------------------------------------------------------------------- batched
def test_step_many_batched():
    r = ProgressReward(3)
    r.reset_many([0, 1, 2], [_blank(38), _blank(38), _blank(38)])
    rew = r.step_many([0, 1, 2], [_blank(37), _blank(38), _blank(0)])
    assert rew.shape == (3,)
    assert rew[0] == pytest.approx(YELLOW.w_map)  # env0 new map
    assert rew[1] == 0.0                           # env1 stayed
    assert rew[2] == pytest.approx(YELLOW.w_map)   # env2 new map


def test_started_flag():
    r = ProgressReward(1)
    r.reset(0, _blank(38))
    assert r.started(0) is False
    w = _blank(40); _set(w, A_PARTY, 1)
    r.step(0, w)
    assert r.started(0) is True


# ------------------------------------------- end-to-end demo telescoping (coupled)
ROM = Path("roms/pokemon_yellow.gb")
STATE = Path("roms/yellow_newgame.state")
DEMO = Path("assets/demo_pikachu")
_HAVE_DEMO = ROM.exists() and STATE.exists() and (DEMO / "demo_actions.npy").exists()


@pytest.mark.skipif(not _HAVE_DEMO, reason="demo corpus not present")
def test_demo_reward_telescopes_to_total_progress():
    """Replay the newgame->Pikachu demo through the reward; the SUM of dense
    per-step rewards must equal the total real progress made (Phi_end - Phi_start),
    and every step's reward is >= 0. This ties the dense signal to the corrected
    from-boot milestones end to end."""
    from pokeio.emu.env import PokeEnv
    from pokeio.reward.from_boot import measure

    acts = np.load(DEMO / "demo_actions.npy")
    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    r = ProgressReward(1)
    total = 0.0
    try:
        env.reset(str(STATE))
        w0 = env.raw_wram().copy()
        r.reset(0, w0)
        snaps = [w0]
        for a in acts:
            env.step(int(a))
            w = env.raw_wram().copy()
            step_r = r.step(0, w)
            assert step_r >= 0.0            # monotone: never negative
            total += step_r
            snaps.append(w)
    finally:
        env.close()

    # the metric's own composite over the whole rollout = the telescoped total,
    # minus the spawn baseline (from_boot counts maps beyond spawn identically).
    m = measure(snaps)
    assert total == pytest.approx(m["progress_score"], abs=1e-4)
    assert total > 0.0                       # real progress was made
    assert r.started(0) is True              # got the starter
