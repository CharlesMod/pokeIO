"""From-boot progress metric tests (pokeio.reward.from_boot, task #28).

Two layers:
  1. Pure decode/compose arithmetic on synthetic WRAM (fast, hermetic) — the
     signals, the uninit/sentinel guards, rollout aggregation, the map cap, the
     game-agnostic generic-spec hook, and determinism.
  2. Env-backed validation (skips without the ROM): the load-bearing check that
     the metric reads ~0 on a random from-newgame policy and reads clearly higher
     on a genuinely progressed state (party/badges/levels poked at the VERIFIED
     addresses, saved + reloaded through the real emulator).

Validation finding (see the module + the returned report): NO in-repo policy or
script makes real from-newgame progress (the project's open hard-exploration
problem), so the competent-vs-random separation on the milestone signals is shown
via a genuinely-progressed emulator state rather than a live competent rollout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokeio.reward.from_boot import (
    YELLOW,
    GameProgressSpec,
    decode_state,
    generic_spec,
    measure,
    progress_scalar,
)

WRAM_BASE = 0xC000
WRAM_SIZE = 0x2000  # 0xC000..0xDFFF


def _blank() -> np.ndarray:
    return np.zeros(WRAM_SIZE, dtype=np.uint8)


def _set(w: np.ndarray, addr: int, val: int) -> None:
    w[addr - WRAM_BASE] = val & 0xFF


# --------------------------------------------------------------------------
# 1. pure decode / compose arithmetic
# --------------------------------------------------------------------------
def test_blank_wram_is_floor():
    w = _blank()  # map 0 (real), party 0, no badges/events
    m = measure(w)
    assert m["progress_score"] == 0.0
    assert m["maps"] == 1  # the (real) spawn map, but 0 maps *beyond* it
    assert m["party_count"] == 0 and m["badges"] == 0 and m["events"] == 0
    assert m["started"] is False


def test_full_signal_decode_and_composition():
    w = _blank()
    _set(w, 0xD163, 2)              # party count
    _set(w, 0xD18C, 12)            # mon1 level
    _set(w, 0xD18C + 0x2C, 8)      # mon2 level
    _set(w, 0xD356, 0b0000_0011)  # 2 badges
    _set(w, 0xD747, 0b0000_0111)  # 3 event flags
    m = measure(w)
    assert m["party_count"] == 2
    assert m["party_level"] == 20
    assert m["badges"] == 2
    assert m["events"] == 3
    assert m["started"] is True
    # composite: w_map*0 + 5*2 + 1*20 + 50*2 + 2*3 = 136
    assert m["progress_score"] == pytest.approx(136.0)


def test_party_count_uninit_and_garbage_are_zero():
    for bad in (0xFF, 9, 200):
        w = _blank()
        _set(w, 0xD163, bad)
        _set(w, 0xD18C, 50)  # would-be level; must be ignored
        d = decode_state(w)
        assert d["party_count"] == 0
        assert d["party_level"] == 0


def test_sentinel_maps_do_not_count_as_progress():
    w = _blank()
    _set(w, 0xD35E, 0xF5)  # loader/transition sentinel
    m = measure(w)
    assert m["maps"] == 0
    assert m["progress_score"] == 0.0


def test_rollout_unions_maps_and_maxes_milestones():
    a = _blank()  # map 0, empty party
    b = _blank()
    _set(b, 0xD35E, 1)   # a different real map
    _set(b, 0xD163, 1)   # got a Pokemon
    _set(b, 0xD18C, 5)   # level 5
    c = _blank()
    _set(c, 0xD35E, 0xFA)  # sentinel mid-transition — ignored
    _set(c, 0xD163, 0xFF)  # transient uninit read — must not zero the maxed party
    m = measure([a, b, c])
    assert m["maps"] == 2          # {0, 1}; sentinel excluded
    assert m["party_count"] == 1   # max over the rollout, not the last read
    assert m["party_level"] == 5
    # w_map*min(2-1,6)=1 + 5*1 + 1*5 = 11
    assert m["progress_score"] == pytest.approx(11.0)


def test_map_term_is_capped_against_glitch_thrash():
    # 12 distinct real maps but NO milestone progress (the glitch-thrash pattern).
    snaps = []
    for mid in range(12):
        w = _blank()
        _set(w, 0xD35E, mid)
        snaps.append(w)
    m = measure(snaps)
    assert m["maps"] == 12
    assert m["party_count"] == 0 and m["badges"] == 0
    # maps_beyond=11 capped to YELLOW.map_cap (6): score == 6, not 11.
    assert m["progress_score"] == pytest.approx(6.0)
    assert m["started"] is False


def test_money_bcd_valid_and_invalid():
    w = _blank()
    _set(w, 0xD347, 0x12)
    _set(w, 0xD348, 0x34)
    _set(w, 0xD349, 0x56)
    assert decode_state(w)["money"] == 123456
    _set(w, 0xD347, 0x1A)  # 'A' is not a BCD digit
    assert decode_state(w)["money"] is None


def test_determinism():
    w = _blank()
    _set(w, 0xD163, 3)
    _set(w, 0xD18C, 7)
    assert measure(w) == measure(w)
    assert progress_scalar(w) == progress_scalar(w)


def test_progress_scalar_matches_measure():
    w = _blank()
    _set(w, 0xD356, 0xFF)  # all 8 badges
    assert progress_scalar(w) == measure(w)["progress_score"]
    assert measure(w)["badges"] == 8


def test_empty_rollout_is_zero():
    m = measure([])
    assert m["progress_score"] == 0.0 and m["n_snapshots"] == 0


# --------------------------------------------------------------------------
# game-agnostic hook: a 2nd game plugs its own spec (no Yellow semantics)
# --------------------------------------------------------------------------
def test_generic_spec_uses_map_plus_miner_counters_only():
    spec = generic_spec("Game2", map_addr=0xC050, progress_counters=[(0xC100, 2)],
                         map_cap=None)
    w = _blank()
    _set(w, 0xC050, 3)      # some real map
    _set(w, 0xC100, 0x10)   # counter low byte
    _set(w, 0xC101, 0x01)   # counter high byte -> 0x0110 = 272
    m = measure(w, spec)
    assert m["counter_sum"] == 272
    # no Yellow semantics leak: party/badges/events stay 0 (addrs are None)
    assert m["party_count"] == 0 and m["badges"] == 0 and m["events"] == 0
    # single snapshot -> maps_beyond 0; score is just the counter term (w_counter=1)
    assert m["progress_score"] == pytest.approx(272.0)


def test_generic_spec_has_no_yellow_addresses():
    spec = generic_spec("Game2", map_addr=0xC050, progress_counters=[])
    assert spec.party_count_addr is None
    assert spec.badge_addr is None
    assert spec.event_regions == ()


# --------------------------------------------------------------------------
# 2. env-backed validation (skips cleanly without the ROM)
# --------------------------------------------------------------------------
ROM = Path("roms/pokemon_yellow.gb")
STATE = Path("roms/yellow_newgame.state")
_HAVE_ROM = ROM.exists() and STATE.exists()


def _rollout_snaps(env, policy, n, seed=0):
    env.reset(str(STATE))
    snaps = [env.raw_wram().copy()]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = 8 if policy == "noop" else int(rng.integers(0, 9))
        env.step(a)
        snaps.append(env.raw_wram().copy())
    return snaps


@pytest.mark.skipif(not _HAVE_ROM, reason="ROM/state assets not present")
def test_random_from_newgame_stays_at_floor():
    """The load-bearing floor check: a random policy earns NO milestone progress
    (party/badges/levels/events all 0) — only bounded local-map wandering."""
    from pokeio.emu.env import PokeEnv

    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        m = measure(_rollout_snaps(env, "random", 200, seed=1))
    finally:
        env.close()
    assert m["party_count"] == 0
    assert m["badges"] == 0
    assert m["party_level"] == 0
    assert m["events"] == 0
    assert m["started"] is False
    # bounded by the map cap regardless of how far random thrashes.
    assert m["progress_score"] <= YELLOW.w_map * YELLOW.map_cap


@pytest.mark.skipif(not _HAVE_ROM, reason="ROM/state assets not present")
def test_newgame_single_state_is_floor_and_uninit_party():
    from pokeio.emu.env import PokeEnv

    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        env.reset(str(STATE))
        m = measure(env)  # env-object path (single current state)
    finally:
        env.close()
    # newgame bedroom: party uninit (0xFF -> 0), no badges/events, started False.
    assert m["party_count"] == 0
    assert m["started"] is False


@pytest.mark.skipif(not _HAVE_ROM, reason="ROM/state assets not present")
def test_genuine_progress_reads_clearly_above_random():
    """Decode-correctness on a REAL progressed emulator state: poke the VERIFIED
    addresses on top of the newgame state, save + reload through PyBoy, and
    confirm the metric reads the milestones and scores far above the random floor.
    """
    import io

    from pyboy import PyBoy

    from pokeio.emu.env import PokeEnv

    p = PyBoy(str(ROM), window="null", sound_emulated=False)
    with open(STATE, "rb") as fh:
        p.load_state(fh)
    p.tick(1, True)
    p.memory[0xD163] = 2            # party count
    p.memory[0xD18C] = 12          # mon1 level
    p.memory[0xD18C + 0x2C] = 8    # mon2 level
    p.memory[0xD356] = 0x01        # Boulder badge
    p.memory[0xD747] = 0b0000_0111  # 3 event flags
    p.tick(1, False)
    buf = io.BytesIO()
    p.save_state(buf)
    p.stop(save=False)

    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        env.load_state(buf.getvalue())
        env.pyboy.tick(1, True)
        m = measure(env)
        rand = measure(_rollout_snaps(env, "random", 200, seed=3))
    finally:
        env.close()

    assert m["party_count"] == 2
    assert m["party_level"] == 20
    assert m["badges"] == 1
    assert m["events"] == 3
    assert m["started"] is True
    # genuine progress (a badge + two leveled Pokemon) dwarfs the random floor.
    assert m["progress_score"] > 50.0
    assert m["progress_score"] > 5 * rand["progress_score"] + 10
