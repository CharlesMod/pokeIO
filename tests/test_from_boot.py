"""From-boot progress metric tests (pokeio.reward.from_boot, task #28).

Three layers:
  1. Pure decode/compose arithmetic on synthetic WRAM (fast, hermetic) — the
     signals, the uninit guard, rollout aggregation, distinct-map counting, the
     game-agnostic generic-spec hook, and determinism.
  2. Env-backed validation (skips without the ROM): a random from-newgame policy
     earns NO milestone progress, and a genuinely-progressed emulator state (poked
     at the CORRECTED Yellow addresses) reads clearly higher.
  3. Pixel-grounded regression guard (skips without the demo corpus): replay the
     real newgame->Pikachu demonstration and confirm the metric reads the first
     milestone. This is the mandatory eyes-on-pixels rule as a permanent test — it
     fails the instant any tap regresses to a Red/Blue (off-by-one) address.

Addresses here are the CORRECTED Yellow taps (Yellow's WRAM save block = Red/Blue
- 1): party $D162, mon1 level $D18B, badges $D355, events $D746.., map $D35D,
money $D346. The historical R/B taps read neighbouring bytes (party-count read the
species byte 84; the map tap read a pointer low byte that minted phantom "glitch
maps"). See the module docstring in from_boot.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokeio.reward.from_boot import (
    YELLOW,
    decode_state,
    generic_spec,
    measure,
    progress_scalar,
)

WRAM_BASE = 0xC000
WRAM_SIZE = 0x2000  # 0xC000..0xDFFF

# Corrected Yellow taps (mirrors from_boot.YELLOW) — used to poke synthetic WRAM.
A_PARTY = 0xD162
A_LEVEL = 0xD18B
A_STRIDE = 0x2C
A_BADGE = 0xD355
A_EVENT = 0xD746
A_MAP = 0xD35D
A_MONEY = 0xD346


def _blank() -> np.ndarray:
    return np.zeros(WRAM_SIZE, dtype=np.uint8)


def _set(w: np.ndarray, addr: int, val: int) -> None:
    w[addr - WRAM_BASE] = val & 0xFF


# --------------------------------------------------------------------------
# 1. pure decode / compose arithmetic
# --------------------------------------------------------------------------
def test_blank_wram_is_floor():
    w = _blank()  # map 0 (Pallet Town), party 0, no badges/events
    m = measure(w)
    assert m["progress_score"] == 0.0
    assert m["maps"] == 1  # the spawn map, but 0 maps *beyond* it
    assert m["party_count"] == 0 and m["badges"] == 0 and m["events"] == 0
    assert m["started"] is False


def test_full_signal_decode_and_composition():
    w = _blank()
    _set(w, A_PARTY, 2)               # party count
    _set(w, A_LEVEL, 12)             # mon1 level
    _set(w, A_LEVEL + A_STRIDE, 8)   # mon2 level
    _set(w, A_BADGE, 0b0000_0011)    # 2 badges
    _set(w, A_EVENT, 0b0000_0111)    # 3 event flags
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
        _set(w, A_PARTY, bad)
        _set(w, A_LEVEL, 50)  # would-be level; must be ignored
        d = decode_state(w)
        assert d["party_count"] == 0
        assert d["party_level"] == 0


def test_rollout_unions_maps_and_maxes_milestones():
    a = _blank()  # map 0, empty party
    b = _blank()
    _set(b, A_MAP, 1)     # a different map
    _set(b, A_PARTY, 1)   # got a Pokemon
    _set(b, A_LEVEL, 5)   # level 5
    c = _blank()
    _set(c, A_MAP, 40)          # yet another map (Oak's lab)
    _set(c, A_PARTY, 0xFF)      # transient uninit read — must not zero the maxed party
    m = measure([a, b, c])
    assert m["maps"] == 3          # {0, 1, 40}
    assert m["party_count"] == 1   # max over the rollout, not the last read
    assert m["party_level"] == 5
    # w_map*(3-1) + 5*1 + 1*5 = 12
    assert m["progress_score"] == pytest.approx(12.0)


def test_distinct_maps_count_uncapped():
    # 12 distinct maps, no milestone progress: each real map beyond spawn is 1
    # point (no cap — the map byte is the REAL map id now, not pointer-byte noise).
    snaps = []
    for mid in range(12):
        w = _blank()
        _set(w, A_MAP, mid)
        snaps.append(w)
    m = measure(snaps)
    assert m["maps"] == 12
    assert m["party_count"] == 0 and m["badges"] == 0
    # maps_beyond = 11, uncapped
    assert m["progress_score"] == pytest.approx(11.0)
    assert m["started"] is False


def test_money_bcd_valid_and_invalid():
    w = _blank()
    _set(w, A_MONEY, 0x12)
    _set(w, A_MONEY + 1, 0x34)
    _set(w, A_MONEY + 2, 0x56)
    assert decode_state(w)["money"] == 123456
    _set(w, A_MONEY, 0x1A)  # 'A' is not a BCD digit
    assert decode_state(w)["money"] is None


def test_determinism():
    w = _blank()
    _set(w, A_PARTY, 3)
    _set(w, A_LEVEL, 7)
    assert measure(w) == measure(w)
    assert progress_scalar(w) == progress_scalar(w)


def test_progress_scalar_matches_measure():
    w = _blank()
    _set(w, A_BADGE, 0xFF)  # all 8 badges
    assert progress_scalar(w) == measure(w)["progress_score"]
    assert measure(w)["badges"] == 8


def test_empty_rollout_is_zero():
    m = measure([])
    assert m["progress_score"] == 0.0 and m["n_snapshots"] == 0


# --------------------------------------------------------------------------
# game-agnostic hook: a 2nd game plugs its own spec (no Yellow semantics)
# --------------------------------------------------------------------------
def test_generic_spec_uses_map_plus_miner_counters_only():
    spec = generic_spec("Game2", map_addr=0xC050, progress_counters=[(0xC100, 2)])
    w = _blank()
    _set(w, 0xC050, 3)      # some map
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
DEMO = Path("assets/demo_pikachu")
_HAVE_ROM = ROM.exists() and STATE.exists()
_HAVE_DEMO = _HAVE_ROM and (DEMO / "demo_actions.npy").exists()


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
    (party/badges/levels/events all 0), and its whole score is just the map term.
    The map count stays SMALL — a regression to the pointer-byte tap ($D35E) would
    balloon it (30+ phantom maps in a few rooms), so a tight bound guards it."""
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
    # only the map term contributes (no milestones)
    assert m["progress_score"] == pytest.approx(YELLOW.w_map * max(0, m["maps"] - 1))
    # real map byte => a random bedroom-bound policy touches only a handful of maps
    assert m["maps"] <= 8


@pytest.mark.skipif(not _HAVE_ROM, reason="ROM/state assets not present")
def test_newgame_single_state_is_floor_and_clean_party():
    from pokeio.emu.env import PokeEnv

    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        env.reset(str(STATE))
        m = measure(env)  # env-object path (single current state)
    finally:
        env.close()
    # newgame bedroom (map 38): party 0 (clean init), no badges/events, not started.
    assert m["map_id"] == 38
    assert m["party_count"] == 0
    assert m["badges"] == 0
    assert m["events"] == 0
    assert m["started"] is False


@pytest.mark.skipif(not _HAVE_ROM, reason="ROM/state assets not present")
def test_genuine_progress_reads_clearly_above_random():
    """Decode-correctness on a REAL progressed emulator state: poke the CORRECTED
    addresses on top of the newgame state, save + reload through PyBoy, and confirm
    the metric reads the milestones and scores far above the random floor.
    """
    import io

    from pyboy import PyBoy

    from pokeio.emu.env import PokeEnv

    p = PyBoy(str(ROM), window="null", sound_emulated=False)
    with open(STATE, "rb") as fh:
        p.load_state(fh)
    p.tick(1, True)
    p.memory[A_PARTY] = 2             # party count
    p.memory[A_LEVEL] = 12           # mon1 level
    p.memory[A_LEVEL + A_STRIDE] = 8  # mon2 level
    p.memory[A_BADGE] = 0x01         # Boulder badge
    p.memory[A_EVENT] = 0b0000_0111  # 3 event flags
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


# --------------------------------------------------------------------------
# 3. pixel-grounded regression guard (skips without the demo corpus)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_DEMO, reason="demo corpus (assets/demo_pikachu) not present")
def test_demo_replay_reads_first_milestone():
    """Replay the deterministic newgame->Pikachu demonstration and confirm the
    metric reads the REAL first milestone from the actual rollout. This is the
    eyes-on-pixels rule as a permanent guard: if a tap regresses to a Red/Blue
    off-by-one address, party would read the species byte (84 -> uninit-zeroed),
    the map would balloon into pointer-byte noise, and these asserts fail."""
    from pokeio.emu.env import PokeEnv

    acts = np.load(DEMO / "demo_actions.npy")
    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        env.reset(str(STATE))
        snaps = [env.raw_wram().copy()]
        for a in acts:
            env.step(int(a))
            snaps.append(env.raw_wram().copy())
        m = measure(snaps)
    finally:
        env.close()

    assert m["party_count"] == 1        # got the starter (Pikachu)
    assert m["party_level"] == 5        # Pikachu L5
    assert m["events"] >= 1             # story flags fired en route
    assert m["map_id"] == 40            # ends in Oak's lab
    assert m["maps"] == 4               # bedroom(38), house 1F(37), Pallet(0), lab(40)
    assert m["started"] is True
    assert m["progress_score"] > 5.0
