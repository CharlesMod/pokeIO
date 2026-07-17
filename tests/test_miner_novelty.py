"""WRAM progress-miner statistics and novelty rarity math (audit A10 / findings
#12, #26, #31).

Both feed selection fitness — the miner promotes mined counters to progress
taps, and the novelty rarity weight is a primary fitness term — yet neither had
a unit test. These pin the properties the reward stack depends on:

Miner (``pokeio.reward.miner.mine``):
  * a clean monotone counter outranks a noisier (higher-increment-entropy) but
    still directional byte;
  * a static byte (never changes) is excluded;
  * an autonomous / free-running byte (changes on essentially every step — a
    per-frame timer) is excluded by the activity band even though it is
    perfectly monotone. Anything that advances on its own is not a *progress*
    counter.

Novelty (``pokeio.reward.novelty.WaveNovelty`` + ``NoveltyArchive``):
  * rarity credit for a distinct reached cell is ``floor + 1/sqrt(1 + prior)``
    and decays monotonically toward ``floor`` as prior visits rise;
  * the spawn (handed) cell is never credited — only cells an agent *reaches*;
  * a distinct cell is credited at most once per genome per episode.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pokeio.reward.archive import NoveltyArchive
from pokeio.reward.miner import Candidate, MinerConfig, WRAM_BASE, mine
from pokeio.reward.novelty import WaveNovelty


# --------------------------------------------------------------------------
# miner
# --------------------------------------------------------------------------
_T = 210  # steps per rollout


def _synth_rollout(seed: int) -> np.ndarray:
    """A (T, 6) WRAM trace with columns of known character:

    col0 clean counter   +1 every 6 steps          -> ideal progress counter
    col1 irregular       +[1..5] every 6 steps, up  -> monotone but higher entropy
    col2 static          constant                    -> excluded (never active)
    col3 autonomous      +1 EVERY step (timer)       -> excluded (hyperactive)
    col4 noise           random value every 6 steps  -> not directional
    col5 static          constant                    -> excluded
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((_T, 6), dtype=np.int32)
    c0 = c1 = n4 = 0
    for t in range(_T):
        if t and t % 6 == 0:
            c0 += 1
            c1 += int(rng.integers(1, 6))
            n4 = int(rng.integers(0, 256))
        m[t, 0] = c0 % 256
        m[t, 1] = c1 % 256
        m[t, 2] = 42
        m[t, 3] = t % 256
        m[t, 4] = n4
        m[t, 5] = 7
    return m


def _mine_default_bytes():
    rolls = [_synth_rollout(s) for s in (1, 2, 3)]
    cands = mine(rolls, cfg=MinerConfig(consider_pairs=False))
    return cands, {c.address: c for c in cands}


def test_clean_counter_is_top_candidate():
    cands, by_addr = _mine_default_bytes()
    assert cands, "miner returned no candidates"
    assert cands[0].address == WRAM_BASE + 0
    assert cands[0].direction == "increasing"
    assert cands[0].stats["monotonicity"] == pytest.approx(1.0)


def test_counter_outranks_higher_entropy_byte():
    _, by_addr = _mine_default_bytes()
    clean = by_addr[WRAM_BASE + 0]
    irregular = by_addr[WRAM_BASE + 1]
    # both are directional (monotone), but the irregular byte's varied
    # increments raise its delta-entropy and cost it score.
    assert irregular.stats["delta_entropy"] > clean.stats["delta_entropy"]
    assert clean.score > irregular.score


def test_static_byte_excluded():
    _, by_addr = _mine_default_bytes()
    assert WRAM_BASE + 2 not in by_addr
    assert WRAM_BASE + 5 not in by_addr


def test_autonomous_freerunning_byte_excluded():
    """A byte that advances every step is a timer, not progress — the activity
    band (max_activity) must drop it despite perfect monotonicity."""
    _, by_addr = _mine_default_bytes()
    assert WRAM_BASE + 3 not in by_addr


def test_activity_band_rejects_hyperactive_directly():
    # A single column that increments on EVERY step: monotone but activity ~1.0.
    T = 120
    col = np.arange(T, dtype=np.int32).reshape(T, 1) % 256
    cands = mine([col, col + 0], cfg=MinerConfig(consider_pairs=False))
    assert all(c.address != WRAM_BASE + 0 for c in cands)


def test_scores_are_bounded_unit_interval():
    cands, _ = _mine_default_bytes()
    for c in cands:
        assert 0.0 <= c.score <= 1.0
        assert isinstance(c, Candidate)


def test_empty_or_too_short_rollouts_yield_nothing():
    assert mine([]) == []
    # single-snapshot rollouts (T<2) carry no deltas.
    assert mine([np.zeros((1, 8), dtype=np.uint8)]) == []


# --------------------------------------------------------------------------
# novelty rarity math
# --------------------------------------------------------------------------
def _wave(n=4, floor=0.01):
    arch = NoveltyArchive()
    return arch, WaveNovelty(arch, n, mode="rarity", floor=floor)


def _spawn(wave, player, key=b"spawn"):
    # first observation for a player is its spawn cell (never credited).
    credited = wave.observe_key(player, key, globally_new=True, prior_visits=0)
    assert credited is False
    assert wave.fitness[player] == 0.0


def test_rarity_credit_formula_for_fresh_cell():
    floor = 0.01
    _, wave = _wave(floor=floor)
    _spawn(wave, 0)
    credited = wave.observe_key(0, b"cellA", globally_new=True, prior_visits=0)
    assert credited is True
    # brand-new cell: floor + 1/sqrt(1+0) == floor + 1.0
    assert wave.fitness[0] == pytest.approx(floor + 1.0)


def test_rarity_credit_decays_monotonically_toward_floor():
    floor = 0.02
    priors = [0, 1, 3, 9, 99, 9999]
    credits = []
    for p in priors:
        _, wave = _wave(floor=floor)
        _spawn(wave, 0)
        wave.observe_key(0, b"cell", globally_new=True, prior_visits=p)
        credits.append(float(wave.fitness[0]))
    # strictly decreasing as the cell gets more well-trodden ...
    assert all(a > b for a, b in zip(credits, credits[1:]))
    # ... bounded below by the floor, approaching it for a heavily-visited cell.
    assert credits[-1] > floor
    assert credits[-1] == pytest.approx(floor + 1.0 / math.sqrt(1 + 9999))
    assert credits[-1] < floor + 0.02


def test_spawn_cell_never_credited_even_if_rare():
    # A do-nothing genome restored into a rare frontier cell must not be paid
    # for merely standing where it was placed (the 'couch potato' exploit).
    _, wave = _wave()
    got = wave.observe_key(0, b"rare-frontier", globally_new=True, prior_visits=0)
    assert got is False
    assert wave.fitness[0] == 0.0


def test_distinct_cell_credited_once_per_genome():
    _, wave = _wave()
    _spawn(wave, 0)
    first = wave.observe_key(0, b"cellX", globally_new=True, prior_visits=0)
    second = wave.observe_key(0, b"cellX", globally_new=False, prior_visits=1)
    assert first is True
    assert second is False  # same cell again -> no double credit
    assert wave.fitness[0] == pytest.approx(0.01 + 1.0)


def test_players_are_independent():
    _, wave = _wave(n=2)
    _spawn(wave, 0)
    _spawn(wave, 1, key=b"spawn1")
    wave.observe_key(0, b"a", globally_new=True, prior_visits=0)
    # player 1 touched nothing beyond its spawn -> zero credit.
    assert wave.fitness[0] > 0.0
    assert wave.fitness[1] == 0.0


def test_archive_visit_returns_prior_then_increments():
    arch = NoveltyArchive()
    k = b"k"
    assert arch.visit(k) == 0  # first sighting
    assert arch.visit(k) == 1
    assert arch.visit(k) == 2
