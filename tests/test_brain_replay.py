"""Backward-robustification curriculum + demo trajectory tests
(pokeio.brain.replay, task #31)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokeio.brain.replay import BackwardCurriculum, DemoTrajectory


# ------------------------------------------------------------- curriculum
def test_frontier_starts_near_the_end():
    c = BackwardCurriculum(length=657, start_frac=0.97)
    assert 600 < c.frontier <= 657
    assert c.at_boot is False


def test_success_recedes_frontier_failure_stalls():
    win = BackwardCurriculum(length=657, seed=1)
    f0 = win.frontier
    for _ in range(20):
        win.report(True)
    lose = BackwardCurriculum(length=657, seed=1)
    for _ in range(20):
        lose.report(False)
    assert win.frontier < f0                 # winning recedes the start toward boot
    assert lose.frontier == pytest.approx(f0)  # failing does not advance (ema ~0)
    assert win.frontier < lose.frontier


def test_report_many_recession_is_nenvs_independent():
    """The bug that skipped the whole curriculum at 64 envs: recession must be
    per-ITERATION (gated by the batch success rate), not per-env — else 64 envs
    recede 64x faster than 4 and jump straight to boot in one iteration."""
    big = BackwardCurriculum(length=657, advance_gain=8.0)
    small = BackwardCurriculum(length=657, advance_gain=8.0)
    f0 = big.frontier
    big.report_many([True] * 64)    # 64 successes, one iteration
    small.report_many([True] * 4)   # 4 successes, one iteration
    assert (f0 - big.frontier) == pytest.approx(8.0)      # recede by rate(1.0)*gain
    assert big.frontier == pytest.approx(small.frontier)  # n_envs-independent


def test_report_many_recedes_by_batch_rate():
    c = BackwardCurriculum(length=657, advance_gain=8.0)
    f0 = c.frontier
    c.report_many([True, True, False, False])   # 50% success this iteration
    assert (f0 - c.frontier) == pytest.approx(4.0)  # 0.5 * 8


def test_frontier_reaches_boot_under_sustained_success():
    c = BackwardCurriculum(length=657, advance_gain=8.0, seed=2)
    for _ in range(2000):
        c.report(True)
        if c.at_boot:
            break
    assert c.at_boot is True
    assert c.frontier == 0.0
    assert c.sample_depths(4) == [0, 0, 0, 0]  # all cold boots once receded


def test_sample_depths_cluster_at_frontier():
    c = BackwardCurriculum(length=657, spread_frac=0.1, seed=3)
    depths = c.sample_depths(200)
    assert all(0 <= d <= 657 for d in depths)
    assert abs(np.mean(depths) - c.frontier) < 0.2 * 657  # clustered near frontier


def test_curriculum_state_report():
    c = BackwardCurriculum(length=657)
    st = c.state()
    assert set(st) >= {"frontier", "frontier_frac", "success_ema", "at_boot", "seen"}


# ------------------------------------------------------- demo trajectory (needs ROM)
ROM = Path("roms/pokemon_yellow.gb")
STATE = Path("roms/yellow_newgame.state")
DEMO = Path("assets/demo_pikachu/demo_actions.npy")
_HAVE = ROM.exists() and STATE.exists() and DEMO.exists()


@pytest.mark.skipif(not _HAVE, reason="ROM/demo assets not present")
def test_demo_len_and_blob_capture():
    t = DemoTrajectory()
    assert len(t) == 657
    boot = t.blob_at(0)
    end = t.blob_at(len(t))
    assert isinstance(boot, bytes) and len(boot) > 1000
    assert boot != end                       # different states along the trajectory
    assert t.blob_at(0) is boot               # cached (same object)


@pytest.mark.skipif(not _HAVE, reason="ROM/demo assets not present")
def test_restore_map_drops_boot_depth():
    t = DemoTrajectory()
    m = t.restore_map({0: 0, 1: 100, 2: 300})
    assert 0 not in m                        # depth 0 = cold boot, no restore blob
    assert set(m) == {1, 2}
    assert all(isinstance(v, bytes) for v in m.values())


@pytest.mark.skipif(not _HAVE, reason="ROM/demo assets not present")
def test_blob_at_end_reaches_the_milestone():
    """The blob at full depth is the progressed (Pikachu) state — the reward's
    started() reads party>0 on it."""
    from pokeio.brain.reward import ProgressReward
    from pokeio.emu.env import PokeEnv

    t = DemoTrajectory()
    blob = t.blob_at(len(t))
    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        env.load_state(blob)
        env.pyboy.tick(1, True)
        r = ProgressReward(1)
        r.reset(0, env.raw_wram())
    finally:
        env.close()
    assert r.started(0) is True
