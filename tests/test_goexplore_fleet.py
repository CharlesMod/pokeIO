"""Correctness/robustness regressions for the Go-Explore + fleet audit fixes.

Covers the four audit items whose fixes live in
``pokeio/emu/fleet.py``, ``pokeio/reward/goexplore.py`` and
``pokeio/reward/novelty.py``:

* **A5** save_state truncation — buffers sized from a probed state; captures are
  SKIPPED (never truncated) when a blob outgrows the buffer.
* **A6** Go-Explore detachment — a proven deep multi-visit hub survives many
  idle generations of shallow churn instead of being recency-evicted.
* **A7** rarity-credit order-dependence — co-visitors of a fresh cell earn equal
  credit regardless of player index / order.
* **A8** hung-worker watchdog — a hung-but-alive worker trips a wall-clock
  deadline that names the offending env index + pid.

The fleet tests spawn real PokeEnv workers, so they need the Yellow ROM; they
skip cleanly if it is absent.
"""

from __future__ import annotations

import os
import signal
import time

import numpy as np
import pytest

from pokeio.emu.fleet import (
    BarrierFleet,
    _probe_state_len,
    _state_capacity,
    _store_capture,
)
from pokeio.reward.archive import NoveltyArchive
from pokeio.reward.goexplore import GoExplore
from pokeio.reward.novelty import WaveNovelty

_ROM = os.path.join(os.path.dirname(__file__), os.pardir, "roms", "pokemon_yellow.gb")
_STATE = os.path.join(os.path.dirname(__file__), os.pardir, "roms", "yellow_newgame.state")
_HAVE_ROM = os.path.exists(_ROM) and os.path.exists(_STATE)
_rom_only = pytest.mark.skipif(not _HAVE_ROM, reason="Yellow ROM/state not present")

# Small obs geometry so the shared-memory blocks stay tiny in tests.
_OBS_RES = 8
_OBS_RAM = 4
_OBS_DIM = _OBS_RES * _OBS_RES + _OBS_RAM


def _fleet(goexplore: bool, n_envs: int = 1, **kw) -> BarrierFleet:
    return BarrierFleet(
        n_envs=n_envs,
        obs_dim=_OBS_DIM,
        obs_res=_OBS_RES,
        obs_ram=_OBS_RAM,
        rom_path=_ROM,
        frame_skip=1,
        hold_frames=0,
        reset_state=_STATE,
        archive_kwargs={},
        wram_stride=64,
        goexplore=goexplore,
        envs_per_worker=1,
        **kw,
    )


# =====================================================================  A5
def test_store_capture_never_truncates() -> None:
    """The store helper writes a blob that fits and SKIPS (never truncates) one
    that does not — a truncated blob would corrupt every restore made from it."""
    cap = 32
    row = np.zeros(cap, dtype=np.uint8)

    fits = bytes(range(20))
    n = _store_capture(row, fits, cap)
    assert n == 20
    assert bytes(row[:20]) == fits

    row2 = np.zeros(cap, dtype=np.uint8)
    oversize = bytes([7] * (cap + 1))
    n2 = _store_capture(row2, oversize, cap)
    assert n2 == -1  # signals SKIP
    assert not row2.any()  # nothing was written — no partial/truncated blob


def test_state_capacity_has_headroom_over_probe() -> None:
    assert _state_capacity(None) > 0  # falls back to a sane bound
    assert _state_capacity(0) == _state_capacity(None)
    probe = 200_592  # measured Yellow save_state size
    cap = _state_capacity(probe)
    assert cap > probe  # strictly larger so the real state always fits
    assert cap - probe >= 65_536  # generous headroom


@_rom_only
def test_probe_reports_real_state_size() -> None:
    probe = _probe_state_len(_ROM, 1, 0, _STATE)
    assert probe is not None and probe > 100_000  # a real GB save_state


@_rom_only
def test_capture_stores_full_untruncated_blob() -> None:
    """A real Go-Explore capture round stores a FULL-length blob (buffers sized
    from the probe) and never records a truncation."""
    probe = _probe_state_len(_ROM, 1, 0, _STATE)
    fleet = _fleet(goexplore=True, n_envs=1)
    try:
        assert fleet.state_cap > probe  # buffer strictly bigger than the state
        fleet.reset_all()
        # Request a capture of the (post-reset) state before this step applies.
        _obs, _keys, _dones, captured = fleet.step_all(
            np.zeros(1, dtype=np.int32), capture_flags=np.ones(1, dtype=np.uint8)
        )
        assert 0 in captured
        assert len(captured[0]) == probe  # full state, NOT truncated to a cap
        assert fleet.state_truncations == 0
        # The captured blob round-trips as a restore without error.
        obs = fleet.reset_all(restore={0: captured[0]})
        assert obs.shape == (1, _OBS_DIM)
    finally:
        fleet.close()


@_rom_only
def test_restore_guard_skips_oversize_blob() -> None:
    """An oversize restore blob is skipped, never loaded truncated (A5)."""
    fleet = _fleet(goexplore=True, n_envs=1)
    try:
        fleet.reset_all()
        huge = bytes(fleet.state_cap + 1)
        fleet.reset_all_begin(restore={0: huge})
        assert int(fleet.arr["res_flag"][0]) == 0  # skipped: not flagged
        assert int(fleet.arr["res_len"][0]) == 0
        fleet.reset_all_end()
    finally:
        fleet.close()


# =====================================================================  A6
def _add_cell(go: GoExplore, key: bytes, depth: int, visits: int = 1) -> None:
    go.store_captured(key, b"state", depth=depth)
    for _ in range(visits - 1):
        go.revisit(key)


def test_deep_hub_survives_idle_shallow_churn() -> None:
    """A proven deep multi-visit hub is NOT recency-evicted across ~30 idle gens
    of shallow one-off churn (the A6 detachment regression)."""
    go = GoExplore(
        capacity=64,
        recency_halflife=4.0,
        rng=np.random.default_rng(0),
    )
    go.begin_generation(0)
    hub = b"DEEP-HUB"
    _add_cell(go, hub, depth=5000, visits=21)  # deep + proven multi-visit
    assert go.cells[hub].visits > 1

    # 30 idle generations: each floods the archive with shallow one-off cells
    # (visits=1, small depth) and NEVER revisits the hub, so its recency decays.
    for gen in range(1, 31):
        go.begin_generation(gen)
        for j in range(200):  # >> capacity -> forces eviction every gen
            _add_cell(go, f"churn-{gen}-{j}".encode(), depth=j % 10)

    assert hub in go.cells, "deep multi-visit hub was evicted (detachment)"
    assert go.cells[hub].depth == 5000
    assert go.size <= go.capacity  # capacity still bounded
    assert go.n_evicted > 0  # churn really did trigger evictions


def test_shallow_oneoffs_are_the_ones_evicted() -> None:
    """Sanity: with the hub protected, it is the stale shallow one-offs that go —
    eviction still removes something every time the archive is full."""
    go = GoExplore(capacity=32, recency_halflife=4.0, rng=np.random.default_rng(1))
    go.begin_generation(0)
    _add_cell(go, b"HUB", depth=9000, visits=15)
    for gen in range(1, 11):
        go.begin_generation(gen)
        for j in range(64):
            _add_cell(go, f"c-{gen}-{j}".encode(), depth=1)
    assert b"HUB" in go.cells
    assert go.size == go.capacity


# =====================================================================  A7
def _rarity_run_key(order: list[int]) -> np.ndarray:
    """Two players co-visit the SAME fresh cell in ``order`` (observe_key path)."""
    arch = NoveltyArchive()
    wave = WaveNovelty(arch, n_players=2, mode="rarity", floor=0.01)
    # Distinct spawn cells first (the spawn is archived but never credited).
    for p in (0, 1):
        spawn = f"spawn-{p}".encode()
        gnew = arch.add(spawn)
        prior = arch.visit(spawn)
        wave.observe_key(p, spawn, gnew, prior)
    fresh = b"SHARED-FRESH-CELL"
    for p in order:
        gnew = arch.add(fresh)
        prior = arch.visit(fresh)  # MONOTONIC shared counter (the bug source)
        wave.observe_key(p, fresh, gnew, prior)
    return wave.fitness.copy()


def test_covisitors_equal_credit_observe_key() -> None:
    f_ab = _rarity_run_key([0, 1])
    f_ba = _rarity_run_key([1, 0])
    # Within a run: both co-visitors of the fresh cell earn EQUAL credit.
    assert f_ab[0] == pytest.approx(f_ab[1])
    assert f_ba[0] == pytest.approx(f_ba[1])
    # Across runs: credit is independent of who reached the cell first.
    assert f_ab[0] == pytest.approx(f_ba[0])
    # And it is the full fresh-cell credit (floor + 1/sqrt(1+0)).
    assert f_ab[0] == pytest.approx(0.01 + 1.0)


def _rarity_run_obs(order: list[int]) -> np.ndarray:
    """Same test via the serial observe() path (crafted screens/wram)."""
    arch = NoveltyArchive()
    wave = WaveNovelty(arch, n_players=2, mode="rarity", floor=0.01)
    wram = np.zeros(8192, dtype=np.uint8)
    for p in (0, 1):
        spawn = np.full((144, 160), 10 * (p + 1), dtype=np.uint8)
        wave.observe(p, spawn, wram)
    fresh = np.full((144, 160), 200, dtype=np.uint8)
    for p in order:
        wave.observe(p, fresh, wram)
    return wave.fitness.copy()


def test_covisitors_equal_credit_observe() -> None:
    f_ab = _rarity_run_obs([0, 1])
    f_ba = _rarity_run_obs([1, 0])
    assert f_ab[0] == pytest.approx(f_ab[1])
    assert f_ba[0] == pytest.approx(f_ba[1])
    assert f_ab[0] == pytest.approx(f_ba[0])


# =====================================================================  A8
@_rom_only
def test_hung_worker_trips_deadline() -> None:
    """A hung-but-alive worker (SIGSTOP: alive, never flips its flag) trips the
    per-round wall-clock deadline and raises a diagnostic naming its env + pid."""
    fleet = _fleet(goexplore=False, n_envs=2, round_deadline_s=2.0)
    stuck_pid = None
    try:
        fleet.reset_all()  # completes normally
        stuck_pid = fleet._procs[0].pid
        os.kill(stuck_pid, signal.SIGSTOP)  # freeze env 0's worker (still alive)
        t0 = time.monotonic()
        with pytest.raises(RuntimeError) as ei:
            fleet.step_all(np.zeros(2, dtype=np.int32))
        elapsed = time.monotonic() - t0
        msg = str(ei.value)
        assert "deadline" in msg
        assert str(stuck_pid) in msg  # names the offending worker
        assert "envs[0]" in msg  # names the offending env index
        # It fired on the deadline, not instantly and not never.
        assert 1.0 < elapsed < 20.0
    finally:
        if stuck_pid is not None:
            try:
                os.kill(stuck_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        fleet.close()
