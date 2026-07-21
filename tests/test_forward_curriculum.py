"""Forward-progression (brain6) tests: ForwardCurriculum source mixing, the
Go-Explore capture/restore wiring in the trainer, teleport-decoupling on archive
spawns, spawn-blind obs, --warm-start cross-arch loading, and the pipeline script.

Hermetic: FakeForwardFleet (no furnace/ROM/GPU) supplies worker-style cell keys,
deferred capture semantics, and restore-blob -> progressed-WRAM behaviour, so the
full forward rollout path runs on CPU.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from pokeio.brain.replay import ForwardCurriculum
from pokeio.brain.rl_loop import BrainConfig, BrainTrainer

WRAM_BASE = 0xC000
REPO = Path(__file__).resolve().parents[1]


class FakeDemo:
    """No-ROM stand-in: fixed length, and demo depths never produce a restore."""

    def __len__(self):
        return 657

    def restore_map(self, env_depths):
        return {}


class FakeForwardFleet:
    """Fleet stub exercising the FORWARD plumbing end to end.

    * worker-style cell keys, fresh per (env, round) -> every round mints
      globally-new cells (capture candidates);
    * DEFERRED capture semantics: flags passed to ``step_all`` yield blobs in the
      same call's ``captured`` dict (the parent-side pattern only needs the
      flag->blob round trip, not the exact one-round lag);
    * restore blobs "teleport" an env to a PROGRESSED state (party=1, lvl 5,
      Oak's lab) -- the teleport-decoupling fixture;
    * every episode, step 1 enters map 37 (part of every capturing trajectory's
      own history); optional knobs: ``second_map`` -> a genuinely NEW map at
      step 2, ``party_at`` -> ALL envs gain party>=1 at step t (the milestone).
    """

    def __init__(self, n_envs=4, obs_dim=454, obs_ram=8, seed=0,
                 second_map=None, party_at=None):
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.obs_ram = obs_ram
        self.second_map = second_map
        self.party_at = party_at
        self.arr = {"wram": np.zeros((n_envs, 8192), np.uint8)}
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._keys = np.zeros((n_envs, 8), np.uint8)
        self.restore_log: list[dict] = []

    def _obs(self):
        return self._rng.random((self.n_envs, self.obs_dim), dtype=np.float32)

    def reset_all(self, restore=None):
        self._t = 0
        restore = dict(restore or {})
        self.restore_log.append(restore)
        w = self.arr["wram"]
        w[:] = 0
        w[:, 0xD35D - WRAM_BASE] = 38                 # boot: bedroom
        for i in restore:                             # restored: progressed state
            w[i, 0xD35D - WRAM_BASE] = 40             # Oak's lab
            w[i, 0xD162 - WRAM_BASE] = 1              # party count
            w[i, 0xD163 - WRAM_BASE] = 84             # Pikachu
            w[i, 0xD18B - WRAM_BASE] = 5              # level 5
        return self._obs()

    def step_all(self, buttons, capture_flags=None, gaze_dx=None, gaze_dy=None):
        self._t += 1
        w = self.arr["wram"]
        if self._t == 1:                              # one EARNED map for everyone
            w[:, 0xD35D - WRAM_BASE] = 37
        if self._t == 2 and self.second_map is not None:
            w[:, 0xD35D - WRAM_BASE] = int(self.second_map)
        if self.party_at is not None and self._t >= self.party_at:
            w[:, 0xD162 - WRAM_BASE] = np.maximum(w[:, 0xD162 - WRAM_BASE], 1)
            w[:, 0xD163 - WRAM_BASE] = 84
            w[:, 0xD18B - WRAM_BASE] = np.maximum(w[:, 0xD18B - WRAM_BASE], 5)
        self._keys[:, 0] = np.arange(self.n_envs)
        self._keys[:, 1] = self._t % 256
        self._keys[:, 2] = self._t // 256
        captured = {}
        if capture_flags is not None:
            for i in np.nonzero(capture_flags)[0]:
                captured[int(i)] = f"blob-{i}-{self._t}".encode()
        return self._obs(), self._keys, np.zeros(self.n_envs, bool), captured


def _forward_trainer(n_envs=4, seed=1, caps_per_round=4.0, **fleet_kw):
    fleet = FakeForwardFleet(n_envs=n_envs, **fleet_kw)
    cfg = BrainConfig(horizon_min=4, horizon_max=8, minibatches=2, epochs=2,
                      eval_every=0, seed=seed,
                      goexplore_caps_per_round=caps_per_round)
    return BrainTrainer(fleet, device="cpu", demo=FakeDemo(), cfg=cfg,
                        forward=True)


# ------------------------------------------------- t1: source-mix adapts
def test_source_mix_saturated_loses_mass_frontier_gains():
    c = ForwardCurriculum(657, seed=0)
    heavy = int(2.0 / (1.0 - c.ema_decay))          # >> the EMA's memory length
    c.src_ema["demo"], c.src_seen["demo"] = 1.0, heavy       # saturated (too easy)
    c.src_ema["boot"], c.src_seen["boot"] = 0.0, heavy       # hopeless (too hard)
    c.src_ema["archive"], c.src_seen["archive"] = 0.5, heavy  # learning frontier
    m = c.source_masses(archive_size=1)
    assert m["archive"] > 3 * m["demo"]             # frontier dominates saturated
    assert m["archive"] > 3 * m["boot"]             # ... and hopeless
    assert min(m.values()) > 0.0                    # no source is ever abandoned
    # sampling follows the masses
    srcs = [s for s, _ in c.sample_spawns(600, archive_size=1)]
    assert srcs.count("archive") > 2 * srcs.count("demo")


def test_source_mix_adapts_from_reports():
    """Reporting saturating success on one source drains its mass toward the
    others -- the recede-on-success rule generalized to the mix."""
    c = ForwardCurriculum(657, seed=0)
    m0 = c.source_masses(archive_size=1)["demo"]
    for _ in range(40):
        c.report_spawns(["demo"] * 8, [True] * 8)
    m1 = c.source_masses(archive_size=1)["demo"]
    assert m1 < m0                                  # saturated source loses mass
    assert c.success_ema > 0.9                      # global EMA (ent_coef) fed too


def test_unexplored_source_starts_at_the_frontier():
    """Zero-evidence sources sit AT max mass (prior p=0.5 -> variance 0.25):
    the mix explores every source before judging it."""
    c = ForwardCurriculum(657, seed=0)
    m = c.source_masses(archive_size=1)
    assert m["boot"] == pytest.approx(m["demo"])
    assert m["boot"] == pytest.approx(m["archive"])


# ------------------------------------------------- t2: empty-archive fallback
def test_empty_archive_degrades_to_boot_plus_demo():
    c = ForwardCurriculum(657, seed=0)
    spawns = c.sample_spawns(200, archive_size=0)
    assert all(s in ("boot", "demo") for s, _ in spawns)
    assert set(c.source_masses(archive_size=0)) == {"boot", "demo"}
    # ... even if the archive source has (stale, e.g. pre-reboot) statistics
    c.src_ema["archive"], c.src_seen["archive"] = 0.5, 10
    assert all(s != "archive" for s, _ in c.sample_spawns(200, archive_size=0))


def test_demo_depths_come_from_the_inner_backward_curriculum():
    c = ForwardCurriculum(657, seed=3)
    c.backward.frontier = 400.0
    depths = [d for s, d in c.sample_spawns(400, archive_size=0) if s == "demo"]
    assert depths, "with two max-mass sources some demo spawns must be drawn"
    assert all(0 <= d <= 657 for d in depths)
    assert abs(np.mean(depths) - 400.0) < 0.2 * 657   # clustered at the frontier
    assert all(d == 0 for s, d in c.sample_spawns(50, 0) if s != "demo")


# ------------------------------------------------- t3: determinism under seed
def test_sampling_deterministic_under_fixed_seed():
    a = ForwardCurriculum(657, seed=7)
    b = ForwardCurriculum(657, seed=7)
    assert a.sample_spawns(100, archive_size=1) == b.sample_spawns(100, archive_size=1)
    a.report_spawns(["boot", "demo"], [True, False])
    b.report_spawns(["boot", "demo"], [True, False])
    assert a.sample_spawns(100, archive_size=0) == b.sample_spawns(100, archive_size=0)
    assert ForwardCurriculum(657, seed=8).sample_spawns(100, 1) != \
        ForwardCurriculum(657, seed=9).sample_spawns(100, 1)


# ------------------------------------------------- t4: serialize/restore
def test_full_state_roundtrip():
    a = ForwardCurriculum(657, seed=5)
    for k in range(30):
        srcs = ["boot", "demo", "archive"][k % 3], "demo"
        a.report_spawns(srcs, [k % 2 == 0, k % 3 == 0])
    b = ForwardCurriculum(657, seed=5)
    b.load_state(a.full_state())
    assert b.success_ema == a.success_ema and b._seen == a._seen
    assert b.src_ema == a.src_ema and b.src_seen == a.src_seen
    assert b.backward.frontier == a.backward.frontier
    assert b.backward.success_ema == a.backward.success_ema
    assert b.backward._seen == a.backward._seen
    assert b.state() == a.state()                   # metrics view identical too
    assert b.full_state() == a.full_state()


def test_state_keeps_backward_compat_keys():
    """The metrics/checkpoint surface the rest of the stack reads (brain_loop
    _checkpoint/_log, rl_loop metrics merge) must keep BackwardCurriculum's keys."""
    st = ForwardCurriculum(657, seed=0).state()
    assert set(st) >= {"frontier", "frontier_frac", "success_ema", "at_boot",
                       "seen", "sources"}


# --------------------------------------- forward rollout: capture + restore
def test_forward_rollout_captures_cells_into_the_archive():
    tr = _forward_trainer()
    batch = tr.rollout()
    assert batch["H"] == tr.cfg.horizon_max         # non-demo spawns -> explore cap
    assert tr.archive.size > 0                      # parent novelty bookkeeping ran
    assert tr.goexplore.size > 0                    # deferred captures were stored
    assert tr.goexplore.n_captured == tr.goexplore.size
    for e in tr.goexplore.cells.values():           # blobs are the worker bytes
        assert e.state.startswith(b"blob-")


def test_capture_budget_throttles_saves():
    tr = _forward_trainer(caps_per_round=1.0)       # tiny budget
    tr.rollout()
    hi = _forward_trainer(caps_per_round=8.0)
    hi.rollout()
    assert tr.goexplore.n_throttled > 0             # budget actually metered
    assert tr.goexplore.n_captured < hi.goexplore.n_captured


def test_archive_spawns_restore_blobs_into_the_fleet():
    tr = _forward_trainer()
    tr.rollout()                                    # fill the archive
    assert tr.goexplore.size >= tr.n_envs
    tr.curriculum.sample_spawns = lambda n, a: [("archive", 0)] * n
    tr.rollout()
    restored = tr.fleet.restore_log[-1]
    assert set(restored) == set(range(tr.n_envs))   # every env got a cell blob
    assert all(v.startswith(b"blob-") for v in restored.values())
    assert tr.goexplore.n_restores >= tr.n_envs


# ------------------------------------- t5: teleport-decoupling regression
def test_restored_progress_is_never_paid():
    """An env restored into a progressed state (party=1, lvl 5, map 40 --
    worth 10+ progress points if paid) whose SPAWNING trajectory already
    traversed maps {38, 37} must earn EXACTLY the one genuinely-new map 12:
    +1.0. Neither the handed milestones (re-baselined from the restored WRAM)
    NOR backtracking through the trajectory's own maps (37 -- invisible to the
    WRAM alone, carried by the cell's progress snapshot) is ever paid."""
    tr = _forward_trainer()
    tr.rollout()
    # every capture carries the discovering trajectory's progress snapshot
    for e in tr.goexplore.cells.values():
        assert e.progress is not None
        assert {38, 37} <= set(e.progress["maps"])
    tr.curriculum.sample_spawns = lambda n, a: [("archive", 0)] * n
    tr.fleet.second_map = 12                     # a map NO trajectory has seen
    batch = tr.rollout()
    assert batch["rew_sum_mean"] == pytest.approx(1.0)   # ONLY map 12 pays
    # the handed progress IS in the baseline (recognized, unpaid)
    for i in range(tr.n_envs):
        assert tr.reward.progress(i)["party_count"] == 1
        assert tr.reward.started(i)
        assert 37 in tr.reward._envs[i].maps     # history seeded, hence unpaid


def test_reward_reset_rebaselines_from_progressed_wram():
    """Unit-level teleport-decoupling: reset on a progressed WRAM, then a step
    on the SAME wram pays ~0."""
    from pokeio.brain.reward import ProgressReward

    w = np.zeros(8192, np.uint8)
    w[0xD35D - WRAM_BASE] = 40
    w[0xD162 - WRAM_BASE] = 1
    w[0xD163 - WRAM_BASE] = 84
    w[0xD18B - WRAM_BASE] = 5
    r = ProgressReward(1)
    r.reset(0, w)
    assert r.step(0, w) == pytest.approx(0.0)


def test_history_maps_never_repay():
    """Unit-level maps-history decoupling: WRAM only shows the CURRENT map, so
    the spawning trajectory's traversed maps come from the history snapshot --
    backtracking through them pays 0; a genuinely new map still pays w_map."""
    from pokeio.brain.reward import ProgressReward

    w40 = np.zeros(8192, np.uint8)
    w40[0xD35D - WRAM_BASE] = 40
    hist = {"party_count": 0, "party_level": 0, "badges": 0, "events": 0,
            "counter_sum": 0, "maps": {38, 37}}
    r = ProgressReward(1)
    r.reset(0, w40, history=hist)
    w37 = w40.copy()
    w37[0xD35D - WRAM_BASE] = 37                 # backtrack into a history map
    assert r.step(0, w37) == pytest.approx(0.0)
    w12 = w40.copy()
    w12[0xD35D - WRAM_BASE] = 12                 # genuinely new map
    assert r.step(0, w12) == pytest.approx(1.0)


def test_earned_milestone_tiers():
    """Unit-level outcome ladder: a tier-0 spawn succeeds only by EARNING
    started(); a spawn handed party>0 is NOT a success (leak immunity) and must
    earn the next badge/event tier; map churn never counts."""
    from pokeio.brain.reward import ProgressReward

    boot = np.zeros(8192, np.uint8)
    boot[0xD35D - WRAM_BASE] = 38
    started = boot.copy()
    started[0xD35D - WRAM_BASE] = 40
    started[0xD162 - WRAM_BASE] = 1
    started[0xD163 - WRAM_BASE] = 84
    r = ProgressReward(2)
    r.reset(0, boot)
    assert not r.earned_milestone(0)             # nothing earned yet
    w37 = boot.copy()
    w37[0xD35D - WRAM_BASE] = 37
    r.step(0, w37)
    assert not r.earned_milestone(0)             # a map transition is NOT one
    r.step(0, started)
    assert r.earned_milestone(0)                 # tier-0 spawn EARNED party>0
    r.reset(1, started)                          # spawned past the milestone
    r.step(1, started)
    assert r.started(1)
    assert not r.earned_milestone(1)             # handed party is no success
    ev = started.copy()
    ev[0xD746 - WRAM_BASE] = 0b1                 # one story event flag
    r.step(1, ev)
    assert r.earned_milestone(1)                 # next tier: a NEW event flag


def test_forward_outcome_is_milestone_not_map_or_started():
    """The outcome driving frontier recession / ent_coef / source masses is a
    spawn-relative MILESTONE crossing: (a) a mere map transition is NOT a
    success (else 'walk out the door' saturates every EMA, flips ent_coef into
    its penalty regime, and recedes the frontier ~8/iter to boot unearned);
    (b) an env restored past the milestone is NOT an auto-success (started()
    leak); (c) a tier-0 spawn that EARNS party>0 IS one."""
    # (a) map-only progress: no outcome fires anywhere, frontier holds
    tr = _forward_trainer()
    f0 = tr.curriculum.backward.frontier
    tr.rollout()
    assert tr.curriculum.success_ema == 0.0
    assert all(v == 0.0 for v in tr.curriculum.src_ema.values())
    assert tr.curriculum.backward.frontier == f0
    # (b) restored past the milestone: started() true, success false
    seen0 = dict(tr.curriculum.src_seen)
    tr.curriculum.sample_spawns = lambda n, a: [("archive", 0)] * n
    tr.rollout()
    assert all(tr.reward.started(i) for i in range(tr.n_envs))
    assert tr.curriculum.src_seen["archive"] == seen0["archive"] + tr.n_envs
    assert tr.curriculum.src_ema["archive"] == 0.0  # reported, as FAILURES
    # (c) tier-0 spawns that EARN the milestone feed the EMAs
    tr2 = _forward_trainer(seed=2, party_at=2)
    tr2.curriculum.sample_spawns = lambda n, a: [("boot", 0)] * n
    tr2.rollout()
    assert tr2.curriculum.src_ema["boot"] > 0.0
    assert tr2.curriculum.success_ema > 0.0


# ------------------------------------------------- t7: spawn-blind obs
def test_obs_invariant_to_curriculum_internals():
    """Regression: the obs builder consumes NOTHING from curriculum state. With
    the spawn draw pinned, wildly different curriculum internals must produce
    bit-identical rollout observations (and hence actions)."""
    pinned = lambda n, a: [("boot", 0)] * n

    tr_a = _forward_trainer(seed=2)
    tr_a.curriculum.sample_spawns = pinned
    batch_a = tr_a.rollout()

    tr_b = _forward_trainer(seed=2)                 # re-seeds torch identically
    tr_b.curriculum.sample_spawns = pinned
    tr_b.curriculum.success_ema = 0.93              # junk internals, all of them
    tr_b.curriculum._seen = 4096
    for s in tr_b.curriculum.SOURCES:
        tr_b.curriculum.src_ema[s] = 0.42
        tr_b.curriculum.src_seen[s] = 777
    tr_b.curriculum.backward.frontier = 123.0
    batch_b = tr_b.rollout()

    assert np.array_equal(batch_a["obs"], batch_b["obs"])
    assert np.array_equal(batch_a["act"], batch_b["act"])
    # and the fleet was never handed anything beyond the (empty) restore map
    assert tr_a.fleet.restore_log == tr_b.fleet.restore_log == [{}]


# ------------------------------------------------- t6: --warm-start
def _ckpt_of(tr, tmp_path, name, wrap="policy"):
    p = tmp_path / name
    torch.save({wrap: tr.policy.state_dict()}, str(p))
    return str(p)


def test_warm_start_gaze_ckpt_into_gazeless_model(tmp_path):
    from pokeio.train.brain_loop import _apply_warm_start

    src = BrainTrainer(FakeForwardFleet(), device="cpu", demo=FakeDemo(),
                       cfg=BrainConfig(seed=3), learned_gaze=True, forward=True)
    path = _ckpt_of(src, tmp_path, "gaze.pt")

    dst = _forward_trainer(seed=4)                  # learned_gaze=False
    missing, unexpected = _apply_warm_start(dst, path, "cpu")
    assert sorted(unexpected) == ["gaze_log_std", "gaze_mu.bias", "gaze_mu.weight"]
    assert missing == []                            # every dst tensor was fed
    # weights actually copied across
    assert torch.equal(dst.policy.pi.weight, src.policy.pi.weight)
    # optimizer FRESH (no loaded moments), ent machinery at its INITIAL state
    assert len(dst.opt.state) == 0
    assert dst.curriculum.success_ema == 0.0 and dst.curriculum._seen == 0
    ema = dst.curriculum.success_ema
    assert dst.cfg.ent_coef_base * (1 - ema) - dst.cfg.decisiveness * ema == \
        pytest.approx(dst.cfg.ent_coef_base)        # full exploration bonus
    assert dst.iter == 0


def test_warm_start_gazeless_ckpt_into_gaze_model(tmp_path):
    from pokeio.train.brain_loop import _apply_warm_start

    src = _forward_trainer(seed=5)                  # learned_gaze=False
    path = _ckpt_of(src, tmp_path, "nogaze.pt", wrap="state_dict")  # champion format
    dst = BrainTrainer(FakeForwardFleet(), device="cpu", demo=FakeDemo(),
                       cfg=BrainConfig(seed=6), learned_gaze=True, forward=True)
    before_gaze = dst.policy.gaze_mu.weight.detach().clone()
    missing, unexpected = _apply_warm_start(dst, path, "cpu")
    assert sorted(missing) == ["gaze_log_std", "gaze_mu.bias", "gaze_mu.weight"]
    assert unexpected == []
    # the gaze head keeps its near-zero init (reflex dominates: intended cold start)
    assert torch.equal(dst.policy.gaze_mu.weight.detach(), before_gaze)
    assert torch.equal(dst.policy.pi.weight, src.policy.pi.weight)


# ------------------------------------------------- checkpoint plumbing
def test_checkpoint_roundtrips_forward_curriculum(tmp_path):
    from pokeio.train.brain_loop import _checkpoint

    tr = _forward_trainer(seed=8)
    tr.rollout()
    tr.iter = 17
    p = tmp_path / "brain.pt"
    _checkpoint(p, tr)
    ckpt = torch.load(str(p), weights_only=False)
    assert ckpt["curriculum_full"]["kind"] == "forward"

    fresh = _forward_trainer(seed=8)
    fresh.curriculum.load_state(ckpt["curriculum_full"])
    assert fresh.curriculum.full_state() == tr.curriculum.full_state()
    assert fresh.curriculum.state() == tr.curriculum.state()


def test_checkpoint_write_is_atomic_with_prev_fallback(tmp_path):
    """Power-cut safety: the write goes to .pt.tmp then renames over brain.pt
    (never truncated in place), the temp never lingers, and the previous good
    checkpoint survives as .pt.prev for the --resume fallback."""
    from pokeio.train.brain_loop import _checkpoint

    tr = _forward_trainer(seed=9)
    p = tmp_path / "brain.pt"
    _checkpoint(p, tr)
    assert p.exists()
    assert not p.with_suffix(".pt.tmp").exists()        # temp renamed away
    assert not p.with_suffix(".pt.prev").exists()       # no prior to keep yet
    tr.iter = 5
    _checkpoint(p, tr)                                  # second write rotates
    assert not p.with_suffix(".pt.tmp").exists()
    prev = torch.load(str(p.with_suffix(".pt.prev")), weights_only=False)
    assert prev["iter"] == 0                            # the previous GOOD ckpt
    assert torch.load(str(p), weights_only=False)["iter"] == 5


# ------------------------------------------------- t8: pipeline script
def test_pipeline_script_parses():
    r = subprocess.run(["bash", "-n", str(REPO / "scripts/train_pipeline.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_pipeline_script_brain6_invariants():
    """The concluded runs must never be auto-resumed; the arms must be present
    with the approved flags."""
    text = (REPO / "scripts/train_pipeline.sh").read_text()
    assert "--run-id brain4" not in text and "--run-id brain5" not in text
    assert "arm brain6r cuda:0" in text
    assert "arm brain6g cuda:1 --learned-gaze" in text
    assert "--forward" in text and "--warm-start" in text
    assert "runs/_pipeline.stop" in text            # stop sentinel kept
    assert "pokeio.dash.serve" in text              # dashboard keepalive kept
    # concurrent arms must pin to DISJOINT NUMA nodes (dual-fleet collision fix)
    assert "arm brain6r cuda:0 --numa-node 0" in text
    assert "arm brain6g cuda:1 --learned-gaze --numa-node 1" in text
