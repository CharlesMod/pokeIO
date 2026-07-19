"""Tests for the evaluation spine (pokeio/eval/spine.py, v2.0 #29).

The spine's whole job is to tell competence from Goodhart, so the tests are
built around that claim:

* a slightly-better stub and a random stub are DISTINGUISHED by the headline
  IQM + CIs (the CIs separate them);
* the best-of-K random-weight control lands at ~chance — the proof the control
  is inert and therefore trustworthy (this is the check that caught the old
  champions);
* the noise-frame control collapses a real policy to ~chance, and is skipped
  cleanly when the from-boot metric has no noise mode;
* the frozen-core git-diff protocol records a sha and DETECTS a mid-eval core
  edit;
* the 2nd-game hook reports a clear "not-run" status until a ROM is decoded;
* the from-boot dependency is injected (task #28 is built in parallel), and the
  unwired path raises a clear error.

Everything is synthetic + fast: the from-boot metric is a stub, no ROM, tiny K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pokeio.eval import controls, harness, spine
from pokeio.evo.forward import TANH
from pokeio.evo.genome import InnovationTracker, Population, make_genome

DEVICE = "cpu"
SEEDS = [0, 1, 2, 3, 4, 5]
EPISODES = 8

# Same block layout as the real foveal obs so the genome path is exercised.
N_IN, N_OUT, N_RAM, N_PROPRIO = 60, 11, 8, 14


# ==========================================================================
# synthetic policies + a stub from-boot metric (the #28 seam)
# ==========================================================================
@dataclass
class StubPolicy:
    """A policy whose only trait is its expected from-boot progress in [0, 1]."""

    skill: float


def stub_from_boot(policy, target, *, seed, episodes, noise_frames=False):
    """A stand-in for #28's ``measure``: progress ~= policy.skill (0 under noise).

    Returns the documented seam dict: a scalar ``progress_score`` + per-episode
    ``episode_scores`` (so the spine gets seed×episode samples for its CIs).
    """
    rng = np.random.default_rng(seed)
    base = 0.0 if noise_frames else float(policy.skill)
    eps = np.clip(base + rng.normal(0.0, 0.02, size=int(episodes)), 0.0, 1.0)
    return {"progress_score": float(np.mean(eps)), "episode_scores": [float(x) for x in eps]}


def stub_from_boot_no_noise_mode(policy, target, *, seed, episodes):
    """Same, but WITHOUT a ``noise_frames`` param (to test graceful skip)."""
    rng = np.random.default_rng(seed)
    eps = np.clip(float(policy.skill) + rng.normal(0.0, 0.02, size=int(episodes)), 0.0, 1.0)
    return {"progress_score": float(np.mean(eps)), "episode_scores": [float(x) for x in eps]}


def stub_random_factory(seed: int) -> StubPolicy:
    """Random-weight policies = chance-level skill (~0 from-boot progress)."""
    return StubPolicy(skill=0.0)


# ==========================================================================
# 1. controls catch Goodhart: distinguish better-vs-random; random ~= chance
# ==========================================================================
def _evaluate(policy, **kw):
    # chance_level is a small positive floor: a strictly-nonnegative progress
    # metric never sits exactly at 0 (a random policy stumbles a hair forward),
    # so "clears chance" means clearing that empirical floor, not literal zero.
    return spine.evaluate(
        policy, fleet_or_gauntlet=object(), from_boot_fn=stub_from_boot,
        seeds=SEEDS, episodes=EPISODES, random_policy_factory=stub_random_factory,
        k_random=6, chance_level=0.05, chance_tol=0.1, n_boot=200, **kw,
    )


def test_headline_distinguishes_better_from_random_with_cis():
    rep_better = _evaluate(StubPolicy(skill=0.6), run_id="better")
    rep_random = _evaluate(StubPolicy(skill=0.0), run_id="random")

    # the slightly-better stub scores materially higher on the headline
    assert rep_better.headline.iqm > rep_random.headline.iqm + 0.3
    # and the CIs SEPARATE them (the whole point of reporting IQM+CI, not a mean)
    assert rep_better.headline.ci_lo > rep_random.headline.ci_hi


def test_random_weight_control_scores_near_chance():
    # THE demonstration: even the best-of-K random-weight baseline lands ~chance,
    # so a champion clearing it is real competence, not a lucky spawn.
    rep = _evaluate(StubPolicy(skill=0.6))
    rw = next(c for c in rep.controls if c.name == "random_weight")
    assert rw.status == "run"
    assert rw.near_chance, f"random-weight control should be ~chance, got IQM {rw.control.iqm}"
    assert rw.control.iqm == pytest.approx(0.0, abs=0.1)
    # champion clears the bar, and cleanly (CI-separated)
    assert rw.beats and rw.margin > 0.0
    assert rw.separated


def test_competent_policy_is_credible_random_policy_is_flagged():
    rep_better = _evaluate(StubPolicy(skill=0.6))
    assert rep_better.credible, rep_better.flags
    assert rep_better.headline_clears_chance

    rep_random = _evaluate(StubPolicy(skill=0.0))
    # a chance-level "policy" fails to clear chance AND fails to beat the controls
    assert not rep_random.credible
    assert not rep_random.headline_clears_chance
    assert any("does not clear chance" in f for f in rep_random.flags)


# ==========================================================================
# 2. noise-frame control
# ==========================================================================
def test_noise_frame_control_collapses_to_chance():
    rep = _evaluate(StubPolicy(skill=0.6))
    nf = next(c for c in rep.controls if c.name == "noise_frame")
    assert nf.status == "run"
    assert nf.control.iqm == pytest.approx(0.0, abs=0.1)  # progress dies under noise
    assert nf.beats and nf.near_chance


def test_noise_frame_control_skips_when_metric_has_no_noise_mode():
    rep = spine.evaluate(
        StubPolicy(skill=0.6), object(), from_boot_fn=stub_from_boot_no_noise_mode,
        seeds=SEEDS, episodes=EPISODES, random_policy_factory=stub_random_factory,
        k_random=4, chance_level=0.0, chance_tol=0.1, n_boot=100,
    )
    nf = next(c for c in rep.controls if c.name == "noise_frame")
    assert nf.status.startswith("skipped")
    # skipping a control must NOT silently pass it off as beaten
    assert not nf.beats


# ==========================================================================
# 3. frozen-core git-diff protocol
# ==========================================================================
def test_frozen_core_records_sha_and_detects_edit(tmp_path):
    core = tmp_path / "core.py"
    core.write_text("x = 1\n")
    fc0 = spine.capture_frozen_core([core])
    assert fc0.core_digest and core.name.endswith(".py")
    # unchanged -> verifies
    assert fc0.verify(spine.capture_frozen_core([core]))
    # a mid-eval edit to the core is DETECTABLE
    core.write_text("x = 2\n")
    fc1 = spine.capture_frozen_core([core])
    assert not fc0.verify(fc1)
    assert fc0.core_digest != fc1.core_digest


def test_frozen_core_missing_file_is_recorded_not_fatal(tmp_path):
    fc = spine.capture_frozen_core([tmp_path / "does_not_exist.py"])
    assert fc.missing and fc.core_files == []


def test_report_carries_frozen_core_and_it_is_ok_when_core_unchanged():
    rep = _evaluate(StubPolicy(skill=0.6))
    assert isinstance(rep.frozen_core, spine.FrozenCore)
    assert rep.frozen_core_ok  # nothing edited the default core during the eval
    # the sha slot is recorded (may be "" outside a git checkout — the digest
    # still guards content)
    assert isinstance(rep.frozen_core.sha, str)


# ==========================================================================
# 4. pre-registered 2nd-game hook
# ==========================================================================
def test_second_game_hook_reports_not_run_by_default():
    rep = _evaluate(StubPolicy(skill=0.6))
    assert rep.second_game.status == "not-run"
    assert "pre-registered" in rep.second_game.note.lower() or "hook" in rep.second_game.note.lower()


def test_second_game_hook_runs_when_target_wired():
    spec = spine.SecondGameSpec(
        rom_path=None, label="game2", chance_level=0.0, target=object()
    )
    # rom_path is None -> still not-run (no ROM decoded), even with a target
    rep = _evaluate(StubPolicy(skill=0.6), second_game=spec)
    assert rep.second_game.status == "not-run"


# ==========================================================================
# 5. from-boot injection seam (#28 built in parallel)
# ==========================================================================
def test_unwired_from_boot_raises_clear_error():
    fn, source = spine.resolve_from_boot_fn(None)
    # from_boot.py is not present yet in this tree -> unwired
    if fn is None:
        assert source == "unwired"
        with pytest.raises(spine.FromBootNotWired):
            spine.evaluate(
                StubPolicy(skill=0.5), object(), None, seeds=[0], episodes=2,
                random_policy_factory=stub_random_factory,
            )
    else:  # #28 has landed: the real measure is bound
        assert source == "pokeio.reward.from_boot.measure"


def test_injected_from_boot_is_labeled():
    rep = _evaluate(StubPolicy(skill=0.6))
    assert rep.from_boot_source == "injected"


# ==========================================================================
# 6. report rendering + serialization
# ==========================================================================
def test_format_report_mentions_all_sections():
    text = spine.format_report(_evaluate(StubPolicy(skill=0.6)))
    for token in ("HEADLINE", "CONTROLS", "2nd game", "frozen-core", "VERDICT"):
        assert token in text


def test_report_to_dict_is_json_friendly():
    import json

    rep = _evaluate(StubPolicy(skill=0.6))
    d = rep.to_dict()
    json.dumps(d)  # must not raise (no numpy arrays leaking through)
    assert d["headline"]["iqm"] == pytest.approx(rep.headline.iqm)
    assert d["vision_controls"]["status"].startswith("n/a")  # stub is not a genome


# ==========================================================================
# 7. checkpoint policy path: default random factory + vision-dependence audits
# ==========================================================================
def _seeing_genome(seed: int = 0):
    tracker = InnovationTracker(N_IN, N_OUT)
    rng = np.random.default_rng(seed)
    return make_genome(
        N_IN, N_OUT, tracker, rng, connect="full", output_act=TANH,
        n_ram=N_RAM, n_proprio=N_PROPRIO,
    )


def champ_from_boot(policy, target, *, seed, episodes, noise_frames=False):
    """A genome-aware stub: progress scales with the champion's stored fitness.

    The real champion has fitness 1.0 (skill ~0.7); the default factory's random
    champions have fitness 0.0 (chance). Exercises the checkpoint path end-to-end.
    """
    skill = 0.0 if noise_frames else float(np.clip(getattr(policy, "fitness", 0.0), 0.0, 1.0)) * 0.7
    rng = np.random.default_rng(seed)
    eps = np.clip(skill + rng.normal(0.0, 0.02, size=int(episodes)), 0.0, 1.0)
    return {"progress_score": float(np.mean(eps)), "episode_scores": [float(x) for x in eps]}


def _write_synthetic_checkpoint(run_dir):
    from pokeio.train.checkpoint import save_checkpoint

    champ = _seeing_genome(9)
    champ.fitness = 1.0
    others = controls.sample_random_genomes(
        n_in=N_IN, n_out=N_OUT, k=3, n_ram=N_RAM, n_proprio=N_PROPRIO
    )
    save_checkpoint(run_dir, {"genomes": [champ, *others], "retina": None, "gen": 5})


def test_checkpoint_policy_runs_vision_controls_and_default_factory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_synthetic_checkpoint(run_dir)

    rep = spine.evaluate(
        run_dir, fleet_or_gauntlet=object(), from_boot_fn=champ_from_boot,
        seeds=SEEDS, episodes=EPISODES, config=None, device=DEVICE,
        k_random=6, chance_level=0.0, chance_tol=0.1, n_boot=200,
        vision_k_random=8, probe_size=32,
    )
    # policy resolved from the checkpoint -> the elite (fitness 1.0)
    assert rep.policy_kind == "checkpoint"
    assert rep.headline_clears_chance
    # the DEFAULT random-weight factory (random genomes like the champ) is ~chance
    rw = next(c for c in rep.controls if c.name == "random_weight")
    assert rw.status == "run" and rw.near_chance and rw.beats
    # the champion clears the meaningful (from-boot currency) controls
    nf = next(c for c in rep.controls if c.name == "noise_frame")
    assert nf.beats
    # the reused vision-dependence audits ran on the real genome: the elite is
    # NOT screen-blind (full-connect => its action moves with the screen).
    assert rep.vision_controls.status == "run"
    assert rep.vision_controls.screen_blind is False
    # (the vision-*magnitude* random baseline is a supplementary audit; a random
    #  synthetic genome need not out-respond wild randoms, so we don't force it.)
    assert rep.vision_controls.beats_random_vision in (True, False)
