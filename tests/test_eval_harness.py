"""Tests for the standing credibility / eval harness (pokeio/eval).

Fast + synthetic: no ROM, no training, tiny K / few probe obs. The one test that
needs the emulator is ROM-guarded (skips when the ROM is absent).

Coverage:
  * noise ablation flags a screen-blind genome, passes a screen-dependent one;
  * random-weight baseline returns a sane best-of-K (mechanism + separation);
  * geometric-mean metric + bootstrap CI compute correctly on synthetic rates;
  * obs-layout inference + synthetic probe shape/ranges;
  * end-to-end run_all over a synthetic checkpoint + telemetry.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pokeio.eval import controls, harness, metrics
from pokeio.evo.forward import TANH
from pokeio.evo.genome import ConnGene, InnovationTracker, Population, make_genome

DEVICE = "cpu"

# Small synthetic controller dims (same block layout as the real 454-d foveal
# obs: optical | proprio(14) | ram(8)) so the forward is instant.
N_IN = 60
N_OUT = 11
N_RAM = 8
N_PROPRIO = 14
OPTICAL_HI = N_IN - N_PROPRIO - N_RAM  # 38


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _seeing_genome(seed: int = 0):
    tracker = InnovationTracker(N_IN, N_OUT)
    rng = np.random.default_rng(seed)
    return make_genome(
        N_IN, N_OUT, tracker, rng, connect="full", output_act=TANH,
        n_ram=N_RAM, n_proprio=N_PROPRIO,
    )


def _blind_genome():
    """A constant-button genome: bias->output edges only, so zeroing the optical
    block cannot change its output (Δ ≈ 0)."""
    tracker = InnovationTracker(N_IN, N_OUT)
    rng = np.random.default_rng(1)
    g = make_genome(
        N_IN, N_OUT, tracker, rng, connect="none", output_act=TANH,
        n_ram=N_RAM, n_proprio=N_PROPRIO,
    )
    for o in g.output_ids():
        innov = tracker.conn_innov(g.bias_id, o)
        g.conns[innov] = ConnGene(g.bias_id, o, 0.7, True, innov)
    return g


def _compile(genomes):
    pop = Population.from_genomes(
        genomes, max_nodes=N_IN + N_OUT + 16, max_conns=(N_IN + 1) * N_OUT + 16
    )
    return pop.compile(DEVICE)


def _probe(b: int = 64, seed: int = 1) -> np.ndarray:
    layout = controls.infer_obs_layout(N_IN, N_OUT, n_ram=N_RAM)
    return controls.synthetic_probe(layout, b=b, seed=seed)


# --------------------------------------------------------------------------
# Control 1 — noise / blank-frame ablation
# --------------------------------------------------------------------------
def test_noise_ablation_flags_blind_passes_seeing():
    cp = _compile([_seeing_genome(), _blind_genome()])
    res = controls.noise_ablation(cp, _probe(), OPTICAL_HI, device=DEVICE)
    # index 0 = seeing, index 1 = blind
    assert not res.blind[0], "screen-dependent genome must NOT be flagged blind"
    assert res.blind[1], "screen-blind genome must be flagged"
    # the seeing genome's output actually moves when the screen is removed;
    # the blind genome's is ~0 (bias-only drive).
    assert res.delta_blank[0] > res.dmin
    assert res.delta_blank[1] < 1e-5
    # a blanked screen also flips the seeing genome's argmax action sometimes,
    # never the blind one's.
    assert res.action_change_blank[0] > 0.0
    assert res.action_change_blank[1] == 0.0


def test_noise_ablation_gate_separates():
    cp = _compile([_seeing_genome(), _blind_genome()])
    res = controls.noise_ablation(cp, _probe(), OPTICAL_HI, device=DEVICE)
    assert res.gate[0] > 0.9  # seeing saturates high
    # blind Δ==0 exactly -> gate == sigmoid(-beta*dmin), the crush floor
    floor = 1.0 / (1.0 + math.exp(res.beta * res.dmin))
    assert abs(res.gate[1] - floor) < 1e-3


def test_noise_ablation_empty_probe_is_safe():
    cp = _compile([_blind_genome()])
    res = controls.noise_ablation(cp, np.empty((0, N_IN), np.float32), OPTICAL_HI, device=DEVICE)
    assert res.gate.shape == (1,) and not res.blind[0]


# --------------------------------------------------------------------------
# Control 2 — random-weight-search baseline
# --------------------------------------------------------------------------
def test_random_baseline_mechanism_with_known_scores():
    # A deterministic scorer isolates the best-of-K + margin bookkeeping.
    known = np.array([0.1, 0.9, 0.4, 0.2])

    def score_fn(cp):
        return known[: cp.n]

    res = controls.random_weight_baseline(
        n_in=N_IN, n_out=N_OUT, k=4, score_fn=score_fn, champion_score=1.0,
        n_ram=N_RAM, n_proprio=N_PROPRIO, device=DEVICE,
    )
    assert res.best == pytest.approx(0.9)
    assert res.best_index == 1
    assert res.margin == pytest.approx(1.0 - 0.9)
    assert res.beats_random


def test_random_baseline_best_of_k_is_sane():
    # Real vision-dependence scorer over full-connect randoms: best is finite,
    # positive (full-connect genomes DO use the screen), and equals max score.
    probe = _probe()
    scorer = controls.vision_dependence_scorer(probe, OPTICAL_HI, device=DEVICE)
    res = controls.random_weight_baseline(
        n_in=N_IN, n_out=N_OUT, k=8, score_fn=scorer, champion_score=0.0,
        connect="full", n_ram=N_RAM, n_proprio=N_PROPRIO, device=DEVICE, seed=3,
    )
    assert res.k == 8 and res.scores.shape == (8,)
    assert np.isfinite(res.best) and res.best > 0.0
    assert res.best == pytest.approx(float(np.max(res.scores)))
    assert res.mean <= res.best


def test_seeing_champion_beats_blind_random_search():
    # Structured champion (full-connect, sees) vs K unconnected randoms (blind,
    # score 0) -> champion clears the bar.
    probe = _probe()
    scorer = controls.vision_dependence_scorer(probe, OPTICAL_HI, device=DEVICE)
    champ_cp = _compile([_seeing_genome(7)])
    champ_score = float(scorer(champ_cp)[0])
    res = controls.random_weight_baseline(
        n_in=N_IN, n_out=N_OUT, k=8, score_fn=scorer, champion_score=champ_score,
        connect="none", n_ram=N_RAM, n_proprio=N_PROPRIO, device=DEVICE, seed=5,
    )
    assert res.best == pytest.approx(0.0, abs=1e-6)  # unconnected -> no vision
    assert res.beats_random and res.margin > 0.0


def test_blind_champion_does_not_beat_random():
    probe = _probe()
    scorer = controls.vision_dependence_scorer(probe, OPTICAL_HI, device=DEVICE)
    res = controls.random_weight_baseline(
        n_in=N_IN, n_out=N_OUT, k=8, score_fn=scorer, champion_score=0.0,
        connect="full", n_ram=N_RAM, n_proprio=N_PROPRIO, device=DEVICE, seed=2,
    )
    assert not res.beats_random and res.margin < 0.0


def test_sample_random_genomes_shape():
    gs = controls.sample_random_genomes(n_in=N_IN, n_out=N_OUT, k=5, n_ram=N_RAM, n_proprio=N_PROPRIO)
    assert len(gs) == 5
    assert all(g.n_in == N_IN and g.n_out == N_OUT for g in gs)


# --------------------------------------------------------------------------
# Control 3 — geometric-mean-of-milestones metric + bootstrap
# --------------------------------------------------------------------------
def test_geometric_mean_all_ones_is_one():
    assert metrics.geometric_mean([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_geometric_mean_known_values():
    assert metrics.geometric_mean([0.25, 1.0]) == pytest.approx(0.5)
    x = [0.1, 0.4, 0.9]
    assert metrics.geometric_mean(x) == pytest.approx(math.prod(x) ** (1 / len(x)), rel=1e-9)


def test_geometric_mean_zero_and_empty():
    assert metrics.geometric_mean([0.0, 0.5, 1.0]) == 0.0  # a wall drags it to 0
    assert math.isnan(metrics.geometric_mean([]))


def test_iqm_trims_tails():
    assert metrics.iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)
    # too small to trim -> plain mean
    assert metrics.iqm([2.0, 4.0]) == pytest.approx(3.0)


def test_bootstrap_ci_degenerate_and_bounds():
    lo, hi = metrics.bootstrap_ci([0.5, 0.5, 0.5, 0.5], metrics.geometric_mean_np, n_boot=200, seed=0)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    lo2, hi2 = metrics.bootstrap_ci([0.2, 0.5, 0.8, 1.0], metrics.geometric_mean_np, n_boot=500, seed=0)
    assert 0.0 <= lo2 <= hi2 <= 1.0


def test_milestone_report_all_ones():
    rep = metrics.milestone_geomean_report({"a": 1.0, "b": 1.0, "c": 1.0}, n_boot=200, seed=0)
    assert rep.geomean == pytest.approx(1.0)
    assert rep.iqm == pytest.approx(1.0)
    assert rep.ci == pytest.approx((1.0, 1.0))
    assert rep.n_milestones == 3


def test_milestone_report_known_set_ci_brackets_point():
    rep = metrics.milestone_geomean_report([0.25, 1.0], n_boot=500, seed=0)
    assert rep.geomean == pytest.approx(0.5)
    lo, hi = rep.ci
    assert 0.0 <= lo <= hi <= 1.0


def test_generic_milestone_rates_monotone():
    series = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # rises to peak 4 once
    rates = metrics.generic_milestone_rates(series, signal="progress_best", n_levels=4)
    vals = list(rates.values())
    assert len(vals) == 4
    # deeper thresholds are reached in fewer generations -> non-increasing rates
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))
    assert 0.0 < metrics.geometric_mean(vals) <= 1.0


# --------------------------------------------------------------------------
# obs layout + probe
# --------------------------------------------------------------------------
def test_infer_obs_layout_matches_real_controllers():
    assert controls.infer_obs_layout(454, 11).optical_hi == 432  # foveal
    assert controls.infer_obs_layout(102, 11).optical_hi == 80   # retina latent
    lay = controls.infer_obs_layout(584, 9)                       # legacy flat
    assert lay.optical_hi == 576 and lay.n_proprio == 0


def test_synthetic_probe_shape_and_ranges():
    layout = controls.infer_obs_layout(N_IN, N_OUT, n_ram=N_RAM)
    X = controls.synthetic_probe(layout, b=32, seed=0)
    assert X.shape == (32, N_IN) and X.dtype == np.float32
    opt = layout.optical_hi
    assert X[:, :opt].min() >= 0.0 and X[:, :opt].max() <= 1.0
    proprio = X[:, opt : opt + layout.n_proprio]
    assert proprio.min() >= -1.0 and proprio.max() <= 1.0


# --------------------------------------------------------------------------
# end-to-end over a synthetic checkpoint + telemetry
# --------------------------------------------------------------------------
def _write_synthetic_run(run_dir: Path):
    from pokeio.telemetry.schema import GenerationRecord, TelemetryWriter
    from pokeio.train.checkpoint import save_checkpoint

    champ = _seeing_genome(9)
    champ.fitness = 1.0
    others = controls.sample_random_genomes(
        n_in=N_IN, n_out=N_OUT, k=3, n_ram=N_RAM, n_proprio=N_PROPRIO
    )
    save_checkpoint(run_dir, {"genomes": [champ, *others], "retina": None, "gen": 5})

    with TelemetryWriter(run_dir, resume=False) as w:
        for gen in range(6):
            w.write_generation(
                GenerationRecord(
                    gen=gen, wall_time=float(gen), fitness_best=float(gen),
                    fitness_median=0.0, fitness_worst=0.0, n_species=1,
                    archive_cells=10 * (gen + 1), archive_delta=10,
                    champion_id=f"gen{gen}_g0", champion_genome_ref=f"gen{gen}:idx0",
                    reward_terms={"progress_best": float(gen), "goexplore_max_depth": float(gen * 2)},
                )
            )


def test_run_all_end_to_end_synthetic(tmp_path):
    run_dir = tmp_path / "synthrun"
    run_dir.mkdir()
    _write_synthetic_run(run_dir)

    rep = harness.run_all(
        run_dir, config=None, k_random=8, probe_size=48, use_rom=False,
        device=DEVICE, n_boot=200,
    )
    # champion recovered = the elite (fitness 1.0), it sees, so not flagged blind
    assert rep.champion.fitness == pytest.approx(1.0)
    assert not rep.noise.blind[0]
    # milestones extracted from the generic progress ladder
    assert rep.milestones.n_milestones > 0
    assert 0.0 <= rep.milestones.geomean <= 1.0
    assert rep.probe_source == "synthetic"
    # report renders without error and mentions the three sections
    text = harness.format_report(rep)
    assert "noise" in text and "random-weight" in text and "HEADLINE" in text


def test_load_champion_index_override(tmp_path):
    run_dir = tmp_path / "idxrun"
    run_dir.mkdir()
    _write_synthetic_run(run_dir)
    champ = harness.load_champion(run_dir, index=2)
    assert champ.index == 2


def test_load_champion_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        harness.load_champion(tmp_path / "nope")


# --------------------------------------------------------------------------
# ROM-guarded: real emulator probe (skips when the ROM is absent)
# --------------------------------------------------------------------------
_ROM = Path("roms/pokemon_yellow.gb")


@pytest.mark.skipif(not _ROM.exists(), reason="ROM not present")
def test_emulator_probe_foveal_smoke():
    from pokeio.config import Config

    cfg = Config()  # defaults: foveal, obs_ram_bytes=8, rom_path=roms/pokemon_yellow.gb
    layout = controls.infer_obs_layout(454, 11, n_ram=8)
    probe, source = harness.build_probe(layout, config=cfg, b=8, seed=0, use_rom=True)
    assert probe.shape == (8, 454)
    assert source == "emulator(real)"
