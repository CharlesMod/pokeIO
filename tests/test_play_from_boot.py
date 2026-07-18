"""Play-from-boot tests (docs/specs/execution-plan.md [PB]).

The composition-gap cure has three coupled levers; each is exercised here on the
pure, env-free surfaces so the suite stays fast and hermetic (the end-to-end
rollout is covered by the functional --no-live smoke in the changelog):

1. The from-newgame selection eval BLENDS into fitness — a genome that plays
   from newgame outranks a frontier specialist that does not.
2. The backward-shift restore curriculum samples SHALLOW cells early and widens
   to the full frontier as generations progress.
3. The showcased/mined champion is picked by BOOT competence, not raw fitness.
"""

from __future__ import annotations

import numpy as np

from pokeio.reward.goexplore import CellEntry, GoExplore
from pokeio.reward.novelty import WaveNovelty
from pokeio.reward.archive import NoveltyArchive
from pokeio.train.loop import (
    boot_blend_fitness,
    boot_champion_idx,
    boot_selection_rank,
)


# --------------------------------------------------------------------------
# Lever 1 — the from-newgame eval blends into selection fitness
# --------------------------------------------------------------------------
def test_boot_blend_flips_frontier_specialist_below_a_boot_player():
    """A frontier specialist (high wave fitness, ~no from-boot play) must end up
    BELOW a genome that actually plays from newgame once the boot eval blends."""
    # index 0 = frontier specialist: top raw/selection fitness, 1 boot cell.
    # index 1 = boot player: middling selection fitness, opens 30 boot cells.
    sel = np.array([0.90, 0.60, 0.50, 0.55], dtype=np.float64)   # post rank-normalize
    boot_cells = np.array([1.0, 30.0, 5.0, 10.0], dtype=np.float64)
    boot_prog = np.zeros(4, dtype=np.float64)
    evaluated = np.ones(4, dtype=bool)

    # WITHOUT the blend the specialist wins (this is the diagnosed pathology).
    assert sel[0] > sel[1]

    boot_rank = boot_selection_rank(boot_cells, boot_prog, evaluated)
    blended = boot_blend_fitness(sel, boot_rank, w_boot=0.4)

    # WITH the blend the boot player outranks the frontier specialist.
    assert blended[1] > blended[0]
    # everything stays inside the [0, 1] quantile band.
    assert blended.min() >= 0.0 and blended.max() <= 1.0


def test_boot_blend_is_identity_at_w_boot_zero():
    sel = np.array([0.9, 0.6, 0.5, 0.55])
    boot_rank = boot_selection_rank(
        np.array([1.0, 30.0, 5.0, 10.0]), np.zeros(4), np.ones(4, bool)
    )
    out = boot_blend_fitness(sel, boot_rank, w_boot=0.0)
    assert np.allclose(out, sel)


def test_non_evaluated_genomes_impute_to_neutral_quantile():
    """Sampled runs: genomes that were not boot-evaluated get a neutral 0.5."""
    boot_cells = np.array([1.0, 30.0, 0.0, 0.0])
    evaluated = np.array([True, True, False, False])
    rank = boot_selection_rank(boot_cells, np.zeros(4), evaluated)
    assert rank[2] == 0.5 and rank[3] == 0.5
    assert rank[1] > rank[0]  # among the evaluated, more boot cells ranks higher


def test_boot_progress_breaks_into_the_score_when_taps_live():
    """With equal boot cells, more from-boot mined progress ranks higher."""
    boot_cells = np.array([10.0, 10.0, 10.0])
    boot_prog = np.array([0.0, 0.5, 0.9])
    rank = boot_selection_rank(
        boot_cells, boot_prog, np.ones(3, bool), progress_weight=0.5
    )
    assert rank[2] > rank[1] > rank[0]


# --------------------------------------------------------------------------
# Lever 2 — backward-shift restore samples shallow cells early
# --------------------------------------------------------------------------
def _go_with_depths(depths, *, backward: bool) -> GoExplore:
    go = GoExplore(capacity=1024, rng=np.random.default_rng(0), backward=backward)
    for d in depths:
        key = f"cell{d}".encode()
        go.cells[key] = CellEntry(
            key=key, state=b"", depth=int(d), gen_added=0, gen_seen=0, visits=2
        )
    return go


def test_backward_shift_samples_shallow_cells_early():
    depths = list(range(100))  # depth 0..99
    go = _go_with_depths(depths, backward=True)
    go.begin_generation(0)
    go.set_backward_schedule(0, q0=0.3, anneal_gens=60)
    assert go.backward_q == 0.3

    draws = go.sample_many(400)
    sampled = np.array([e.depth for e in draws], dtype=np.float64)
    thresh = np.quantile(np.asarray(depths, dtype=np.float64), 0.3)  # ~29.7
    # every restored cell early on is in the shallow (bottom-30%) frontier.
    assert sampled.max() <= thresh
    assert sampled.min() >= 0.0


def test_backward_shift_anneals_to_the_full_frontier():
    depths = list(range(100))
    go = _go_with_depths(depths, backward=True)
    # at/after the anneal horizon the eligible quantile is the whole frontier.
    go.begin_generation(60)
    go.set_backward_schedule(60, q0=0.3, anneal_gens=60)
    assert go.backward_q == 1.0
    draws = go.sample_many(400)
    sampled = np.array([e.depth for e in draws], dtype=np.float64)
    assert sampled.max() > 50.0  # deep cells are now reachable


def test_backward_off_samples_deep_cells_even_early():
    """The lever is opt-in: with backward off the legacy sampler is unchanged."""
    depths = list(range(100))
    go = _go_with_depths(depths, backward=False)
    go.begin_generation(0)
    go.set_backward_schedule(0, q0=0.3, anneal_gens=60)  # no-op when off
    assert go.backward_q == 1.0
    draws = go.sample_many(400)
    sampled = np.array([e.depth for e in draws], dtype=np.float64)
    assert sampled.max() > 50.0


def test_backward_intermediate_quantile_widens_with_generations():
    go = _go_with_depths(range(100), backward=True)
    go.set_backward_schedule(0, 0.3, 60)
    q_early = go.backward_q
    go.set_backward_schedule(30, 0.3, 60)
    q_mid = go.backward_q
    go.set_backward_schedule(60, 0.3, 60)
    q_late = go.backward_q
    assert q_early < q_mid < q_late == 1.0


# --------------------------------------------------------------------------
# Lever 3 — champion is picked by boot competence, not raw fitness
# --------------------------------------------------------------------------
def test_champion_picked_by_boot_competence_not_raw_fitness():
    # genome 0 is the raw-fitness king (a deep-restore frontier specialist) but
    # opens only 1 cell from boot; genome 1 opens the most cells from newgame.
    boot_cells = np.array([1.0, 30.0, 5.0, 10.0])
    boot_prog = np.zeros(4)
    evaluated = np.ones(4, bool)
    raw_fits = np.array([100.0, 2.0, 3.0, 4.0])
    assert boot_champion_idx(boot_cells, boot_prog, evaluated, raw_fits) == 1


def test_champion_falls_back_to_raw_when_nothing_boot_evaluated():
    boot_cells = np.zeros(4)
    evaluated = np.zeros(4, bool)
    raw_fits = np.array([1.0, 9.0, 3.0, 4.0])
    assert boot_champion_idx(boot_cells, np.zeros(4), evaluated, raw_fits) == 1


def test_champion_tiebreak_prefers_progress_then_raw_fitness():
    boot_cells = np.array([10.0, 10.0, 10.0])   # tie on distinct cells
    boot_prog = np.array([0.0, 0.9, 0.0])       # genome 1 advanced progress
    evaluated = np.ones(3, bool)
    raw_fits = np.array([5.0, 1.0, 2.0])
    assert boot_champion_idx(boot_cells, boot_prog, evaluated, raw_fits) == 1


# --------------------------------------------------------------------------
# WaveNovelty.distinct_counts — the per-genome boot signal the eval reads
# --------------------------------------------------------------------------
def test_distinct_counts_excludes_the_spawn_cell():
    archive = NoveltyArchive()
    wave = WaveNovelty(archive, n_players=2, mode="rarity")
    # player 0: spawn key + two reached keys -> 2 distinct reached.
    wave.observe_key(0, b"spawn", globally_new=True, prior_visits=0)   # spawn
    wave.observe_key(0, b"a", globally_new=True, prior_visits=0)
    wave.observe_key(0, b"b", globally_new=True, prior_visits=0)
    wave.observe_key(0, b"a", globally_new=False, prior_visits=1)      # revisit
    # player 1: only ever sees its spawn -> 0 distinct reached (couch potato).
    wave.observe_key(1, b"spawn", globally_new=False, prior_visits=5)
    counts = wave.distinct_counts()
    assert counts[0] == 2.0
    assert counts[1] == 0.0
