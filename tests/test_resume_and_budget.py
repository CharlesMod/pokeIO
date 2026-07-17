"""Regression tests for the audit correctness/reproducibility fixes.

Covers:
  * A2 — gen-0 topology budget crash: growing the pad budget lets a genome that
    outgrew the gen-0 cap pack + compile + forward instead of raising ValueError.
  * A3 — resume corruption: (a) telemetry/champion streams are truncated on
    gen>=start_gen so a re-run doesn't double-append; (b) load distinguishes
    absent from corrupt so --resume can refuse to silently restart at gen 0.
  * A4 — reproducibility: git_sha stamping, resolved-run.json capture, torch
    seeding determinism.
  * A9 — miner consistency loophole: byte-identical rollouts are de-duplicated
    before the cross-episode vote.

Emulator-free and CPU-only, so the suite runs anywhere.
"""

from __future__ import annotations

import json
import subprocess
from collections import deque

import numpy as np
import pytest
import torch

from pokeio.config import Config
from pokeio.evo.genome import InnovationTracker, Population, make_genome
from pokeio.evo.forward import population_forward_sparse
from pokeio.reward.miner import MinerConfig, mine
from pokeio.train import loop
from pokeio.train.checkpoint import (
    LOAD_ABSENT,
    LOAD_CORRUPT,
    LOAD_OK,
    LOAD_SCHEMA,
    build_state,
    checkpoint_path,
    load_checkpoint_status,
    save_checkpoint,
)
from pokeio.telemetry.schema import (
    ChampionStep,
    GenerationRecord,
    TelemetryWriter,
    read_champion_steps,
    read_generations,
)


# ==========================================================================
# A2 — topology budget growth
# ==========================================================================
def test_grow_topology_budget_headroom_and_monotonicity():
    # well under the cap => no change
    assert loop._grow_topology_budget(850, 4096, 600, 3000) == (850, 4096)
    # within headroom of the cap => grow to next power of two above need*1.25
    gn, gc = loop._grow_topology_budget(850, 4096, 700, 3900)
    assert gn > 850 and gc > 4096
    assert gn >= int(700 * 1.25) and gc >= int(3900 * 1.25)
    # never shrinks
    for _ in range(20):
        gn, gc = loop._grow_topology_budget(gn, gc, gn - 1, gc - 1)
    assert gn >= 1024 and gc >= 8192


def _forward_ok(genomes, max_nodes, max_conns):
    """Pack + compile + forward a population on CPU; return the output tensor."""
    pop = Population.from_genomes(genomes, max_nodes=max_nodes, max_conns=max_conns)
    cp = pop.compile("cpu")
    n_in = genomes[0].n_in
    x = torch.zeros(len(genomes), 1, n_in)
    return population_forward_sparse(cp, x, steps=4)


def test_population_survives_genome_past_initial_cap():
    """A genome that outgrew the gen-0 cap must not kill the run.

    Reproduces A2: 'full' wiring at moderate width blows past a small gen-0
    budget; Population.from_genomes raises. The loop's fix recomputes the need
    and grows the budget, after which packing/compiling/forwarding succeeds and
    yields correct-shaped output.
    """
    rng = np.random.default_rng(0)
    t = InnovationTracker(20, 5)
    genomes = [make_genome(20, 5, t, rng, connect="full") for _ in range(3)]
    need_nodes = max(len(g.nodes) for g in genomes)
    need_conns = max(len(g.conns) for g in genomes)  # (20+1)*5 = 105

    # the frozen gen-0 budget is too small -> the uncaught crash A2 describes
    tiny_nodes, tiny_conns = 64, 64
    assert need_conns > tiny_conns
    with pytest.raises(ValueError):
        Population.from_genomes(genomes, max_nodes=tiny_nodes, max_conns=tiny_conns)

    # the fix: grow the budget at the wave boundary, then it packs fine
    gn, gc = loop._grow_topology_budget(tiny_nodes, tiny_conns, need_nodes, need_conns)
    assert gc >= need_conns and gn >= need_nodes
    out = _forward_ok(genomes, gn, gc)
    assert out.shape == (3, 1, 5)
    assert torch.isfinite(out).all()


def test_grow_budget_loop_simulation_never_raises():
    """Simulate the loop's per-gen growth as genomes accrete conns/nodes."""
    rng = np.random.default_rng(1)
    t = InnovationTracker(8, 3)
    genomes = [make_genome(8, 3, t, rng, connect="sparse", sparse_k=4) for _ in range(4)]
    rates = loop.MutationRates(add_node=1.0, add_conn=1.0)
    max_nodes = max(len(g.nodes) for g in genomes) + 4
    max_conns = max(len(g.conns) for g in genomes) + 4
    for _gen in range(30):
        # each generation grows the budget from the live need, exactly as train()
        need_nodes = max(len(g.nodes) for g in genomes)
        need_conns = max(len(g.conns) for g in genomes)
        max_nodes, max_conns = loop._grow_topology_budget(
            max_nodes, max_conns, need_nodes, need_conns
        )
        # must never raise even as genomes outgrow the gen-0 cap
        Population.from_genomes(genomes, max_nodes=max_nodes, max_conns=max_conns)
        for g in genomes:
            from pokeio.evo import ops
            ops.mutate_genome(g, t, rng, rates)


# ==========================================================================
# A3 — resume: absent-vs-corrupt + stream truncation
# ==========================================================================
def _minimal_state(gen: int) -> dict:
    return build_state(
        gen=gen,
        genomes=[{"g": 1}],
        archive={"cells": 3},
        goexplore=None,
        tracker={"innov": 7},
        spec={"thr": 3.0},
        prev_reps={},
        species_best={},
        rng=np.random.default_rng(gen),
        taps=[],
        miner_rollouts=[np.arange(4)],
        miner_exclude=set(),
        total_agent_steps=gen * 100,
    )


def test_load_status_distinguishes_absent_corrupt_ok(tmp_path):
    # absent
    data, status = load_checkpoint_status(tmp_path)
    assert data is None and status == LOAD_ABSENT

    # corrupt (garbage that is not a valid pickle)
    checkpoint_path(tmp_path).write_bytes(b"\x00 not a pickle \xff")
    data, status = load_checkpoint_status(tmp_path)
    assert data is None and status == LOAD_CORRUPT

    # valid round-trip
    save_checkpoint(tmp_path, _minimal_state(7))
    data, status = load_checkpoint_status(tmp_path)
    assert status == LOAD_OK
    assert data["gen"] == 7
    assert data["archive"] == {"cells": 3}
    assert data["total_agent_steps"] == 700


def test_load_status_flags_schema_mismatch(tmp_path):
    import pickle

    checkpoint_path(tmp_path).write_bytes(
        pickle.dumps({"schema": 999, "gen": 1})
    )
    data, status = load_checkpoint_status(tmp_path)
    assert data is None and status == LOAD_SCHEMA


def _write_run(run_dir, n_gens: int, champ_steps: int = 3):
    """Write telemetry + champion streams for gens 0..n_gens-1."""
    with TelemetryWriter(run_dir, resume=False) as w:
        for gen in range(n_gens):
            w.write_generation(
                GenerationRecord(
                    gen=gen,
                    wall_time=float(gen),
                    fitness_best=float(gen),
                    fitness_median=0.0,
                    fitness_worst=0.0,
                    n_species=1,
                    archive_cells=gen,
                    archive_delta=1,
                    champion_id=f"gen{gen}",
                    champion_genome_ref=f"gen{gen}",
                )
            )
            for step in range(champ_steps):  # one replay block per gen
                w.write_champion_step(
                    ChampionStep(step=step, action=gen, screen_ref=f"g{gen}s{step}")
                )


def test_truncate_streams_on_resume(tmp_path):
    _write_run(tmp_path, n_gens=6, champ_steps=3)
    assert [g.gen for g in read_generations(tmp_path)] == [0, 1, 2, 3, 4, 5]
    assert len(read_champion_steps(tmp_path)) == 18  # 6 gens x 3 steps

    # resume at gen 3: drop gens 3,4,5 from both streams
    loop._truncate_streams_on_resume(tmp_path, start_gen=3)

    assert [g.gen for g in read_generations(tmp_path)] == [0, 1, 2]
    champ = read_champion_steps(tmp_path)
    assert len(champ) == 9  # first 3 blocks kept
    assert sum(1 for c in champ if c.step == 0) == 3  # 3 replay blocks


def test_resume_reappend_is_dedup_free(tmp_path):
    """The double-append A3a describes must not survive truncate + resume."""
    _write_run(tmp_path, n_gens=6, champ_steps=2)
    # crash-then-resume: checkpoint only covered through gen 2
    loop._truncate_streams_on_resume(tmp_path, start_gen=3)
    # re-run gens 3..5, appending as the resumed loop would
    with TelemetryWriter(tmp_path, resume=True) as w:
        for gen in range(3, 6):
            w.write_generation(
                GenerationRecord(
                    gen=gen, wall_time=float(gen), fitness_best=float(gen),
                    fitness_median=0.0, fitness_worst=0.0, n_species=1,
                    archive_cells=gen, archive_delta=1,
                    champion_id=f"gen{gen}", champion_genome_ref=f"gen{gen}",
                )
            )
    gens = [g.gen for g in read_generations(tmp_path)]
    assert gens == [0, 1, 2, 3, 4, 5]  # contiguous, no duplicates


# ==========================================================================
# A4 — reproducibility
# ==========================================================================
def test_git_sha_format_and_non_git_guard(tmp_path):
    sha = loop._git_sha()  # this repo IS a git checkout
    assert isinstance(sha, str)
    if sha:  # allow empty in a non-git CI sandbox
        core = sha[:-6] if sha.endswith("-dirty") else sha
        assert len(core) == 40
        int(core, 16)  # hex

    # a non-git directory must degrade to "" rather than raise
    assert loop._git_sha(tmp_path) == ""


def test_write_resolved_run_captures_cli_knobs(tmp_path):
    cfg = Config()
    cfg.run.git_sha = "deadbeef-dirty"
    kwargs = dict(
        gens=8, episode_steps=300, novelty_mode="rarity", goexplore=True,
        goexplore_capacity=16384, restore_prob=0.5, init_connect="sparse",
        init_k=12, engine="furnace", boot_gauntlet_every=10,
        recurrent_memory=True, checkpoint_every=-1,
    )
    path = loop._write_resolved_run(
        tmp_path, argv=["loop.py", "--gens", "8"], train_kwargs=kwargs, config=cfg
    )
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["argv"] == ["loop.py", "--gens", "8"]
    assert payload["git_sha"] == "deadbeef-dirty"
    # the decisive CLI-only knobs the config.yaml snapshot omits
    for key in ("gens", "episode_steps", "novelty_mode", "goexplore",
                "engine", "boot_gauntlet_every", "recurrent_memory",
                "checkpoint_every"):
        assert key in payload["train_kwargs"]
    assert payload["config"]["evo"]["pop_size"] == cfg.evo.pop_size


def test_torch_seeding_is_deterministic():
    """The mechanism train() now invokes (torch.manual_seed) is reproducible."""
    torch.manual_seed(1234)
    a = torch.rand(5)
    torch.manual_seed(1234)
    b = torch.rand(5)
    assert torch.equal(a, b)


# ==========================================================================
# A9 — miner consistency loophole
# ==========================================================================
def _counter_rollout(change_steps, n_bytes: int = 16, T: int = 20, col: int = 3):
    """A rollout where `col` is a clean +1 counter (low entropy, in-band)."""
    mat = np.zeros((T, n_bytes), dtype=np.uint8)
    val = 0
    for t in range(T):
        if t in change_steps:
            val += 1
        mat[t, col] = val
    return mat


def _find(cands, address):
    for c in cands:
        if c.address == address and c.width == 1:
            return c
    return None


def test_miner_dedup_collapses_identical_traces():
    """Byte-identical replays must not inflate cross-episode consistency (A9)."""
    addr = 0xC000 + 3
    mat = _counter_rollout({5, 10, 15})
    cfg_dedup = MinerConfig(consider_pairs=False, dedup_rollouts=True)
    cfg_raw = MinerConfig(consider_pairs=False, dedup_rollouts=False)

    # feeding 5 identical copies with dedup ON == feeding a single trace
    c_five = _find(mine([mat] * 5, cfg=cfg_dedup), addr)
    c_one = _find(mine([mat], cfg=cfg_dedup), addr)
    assert c_five is not None and c_one is not None
    assert c_five.stats["n_rollouts_active"] == 1
    assert c_five.stats["n_changes"] == c_one.stats["n_changes"]

    # without dedup the duplicates masquerade as 5 independent episodes
    c_raw = _find(mine([mat] * 5, cfg=cfg_raw), addr)
    assert c_raw is not None
    assert c_raw.stats["n_rollouts_active"] == 5
    assert c_raw.stats["n_changes"] == 5 * c_one.stats["n_changes"]


def test_miner_dedup_preserves_distinct_traces():
    """Genuinely distinct rollouts still count as independent episodes."""
    addr = 0xC000 + 3
    rolls = [
        _counter_rollout({5, 10, 15}),
        _counter_rollout({4, 9, 14}),
        _counter_rollout({6, 11, 16}),
        _counter_rollout({3, 8, 13}),
    ]
    cfg = MinerConfig(consider_pairs=False, dedup_rollouts=True)
    c = _find(mine(rolls, cfg=cfg), addr)
    assert c is not None
    assert c.stats["n_rollouts_active"] == 4  # all four are distinct
    assert c.stats["consistency"] == pytest.approx(1.0)
