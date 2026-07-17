"""Checkpoint save -> load -> continue must be lossless (audit A10 / findings
#12, #13, #39).

``checkpoint.py`` carries an explicit *"continue bit-for-bit from the boundary"*
claim in its module docstring and is, per the fix ledger, the hottest bug area
in the trainer — yet it had no test. These pin the contract that resume relies
on:

  * every load-bearing object (population, novelty archive, Go-Explore archive,
    innovation tracker, speciation state, mined taps, miner bookkeeping, cumulative
    step count) survives a real pickle round-trip through disk unchanged;
  * the numpy RNG restores **bit-for-bit** — a fresh generator seeded from the
    saved ``rng_state`` reproduces the exact draw stream the original would have
    continued with;
  * running one more generation of ``reproduce`` from a restored checkpoint is
    byte-identical to running it from the live objects (the actual "resume ==
    keep going" guarantee);
  * the novelty archive's ``seen`` / visit bookkeeping continues correctly after
    a restore;
  * ``load_checkpoint_status`` distinguishes absent / corrupt / schema-mismatch
    (a ``--resume`` path must never silently restart at gen 0 on a *present but
    unreadable* checkpoint).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeio.evo import ops
from pokeio.evo.genome import InnovationTracker, make_genome
from pokeio.reward.archive import NoveltyArchive
from pokeio.reward.goexplore import GoExplore
from pokeio.train.checkpoint import (
    CHECKPOINT_FILENAME,
    LOAD_ABSENT,
    LOAD_CORRUPT,
    LOAD_OK,
    LOAD_SCHEMA,
    build_state,
    checkpoint_path,
    load_checkpoint,
    load_checkpoint_status,
    save_checkpoint,
)

_SCREEN_H, _SCREEN_W = 144, 160
_WRAM = 8192


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _genome_sig(g):
    nodes = tuple(
        sorted((n.id, n.type, n.act, round(float(n.bias), 12)) for n in g.nodes.values())
    )
    conns = tuple(
        sorted(
            (c.in_id, c.out_id, round(float(c.weight), 12), bool(c.enabled), c.innov)
            for c in g.conns.values()
        )
    )
    return (g.n_in, g.n_out, g.age, nodes, conns)


def _sigs(gs):
    return [_genome_sig(g) for g in gs]


def _cell(i: int):
    """A distinct (screen, wram) pair, deterministic in ``i``."""
    rng = np.random.default_rng(1000 + i)
    screen = rng.integers(0, 256, size=(_SCREEN_H, _SCREEN_W), dtype=np.uint8)
    wram = rng.integers(0, 256, size=(_WRAM,), dtype=np.uint8)
    return screen, wram


def _make_state(seed=0):
    """Assemble a small-but-complete live training state to checkpoint."""
    rng = np.random.default_rng(seed)
    tracker = InnovationTracker(3, 2)
    genomes = [make_genome(3, 2, tracker, rng, connect="full") for _ in range(12)]
    for i, g in enumerate(genomes):
        for _ in range(i % 3):
            ops.mutate_add_node(g, tracker, rng)
        g.fitness = float(rng.random())
    spec = ops.Speciation(threshold=2.0)
    species = spec.assign(genomes, rng)  # populates spec.reps

    # Novelty archive: several cells, some visited multiple times.
    archive = NoveltyArchive()
    archive.begin_generation()
    keys = []
    for i in range(6):
        s, w = _cell(i)
        k = archive.cell_key(s, w)
        keys.append(k)
        archive.add(k)
        for _ in range(i):  # cell i gets i prior visits
            archive.visit(k)

    # Go-Explore archive: a few restorable cells with distinct depths.
    go = GoExplore(capacity=64, rng=np.random.default_rng(seed + 1))
    go.begin_generation(3)
    for i in range(4):
        go.store_captured(keys[i], bytes([i]) * 32, depth=i * 5)

    taps = [{"slots": (0,), "dir": 1}, {"slots": (1, 2), "dir": -1}]
    miner_rollouts = [np.arange(6, dtype=np.uint8).reshape(2, 3)]
    miner_exclude = {0xC010, 0xC011, 0xC012}

    return dict(
        rng=rng, tracker=tracker, genomes=genomes, spec=spec, species=species,
        archive=archive, go=go, keys=keys, taps=taps,
        miner_rollouts=miner_rollouts, miner_exclude=miner_exclude,
    )


def _build(st, *, gen=7, total_agent_steps=54321):
    return build_state(
        gen=gen,
        genomes=st["genomes"],
        archive=st["archive"],
        goexplore=st["go"],
        tracker=st["tracker"],
        spec=st["spec"],
        prev_reps=dict(st["spec"].reps),
        species_best={},
        rng=st["rng"],
        taps=st["taps"],
        miner_rollouts=st["miner_rollouts"],
        miner_exclude=st["miner_exclude"],
        total_agent_steps=total_agent_steps,
    )


# --------------------------------------------------------------------------
# round-trip integrity
# --------------------------------------------------------------------------
def test_roundtrip_preserves_every_object(tmp_path):
    st = _make_state()
    state = _build(st)
    save_checkpoint(tmp_path, state)
    data, status = load_checkpoint_status(tmp_path)
    assert status == LOAD_OK

    assert data["gen"] == 7
    assert data["total_agent_steps"] == 54321
    # population
    assert _sigs(data["genomes"]) == _sigs(st["genomes"])
    # innovation tracker (identical global counters + memo tables)
    assert vars(data["tracker"]) == vars(st["tracker"])
    # speciation state (threshold, coeffs, representatives, next id)
    assert data["spec"].reps.keys() == st["spec"].reps.keys()
    assert data["spec"]._next == st["spec"]._next
    # novelty archive: seen set + per-cell visit counts + per-gen frontier
    assert data["archive"].seen == st["archive"].seen
    assert data["archive"]._visits == st["archive"]._visits
    assert data["archive"]._gen_new == st["archive"]._gen_new
    # go-explore: cell keys, depths, and lifetime counters
    assert set(data["goexplore"].cells) == set(st["go"].cells)
    assert {k: e.depth for k, e in data["goexplore"].cells.items()} == {
        k: e.depth for k, e in st["go"].cells.items()
    }
    assert data["goexplore"].n_captured == st["go"].n_captured
    # miner + taps
    assert data["taps"] == st["taps"]
    assert data["miner_exclude"] == st["miner_exclude"]
    assert len(data["miner_rollouts"]) == len(st["miner_rollouts"])
    assert np.array_equal(data["miner_rollouts"][0], st["miner_rollouts"][0])


def test_rng_restores_bit_for_bit(tmp_path):
    st = _make_state()
    # The state snapshot freezes the rng's bit-generator state at this instant.
    state = _build(st)
    save_checkpoint(tmp_path, state)
    # Ground truth: what the ORIGINAL generator produces from here on.
    expected = st["rng"].random(64)

    data = load_checkpoint(tmp_path)
    restored = np.random.default_rng()
    restored.bit_generator.state = data["rng_state"]
    got = restored.random(64)

    assert np.array_equal(got, expected)


def test_resume_continues_one_gen_bit_identical(tmp_path):
    """Evaluate + reproduce one more generation from a restored checkpoint and
    from the live objects; the offspring and the resulting rng state must match.

    This is the concrete meaning of the module's 'continue bit-for-bit' claim."""
    st = _make_state(seed=5)
    # Freeze the resume point P (rng state at checkpoint time).
    state = _build(st)
    save_checkpoint(tmp_path, state)
    resume_point = state["rng_state"]

    def one_gen(genomes, spec, tracker, rng):
        fits = rng.random(len(genomes))  # a deterministic 'evaluation'
        for g, f in zip(genomes, fits):
            g.fitness = float(f)
        species = spec.assign(genomes, rng)
        kids = ops.reproduce(genomes, species, tracker, rng, ops.MutationRates(),
                             pop_size=len(genomes))
        return kids, rng.bit_generator.state

    # Live path: fresh generator wound to P, live in-memory objects.
    rng_live = np.random.default_rng()
    rng_live.bit_generator.state = resume_point
    kids_live, end_live = one_gen(
        st["genomes"], st["spec"], st["tracker"], rng_live
    )

    # Resume path: everything reconstituted from disk.
    data = load_checkpoint(tmp_path)
    rng_res = np.random.default_rng()
    rng_res.bit_generator.state = data["rng_state"]
    kids_res, end_res = one_gen(
        data["genomes"], data["spec"], data["tracker"], rng_res
    )

    assert _sigs(kids_live) == _sigs(kids_res)
    assert end_live == end_res


def test_archive_continuity_after_restore(tmp_path):
    st = _make_state()
    save_checkpoint(tmp_path, _build(st))
    data = load_checkpoint(tmp_path)
    arch = data["archive"]

    # A cell that was in the run is still 'seen' -> re-adding is not novel.
    assert arch.add(st["keys"][0]) is False
    # A genuinely new cell is still detected as new after resume.
    s, w = _cell(999)
    assert arch.observe(s, w) is True
    # Visit counts continued from the restored values (cell 3 had 3 priors).
    assert arch.visit(st["keys"][3]) == 3


# --------------------------------------------------------------------------
# load-status semantics (resume must not silently restart at gen 0)
# --------------------------------------------------------------------------
def test_absent_checkpoint(tmp_path):
    data, status = load_checkpoint_status(tmp_path)
    assert data is None
    assert status == LOAD_ABSENT
    assert load_checkpoint(tmp_path) is None


def test_corrupt_checkpoint_is_not_absent(tmp_path):
    checkpoint_path(tmp_path).write_bytes(b"\x00\x01 not a pickle \xff")
    data, status = load_checkpoint_status(tmp_path)
    assert data is None
    assert status == LOAD_CORRUPT  # crucially NOT absent
    assert load_checkpoint(tmp_path) is None


def test_schema_mismatch_detected(tmp_path):
    import pickle

    with open(checkpoint_path(tmp_path), "wb") as fh:
        pickle.dump({"schema": 999, "gen": 3}, fh)
    data, status = load_checkpoint_status(tmp_path)
    assert data is None
    assert status == LOAD_SCHEMA
    assert load_checkpoint(tmp_path) is None


def test_saved_file_lands_at_expected_path(tmp_path):
    save_checkpoint(tmp_path, _build(_make_state()))
    assert (tmp_path / CHECKPOINT_FILENAME).exists()
