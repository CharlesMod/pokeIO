"""Contract tests for the telemetry, config, and manifest schemas."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pokeio.config import Config, EmuConfig
from pokeio.manifest.schema import Manifest, ProgressDimension
from pokeio.telemetry.schema import (
    SCHEMA_VERSION,
    ChampionStep,
    GenerationRecord,
    TelemetryWriter,
    read_champion_steps,
    read_generations,
)

EXAMPLE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "pokeio"
    / "manifest"
    / "examples"
    / "pokemon_yellow.json"
)


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------
def test_telemetry_generation_round_trip(tmp_path):
    recs = [
        GenerationRecord(
            gen=0,
            wall_time=1.5,
            fitness_best=10.0,
            fitness_median=5.0,
            fitness_worst=0.0,
            n_species=3,
            archive_cells=100,
            archive_delta=100,
            champion_id="agent-0",
            champion_genome_ref="genomes/g0.npz",
            reward_terms={"novelty": 0.7, "progress": 0.3},
            throughput_sps=12345.6,
            cpu_pct=88.0,
            gpu=[{"card": 1, "util": 97, "mem_mb": 15000}],
        ),
        GenerationRecord(
            gen=1,
            wall_time=3.0,
            fitness_best=12.0,
            fitness_median=6.0,
            fitness_worst=0.5,
            n_species=4,
            archive_cells=150,
            archive_delta=50,
            champion_id="agent-7",
            champion_genome_ref="genomes/g1.npz",
        ),
    ]
    with TelemetryWriter(tmp_path) as w:
        for r in recs:
            w.write_generation(r)

    back = read_generations(tmp_path)
    assert back == recs
    assert back[0].gpu[0]["util"] == 97


def test_telemetry_champion_round_trip(tmp_path):
    steps = [
        ChampionStep(
            step=0,
            action=4,
            screen_ref="frames/0.png",
            ram_tap={"badges": 0, "party": 1},
            reward_components={"novelty": 0.1},
        ),
        ChampionStep(step=1, action=0, screen_ref="frames/1.png"),
    ]
    with TelemetryWriter(tmp_path) as w:
        for s in steps:
            w.write_champion_step(s)

    back = read_champion_steps(tmp_path)
    assert back == steps


def test_telemetry_streams_are_separate(tmp_path):
    with TelemetryWriter(tmp_path) as w:
        w.write_generation(
            GenerationRecord(
                gen=0,
                wall_time=0.0,
                fitness_best=1.0,
                fitness_median=1.0,
                fitness_worst=1.0,
                n_species=1,
                archive_cells=1,
                archive_delta=1,
                champion_id="a",
                champion_genome_ref="ref",
            )
        )
        w.write_champion_step(ChampionStep(step=0, action=0, screen_ref="f"))

    # Each reader only picks up its own record type.
    assert len(read_generations(tmp_path)) == 1
    assert len(read_champion_steps(tmp_path)) == 1


def test_telemetry_missing_dir_reads_empty(tmp_path):
    assert read_generations(tmp_path / "does_not_exist") == []
    assert read_champion_steps(tmp_path / "does_not_exist") == []


def test_schema_version_is_int():
    assert isinstance(SCHEMA_VERSION, int)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def test_config_defaults():
    from pokeio.emu.env import ACTIONS

    c = Config()
    assert c.emu.frame_skip == 1  # Phase 1.5 default: per-frame reflex control
    # A NOOP action was added (all buttons released) so the agent can stand
    # still; the default action space is 9, not 8. Keep this pinned to the
    # single source of truth in emu.env so the two can never silently drift.
    assert c.emu.action_space == 9
    assert "noop" in ACTIONS
    assert len(ACTIONS) == c.emu.action_space
    assert c.evo.pop_size > 0
    # The served model name is lowercase (llama.cpp reports "glm-4.7-flash");
    # the old "GLM-4.7-Flash" default case-mismatched and 404'd (audit A11).
    assert c.llm.model == "glm-4.7-flash"
    assert c.llm.base_url.startswith("http")
    # base_url must NOT carry the /v1 suffix (the client appends it) — a
    # double-/v1 was one of the A11 findings.
    assert not c.llm.base_url.rstrip("/").endswith("/v1")
    assert c.vision.n_channels >= 1


def test_config_save_load_equality(tmp_path):
    c = Config()
    c.emu.frame_skip = 16
    c.evo.pop_size = 512
    c.llm.model = "GLM-4.7-Flash"
    c.run.device_map = {"llm": 0, "evo": 1}

    path = c.save(tmp_path / "config.yaml")
    loaded = Config.load(path)
    assert loaded == c


def test_config_snapshot(tmp_path):
    c = Config()
    out = c.snapshot(tmp_path)
    assert out.exists()
    assert out.name == "config.yaml"
    assert Config.load(out) == c


def test_config_nested_dataclass_rebuilds(tmp_path):
    c = Config()
    reloaded = Config.load(c.save(tmp_path / "c.yaml"))
    # The emu section must round-trip as a dataclass, not a raw dict.
    assert isinstance(reloaded.emu, EmuConfig)


def test_config_ignores_unknown_keys():
    c = Config.from_dict({"emu": {"frame_skip": 8, "bogus": 1}, "extra": 42})
    assert c.emu.frame_skip == 8


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def test_example_manifest_loads_and_validates():
    assert EXAMPLE_MANIFEST.exists(), EXAMPLE_MANIFEST
    m = Manifest.load_json(EXAMPLE_MANIFEST)
    assert m.game == "Pokemon Yellow"
    assert m.milestone_label == "Badges"
    assert len(m.progress_dimensions) >= 1
    assert m.validate() == []


def test_manifest_round_trip(tmp_path):
    m = Manifest.load_json(EXAMPLE_MANIFEST)
    out = m.save_json(tmp_path / "m.json")
    reloaded = Manifest.load_json(out)
    assert reloaded == m


def test_broken_manifest_reports_errors():
    m = Manifest(
        game="",  # empty game -> error
        milestone_label="",  # empty label -> error
        progress_dimensions=[
            ProgressDimension(
                id="dup", label="A", source={"ram": {"addr": "0x1"}}, dir="up"
            ),
            ProgressDimension(
                id="dup", label="", source={}, dir="sideways"
            ),  # dup id, empty label, empty source, bad dir
        ],
        spatial={},
    )
    problems = m.validate()
    assert problems  # non-empty
    joined = " ".join(problems)
    assert "game" in joined
    assert "milestone_label" in joined
    assert "duplicate" in joined
    assert "dir" in joined
    assert "source" in joined


def test_valid_minimal_manifest():
    m = Manifest(
        game="Tetris",
        milestone_label="Lines",
        progress_dimensions=[
            ProgressDimension(
                id="lines",
                label="Lines Cleared",
                source={"screen": {"region": [0, 0, 10, 10]}},
                dir="up",
            )
        ],
    )
    assert m.validate() == []


def test_manifest_empty_dimensions_invalid():
    m = Manifest(game="X", milestone_label="Y", progress_dimensions=[])
    assert any("progress_dimensions" in p for p in m.validate())
