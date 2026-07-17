"""pokeIO telemetry schema v0.

Frozen data contract consumed by both the reward stack and the dashboard.
Two streams live under a run directory:

  * ``<run_dir>/telemetry.jsonl``  — one :class:`GenerationRecord` per generation.
  * ``<run_dir>/champion.jsonl``   — one :class:`ChampionStep` per champion step.

Every JSONL line is a self-describing envelope::

    {"type": "generation", "schema": 0, "data": {...}}

so a reader can dispatch on ``type`` and reject unknown ``schema`` versions.
Serialization goes through ``orjson`` (fast, deterministic, numpy-aware).

Dependency-light: stdlib + orjson only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import orjson

# Bump when the wire shape of any record below changes incompatibly.
SCHEMA_VERSION = 0

# Envelope ``type`` tags.
TYPE_GENERATION = "generation"
TYPE_CHAMPION_STEP = "champion_step"

TELEMETRY_FILENAME = "telemetry.jsonl"
CHAMPION_FILENAME = "champion.jsonl"


@dataclass
class GenerationRecord:
    """Per-generation summary of the evolutionary run.

    One record is emitted per generation of the population. Fitness values are
    whatever the active reward stack produces (higher is better by convention).
    """

    gen: int
    wall_time: float  # seconds since run start
    fitness_best: float
    fitness_median: float
    fitness_worst: float
    n_species: int
    archive_cells: int  # total Go-Explore cells discovered so far
    archive_delta: int  # new cells discovered this generation
    champion_id: str
    champion_genome_ref: str  # opaque ref (path/hash) to the champion genome
    reward_terms: dict[str, float] = field(default_factory=dict)
    throughput_sps: float = 0.0  # aggregate agent-steps/second
    cpu_pct: float = 0.0
    gpu: list[dict[str, Any]] = field(default_factory=list)  # per-card util/mem
    # -- boot gauntlet (docs/specs/boot-gauntlet.md); -1.0 = didn't run this gen
    boot_depth: float = -1.0
    boot_depth_frac: float = -1.0
    boot_cells: float = -1.0
    boot_cells_known: float = -1.0
    boot_steps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationRecord":
        return _from_dict(cls, d)


@dataclass
class ChampionStep:
    """Per-step record of the current champion's episode (the live feed)."""

    step: int
    action: int
    screen_ref: str  # opaque ref (path/hash) to the captured frame
    ram_tap: dict[str, int] = field(default_factory=dict)  # mined byte reads
    reward_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChampionStep":
        return _from_dict(cls, d)


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def _envelope(type_tag: str, record: Any) -> bytes:
    payload = {
        "type": type_tag,
        "schema": SCHEMA_VERSION,
        "data": record.to_dict() if hasattr(record, "to_dict") else asdict(record),
    }
    # orjson emits a single line with no embedded newlines; append our own.
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n"


class TelemetryWriter:
    """Append-only writer for the two telemetry streams.

    Opens both JSONL files in binary append mode and flushes each line so a
    tailing dashboard sees records promptly. Usable as a context manager.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.run_dir / TELEMETRY_FILENAME
        self.champion_path = self.run_dir / CHAMPION_FILENAME
        self._gen_fh = open(self.telemetry_path, "ab")
        self._champ_fh = open(self.champion_path, "ab")

    def write_generation(self, record: GenerationRecord) -> None:
        self._gen_fh.write(_envelope(TYPE_GENERATION, record))
        self._gen_fh.flush()

    def write_champion_step(self, step: ChampionStep) -> None:
        self._champ_fh.write(_envelope(TYPE_CHAMPION_STEP, step))
        self._champ_fh.flush()

    def close(self) -> None:
        self._gen_fh.close()
        self._champ_fh.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def read_generations(run_dir: str | Path) -> list[GenerationRecord]:
    """Load all generation records from a run directory."""
    path = Path(run_dir) / TELEMETRY_FILENAME
    out: list[GenerationRecord] = []
    for env in _read_jsonl(path):
        if env.get("type") != TYPE_GENERATION:
            continue
        _check_schema(env)
        out.append(GenerationRecord.from_dict(env["data"]))
    return out


def read_champion_steps(run_dir: str | Path) -> list[ChampionStep]:
    """Load all champion-step records from a run directory."""
    path = Path(run_dir) / CHAMPION_FILENAME
    out: list[ChampionStep] = []
    for env in _read_jsonl(path):
        if env.get("type") != TYPE_CHAMPION_STEP:
            continue
        _check_schema(env)
        out.append(ChampionStep.from_dict(env["data"]))
    return out


def _check_schema(env: dict[str, Any]) -> None:
    ver = env.get("schema")
    if ver != SCHEMA_VERSION:
        raise ValueError(
            f"telemetry schema mismatch: file has {ver!r}, code expects {SCHEMA_VERSION}"
        )


__all__ = [
    "SCHEMA_VERSION",
    "GenerationRecord",
    "ChampionStep",
    "TelemetryWriter",
    "read_generations",
    "read_champion_steps",
    "TYPE_GENERATION",
    "TYPE_CHAMPION_STEP",
]
