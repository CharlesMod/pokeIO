"""pokeIO Progress Manifest — the only game-specific contract.

The manifest is the *single* place a game name, RAM address, or milestone label
is allowed to appear. It is normally **LLM-generated** (see TODO Phase 3,
``manifest/generate.py``); the reward stack and the dashboard both consume it
and neither knows the word "Pokémon".

JSON shape (matches the TODO example)::

    {
      "game": "Pokemon Yellow",
      "milestone_label": "Badges",
      "spatial": {"source": {"ram": {...}}, "grid": [...]},
      "progress_dimensions": [
        {
          "id": "badges",
          "label": "Gym Badges",
          "source": {"ram": {"addr": "0xD356", "kind": "bitfield", "bits": 8}},
          "dir": "up",
          "icon": "medal"
        },
        ...
      ]
    }

``validate()`` returns a list of human-readable problems (empty == valid).

Dependency-light: stdlib + orjson only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import orjson

# Bump when the manifest contract changes incompatibly.
MANIFEST_SCHEMA_VERSION = 0

# A progress dimension improves in one of these directions.
VALID_DIRECTIONS = {"up", "down"}


@dataclass
class ProgressDimension:
    """One tracked axis of game progress (a badge count, party size, ...).

    ``source`` must carry at least one of ``ram`` or ``screen`` describing where
    the value is read from. ``dir`` is the direction of *improvement*.
    """

    id: str
    label: str
    source: dict[str, Any]  # {"ram": {...}} and/or {"screen": {...}}
    dir: str  # "up" | "down"
    icon: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProgressDimension":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Manifest:
    """The full Progress Manifest for one game."""

    game: str
    progress_dimensions: list[ProgressDimension] = field(default_factory=list)
    spatial: dict[str, Any] = field(default_factory=dict)
    milestone_label: str = ""

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA_VERSION,
            "game": self.game,
            "milestone_label": self.milestone_label,
            "spatial": self.spatial,
            "progress_dimensions": [pd.to_dict() for pd in self.progress_dimensions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        return cls(
            game=d.get("game", ""),
            milestone_label=d.get("milestone_label", ""),
            spatial=d.get("spatial", {}) or {},
            progress_dimensions=[
                ProgressDimension.from_dict(pd)
                for pd in d.get("progress_dimensions", []) or []
            ],
        )

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2))
        return p

    @classmethod
    def load_json(cls, path: str | Path) -> "Manifest":
        with open(path, "rb") as fh:
            return cls.from_dict(orjson.loads(fh.read()))

    # -- validation --------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of problems; empty means the manifest is valid."""
        problems: list[str] = []

        if not self.game or not str(self.game).strip():
            problems.append("game: must be a non-empty string")

        if not self.milestone_label or not str(self.milestone_label).strip():
            problems.append("milestone_label: must be a non-empty string")

        if not self.progress_dimensions:
            problems.append("progress_dimensions: must contain at least one dimension")

        seen_ids: set[str] = set()
        for i, pd in enumerate(self.progress_dimensions):
            where = f"progress_dimensions[{i}]"

            if not pd.id or not str(pd.id).strip():
                problems.append(f"{where}.id: must be a non-empty string")
            elif pd.id in seen_ids:
                problems.append(f"{where}.id: duplicate id {pd.id!r}")
            else:
                seen_ids.add(pd.id)

            if not pd.label or not str(pd.label).strip():
                problems.append(f"{where}.label: must be a non-empty string")

            if pd.dir not in VALID_DIRECTIONS:
                problems.append(
                    f"{where}.dir: must be one of {sorted(VALID_DIRECTIONS)}, got {pd.dir!r}"
                )

            if not isinstance(pd.source, dict) or not pd.source:
                problems.append(f"{where}.source: must be a non-empty object")
            elif "ram" not in pd.source and "screen" not in pd.source:
                problems.append(
                    f"{where}.source: must have a 'ram' or 'screen' key"
                )

        # Spatial section is optional, but if present it must carry a source.
        if self.spatial:
            src = self.spatial.get("source")
            if not isinstance(src, dict) or (
                "ram" not in src and "screen" not in src
            ):
                problems.append(
                    "spatial.source: when spatial is present it must have a "
                    "'ram' or 'screen' source"
                )

        return problems


def load_json(path: str | Path) -> Manifest:
    """Module-level convenience wrapper around :meth:`Manifest.load_json`."""
    return Manifest.load_json(path)


def save_json(manifest: Manifest, path: str | Path) -> Path:
    """Module-level convenience wrapper around :meth:`Manifest.save_json`."""
    return manifest.save_json(path)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "VALID_DIRECTIONS",
    "ProgressDimension",
    "Manifest",
    "load_json",
    "save_json",
]
