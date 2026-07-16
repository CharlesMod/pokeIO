"""pokeIO configuration — the single source of truth for every knob.

A nested dataclass ``Config`` with sections that mirror the pipeline:
``emu, vision, evo, reward, llm, run``. Defaults are derived from the roadmap
in ``TODO.md`` (frame-skip 24, Discrete(8) action space, N-channel foveated
vision, GLM-4.7-Flash oversight, etc.).

Round-trips through YAML (pyyaml). Every run snapshots its resolved config to
``<run_dir>/config.yaml`` for reproducibility.

Dependency-light: stdlib + pyyaml only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

CONFIG_FILENAME = "config.yaml"


@dataclass
class EmuConfig:
    """Emulator / environment knobs (see TODO Phase 1)."""

    rom_path: str = "roms/pokemon_yellow.gb"
    reset_state: str = "roms/yellow_newgame.state"
    dmg_mode: bool = True  # monochrome DMG mode
    headless: bool = True  # window="null"
    sound_emulated: bool = False
    frame_skip: int = 1  # ticks advanced per agent step (Phase 1.5: reflex fs=1)
    button_hold_frames: int = 0  # frames a button is held within a step (fs>1 only)
    # Concurrent emulator instances ("players"). Measured operating point on this
    # box (bench_40k sweep): 128 players -> ~23.2k agent-steps/s (~3x realtime per
    # player, ~390x aggregate). Pure-throughput peak is ~28 players @ 26.6k, but
    # 128 buys 4.5x the in-flight behavioral diversity for ~13% less throughput.
    n_players: int = 128
    action_space: int = 8  # Discrete(8): up down left right A B START SELECT
    action_names: list[str] = field(
        default_factory=lambda: [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "A",
            "B",
            "START",
            "SELECT",
        ]
    )


@dataclass
class VisionConfig:
    """N-channel configurable vision pipeline (see TODO Phase 1)."""

    screen_height: int = 144
    screen_width: int = 160
    shades: int = 4  # monochrome normalized to {0, 1/3, 2/3, 1}
    coarse_size: int = 32  # whole screen downscaled to coarse_size x coarse_size
    # -- foveal crop (Phase 1 obs spec) --------------------------------------
    # A center crop of the NATIVE screen at native resolution (no resample):
    # game-agnostic since the player sprite is screen-centered in GB overworld
    # games, and it stays an informative center crop in menus/battles. Padded
    # if the crop exceeds the screen bounds.
    fovea_size: int = 64  # side length of the native-resolution center crop
    # -- legacy foveal knobs (superseded by fovea_size; kept for back-compat) -
    foveal_size: int = 32  # (unused by ObsBuilder) old downscaled foveal res
    foveal_crop: int = 48  # (unused by ObsBuilder) old source-pixel crop side
    motion_channel: bool = True  # coarse_t - coarse_{t-1} difference sheet
    ram_aux_dim: int = 32  # length-K RAM aux vector (placeholder until Phase 3)
    # Resulting sheets: coarse global + foveal + optional motion (+ ram_aux vec).
    n_channels: int = 3
    normalize: bool = True


@dataclass
class EvoConfig:
    """Evolution core knobs (see TODO Phase 2)."""

    # Genomes per generation, evaluated in waves of emu.n_players (1024/128 = 8 waves).
    pop_size: int = 1024
    tournament_size: int = 4
    elitism: int = 2
    species_threshold: float = 3.0  # compatibility distance for speciation
    mutate_add_node: float = 0.03
    mutate_add_conn: float = 0.05
    mutate_weight: float = 0.8
    mutate_toggle: float = 0.01
    crossover_rate: float = 0.75
    fitness_sharing: bool = True
    recurrent: bool = True  # allow evolved recurrent connections
    max_nodes: int = 512  # padding bound for tensorized genome
    max_conns: int = 4096


@dataclass
class RewardConfig:
    """Reward stack / Go-Explore / manifest knobs (see TODO Phase 3)."""

    novelty_floor: float = 0.1  # novelty backbone reward always > 0
    archive_cell_downscale: int = 8  # screen downscale before hashing a cell
    use_ram_hash: bool = True  # prefer RAM-hash over pixel-hash (noisy-TV guard)
    rarity_weighted: bool = True
    manifest_path: str = ""  # runs/<id>/manifest.json (LLM-generated)
    reward_pop_size: int = 32  # co-evolving reward candidate population
    miner_entropy_mask: float = 0.95  # mask addresses above this entropy fraction


@dataclass
class LLMConfig:
    """GLM oversight client knobs (see TODO Phase 0 / Phase 3)."""

    model: str = "GLM-4.7-Flash"
    base_url: str = "http://127.0.0.1:8080/v1"
    # UD-Q3_K_XL (13.78GB): largest GLM-4.7-Flash quant that fits ENTIRELY on one
    # 16GB P100 with a 4k ctx (measured 13.75GB on card 0, card 1 free).
    quant: str = "UD-Q3_K_XL"
    context: int = 4096  # tight ctx so the model fits a single card
    device: int = 0  # card 0 = oversight; card 1 stays free for training
    timeout_s: float = 60.0
    max_retries: int = 3
    temperature: float = 0.2
    cache_dir: str = "runs/llm_cache"
    json_mode: bool = True


@dataclass
class RunConfig:
    """Run-level bookkeeping (see TODO Cross-Cutting)."""

    run_id: str = "dev"
    seed: int = 0
    runs_dir: str = "runs"
    device_map: dict[str, int] = field(
        default_factory=lambda: {"llm": 0, "evo": 1, "autoencoder": 1}
    )
    checkpoint_every_gens: int = 25
    telemetry_enabled: bool = True
    git_sha: str = ""  # filled at run start


@dataclass
class Config:
    """Top-level resolved configuration."""

    emu: EmuConfig = field(default_factory=EmuConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    evo: EvoConfig = field(default_factory=EvoConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return _build_dataclass(cls, d or {})

    def save(self, path: str | Path) -> Path:
        """Write this config to a YAML file, returning the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load a config from a YAML file (missing keys fall back to defaults)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def snapshot(self, run_dir: str | Path) -> Path:
        """Write the resolved config to ``<run_dir>/config.yaml``."""
        return self.save(Path(run_dir) / CONFIG_FILENAME)


def _build_dataclass(cls: type, d: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a dict.

    Unknown keys are ignored; nested dataclass fields recurse into their type.
    """
    # Resolve string annotations (PEP 563) to real types.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _build_dataclass(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


__all__ = [
    "Config",
    "EmuConfig",
    "VisionConfig",
    "EvoConfig",
    "RewardConfig",
    "LLMConfig",
    "RunConfig",
    "CONFIG_FILENAME",
]
