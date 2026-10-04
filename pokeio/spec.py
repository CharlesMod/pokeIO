"""Game spec: the only game-specific contract.

A spec is a YAML file that tells the engine *what* to read and *how much* it is
worth, never *how* to play. The engine (env, rewards, policy, trainer, swarm)
consumes only the parsed dataclasses below and must not contain game constants.

Everything except ``rom``/``platform``/``controls`` is optional. A spec with no
``memory`` section is a pure-pixels spec: exploration falls back to screen-hash
novelty and progress terms are unavailable (see games/generic_gb/spec.yaml).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import yaml

FIELD_TYPES = {"u8", "u16be", "u16le", "u24le", "bcd"}
CONDITION_OPS = {"eq", "ne", "gt", "ge", "lt", "le", "nonzero", "zero", "bit", "any_bits"}
TERM_KINDS = {"bitcount", "value", "sum_softcap", "ratio_gain", "distinct"}
VECTOR_KINDS = {"scalar", "categorical", "bits", "ratio"}


class SpecError(ValueError):
    pass


@dataclass
class MemoryField:
    """A named view onto emulator memory.

    ``addr`` or ``symbol`` locates the first element. ``count``/``stride`` make an
    array (e.g. six party slots 44 bytes apart). ``length`` makes a raw byte block
    (e.g. an event-flag bitfield). ``mask``/``shift`` select sub-byte bits of u8s.
    """

    name: str
    addr: int | None = None
    symbol: str | None = None
    type: str = "u8"
    count: int = 1
    stride: int = 1
    length: int = 0
    mask: int = 0xFF
    shift: int = 0
    bcd_bytes: int = 3
    offset: int = 0  # added to the symbol's address

    @property
    def is_block(self) -> bool:
        return self.length > 0


@dataclass
class Condition:
    field: str
    op: str = "nonzero"
    value: int = 0
    index: int = 0  # bit index for op=bit, element index for arrays

    def describe(self) -> str:
        return f"{self.field} {self.op} {self.value if self.op != 'bit' else self.index}"


@dataclass
class Controls:
    buttons: list[str]
    frames_per_action: int = 24
    press_frames: int = 8
    # Auto-advance while the game ignores input (cutscenes, scrolling text):
    # keep tapping ``wait_button`` and ticking until the condition clears.
    wait_while: Condition | None = None
    wait_button: str = "a"
    wait_max_loops: int = 1000


@dataclass
class Screen:
    downsample: int = 2
    levels: int = 4  # palette levels; GB DMG has exactly 4
    channel: str = "luma"  # luma | r | g | b


@dataclass
class Position:
    """Where the player is, in game cells. Enables the visited-cell memory."""

    room: str | None
    x: str
    y: str
    cell_px: int = 16  # full-resolution pixels per movement cell
    # "follow": the camera scrolls with the player, who sits at a fixed screen
    # position (top-left pixel of their cell = player_cell_px).
    # "fixed": each room is drawn statically; cell (x, y) is at (x, y) * cell_px.
    camera: str = "follow"
    player_cell_px: tuple[int, int] = (64, 64)
    # While this holds, position is not credited (e.g. battles, menus with stale coords).
    pause_when: Condition | None = None


@dataclass
class VectorFeature:
    field: str
    kind: str = "scalar"
    scale: float = 1.0
    num: int = 256  # categorical cardinality
    embed: int = 4  # categorical embedding width
    denom: str | None = None  # ratio denominator field


@dataclass
class ProgressTerm:
    """A potential function over memory; reward = weight * (phi_t - phi_{t-1}).

    kinds:
      bitcount    popcount of a field (or block), minus its value at episode start.
      value       the (first element of the) field value, minus start value.
      sum_softcap sum over the first ``count_field`` elements, linear to ``knee``
                  then ``slope`` per unit beyond it.
      ratio_gain  accumulated positive increases of sum(field)/sum(denom),
                  ignored on steps where ``count_field`` changed (healing).
      distinct    number of distinct values the field has taken.
    ``monotone`` (default on) makes phi a running max so toggles can't be farmed.
    """

    name: str
    kind: str
    field: str
    weight: float = 1.0
    monotone: bool = True
    ignore_bits: list[tuple[int, int]] = dc_field(default_factory=list)  # (byte, bit)
    count_field: str | None = None
    denom: str | None = None
    knee: float = 0.0
    slope: float = 0.25
    swarm: bool = True  # counts toward the swarm frontier score


@dataclass
class Milestone:
    name: str
    when: list[Condition]  # all must hold


@dataclass
class Exploration:
    cell_weight: float = 0.02  # reward for refreshing a fully forgotten cell
    half_life_steps: float = 0.0  # 0 = binary memory; >0 = strength half-life (steps)
    floor: float = 0.15  # visited cells never decay below this
    interaction_weight: float = 0.01  # effective interaction at a new (cell, facing)
    interaction_buttons: list[str] = dc_field(default_factory=lambda: ["a"])
    interaction_min_change: float = 0.05  # fraction of screen pixels that must change
    screen_novelty_weight: float = 0.0  # count-based bonus on coarse screen hashes
    screen_hash_grid: tuple[int, int] = (9, 10)  # rows, cols of the hash thumbnail


@dataclass
class Swarm:
    enabled: bool = True
    score: str = "progress"  # progress | milestones
    min_delta: float = 1.0  # score gain over the global best that triggers migration
    fraction: float = 1.0  # fraction of envs moved to the new frontier state
    min_interval_steps: int = 0  # global agent steps between migrations


@dataclass
class Episode:
    # Exploration memory is wiped every ``memory_reset_steps`` (with jitter); the
    # emulator itself keeps running ("mini-episodes", see docs/puffer-lessons.md).
    memory_reset_steps: int = 19816
    memory_reset_jitter: int = 2000
    # Hard reset back to a start/frontier state (0 = never).
    hard_reset_steps: int = 0
    # Reset if no positive progress reward for this many steps (0 = never).
    stall_reset_steps: int = 0
    # On every reset, idle a random 0..N frames first. Game Boy RNG is driven by
    # timing, so this diversifies otherwise identical starts across envs.
    start_jitter_frames: int = 0


@dataclass
class GameSpec:
    name: str
    platform: str
    rom: Path
    controls: Controls
    rom_md5: str | None = None
    symbols: Path | None = None
    platform_options: dict[str, Any] = dc_field(default_factory=dict)
    start_states: list[Path] = dc_field(default_factory=list)
    boot_macro: list[list[Any]] = dc_field(default_factory=list)
    memory_patches: list[dict[str, Any]] = dc_field(default_factory=list)
    screen: Screen = dc_field(default_factory=Screen)
    memory: dict[str, MemoryField] = dc_field(default_factory=dict)
    position: Position | None = None
    vector: list[VectorFeature] = dc_field(default_factory=list)
    terms: list[ProgressTerm] = dc_field(default_factory=list)
    milestones: list[Milestone] = dc_field(default_factory=list)
    exploration: Exploration = dc_field(default_factory=Exploration)
    swarm: Swarm = dc_field(default_factory=Swarm)
    episode: Episode = dc_field(default_factory=Episode)
    # Dashboard sugar only (never read by the agent): title, room names, palette.
    display: dict[str, Any] = dc_field(default_factory=dict)
    root: Path = Path(".")

    @property
    def fingerprint(self) -> str:
        """Stable hash of the parts that change observation/action shapes."""
        parts = [
            self.platform,
            ",".join(self.controls.buttons),
            str(self.screen.downsample),
            str(self.screen.levels),
            str(self.position is not None),
            ";".join(
                f"{v.field}:{v.kind}:{v.num}:{v.embed}:"
                f"{self.memory[v.field].length or self.memory[v.field].count}"
                for v in self.vector
                if v.field in self.memory
            ),
        ]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# parsing


def _int(v: Any) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return int(v, 0)
    raise SpecError(f"expected an integer, got {v!r}")


def _condition(d: Any, where: str) -> Condition:
    if not isinstance(d, dict) or "field" not in d:
        raise SpecError(f"{where}: condition needs a 'field'")
    c = Condition(
        field=d["field"],
        op=d.get("op", "nonzero"),
        value=_int(d.get("value", 0)),
        index=_int(d.get("index", 0)),
    )
    if c.op not in CONDITION_OPS:
        raise SpecError(f"{where}: unknown op {c.op!r}")
    return c


def _conditions(d: Any, where: str) -> list[Condition]:
    if isinstance(d, list):
        return [_condition(x, where) for x in d]
    return [_condition(d, where)]


def _memory(raw: dict[str, Any]) -> dict[str, MemoryField]:
    out: dict[str, MemoryField] = {}
    for name, d in (raw or {}).items():
        if isinstance(d, (int, str)) and not isinstance(d, bool):
            d = {"addr": d}
        f = MemoryField(
            name=name,
            addr=_int(d["addr"]) if "addr" in d else None,
            symbol=d.get("symbol"),
            type=d.get("type", "u8"),
            count=_int(d.get("count", 1)),
            stride=_int(d.get("stride", 1)),
            length=_int(d.get("length", 0)),
            mask=_int(d.get("mask", 0xFF)),
            shift=_int(d.get("shift", 0)),
            bcd_bytes=_int(d.get("bcd_bytes", 3)),
            offset=_int(d.get("offset", 0)),
        )
        if f.addr is None and f.symbol is None:
            raise SpecError(f"memory.{name}: needs 'addr' or 'symbol'")
        if f.type not in FIELD_TYPES:
            raise SpecError(f"memory.{name}: unknown type {f.type!r}")
        out[name] = f
    return out


def _resolve(root: Path, p: str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def load_spec(path: str | Path, repo_root: str | Path | None = None) -> GameSpec:
    """Parse and validate a spec. Relative paths resolve against ``repo_root``
    (default: the current working directory), so specs can say ``roms/x.gb``."""
    path = Path(path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    ctl = raw.get("controls") or {}
    if not ctl.get("buttons"):
        raise SpecError("controls.buttons is required")
    controls = Controls(
        buttons=list(ctl["buttons"]),
        frames_per_action=_int(ctl.get("frames_per_action", 24)),
        press_frames=_int(ctl.get("press_frames", 8)),
        wait_while=_condition(ctl["wait_while"], "controls.wait_while")
        if ctl.get("wait_while")
        else None,
        wait_button=ctl.get("wait_button", "a"),
        wait_max_loops=_int(ctl.get("wait_max_loops", 1000)),
    )
    if controls.press_frames >= controls.frames_per_action:
        raise SpecError("controls.press_frames must be < frames_per_action")

    pos = None
    if raw.get("position"):
        p = raw["position"]
        pos = Position(
            room=p.get("room"),
            x=p["x"],
            y=p["y"],
            cell_px=_int(p.get("cell_px", 16)),
            camera=p.get("camera", "follow"),
            player_cell_px=tuple(_int(v) for v in p.get("player_cell_px", (64, 64))),
            pause_when=_condition(p["pause_when"], "position.pause_when")
            if p.get("pause_when")
            else None,
        )

    obs = raw.get("observation") or {}
    vector = []
    for v in obs.get("vector") or []:
        vf = VectorFeature(
            field=v["field"],
            kind=v.get("kind", "scalar"),
            scale=float(v.get("scale", 1.0)),
            num=_int(v.get("num", 256)),
            embed=_int(v.get("embed", 4)),
            denom=v.get("denom"),
        )
        if vf.kind not in VECTOR_KINDS:
            raise SpecError(f"observation.vector: unknown kind {vf.kind!r}")
        vector.append(vf)

    rew = raw.get("rewards") or {}
    terms = []
    for t in rew.get("terms") or []:
        term = ProgressTerm(
            name=t["name"],
            kind=t["kind"],
            field=t["field"],
            weight=float(t.get("weight", 1.0)),
            monotone=bool(t.get("monotone", True)),
            ignore_bits=[(_int(b[0]), _int(b[1])) for b in t.get("ignore_bits", [])],
            count_field=t.get("count_field"),
            denom=t.get("denom"),
            knee=float(t.get("knee", 0.0)),
            slope=float(t.get("slope", 0.25)),
            swarm=bool(t.get("swarm", True)),
        )
        if term.kind not in TERM_KINDS:
            raise SpecError(f"rewards.terms.{term.name}: unknown kind {term.kind!r}")
        terms.append(term)

    ex = rew.get("exploration") or {}
    exploration = Exploration(
        cell_weight=float(ex.get("cell_weight", 0.02)),
        half_life_steps=float(ex.get("half_life_steps", 0.0)),
        floor=float(ex.get("floor", 0.15)),
        interaction_weight=float(ex.get("interaction_weight", 0.01)),
        interaction_buttons=list(ex.get("interaction_buttons", ["a"])),
        interaction_min_change=float(ex.get("interaction_min_change", 0.05)),
        screen_novelty_weight=float(ex.get("screen_novelty_weight", 0.0)),
        screen_hash_grid=tuple(_int(v) for v in ex.get("screen_hash_grid", (9, 10))),
    )

    milestones = [
        Milestone(name=m["name"], when=_conditions(m["when"], f"milestones.{m['name']}"))
        for m in raw.get("milestones") or []
    ]

    sw = raw.get("swarm") or {}
    swarm = Swarm(
        enabled=bool(sw.get("enabled", True)),
        score=sw.get("score", "progress"),
        min_delta=float(sw.get("min_delta", 1.0)),
        fraction=float(sw.get("fraction", 1.0)),
        min_interval_steps=_int(sw.get("min_interval_steps", 0)),
    )
    if swarm.score not in ("progress", "milestones"):
        raise SpecError("swarm.score must be 'progress' or 'milestones'")

    ep = raw.get("episode") or {}
    episode = Episode(
        memory_reset_steps=_int(ep.get("memory_reset_steps", 19816)),
        memory_reset_jitter=_int(ep.get("memory_reset_jitter", 2000)),
        hard_reset_steps=_int(ep.get("hard_reset_steps", 0)),
        stall_reset_steps=_int(ep.get("stall_reset_steps", 0)),
        start_jitter_frames=_int(ep.get("start_jitter_frames", 0)),
    )

    scr = raw.get("screen") or {}
    start = raw.get("start") or {}
    spec = GameSpec(
        name=raw.get("name", path.parent.name),
        platform=raw.get("platform", "gameboy"),
        rom=_resolve(root, raw["rom"]),
        rom_md5=raw.get("rom_md5"),
        symbols=_resolve(root, raw.get("symbols")),
        platform_options=dict(raw.get("platform_options") or {}),
        controls=controls,
        start_states=[_resolve(root, s) for s in start.get("states", [])],
        boot_macro=list(start.get("boot_macro") or []),
        memory_patches=list(start.get("memory_patches") or []),
        screen=Screen(
            downsample=_int(scr.get("downsample", 2)),
            levels=_int(scr.get("levels", 4)),
            channel=scr.get("channel", "luma"),
        ),
        memory=_memory(raw.get("memory") or {}),
        position=pos,
        vector=vector,
        terms=terms,
        milestones=milestones,
        exploration=exploration,
        swarm=swarm,
        episode=episode,
        display=dict(raw.get("display") or {}),
        root=root,
    )
    problems = validate(spec)
    if problems:
        raise SpecError("invalid spec:\n  " + "\n  ".join(problems))
    return spec


def validate(spec: GameSpec) -> list[str]:
    """Cross-reference checks: every referenced field must be declared."""
    problems: list[str] = []
    mem = spec.memory

    def need(name: str | None, where: str) -> None:
        if name is not None and name not in mem:
            problems.append(f"{where}: unknown memory field {name!r}")

    if spec.controls.wait_while:
        need(spec.controls.wait_while.field, "controls.wait_while")
    if spec.position:
        if spec.position.camera not in ("follow", "fixed"):
            problems.append("position.camera must be 'follow' or 'fixed'")
        need(spec.position.room, "position.room")
        need(spec.position.x, "position.x")
        need(spec.position.y, "position.y")
        if spec.position.pause_when:
            need(spec.position.pause_when.field, "position.pause_when")
    for v in spec.vector:
        need(v.field, f"observation.vector[{v.field}]")
        need(v.denom, f"observation.vector[{v.field}].denom")
        if v.kind == "ratio" and v.denom is None:
            problems.append(f"observation.vector[{v.field}]: ratio needs 'denom'")
    for t in spec.terms:
        need(t.field, f"rewards.terms.{t.name}")
        need(t.count_field, f"rewards.terms.{t.name}.count_field")
        need(t.denom, f"rewards.terms.{t.name}.denom")
        if t.kind == "ratio_gain" and t.denom is None:
            problems.append(f"rewards.terms.{t.name}: ratio_gain needs 'denom'")
    for m in spec.milestones:
        for c in m.when:
            need(c.field, f"milestones.{m.name}")
    names = [t.name for t in spec.terms]
    if len(set(names)) != len(names):
        problems.append("rewards.terms: duplicate names")
    if spec.swarm.score == "milestones" and not spec.milestones:
        problems.append("swarm.score=milestones but no milestones are defined")
    unknown = set(spec.controls.buttons) - {
        "up", "down", "left", "right", "a", "b", "start", "select", "noop",
    }
    if unknown:
        problems.append(f"controls.buttons: unknown {sorted(unknown)}")
    return problems
