"""From-boot progress metric — scores REAL game progress from RAM game-state.

The prerequisite Fable flagged as co-equal with the reward (PokeIO v2.0 task #28,
``docs/specs/brain-architecture.md`` RED-TEAM REVISION): the old boot gauntlet
reads ~4.4% ≈ noise, and backward-robustification / reward / QD selection all
depend on a from-boot signal that barely exists. Optimizing against a near-zero
metric is optimizing against noise.

Why this and not cell-count / novelty
-------------------------------------
Cell-count and novelty are exactly what failed: a run can log a Go-Explore
frontier *depth of 12402* (``runs/live1`` gen 175) while **zero** of its states
ever held a single Pokémon — the "progress" was the exploration proxy Goodharting
into glitch/menu state-space (verified: ``pokeio.analytics.gameprogress`` already
warns "no agent has legitimately started the game"). This metric instead reads
**RAM game-state milestones** that cannot be faked without competent play:

  * distinct real map-IDs reached (spatial progress),
  * badge count,
  * party count + summed party levels (having & training Pokémon),
  * story event-flag count.

It is a **reward / evaluation signal only** — deterministic, reads RAM, and MUST
NEVER be fed to the agent as an observation input (the miner-Goodhart trap the
red-team revision calls out).

Game-agnostic hook
------------------
The Yellow specifics live in a :class:`GameProgressSpec` (addresses + weights).
A second game plugs in its own spec: the game-blind miner
(:mod:`pokeio.reward.miner`) discovers *progress-correlated* counter addresses,
and a small per-game spec maps which are "progress" (a map-id counter plus the
milestone counters). Fill the semantic fields when the RAM map is known (Yellow),
or leave them ``None`` and pass ``progress_counters`` for a game whose semantics
are still generic. Nothing below hardcodes Yellow so deeply that only Yellow works.

Interface (the seam #29 eval-spine / #30 reward / #31 replay call)
------------------------------------------------------------------
``measure(env_or_rollout, spec=YELLOW) -> {"progress_score": float, "maps": int,
"badges": int, "party_level": int, "party_count": int, "events": int, ...}`` and
``progress_scalar(env_or_rollout, spec=YELLOW) -> float`` for just the composite.

Yellow RAM addresses (source & cross-check)
-------------------------------------------
The addresses below match the verified table in
:mod:`pokeio.analytics.yellow` (map/party/badges/money are validated against real
game states there) and the Gen-1 RAM map (datacrystal Pokémon Red/Blue; Yellow
shares this WRAM control block — confirmed on this ROM: map ``$D35E``, party count
``$D163``, party-mon level ``$D18C`` stride ``$2C``, badges ``$D356``, money
``$D347``). Event flags ``$D747..$D87F`` are the standard Gen-1 ``wEventFlags``
array; see the CAVEAT in :data:`YELLOW`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WRAM_BASE = 0xC000
# Map-id bytes >= this are loader/transition sentinels ($F0-$FF), emitted mid-warp
# — NOT real maps. Counting them would let menu/door-thrashing inflate the score
# (pokeio.analytics.gameprogress drops the same range for the frontier histogram).
SENTINEL_MIN = 0xF0


# --------------------------------------------------------------------------- spec
@dataclass(frozen=True)
class GameProgressSpec:
    """Per-game progress decode + composition config (the game-agnostic hook).

    Semantic fields (``*_addr``) may be ``None`` for a game whose RAM semantics
    are unknown; then only ``map_addr`` (universal) + ``progress_counters``
    (miner-discovered) drive the score. Weights are on a rough "how hard to earn"
    scale so a random policy stays near the floor and real milestones dominate.
    """

    name: str
    map_addr: int                              # current map-id byte
    # --- semantic milestone taps (None => that signal is skipped) ---
    party_count_addr: int | None = None
    party_level_addr: int | None = None        # first party-mon level byte
    party_stride: int = 0x2C                    # bytes between consecutive party mons
    party_max: int = 6
    party_uninit: tuple[int, ...] = (0xFF,)     # count values meaning "empty/uninit"
    badge_addr: int | None = None
    badge_bits: int = 8
    money_addr: int | None = None              # 3-byte BCD start (reported, unweighted)
    event_regions: tuple[tuple[int, int], ...] = ()   # inclusive (start,end) byte ranges
    # --- generic (2nd-game) progress counters the miner flagged: (addr, width) ---
    progress_counters: tuple[tuple[int, int], ...] = ()
    sentinel_min: int = SENTINEL_MIN
    # Cap on the distinct-map term (None => uncapped). Raw distinct-map-byte count
    # is a WEAK signal: glitch/menu state-space thrashing collides with real map
    # ids (e.g. runs/live1 shows 30 distinct non-sentinel map-bytes at party=0,
    # though you cannot legitimately leave Pallet without a Pokémon). Capping keeps
    # map-thrash bounded well below a single milestone so it cannot fake progress;
    # once you truly explore widely, party/level/badges dominate anyway. ~= the
    # count of maps legitimately reachable before the first Pokémon (bedroom, 1F,
    # Pallet, rival's house, Oak's lab).
    map_cap: int | None = 6
    # --- composite weights (progress "points") ---
    w_map: float = 1.0     # each distinct real map beyond the spawn map (capped)
    w_party: float = 5.0   # each Pokémon in the party (requires reaching Oak)
    w_level: float = 1.0   # each summed party level
    w_badge: float = 50.0  # each gym badge (a major, multi-hour milestone)
    w_event: float = 2.0   # each story event flag
    w_counter: float = 1.0 # per-unit of a generic miner-discovered progress counter
    notes: str = ""


# Verified Pokémon YELLOW spec. Addresses cross-checked against
# pokeio.analytics.yellow (validated on this ROM) + the Gen-1 RAM map.
#
# EVENT-FLAG CAVEAT: $D747..$D87F is the standard Gen-1 ``wEventFlags`` array.
# WRAM layout (unlike the map-id *enumeration*, which this ROM revision shifts by
# -20 on interiors) is stable across Gen-1 revisions, and the region reads a clean
# 0 at newgame — consistent with the new-game routine zeroing the event array
# (uninitialised RAM here would be nonzero garbage, as the money bytes are). That
# is strong-but-not-proof evidence it is the right region on THIS ROM: no reached
# story event was available to exercise it (no policy/script in-repo starts the
# game). Treated as a *soft* signal (low weight); the load-bearing discriminators
# are party / badges / levels / maps.
YELLOW = GameProgressSpec(
    name="Pokemon Yellow",
    map_addr=0xD35E,
    party_count_addr=0xD163,
    party_level_addr=0xD18C,
    party_stride=0x2C,
    party_max=6,
    party_uninit=(0xFF,),
    badge_addr=0xD356,
    badge_bits=8,
    money_addr=0xD347,
    event_regions=((0xD747, 0xD87F),),
    notes="event flags soft/unverified-on-reached-event; see module CAVEAT",
)


# --------------------------------------------------------------------------- reads
def _b(wram: np.ndarray, addr: int) -> int:
    i = addr - WRAM_BASE
    return int(wram[i]) if 0 <= i < wram.size else 0


def _popcount_regions(wram: np.ndarray, regions) -> int:
    """Total set bits across the inclusive byte ranges (event-flag count)."""
    total = 0
    for start, end in regions:
        lo, hi = start - WRAM_BASE, end - WRAM_BASE + 1
        seg = wram[max(0, lo): min(wram.size, hi)]
        if seg.size:
            total += int(np.unpackbits(seg.astype(np.uint8)).sum())
    return total


def _party(wram: np.ndarray, spec: GameProgressSpec) -> tuple[int, int]:
    """``(party_count, party_level_sum)``; uninit/garbage count => (0, 0)."""
    if spec.party_count_addr is None:
        return 0, 0
    pc = _b(wram, spec.party_count_addr)
    if pc in spec.party_uninit or pc > spec.party_max:
        return 0, 0
    lvl_sum = 0
    if spec.party_level_addr is not None:
        lvl_sum = sum(
            _b(wram, spec.party_level_addr + i * spec.party_stride) for i in range(pc)
        )
    return pc, lvl_sum


def _bcd3(wram: np.ndarray, addr: int) -> int | None:
    """3-byte packed-BCD (Gen-1 money). None if any nibble is not a BCD digit."""
    digits = "".join(f"{_b(wram, addr + i):02x}" for i in range(3))
    if any(c not in "0123456789" for c in digits):
        return None
    return int(digits)


def _counter_sum(wram: np.ndarray, counters) -> int:
    """Sum of generic miner-discovered progress counters (little-endian widths)."""
    total = 0
    for addr, width in counters:
        val = 0
        for k in range(width):
            val += _b(wram, addr + k) << (8 * k)
        total += val
    return total


def decode_state(wram: np.ndarray, spec: GameProgressSpec = YELLOW) -> dict:
    """Decode ONE WRAM snapshot into raw progress sub-signals (no map-set)."""
    mid = _b(wram, spec.map_addr)
    pc, lvl = _party(wram, spec)
    badges = _b(wram, spec.badge_addr) if spec.badge_addr is not None else 0
    badge_ct = bin(badges).count("1") if spec.badge_addr is not None else 0
    return {
        "map_id": mid,
        "real_map": mid < spec.sentinel_min,
        "party_count": pc,
        "party_level": lvl,
        "badges": badge_ct,
        "badge_bits": badges,
        "events": _popcount_regions(wram, spec.event_regions),
        "money": _bcd3(wram, spec.money_addr) if spec.money_addr is not None else None,
        "counter_sum": _counter_sum(wram, spec.progress_counters),
    }


# ------------------------------------------------------------------- composition
def _compose(spec: GameProgressSpec, *, maps: int, party_count: int,
             party_level: int, badges: int, events: int, counter_sum: int) -> float:
    """Weighted 'progress points'. maps counts distinct real maps beyond spawn,
    capped by ``spec.map_cap`` so glitch/menu map-thrash cannot inflate the score."""
    maps_beyond = max(0, maps - 1)
    if spec.map_cap is not None:
        maps_beyond = min(maps_beyond, spec.map_cap)
    return float(
        spec.w_map * maps_beyond
        + spec.w_party * party_count
        + spec.w_level * party_level
        + spec.w_badge * badges
        + spec.w_event * events
        + spec.w_counter * counter_sum
    )


# --------------------------------------------------------------------- normalize
def _snapshots(source) -> list[np.ndarray]:
    """Coerce ``source`` into a list of 1-D WRAM snapshots.

    Accepts: an env-like object exposing ``raw_wram()`` (a single current state);
    a 1-D ndarray (single snapshot); a 2-D ``(T, N)`` ndarray (a rollout); or a
    sequence of 1-D snapshots.
    """
    if hasattr(source, "raw_wram"):
        return [np.asarray(source.raw_wram())]
    if not isinstance(source, np.ndarray):
        try:
            if len(source) == 0:  # empty rollout
                return []
        except TypeError:
            pass
    arr = np.asarray(source) if not isinstance(source, np.ndarray) else source
    if arr.size == 0:
        return []
    if arr.dtype == object or (arr.ndim == 1 and arr.size and np.ndim(arr[0]) == 1):
        return [np.asarray(s).ravel() for s in source]
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[t] for t in range(arr.shape[0])]
    raise ValueError(f"unrecognized source with shape {getattr(arr, 'shape', None)}")


# ---------------------------------------------------------------------- public API
def measure(env_or_rollout, spec: GameProgressSpec = YELLOW) -> dict:
    """Score real from-boot progress. THE seam #29/#30/#31 call.

    ``env_or_rollout`` is a from-newgame rollout's WRAM snapshots (preferred:
    every step's ``PokeEnv.raw_wram()``), or a single env / snapshot for the
    current state. Monotone milestones (party/level/badges/events/counters) take
    their **max over the rollout** (progress reached counts even if later lost,
    e.g. a faint); ``maps`` is the count of **distinct real (non-sentinel) map-IDs**
    seen across the whole rollout.

    Deterministic (same snapshots -> same score). Reads RAM only. NEVER feed the
    result to the agent as an observation input — it is a reward/eval signal.

    Returns a dict: ``progress_score`` (composite float) plus the raw sub-signals
    ``maps, badges, party_level, party_count, events`` (and ``map_id, money,
    counter_sum, started, n_snapshots``).
    """
    snaps = _snapshots(env_or_rollout)
    if not snaps:
        return _empty(spec)

    real_maps: set[int] = set()
    party_count = party_level = badges = events = counter_sum = 0
    money = None
    last_mid = 0
    for w in snaps:
        d = decode_state(w, spec)
        last_mid = d["map_id"]
        if d["real_map"]:
            real_maps.add(d["map_id"])
        # max over the rollout for the monotone-ish milestone signals
        if d["party_count"] > party_count:
            party_count = d["party_count"]
        if d["party_level"] > party_level:
            party_level = d["party_level"]
        if d["badges"] > badges:
            badges = d["badges"]
        if d["events"] > events:
            events = d["events"]
        if d["counter_sum"] > counter_sum:
            counter_sum = d["counter_sum"]
        if d["money"] is not None:
            money = d["money"]

    maps = len(real_maps)
    score = _compose(
        spec, maps=maps, party_count=party_count, party_level=party_level,
        badges=badges, events=events, counter_sum=counter_sum,
    )
    started = bool(party_count > 0 or badges > 0)
    return {
        "progress_score": score,
        "maps": maps,
        "badges": badges,
        "party_level": party_level,
        "party_count": party_count,
        "events": events,
        "counter_sum": counter_sum,
        "money": money,
        "map_id": last_mid,
        "started": started,
        "n_snapshots": len(snaps),
    }


def progress_scalar(env_or_rollout, spec: GameProgressSpec = YELLOW) -> float:
    """Convenience: just the composite ``progress_score`` float."""
    return measure(env_or_rollout, spec)["progress_score"]


def _empty(spec: GameProgressSpec) -> dict:
    return {
        "progress_score": 0.0, "maps": 0, "badges": 0, "party_level": 0,
        "party_count": 0, "events": 0, "counter_sum": 0, "money": None,
        "map_id": 0, "started": False, "n_snapshots": 0,
    }


def generic_spec(name: str, map_addr: int, progress_counters, **weights) -> GameProgressSpec:
    """Build a semantic-free spec for a 2nd game: distinct maps + miner counters.

    ``progress_counters`` is a sequence of ``(addr, width)`` the game-blind miner
    flagged as progress-correlated. This is the plug point for cross-game transfer
    — no Yellow semantics required.
    """
    return GameProgressSpec(
        name=name, map_addr=map_addr,
        progress_counters=tuple((int(a), int(w)) for a, w in progress_counters),
        **weights,
    )


__all__ = [
    "GameProgressSpec", "YELLOW", "SENTINEL_MIN",
    "measure", "progress_scalar", "decode_state", "generic_spec",
]
