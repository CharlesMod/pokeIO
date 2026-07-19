"""From-boot progress metric — scores REAL game progress from RAM game-state.

The prerequisite Fable flagged as co-equal with the reward (PokeIO v2.0 task #28,
``docs/specs/brain-architecture.md`` RED-TEAM REVISION): backward-robustification /
reward / QD selection all depend on a from-boot signal of *real* progress, not a
proxy. Optimizing against a proxy is what failed (below).

Why this and not cell-count / novelty
-------------------------------------
Cell-count / novelty are exactly what Goodharted. ``runs/live1`` logged a
Go-Explore frontier *depth of 12402* while **zero** of its states ever held a
single Pokémon. Two things drove that: (1) novelty was measured partly on a WRAM
byte that is a *pointer low byte* ($D35E), which ticks every step you walk — so a
4-room walk minted ~30 distinct "maps" of pure noise; and (2) even the real
underlying play never completed the Oak sequence to get a starter. This metric
instead reads **RAM game-state milestones** that cannot be faked without competent
play:

  * distinct map-IDs reached (spatial progress, read from the REAL map byte),
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

Yellow RAM addresses (VERIFIED against pixels — the mandatory protocol)
----------------------------------------------------------------------
Pokémon **Yellow's** WRAM save block is shifted **-1 vs Red/Blue**; the repo
historically used the Red/Blue addresses and every semantic tap read its neighbour
byte (the "-20 map shift", "glitch map 245/253", and the party-count-reads-84 bugs
were all this one root cause). The table below is the corrected Yellow map, pinned
by replaying the first-milestone demonstration (``assets/demo_pikachu``, 657 actions
newgame->Pikachu) and reading RAM *while watching the pixels* — the permanent
eyes-on-pixels rule now in the eval spine (no RAM tap feeds a metric until it
agrees with the screen on a ground-truth transition):

    map id      $D35D   (newgame bedroom=38; walks 38->37->0->40 = bedroom->1F->
                         Pallet->Oak's lab; NO $F0+ sentinels ever appear)
    party count $D162   (0 at newgame, 1 after the starter)
    species[0]  $D163   (84 = Pikachu's Gen-1 internal id; = the old party-count tap)
    mon1 level  $D18B   (stride $2C x6)
    badges      $D355   (popcount)
    money       $D346   (3-byte BCD; reads 003000 at newgame)
    events      $D746..$D87E  (wEventFlags popcount; 0 at newgame, 5 at the starter)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRAM_BASE = 0xC000


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
    # --- composite weights (progress "points") ---
    w_map: float = 1.0     # each distinct map beyond the spawn map
    w_party: float = 5.0   # each Pokémon in the party (requires reaching Oak)
    w_level: float = 1.0   # each summed party level
    w_badge: float = 50.0  # each gym badge (a major, multi-hour milestone)
    w_event: float = 2.0   # each story event flag
    w_counter: float = 1.0 # per-unit of a generic miner-discovered progress counter
    notes: str = ""


# Verified Pokémon YELLOW spec (see module docstring: pixels-validated, Yellow's
# save block is Red/Blue - 1). Event flags fire on real progress (verified: 0 at
# newgame, first flag by step ~47 of the demo, 5 at the starter) — they are a
# load-bearing signal now, not the soft/unverified guess the old R/B tap forced.
YELLOW = GameProgressSpec(
    name="Pokemon Yellow",
    map_addr=0xD35D,
    party_count_addr=0xD162,
    party_level_addr=0xD18B,
    party_stride=0x2C,
    party_max=6,
    party_uninit=(0xFF,),
    badge_addr=0xD355,
    badge_bits=8,
    money_addr=0xD346,
    event_regions=((0xD746, 0xD87E),),
    notes="Yellow WRAM save block = Red/Blue - 1; taps pixel-validated on assets/demo_pikachu",
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
    """Weighted 'progress points'. ``maps`` counts distinct map-IDs beyond spawn.

    No cap: the map byte ($D35D) is the REAL current-map id, so distinct maps are
    genuine spatial progress (the old cap fought the pointer-byte noise from the
    wrong tap — see the module docstring — which no longer exists). Party / badges
    / events dominate the composite anyway.
    """
    maps_beyond = max(0, maps - 1)
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
    e.g. a faint); ``maps`` is the count of **distinct map-IDs** seen across the
    whole rollout.

    Deterministic (same snapshots -> same score). Reads RAM only. NEVER feed the
    result to the agent as an observation input — it is a reward/eval signal.

    Returns a dict: ``progress_score`` (composite float) plus the raw sub-signals
    ``maps, badges, party_level, party_count, events`` (and ``map_id, money,
    counter_sum, started, n_snapshots``).
    """
    snaps = _snapshots(env_or_rollout)
    if not snaps:
        return _empty(spec)

    maps_seen: set[int] = set()
    party_count = party_level = badges = events = counter_sum = 0
    money = None
    last_mid = 0
    for w in snaps:
        d = decode_state(w, spec)
        last_mid = d["map_id"]
        maps_seen.add(d["map_id"])
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

    maps = len(maps_seen)
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
    "GameProgressSpec", "YELLOW",
    "measure", "progress_scalar", "decode_state", "generic_spec",
]
