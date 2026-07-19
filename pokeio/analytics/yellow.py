"""Pokémon Yellow game-state decoder (TRAINER-FACING, not for the ML system).

Decodes a raw 8 KB WRAM snapshot (``PokeEnv.raw_wram()``, GB $C000-$DFFF →
index 0) into a human-readable :class:`GameState`. This is deliberately
game-specific — it exists so WE can see how far into Pokémon the swarm has
gotten. It never feeds the reward stack (which stays game-agnostic).

Addresses are Pokémon YELLOW-specific (pret/pokeyellow): Yellow's WRAM save
block is shifted **-1 vs Red/Blue**, so every semantic tap is one byte below
its R/B address. Every field is pixel-validated against real game states (the
newgame->Pikachu demo) — the mandatory eyes-on-pixels rule — and
unverified/uninitialised reads surface as ``None`` rather than garbage. See
MAP_NAMES / _ADDR below for the source-of-truth table.

QUARANTINE: the hardcoded Yellow tables (``_ADDR``, ``MAP_NAMES``,
``BADGE_NAMES``) are the game-specific fallback. When a Progress Manifest is
passed, :func:`decode` only applies them if the manifest's ``game`` is Yellow
(:func:`game_is_yellow`); for any other game it decodes generically from the
manifest's spatial RAM taps, so Yellow labels can never leak onto a second
game. With no manifest (the current default) the Yellow path is used, keeping
the existing dashboard working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRAM_BASE = 0xC000


def _b(wram: np.ndarray, addr: int) -> int:
    i = addr - WRAM_BASE
    return int(wram[i]) if 0 <= i < wram.size else 0


# --- verified Yellow WRAM addresses (pixels-validated on assets/demo_pikachu) ----
# Pokémon YELLOW's WRAM save block is shifted -1 vs Red/Blue; these are the
# corrected Yellow taps (the repo previously used the R/B addresses, one byte
# high, which read neighbouring bytes — e.g. $D35E is the tile-view pointer low
# byte, the source of the phantom "glitch map 245/253" / "-20 offset" readings).
# Pinned by replaying the newgame->Pikachu demo and reading RAM against the screen.
_ADDR = {
    "map_id": 0xD35D,        # newgame bedroom=38; 38→37→0→40 to Oak's lab
    "pos_y": 0xD360,
    "pos_x": 0xD361,
    "party_count": 0xD162,   # 0 at newgame, 1 after the starter; 0xFF/uninit → None
    "badges": 0xD355,        # bitfield (popcount)
    "money": 0xD346,         # 3-byte BCD (D346..D348); reads 003000 at newgame
    "in_battle": 0xD056,
}

# Map id → human name (decimal id = byte $D35D). From pret/pokeyellow
# constants/map_constants.asm — VALIDATED: fed the correct byte, the newgame
# bedroom reads 38 ("Player's bedroom (2F)") and the demo walks 38→37→0→40, each
# naming correctly. (The old "-20 offset, names may be misaligned" caveat was an
# artifact of reading the wrong byte $D35E, which is a pointer low byte: 38-20=18.)
# Unknown ids render as "map $NN".
MAP_NAMES: dict[int, str] = {
    0: "Pallet Town", 1: "Viridian City", 2: "Pewter City",
    3: "Cerulean City", 4: "Lavender Town", 5: "Vermilion City",
    6: "Celadon City", 7: "Fuchsia City", 8: "Cinnabar Island",
    9: "Indigo Plateau", 10: "Saffron City",
    12: "Route 1", 13: "Route 2", 18: "Route 7",
    37: "Player's house (1F)", 38: "Player's bedroom (2F)",
    39: "Rival's house", 40: "Oak's Lab",
    41: "Viridian PokéCenter", 42: "Viridian Mart", 45: "Viridian Gym",
    51: "Viridian Forest", 54: "Pewter Gym", 65: "Cerulean Gym",
    92: "Vermilion Gym", 134: "Celadon Gym", 157: "Fuchsia Gym",
    178: "Saffron Gym", 166: "Cinnabar Gym",
    108: "Victory Road 1F", 194: "Victory Road 2F", 198: "Victory Road 3F",
    174: "Indigo Plateau Lobby",
    245: "Elite Four — Lorelei", 246: "Elite Four — Bruno",
    247: "Elite Four — Agatha", 113: "Elite Four — Lance",
    120: "Champion's Room",
}

# Badge bit (in $D356) → badge name (constants/ram_constants.asm order).
BADGE_NAMES = (
    "Boulder", "Cascade", "Thunder", "Rainbow",
    "Soul", "Marsh", "Volcano", "Earth",
)

# Ordered milestone ladder (map-id or badge-bit keyed). Filled from research;
# each entry: (label, predicate over GameState). Kept data-driven so the
# report can show a "furthest reached" ladder.
# (populated in milestones.py once the verified map table is in.)


@dataclass
class GameState:
    map_id: int
    map_name: str
    pos: tuple[int, int]
    party_count: int | None
    badges: int | None
    badge_count: int | None
    money: int | None
    in_battle: bool

    def one_line(self) -> str:
        parts = [self.map_name, f"({self.pos[0]},{self.pos[1]})"]
        if self.party_count is not None:
            parts.append(f"party {self.party_count}")
        if self.badge_count is not None:
            parts.append(f"{self.badge_count} badges")
        if self.money is not None:
            parts.append(f"¥{self.money}")
        if self.in_battle:
            parts.append("IN BATTLE")
        return " · ".join(parts)


def _bcd3(wram: np.ndarray, addr: int) -> int | None:
    """Decode a 3-byte packed-BCD value (Yellow money format)."""
    b0, b1, b2 = _b(wram, addr), _b(wram, addr + 1), _b(wram, addr + 2)
    digits = f"{b0:02x}{b1:02x}{b2:02x}"
    if any(c not in "0123456789" for c in digits):  # not valid BCD → uninit
        return None
    return int(digits)


def game_is_yellow(manifest) -> bool:
    """True iff ``manifest`` names Pokémon Yellow (case-insensitive substring).

    The gate that quarantines the Yellow tables: only a Yellow manifest may
    apply Yellow map/badge labels. ``None`` returns False — callers treat a
    missing manifest as the legacy Yellow default separately.
    """
    if manifest is None:
        return False
    game = str(getattr(manifest, "game", "") or "")
    return "yellow" in game.lower()


def _parse_addr(value) -> int | None:
    """Coerce ``0xD35E`` / ``"0xD35E"`` / ``54622`` into an int, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s, 16) if s.lower().startswith("0x") else int(s)
        except ValueError:
            return None
    return None


def _manifest_spatial_addrs(manifest) -> dict[str, int]:
    """Extract map_id/x/y taps from ``manifest.spatial.source.ram`` (best-effort)."""
    out: dict[str, int] = {}
    spatial = getattr(manifest, "spatial", None) or {}
    if not isinstance(spatial, dict):
        return out
    src = spatial.get("source", {})
    ram = src.get("ram", {}) if isinstance(src, dict) else {}
    if isinstance(ram, dict):
        for key in ("map_id", "x", "y"):
            addr = _parse_addr(ram.get(key))
            if addr is not None:
                out[key] = addr
    return out


def _decode_yellow(wram: np.ndarray) -> GameState:
    """Decode using the hardcoded Yellow tables (the game-specific fallback)."""
    mid = _b(wram, _ADDR["map_id"])
    party = _b(wram, _ADDR["party_count"])
    party_out = None if party in (0xFF,) else party if party <= 6 else None
    badges = _b(wram, _ADDR["badges"])
    badge_ct = bin(badges).count("1") if badges is not None else None
    return GameState(
        map_id=mid,
        map_name=MAP_NAMES.get(mid, f"map ${mid:02X}"),
        pos=(_b(wram, _ADDR["pos_x"]), _b(wram, _ADDR["pos_y"])),
        party_count=party_out,
        badges=badges,
        badge_count=badge_ct,
        money=_bcd3(wram, _ADDR["money"]),
        in_battle=bool(_b(wram, _ADDR["in_battle"])),
    )


def _decode_generic(wram: np.ndarray, manifest) -> GameState:
    """Decode a non-Yellow game from the manifest's spatial taps only.

    No Yellow labels: map names are numeric and the semantic fields (party,
    badges, money) surface as ``None`` since their meaning is game-specific and
    unknown here. This is the leak-proof path for a second game.
    """
    addrs = _manifest_spatial_addrs(manifest)
    mid = _b(wram, addrs["map_id"]) if "map_id" in addrs else 0
    x = _b(wram, addrs["x"]) if "x" in addrs else 0
    y = _b(wram, addrs["y"]) if "y" in addrs else 0
    return GameState(
        map_id=mid,
        map_name=f"map ${mid:02X}",
        pos=(x, y),
        party_count=None,
        badges=None,
        badge_count=None,
        money=None,
        in_battle=False,
    )


def decode(wram: np.ndarray, manifest=None) -> GameState:
    """Decode one WRAM snapshot into a :class:`GameState`.

    With no manifest (the legacy default) or a Yellow manifest, the hardcoded
    Yellow tables are used. With a non-Yellow manifest, decoding falls back to
    the manifest's spatial RAM taps and emits no Yellow-specific labels.
    """
    if manifest is None or game_is_yellow(manifest):
        return _decode_yellow(wram)
    return _decode_generic(wram, manifest)


__all__ = ["GameState", "decode", "game_is_yellow", "MAP_NAMES", "WRAM_BASE"]
