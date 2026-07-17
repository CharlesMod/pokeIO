"""Pokémon Yellow game-state decoder (TRAINER-FACING, not for the ML system).

Decodes a raw 8 KB WRAM snapshot (``PokeEnv.raw_wram()``, GB $C000-$DFFF →
index 0) into a human-readable :class:`GameState`. This is deliberately
game-specific — it exists so WE can see how far into Pokémon the swarm has
gotten. It never feeds the reward stack (which stays game-agnostic).

Addresses are Pokémon YELLOW-specific (pret/pokeyellow); Yellow shifts a few
vs Red/Blue. Every field is validated against real game states before being
trusted — unverified/uninitialised reads surface as ``None`` rather than
garbage. See MAP_NAMES / _ADDR below for the source-of-truth table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRAM_BASE = 0xC000


def _b(wram: np.ndarray, addr: int) -> int:
    i = addr - WRAM_BASE
    return int(wram[i]) if 0 <= i < wram.size else 0


# --- verified Yellow WRAM addresses (filled/confirmed from pokeyellow) -------
# map id byte is confirmed working (env.py taps it and map transitions track).
_ADDR = {
    "map_id": 0xD35E,
    "pos_y": 0xD361,
    "pos_x": 0xD362,
    "party_count": 0xD163,   # 0xFF/uninit → None
    "badges": 0xD356,        # bitfield; verify Yellow bit→badge order
    "money": 0xD347,         # 3-byte BCD (D347..D349)
    "in_battle": 0xD057,
}

# Map id → human name. Populated from the pokeyellow map-constant table.
# Numeric ids are what byte $D35E holds (decimal). Unknown ids render as
# "map $NN" so the report is always usable even before the table is complete.
MAP_NAMES: dict[int, str] = {
    18: "Player's bedroom (2F)",  # verified: newgame state reads map 18 here
}

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


def decode(wram: np.ndarray) -> GameState:
    """Decode one WRAM snapshot into a :class:`GameState`."""
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


__all__ = ["GameState", "decode", "MAP_NAMES", "WRAM_BASE"]
