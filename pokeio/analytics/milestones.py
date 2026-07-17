"""Pokémon Yellow milestone ladder — how far into the game the swarm has gotten.

An ordered list of human checkpoints keyed to map ids / badge bits (from the
verified pokeyellow layout). Given the decoded game-states of the frontier,
:func:`furthest_milestone` reports the deepest checkpoint reached. Trainer-facing
only — never touches the reward path.

QUARANTINE: the ladder below is entirely Yellow-specific (map ids + badge
bits). The public functions take an optional ``manifest``; the ladder is only
applied when there is no manifest (the legacy default) or the manifest names
Yellow (:func:`game_is_yellow`). For any other game they return no milestones,
so Yellow labels can never leak onto a second game.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokeio.analytics.yellow import GameState, game_is_yellow

# (order index, label, predicate) — predicate is over a decoded GameState.
# Ordered by natural playthrough progression; "reached" = any frontier state
# satisfies it. Map-id checks are exact; badge checks test the $D356 bit.
_LADDER = [
    ("In the bedroom", lambda s: s.map_id == 38),
    ("Left the bedroom", lambda s: s.map_id == 37),
    ("Reached Pallet Town", lambda s: s.map_id == 0),
    ("Entered Oak's Lab", lambda s: s.map_id == 40),
    ("Got a starter Pokémon", lambda s: (s.party_count or 0) >= 1),
    ("Reached Route 1", lambda s: s.map_id == 12),
    ("Reached Viridian City", lambda s: s.map_id == 1),
    ("Reached Route 2", lambda s: s.map_id == 13),
    ("Entered Viridian Forest", lambda s: s.map_id == 51),
    ("Reached Pewter City", lambda s: s.map_id == 2),
    ("Boulder Badge", lambda s: bool((s.badges or 0) & (1 << 0))),
    ("Cascade Badge", lambda s: bool((s.badges or 0) & (1 << 1))),
    ("Thunder Badge", lambda s: bool((s.badges or 0) & (1 << 2))),
    ("Rainbow Badge", lambda s: bool((s.badges or 0) & (1 << 3))),
    ("Soul Badge", lambda s: bool((s.badges or 0) & (1 << 4))),
    ("Marsh Badge", lambda s: bool((s.badges or 0) & (1 << 5))),
    ("Volcano Badge", lambda s: bool((s.badges or 0) & (1 << 6))),
    ("Earth Badge", lambda s: bool((s.badges or 0) & (1 << 7))),
    ("Victory Road", lambda s: s.map_id in (108, 194, 198)),
    ("Indigo Plateau", lambda s: s.map_id == 174),
    ("Elite Four", lambda s: s.map_id in (245, 246, 247, 113)),
    ("Champion fight", lambda s: s.map_id == 120),
]


@dataclass
class Milestone:
    index: int
    label: str


def milestones_reached(states: list[GameState], manifest=None) -> list[Milestone]:
    """All ladder checkpoints satisfied by at least one of ``states``.

    Returns ``[]`` when a non-Yellow manifest is supplied (the Yellow ladder
    must not label another game); no manifest or a Yellow manifest applies it.
    """
    if not (manifest is None or game_is_yellow(manifest)):
        return []
    hit = []
    for i, (label, pred) in enumerate(_LADDER):
        if any(_safe(pred, s) for s in states):
            hit.append(Milestone(i, label))
    return hit


def furthest_milestone(states: list[GameState], manifest=None) -> Milestone | None:
    """The deepest ladder checkpoint any state reached (highest index hit)."""
    hit = milestones_reached(states, manifest)
    return hit[-1] if hit else None


def ladder_size() -> int:
    return len(_LADDER)


def _safe(pred, s) -> bool:
    try:
        return bool(pred(s))
    except Exception:
        return False


__all__ = ["Milestone", "milestones_reached", "furthest_milestone", "ladder_size"]
