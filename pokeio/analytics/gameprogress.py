"""Decode furthest game progress from a run's Go-Explore checkpoint.

The Go-Explore archive stores emulator save_states for its frontier cells; the
DEEPEST cells are the furthest any restore-chain reached — a far better "how far
into Pokémon are we" signal than the champion cage. This loads those states,
decodes each with :mod:`pokeio.analytics.yellow`, and summarizes where the
frontier actually is (furthest location, map histogram, deepest game-state).

Trainer-facing only. Loading states spins a throwaway PyBoy, so this is an
on-demand report step, never on the training path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pokeio.analytics.milestones import furthest_milestone, ladder_size
from pokeio.analytics.yellow import decode
from pokeio.train.checkpoint import load_checkpoint


def checkpoint_progress_summary(run_dir: str | Path, top: int = 60) -> str | None:
    """Decode the deepest ``top`` frontier states; return a report block or None.

    ``None`` when there's no checkpoint or no Go-Explore archive to read.
    """
    ckpt = load_checkpoint(run_dir)
    if not ckpt:
        return None
    go = ckpt.get("goexplore")
    if go is None or not getattr(go, "cells", None):
        return None

    # deepest cells first — the furthest the frontier reached. Milestones are
    # early-game, so ALSO stride a spread across shallower cells (else a sample
    # of only the deepest states misses that the swarm reached Pallet/Viridian).
    cells = sorted(go.cells.values(), key=lambda e: -e.depth)
    deep = cells[: max(1, top)]
    rest = cells[max(1, top):]
    stride = rest[:: max(1, len(rest) // max(1, top))] if rest else []
    sample = deep + stride

    # decode each state's WRAM (lazy PyBoy import: keep analytics import-light)
    from pokeio.emu.env import PokeEnv

    # ROM path is not stored in the checkpoint; use the project default.
    env = PokeEnv(rom_path="roms/pokemon_yellow.gb", frame_skip=24)
    states = []
    try:
        for e in sample:
            try:
                env.load_state(e.state)
                env.pyboy.tick(1, True)
                states.append((e.depth, decode(env.raw_wram())))
            except Exception:
                continue
    finally:
        env.close()

    if not states:
        return None

    # The corrected map tap ($D35D — Yellow save block = R/B - 1) reads the REAL
    # current map, so there is no pointer-byte noise to filter (the old $F0+
    # "sentinel" drop was an artifact of the wrong byte, and would wrongly discard
    # real high-id endgame maps — the Elite Four rooms are 245/246/247).
    deepest_depth, deepest = states[0]
    map_hist = Counter(gs.map_name for _, gs in states)
    # furthest-progress signals across the sampled frontier
    max_party = max((gs.party_count or 0 for _, gs in states), default=0)
    max_badges = max((gs.badge_count or 0 for _, gs in states), default=0)
    max_money = max((gs.money or 0 for _, gs in states), default=0)
    any_battle = sum(1 for _, gs in states if gs.in_battle)

    all_states = [gs for _, gs in states]
    ms = furthest_milestone(all_states)
    # "real game start" tell: party>0 only after completing the Oak sequence to
    # get the starter. If no sampled state has a party, the frontier has explored
    # (real maps: bedroom/house/Pallet/…) but never obtained a Pokémon — flag it
    # so raw archive depth isn't misread as game progress.
    legit = any((gs.party_count or 0) > 0 for gs in all_states)

    lines = ["GAME PROGRESS  (decoded from the frontier states)"]
    if ms is not None:
        lines.append(f"  furthest milestone  [{ms.index + 1}/{ladder_size()}] "
                     f"{ms.label}")
    elif not legit:
        lines.append("  furthest milestone  none yet — no sampled state has a "
                     "Pokémon (never got the starter)")
        lines.append("                      (frontier reached real maps but the "
                     "Oak sequence was never completed; party empty)")
    lines.append(f"  deepest chain     depth {deepest_depth} → {deepest.one_line()}")
    top_maps = ", ".join(
        f"{name}×{cnt}" for name, cnt in map_hist.most_common(6)
    )
    lines.append(f"  frontier spread   {top_maps}")
    reach = []
    if max_party:
        reach.append(f"party≤{max_party}")
    if max_badges:
        reach.append(f"badges≤{max_badges}")
    if max_money:
        reach.append(f"money≤¥{max_money}")
    if any_battle:
        reach.append(f"{any_battle}/{len(states)} frontier states mid-battle")
    if reach:
        lines.append(f"  furthest reached  {' · '.join(reach)}")
    lines.append(f"  (sampled {len(states)} of {len(go.cells)} archived states)")
    return "\n".join(lines)


__all__ = ["checkpoint_progress_summary"]
