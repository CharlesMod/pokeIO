"""Bench / correctness check for the sticky-input model in PokeEnv.

Run (venv must be active):
    cd /home/cmod/pokeIO && . .venv/bin/activate
    PYTHONPATH=/home/cmod/pokeIO python -m scripts.bench_sticky_input

Checks:
  (a) sticky frame_skip=1 registers input   -> player position CHANGES
  (b) legacy (non-sticky) frame_skip=1, hold=0 (the reported bug) -> NO change
  (c) legacy frame_skip=24 path still moves the player

Each check prints a clear PASS/FAIL line with the before/after positions.
Uses a scratch run; does not touch the live training run or any shared state.
"""

from __future__ import annotations

from pokeio.emu.env import PokeEnv

ROM = "roms/pokemon_yellow.gb"
STATE = "roms/yellow_newgame.state"

# Discrete(8) index for "right" (moves the player in the overworld).
RIGHT = 3

# WRAM position bytes for Pokemon Yellow (verified via raw_wram()).
MAP_ID = 0xD35E
PLAYER_X = 0xD361
PLAYER_Y = 0xD362
WRAM_START = 0xC000  # raw_wram() is the 0xC000-0xDFFF block


def pos(env: PokeEnv) -> tuple[int, int, int]:
    """(map_id, x, y) read straight out of the raw WRAM block."""
    ram = env.raw_wram()
    return (
        int(ram[MAP_ID - WRAM_START]),
        int(ram[PLAYER_X - WRAM_START]),
        int(ram[PLAYER_Y - WRAM_START]),
    )


def hold_right(env: PokeEnv, steps: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    env.reset(STATE)
    start = pos(env)
    for _ in range(steps):
        env.step(RIGHT)
    return start, pos(env)


def check(label: str, before, after, want_change: bool) -> bool:
    changed = before != after
    ok = changed == want_change
    verdict = "PASS" if ok else "FAIL"
    what = "CHANGED" if changed else "no change"
    expect = "expect change" if want_change else "expect NO change"
    print(f"[{verdict}] {label}: {before} -> {after} ({what}; {expect})")
    return ok


def main() -> int:
    results = []

    # (a) sticky fs=1 — button held across frame boundaries -> input registers.
    with PokeEnv(ROM, frame_skip=1, hold_frames=0, sticky_input=True) as env:
        before, after = hold_right(env, 40)
    results.append(check("sticky fs=1 (hold RIGHT x40)", before, after, want_change=True))

    # (b) legacy fs=1, hold=0 — press+release inside one tick -> the reported bug.
    with PokeEnv(ROM, frame_skip=1, hold_frames=0, sticky_input=False) as env:
        before, after = hold_right(env, 40)
    results.append(check("legacy fs=1 hold=0 (RIGHT x40)", before, after, want_change=False))

    # (c) legacy fs=24 path still works (the current training fallback).
    with PokeEnv(ROM, frame_skip=24, hold_frames=8, sticky_input=False) as env:
        before, after = hold_right(env, 8)
    results.append(check("legacy fs=24 hold=8 (RIGHT x8)", before, after, want_change=True))

    # Bonus: sticky fs=24 also works (sticky is the new default across any fs).
    with PokeEnv(ROM, frame_skip=24, hold_frames=8, sticky_input=True) as env:
        before, after = hold_right(env, 8)
    results.append(check("sticky fs=24 (RIGHT x8)", before, after, want_change=True))

    ok = all(results)
    print()
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
