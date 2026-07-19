"""Replay the first-milestone demonstration (get Pikachu from newgame boot).

demo_actions.npy: 657 actions in the training action space (PokeEnv.step,
frame_skip=24, sticky input) that deterministically take roms/yellow_newgame.state
to party=1 (Pikachu, L5) in Oak's lab. pikachu_purestep.state is the resulting
progressed emulator state (backward-robustification seed / eval reference).

Produced 2026-07-18 by the corrected-tap feedback navigator (Fable diagnostic
session). CORRECT Pokemon Yellow WRAM taps (Yellow save block = Red/Blue - 1):
  map=$D35D  y=$D360 x=$D361  party_count=$D162  species0=$D163  mon1_level=$D18B
  badges=$D355  money(BCD3)=$D346  events=$D746..$D87E  in_battle=$D056
(The repo's historical taps -- $D35E/$D163/$D18C/$D356/$D747 -- are the R/B
addresses and read neighboring bytes on Yellow: $D35E is the tile-view pointer
low byte, source of the phantom "glitch map 245/253" readings.)

Usage:  PYTHONPATH=. .venv/bin/python assets/demo_pikachu/replay_demo.py
Expected output: party=1 species=84 level=5 map=40 events=5
"""
from __future__ import annotations

import os

import numpy as np

from pokeio.emu.env import PokeEnv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    acts = np.load(os.path.join(HERE, "demo_actions.npy"))
    env = PokeEnv(rom_path=os.path.join(ROOT, "roms", "pokemon_yellow.gb"), frame_skip=24)
    try:
        env.reset(os.path.join(ROOT, "roms", "yellow_newgame.state"))
        for a in acts:
            env.step(int(a))
        w = env.raw_wram()
        b = lambda addr: int(w[addr - 0xC000])
        ev = int(np.unpackbits(w[0xD746 - 0xC000:0xD87E - 0xC000 + 1]).sum())
        print(f"party={b(0xD162)} species={b(0xD163)} level={b(0xD18B)} "
              f"map={b(0xD35D)} events={ev}")
        assert b(0xD162) == 1 and b(0xD163) == 84, "demo replay diverged!"
        print("OK: deterministic replay reached the first real milestone (Pikachu).")
    finally:
        env.close()


if __name__ == "__main__":
    main()
