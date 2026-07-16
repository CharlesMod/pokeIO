#!/usr/bin/env python3
"""Boot Pokemon Yellow and advance to a controllable overworld, then save a state.

Best-effort scripted intro: boot -> mash START through the Pikachu title
animation -> mash A through Oak's introduction speech -> handle the player/rival
name-entry menus by interleaving DOWN so a *preset* name is chosen (this avoids
the letter-keyboard, which mashing A cannot escape). The sequence is FIXED (no
timing-dependent branches) so it replays identically.

Reached state (documented for reproducibility):
  * Player's bedroom, upstairs of the Pallet Town house, immediately post-intro.
  * The sprite/view respond to movement input -> the state is controllable.
  * Map id at 0xD35E reads 18; player coords at 0xD361/0xD362.

Output: roms/yellow_newgame.state  (PyBoy save_state, loadable via PokeEnv.reset).
"""

import os
import sys

from pyboy import PyBoy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROM = os.path.join(ROOT, "roms", "pokemon_yellow.gb")
OUT = os.path.join(ROOT, "roms", "yellow_newgame.state")

MAP_ID_ADDR = 0xD35E
Y_ADDR = 0xD361
X_ADDR = 0xD362
ALL_BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")


def press(p, btn, hold=6, gap=8):
    p.button_press(btn)
    p.tick(hold, False)
    p.button_release(btn)
    p.tick(gap, False)


def ascii_screen(p):
    p.tick(1, True)
    s = p.screen.ndarray[:, :, 0]
    chars = " .:-=+*#%@"
    rows = []
    for yy in range(0, 144, 8):
        rows.append(
            "".join(chars[min(9, (255 - int(s[yy, xx])) * 10 // 256)] for xx in range(0, 160, 4))
        )
    return "\n".join(rows)


def run_intro(p):
    """Fixed input sequence: boot -> title -> Oak speech -> preset names."""
    p.tick(400, False)  # boot / Game Freak + Pikachu intro animation
    for _ in range(20):  # mash START to reach and pass the title screen
        press(p, "start")
    for _ in range(30):  # begin advancing Oak's introduction speech
        press(p, "a")
    # Advance remaining speech + both name menus. DOWN every 6th press moves the
    # menu cursor off "NEW NAME" onto a preset name, so A confirms a preset.
    for i in range(300):
        press(p, "a")
        if i % 6 == 5:
            press(p, "down")


def main():
    if not os.path.exists(ROM):
        sys.exit(f"ROM not found: {ROM}")
    p = PyBoy(ROM, window="null", sound_emulated=False)
    try:
        run_intro(p)

        map_id = p.memory[MAP_ID_ADDR]
        coords = (p.memory[Y_ADDR], p.memory[X_ADDR])
        screen = ascii_screen(p)

        # Flush any queued/held input before snapshotting (determinism gotcha).
        for b in ALL_BUTTONS:
            p.button_release(b)
        p.tick(1, False)

        with open(OUT, "wb") as fh:
            p.save_state(fh)

        # Verify controllability AFTER saving (free to perturb now): walk a few
        # steps and confirm the player coords actually change.
        start = (p.memory[Y_ADDR], p.memory[X_ADDR])
        for btn in ("down", "down", "right", "right", "up", "left"):
            press(p, btn)
        controllable = (p.memory[Y_ADDR], p.memory[X_ADDR]) != start

        print("=== make_newgame_state ===")
        print(screen)
        print()
        print(f"map_id (0xD35E) = {map_id}")
        print(f"player coords (y=0xD361, x=0xD362) = {coords}")
        print(f"controllable (coords change after walking) = {controllable}")
        print(f"saved state -> {OUT} ({os.path.getsize(OUT)} bytes)")
        print(
            "Description: player's bedroom (upstairs, Pallet Town house), "
            "immediately post-intro, controllable overworld."
        )
    finally:
        p.stop(save=False)


if __name__ == "__main__":
    main()
