#!/usr/bin/env python3
"""Determinism smoke test for PokeEnv.

Load roms/yellow_newgame.state, run a FIXED seeded 1000-action script, and
capture (final-screen hash, final-WRAM digest). Repeat from the same state in a
fresh env. The two runs must be byte-identical -> PASS, else FAIL.
"""

import hashlib
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pokeio.emu.env import PokeEnv  # noqa: E402

ROM = os.path.join(ROOT, "roms", "pokemon_yellow.gb")
STATE = os.path.join(ROOT, "roms", "yellow_newgame.state")

SEED = 1337
N_ACTIONS = 1000


def action_script(seed=SEED, n=N_ACTIONS):
    """Fixed pseudo-random action sequence over the Discrete(8) space."""
    rng = random.Random(seed)
    return [rng.randrange(8) for _ in range(n)]


def run(script):
    env = PokeEnv(ROM)
    try:
        env.reset(STATE)
        last_obs = None
        last_ram = None
        for a in script:
            last_obs, last_ram, _, _ = env.step(a)
        screen_hash = hashlib.sha256(np.ascontiguousarray(last_obs).tobytes()).hexdigest()
        wram_hash = hashlib.sha256(np.ascontiguousarray(last_ram).tobytes()).hexdigest()
        return screen_hash, wram_hash
    finally:
        env.close()


def main():
    if not os.path.exists(STATE):
        sys.exit(f"missing state {STATE} — run scripts/make_newgame_state.py first")

    script = action_script()
    print(f"determinism smoke test: seed={SEED}, actions={N_ACTIONS}")

    s1, w1 = run(script)
    print(f"run 1: screen={s1[:16]}...  wram={w1[:16]}...")
    s2, w2 = run(script)
    print(f"run 2: screen={s2[:16]}...  wram={w2[:16]}...")

    ok = (s1 == s2) and (w1 == w2)
    print(f"screen match: {s1 == s2}")
    print(f"wram   match: {w1 == w2}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
