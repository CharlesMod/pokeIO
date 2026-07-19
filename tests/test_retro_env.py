"""RetroPokeEnv contract-conformance test (pokeio.emu.retro_env, task #36).

The libretro-backed multi-console drop-in for PokeEnv. Gated on the gambatte core
being fetched (scripts/fetch_cores.py) + the ROM present; skips cleanly otherwise.

libretro cores are SINGLE-INSTANCE per process (global core state) — the fleet runs
one emulator per worker, matching this. So ALL checks run on ONE env instance;
save/load fidelity + determinism are verified via save -> replay-twice-from-the-blob
rather than a second instance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokeio.emu.retro_env import ConsoleSpec, RetroPokeEnv, core_path

ROM = Path("roms/pokemon_yellow.gb")
CORE = Path(core_path("gambatte_libretro.so"))
_HAVE = ROM.exists() and CORE.exists()


@pytest.mark.skipif(not _HAVE, reason="gambatte core (scripts/fetch_cores.py) or ROM absent")
def test_retro_pokeenv_contract_conformance():
    env = RetroPokeEnv(str(ROM), ConsoleSpec.game_boy(), frame_skip=24)
    try:
        obs = env.reset()
        w = env.raw_wram()

        # --- shapes / dtypes (the encoder + reward stack rely on these) ---
        assert obs.shape == (144, 160) and obs.dtype == np.uint8
        assert w.shape == (8192,) and w.dtype == np.uint8          # DMG SYSTEM_RAM
        assert np.array_equal(env.wram_strided(64), w[::64])       # compact-path identity

        # --- the .pyboy shim reach-throughs consumers depend on ---
        assert 0 <= env.pyboy.memory[0xD35D] <= 255                # single-byte tap
        assert len(env.pyboy.memory[0xC000:0xE000]) == 8192        # slice
        assert env.pyboy.screen.ndarray.shape == (144, 160, 4)
        assert np.array_equal(obs, env.pyboy.screen.ndarray[:, :, 0])

        # --- sticky input actuates + edge-read re-tap timing ---
        o0 = env._obs().copy()
        for _ in range(20):
            env.step(6)                                            # START x20
        assert not np.array_equal(o0, env._obs())                 # screen advanced
        assert env.hold(4) == 0 and env.hold(4) == 2              # face-button repeat -> _TAP_GAP

        # --- save/load fidelity + intra-core determinism ---
        for _ in range(30):
            env.step(6)
        blob = env.save_state()
        wram_at_save = env.raw_wram().copy()
        assert isinstance(blob, bytes) and 1000 < len(blob) < 500000
        seq = [4, 6, 5, 1, 7, 0] * 8

        def replay_from_blob():
            env.load_state(blob)
            assert np.array_equal(env.raw_wram(), wram_at_save)    # load restores exactly
            for a in seq:
                env.step(a)
            return env.raw_wram().copy()

        r1 = replay_from_blob()
        r2 = replay_from_blob()
        assert np.array_equal(r1, r2)                              # deterministic replay
        assert not np.array_equal(r1, wram_at_save)               # replay progressed state
    finally:
        env.close()
