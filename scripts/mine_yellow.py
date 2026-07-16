#!/usr/bin/env python3
"""Verify the WRAM progress-counter miner on Pokemon Yellow.

Collects a handful of short *random-policy* rollouts from the canonical newgame
state, records raw WRAM at every step, runs :func:`pokeio.reward.miner.mine`,
and prints the top candidate progress addresses with their per-feature stats and
inferred direction.

The miner is game-agnostic — it is handed nothing but raw bytes.  This script's
"sanity" section lists a few *known* Yellow addresses purely for human commentary
(does the miner's output overlap them?).  Those addresses are NOT fed to the
miner and NOT asserted; the miner must find them (or not) on its own.

Run:
    cd /home/cmod/pokeIO && . .venv/bin/activate
    PYTHONPATH=/home/cmod/pokeIO python scripts/mine_yellow.py
"""

from __future__ import annotations

import os
import time

import numpy as np

from pokeio.config import Config
from pokeio.emu.env import PokeEnv
from pokeio.reward.miner import MinerConfig, mine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_ROLLOUTS = 5
STEPS_PER_ROLLOUT = 300

# Known Yellow addresses — for HUMAN commentary only. Never fed to the miner.
KNOWN = {
    0xD347: "money (3-byte BCD, low byte)",
    0xD348: "money (BCD, mid byte)",
    0xD349: "money (BCD, high byte)",
    0xD163: "party count",
    0xD356: "badges bitfield",
    0xD35E: "current map id",
    0xD361: "player Y coord",
    0xD362: "player X coord",
}


def collect_rollouts(cfg: Config, n_rollouts: int, steps: int):
    """Run random-policy rollouts; return a list of (steps+1, 8192) uint8 arrays."""
    rom = os.path.join(ROOT, cfg.emu.rom_path)
    state = os.path.join(ROOT, cfg.emu.reset_state)
    n_actions = cfg.emu.action_space

    rollouts = []
    env = PokeEnv(rom)  # default frame_skip=24 -> button actually registers
    try:
        for r in range(n_rollouts):
            rng = np.random.default_rng(1000 + r)  # distinct, reproducible policy per rollout
            env.reset(state)
            frames = [env.raw_wram().copy()]
            for _ in range(steps):
                a = int(rng.integers(0, n_actions))
                _, ram, _, _ = env.step(a)
                frames.append(np.asarray(ram, dtype=np.uint8).copy())
            rollouts.append(np.stack(frames, axis=0))
    finally:
        env.close()
    return rollouts


def main() -> None:
    cfg = Config()
    print("=== mine_yellow: collecting random-policy rollouts ===")
    print(f"rollouts={N_ROLLOUTS}  steps_each={STEPS_PER_ROLLOUT}  "
          f"frame_skip={PokeEnv.__init__.__defaults__[0]}  actions={cfg.emu.action_space}")

    t0 = time.time()
    rollouts = collect_rollouts(cfg, N_ROLLOUTS, STEPS_PER_ROLLOUT)
    t_collect = time.time() - t0
    shape = rollouts[0].shape
    print(f"collected {len(rollouts)} rollouts, each {shape} "
          f"({shape[1]} WRAM bytes) in {t_collect:.1f}s")

    # entropy mask comes straight from the project RewardConfig (0.95)
    mcfg = MinerConfig.from_reward_config(cfg.reward)
    print(f"MinerConfig: entropy_mask={mcfg.entropy_mask}  "
          f"activity_band[{mcfg.min_activity}, {mcfg.max_activity}] "
          f"target={mcfg.activity_target}  consider_pairs={mcfg.consider_pairs}")

    t0 = time.time()
    cands = mine(rollouts, mcfg)
    t_mine = time.time() - t0
    print(f"mined {len(cands)} surviving candidates in {t_mine:.2f}s\n")

    # ---- ranked table -----------------------------------------------------
    print("=== top candidate progress addresses ===")
    header = (f"{'rank':>4} {'addr':>7} {'w':>2} {'score':>6} {'dir':>4} "
              f"{'mono':>5} {'act':>6} {'dent':>5} {'cons':>5} {'dagr':>5} "
              f"{'rsan':>5} {'nchg':>6} {'range':>13}  known?")
    print(header)
    print("-" * len(header))
    for i, c in enumerate(cands[:20]):
        s = c.stats
        known = ""
        for a in ([c.address] if c.width == 1 else [c.address, c.address + 1]):
            if a in KNOWN:
                known = f"<- {KNOWN[a]}"
                break
        rng = f"{s['value_min']}..{s['value_max']}"
        d = "inc" if c.direction == "increasing" else "dec"
        print(f"{i+1:>4} {c.addr_hex:>7} {c.width:>2} {c.score:>6.3f} {d:>4} "
              f"{s['monotonicity']:>5.2f} {s['activity']:>6.3f} {s['delta_entropy']:>5.2f} "
              f"{s['consistency']:>5.2f} {s['direction_agreement']:>5.2f} "
              f"{s['range_sanity']:>5.2f} {s['n_changes']:>6} {rng:>13}  {known}")

    # ---- sanity commentary (NOT baked into the miner) ---------------------
    # For each known address, report where it landed (if at all) AND classify its
    # raw behavior in the collected rollouts, so a non-appearance is explained by
    # data rather than a canned excuse.
    print("\n=== sanity commentary (known Yellow addresses; not fed to miner) ===")
    found = {}
    for rank, c in enumerate(cands):
        addrs = [c.address] if c.width == 1 else [c.address, c.address + 1]
        for a in addrs:
            if a in KNOWN and a not in found:
                found[a] = (rank, c)

    for a in sorted(KNOWN):
        i = a - 0xC000  # column index into a snapshot
        # classify raw behavior across rollouts
        n_changes = 0
        mono_frac = []
        vmin, vmax = 255, 0
        for m in rollouts:
            col = m[:, i].astype(np.int64)
            d = np.diff(col)
            nz = d[d != 0]
            n_changes += int(nz.size)
            if nz.size:
                mono_frac.append(max((nz > 0).sum(), (nz < 0).sum()) / nz.size)
            vmin, vmax = min(vmin, int(col.min())), max(vmax, int(col.max()))
        avg_mono = float(np.mean(mono_frac)) if mono_frac else 0.0
        if n_changes == 0:
            behavior = f"STATIC (never changes; const={vmin}) -> nothing earns it under random play"
        elif avg_mono < 0.75:
            behavior = (f"CHANGES but NON-MONOTONE (mono~{avg_mono:.2f}, range {vmin}..{vmax}) "
                        f"-> bounded random-walk, not a progress counter under random play")
        else:
            behavior = f"changes monotonically (mono~{avg_mono:.2f}, range {vmin}..{vmax})"

        if a in found:
            rank, c = found[a]
            print(f"  0x{a:04X} {KNOWN[a]:<28} surfaced @ rank {rank+1} "
                  f"({c.addr_hex} w{c.width}, {c.score:.3f}, {c.direction})")
        else:
            print(f"  0x{a:04X} {KNOWN[a]:<28} not surfaced -- {behavior}")


if __name__ == "__main__":
    main()
