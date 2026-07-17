"""Does sleeping between steps inflate step_fast time? (single process, pinned)

Modes per phase: continuous stepping / time.sleep(gap) between steps /
busy-wait(gap) between steps. Same env, same core, interleaved phases.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
import psutil

ROOT = "/home/cmod/pokeIO"

psutil.Process().cpu_affinity([5])  # a plain physical core, node0

from pokeio.emu.env import PokeEnv  # noqa: E402

env = PokeEnv(f"{ROOT}/roms/pokemon_yellow.gb", frame_skip=24, hold_frames=8)
env.reset(f"{ROOT}/roms/yellow_newgame.state")
rng = np.random.default_rng(0)

def busy(sec):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        pass

def run(mode, n=400, gap=2.5e-3):
    ts = []
    for _ in range(n):
        if mode == "sleep":
            time.sleep(gap)
        elif mode == "busy":
            busy(gap)
        a = int(rng.integers(0, 8))
        t0 = time.perf_counter()
        env.step_fast(a, 64)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    print(f"  {mode:6s}: mean={ts.mean():.2f} p50={np.percentile(ts,50):.2f} "
          f"p90={np.percentile(ts,90):.2f} max={ts.max():.2f} ms")

for _ in range(2):  # repeat to rule out game-state drift
    for mode in ("cont", "sleep", "busy"):
        run(mode)
