"""Isolated round-latency benchmark for BarrierFleet (no GPU forward).

Run: PYTHONPATH=/home/cmod/pokeIO python scripts/bench_barrier.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time

import numpy as np

from pokeio.emu.fleet import BarrierFleet, ObsEncoder
from pokeio.reward.archive import NoveltyArchive


def main():
    a = NoveltyArchive()
    ak = dict(
        screen_cells=a.screen_cells, screen_levels=a.screen_levels,
        wram_stride=a.wram_stride, wram_levels=a.wram_levels,
    )
    for P in (8, 32, 56):
        enc = ObsEncoder(24, 8)
        f = BarrierFleet(
            n_envs=P, obs_dim=enc.dim, obs_res=24, obs_ram=8,
            rom_path="roms/pokemon_yellow.gb", frame_skip=24, hold_frames=8,
            reset_state="roms/yellow_newgame.state", archive_kwargs=ak,
            wram_stride=64, goexplore=False, envs_per_worker=1,
        )
        f.reset_all()
        acts = np.zeros(P, dtype=np.int32)
        for _ in range(20):
            f.step_all(acts)
        N = 300
        t = time.perf_counter()
        for i in range(N):
            acts[:] = i % 8
            f.step_all(acts)
        dt = time.perf_counter() - t
        print(f"P={P:3d}: {dt / N * 1000:6.2f} ms/round  "
              f"{P * N / dt:8.0f} env-steps/s  ({f.n_workers} workers)")
        f.close()


if __name__ == "__main__":
    main()
