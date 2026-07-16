#!/usr/bin/env python3
"""Phase 0 throughput benchmark for the PokeEnv / PyBoy fleet.

Measures:
  (a) single-core RAW headless fps  -- bare pyboy.tick(1, False) loop
  (b) single-core agent-steps/s     -- through PokeEnv.step (frame-skip=24)
  (c) multiprocess scaling at N in {1,7,14,28,56} processes, each running
      PokeEnv.step for a few seconds; reports AGGREGATE agent-steps/s and
      mean per-instance RSS (psutil).

Results -> runs/bench0.json, plus a readable summary table on stdout.
"""

import json
import multiprocessing as mp
import os
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pokeio.emu.env import PokeEnv  # noqa: E402

ROM = os.path.join(ROOT, "roms", "pokemon_yellow.gb")
STATE = os.path.join(ROOT, "roms", "yellow_newgame.state")
OUT = os.path.join(ROOT, "runs", "bench0.json")

FRAME_SKIP = 24
PROC_COUNTS = [1, 7, 14, 28, 56]
DURATION = 4.0        # timed seconds per worker
RAW_FRAMES = 30000    # frames for the single-core raw fps measurement


# ------------------------------------------------------------------ (a) raw fps
def bench_raw_fps():
    from pyboy import PyBoy

    p = PyBoy(ROM, window="null", sound_emulated=False)
    try:
        with open(STATE, "rb") as fh:
            p.load_state(fh)
        p.tick(120, False)  # warm up
        t0 = time.perf_counter()
        p.tick(RAW_FRAMES, False)  # tightest possible loop (batched, no render)
        dt = time.perf_counter() - t0
    finally:
        p.stop(save=False)
    return RAW_FRAMES / dt


# ---------------------------------------------------- (b) single-core env steps
def bench_single_env(duration=DURATION):
    env = PokeEnv(ROM, frame_skip=FRAME_SKIP)
    try:
        env.reset(STATE)
        for i in range(10):  # warm up
            env.step(i % 8)
        steps = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            env.step(steps % 8)
            steps += 1
        dt = time.perf_counter() - t0
    finally:
        env.close()
    return steps / dt


# ------------------------------------------------------- (c) multiprocess scale
def _worker(barrier, duration, ret_q, widx):
    env = PokeEnv(ROM, frame_skip=FRAME_SKIP)
    try:
        env.reset(STATE)
        # per-worker deterministic-ish varied actions
        a = widx % 8
        for _ in range(10):  # warm up (load state, page in)
            a = (a + 1) % 8
            env.step(a)
        rss = psutil.Process().memory_info().rss
        barrier.wait()  # all workers start the timed loop together
        steps = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            a = (a + 1) % 8
            env.step(a)
            steps += 1
        dt = time.perf_counter() - t0
        ret_q.put((steps, dt, rss))
    finally:
        env.close()


def bench_multiproc(n, duration=DURATION):
    barrier = mp.Barrier(n)
    ret_q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(barrier, duration, ret_q, i)) for i in range(n)]
    for p in procs:
        p.start()
    results = [ret_q.get() for _ in range(n)]
    for p in procs:
        p.join()
    total_steps = sum(r[0] for r in results)
    mean_dt = sum(r[1] for r in results) / n
    agg_sps = total_steps / mean_dt
    rss_list = [r[2] for r in results]
    return {
        "n": n,
        "aggregate_steps_per_s": agg_sps,
        "per_proc_steps_per_s": agg_sps / n,
        "total_steps": total_steps,
        "mean_duration_s": mean_dt,
        "rss_mean_mb": sum(rss_list) / n / 1e6,
        "rss_max_mb": max(rss_list) / 1e6,
    }


def main():
    if not os.path.exists(STATE):
        sys.exit(f"missing state {STATE} — run scripts/make_newgame_state.py first")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = {
        "meta": {
            "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "frame_skip": FRAME_SKIP,
            "duration_s": DURATION,
            "raw_frames": RAW_FRAMES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    }

    print("(a) single-core raw headless fps ...")
    raw_fps = bench_raw_fps()
    results["single_core_raw_fps"] = raw_fps
    print(f"    {raw_fps:,.0f} frames/s")

    print("(b) single-core agent-steps/s (frame-skip=%d) ..." % FRAME_SKIP)
    single_sps = bench_single_env()
    results["single_core_steps_per_s"] = single_sps
    print(f"    {single_sps:,.1f} steps/s  ({single_sps * FRAME_SKIP:,.0f} eff. frames/s)")

    print("(c) multiprocess scaling ...")
    results["multiproc"] = []
    for n in PROC_COUNTS:
        r = bench_multiproc(n)
        results["multiproc"].append(r)
        print(
            f"    N={n:<3d} agg={r['aggregate_steps_per_s']:>10,.0f} steps/s"
            f"  per-proc={r['per_proc_steps_per_s']:>7,.0f}"
            f"  rss/inst={r['rss_mean_mb']:>6.1f} MB"
        )

    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- summary table
    print("\n===== bench0 summary =====")
    print(f"machine: {results['meta']['physical_cpus']} phys / "
          f"{results['meta']['logical_cpus']} logical cpus")
    print(f"(a) single-core RAW fps        : {raw_fps:,.0f} frames/s")
    print(f"(b) single-core agent steps/s  : {single_sps:,.1f} steps/s")
    print(f"{'N':>4} | {'agg steps/s':>12} | {'per-proc s/s':>12} | "
          f"{'rss/inst MB':>11} | {'scaling':>8}")
    print("-" * 62)
    base = results["multiproc"][0]["aggregate_steps_per_s"]
    for r in results["multiproc"]:
        print(f"{r['n']:>4} | {r['aggregate_steps_per_s']:>12,.0f} | "
              f"{r['per_proc_steps_per_s']:>12,.0f} | {r['rss_mean_mb']:>11.1f} | "
              f"{r['aggregate_steps_per_s'] / base:>7.2f}x")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
