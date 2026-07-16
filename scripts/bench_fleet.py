#!/usr/bin/env python3
"""Phase 1 fleet benchmark — aggregate agent-steps/s WITH the real vision obs
pipeline, so we can quantify the obs-pipeline overhead vs the raw Phase 0 numbers
(bench0.json: ~10,183 steps/s at 56 procs).

For each N in {28, 56} it runs three variants so the overhead is attributable:

  raw      -- PokeEnv.step only (no obs build)          [baseline, ~bench0]
  vision   -- VisionEnv.step (PokeEnv + ObsBuilder)     [obs pipeline cost]
  fleet    -- VecFleet.step_all (workers + shared-mem)  [end-to-end w/ transport]

Reports steps/s + RSS/instance to runs/bench_fleet.json and a stdout table.
"""

import os

# Pin numeric thread pools to 1 BEFORE numpy is imported anywhere: each worker
# owns a single core, so multi-threaded BLAS is pure oversubscription (collapses
# throughput ~5x at 56 procs). Spawn children inherit these.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import psutil  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pokeio.config import Config  # noqa: E402
from pokeio.emu.env import PokeEnv  # noqa: E402
from pokeio.emu.fleet import NUMA_NODES, VecFleet  # noqa: E402
from pokeio.vision.env_wrap import VisionEnv  # noqa: E402

CFG = Config()
ROM = os.path.join(ROOT, CFG.emu.rom_path)
STATE = os.path.join(ROOT, CFG.emu.reset_state)
OUT = os.path.join(ROOT, "runs", "bench_fleet.json")

FRAME_SKIP = CFG.emu.frame_skip
PROC_COUNTS = [28, 56]
DURATION = 4.0
WARMUP = 10

_CORE_ORDER = []
for a, b in zip(NUMA_NODES[0], NUMA_NODES[1]):
    _CORE_ORDER += [a, b]


# --------------------------------------------------------- per-process workers
def _pin(widx):
    try:
        psutil.Process().cpu_affinity([_CORE_ORDER[widx % len(_CORE_ORDER)]])
    except Exception:
        pass


def _raw_worker(barrier, duration, ret_q, widx):
    _pin(widx)
    env = PokeEnv(ROM, frame_skip=FRAME_SKIP)
    try:
        env.reset(STATE)
        a = widx % 8
        for _ in range(WARMUP):
            a = (a + 1) % 8
            env.step(a)
        rss = psutil.Process().memory_info().rss
        barrier.wait()
        steps = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            a = (a + 1) % 8
            env.step(a)
            steps += 1
        ret_q.put((steps, time.perf_counter() - t0, rss))
    finally:
        env.close()


def _vision_worker(barrier, duration, ret_q, widx):
    _pin(widx)
    env = VisionEnv(ROM, config=CFG)
    try:
        env.reset(STATE)
        a = widx % 8
        for _ in range(WARMUP):
            a = (a + 1) % 8
            env.step(a)
        rss = psutil.Process().memory_info().rss
        barrier.wait()
        steps = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            a = (a + 1) % 8
            env.step(a)
            steps += 1
        ret_q.put((steps, time.perf_counter() - t0, rss))
    finally:
        env.close()


def _bench_procs(worker, n, duration=DURATION):
    barrier = mp.Barrier(n)
    ret_q = mp.Queue()
    procs = [mp.Process(target=worker, args=(barrier, duration, ret_q, i)) for i in range(n)]
    for p in procs:
        p.start()
    results = [ret_q.get() for _ in range(n)]
    for p in procs:
        p.join()
    total_steps = sum(r[0] for r in results)
    mean_dt = sum(r[1] for r in results) / n
    rss_list = [r[2] for r in results]
    return {
        "n": n,
        "aggregate_steps_per_s": total_steps / mean_dt,
        "per_proc_steps_per_s": total_steps / mean_dt / n,
        "total_steps": total_steps,
        "mean_duration_s": mean_dt,
        "rss_mean_mb": sum(rss_list) / n / 1e6,
        "rss_max_mb": max(rss_list) / 1e6,
    }


# ----------------------------------------------------------- VecFleet end-to-end
def _bench_fleet(n, duration=DURATION):
    """End-to-end: VecFleet.step_all including shared-mem batching in the parent."""
    fleet = VecFleet(n, ROM, config=CFG, state_path=STATE)
    try:
        fleet.reset_all()
        acts = [i % 8 for i in range(n)]
        for _ in range(WARMUP):  # warm up (page in, JIT paths)
            acts = [(a + 1) % 8 for a in acts]
            fleet.step_all(acts)
        # RSS across the live worker processes.
        rss = [
            psutil.Process(p.pid).memory_info().rss
            for p in fleet._procs
            if p is not None and p.is_alive()
        ]
        steps = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            acts = [(a + 1) % 8 for a in acts]
            fleet.step_all(acts)
            steps += 1  # one batched step == n agent-steps
        dt = time.perf_counter() - t0
    finally:
        fleet.close()
    agg = steps * n / dt
    return {
        "n": n,
        "batched_steps": steps,
        "aggregate_steps_per_s": agg,
        "per_proc_steps_per_s": agg / n,
        "mean_duration_s": dt,
        "rss_mean_mb": (sum(rss) / len(rss) / 1e6) if rss else 0.0,
        "rss_max_mb": (max(rss) / 1e6) if rss else 0.0,
    }


def main():
    if not (os.path.exists(ROM) and os.path.exists(STATE)):
        sys.exit(f"missing rom/state ({ROM} / {STATE})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # Reference raw number from Phase 0.
    raw_ref = None
    bench0 = os.path.join(ROOT, "runs", "bench0.json")
    if os.path.exists(bench0):
        with open(bench0) as fh:
            b0 = json.load(fh)
        for r in b0.get("multiproc", []):
            if r["n"] == 56:
                raw_ref = r["aggregate_steps_per_s"]

    results = {
        "meta": {
            "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "frame_skip": FRAME_SKIP,
            "duration_s": DURATION,
            "obs_spec": {
                "coarse": [CFG.vision.coarse_size, CFG.vision.coarse_size],
                "fovea": [CFG.vision.fovea_size, CFG.vision.fovea_size],
                "motion": [CFG.vision.coarse_size, CFG.vision.coarse_size],
                "ram_aux": [CFG.vision.ram_aux_dim],
            },
            "bench0_raw_56_steps_per_s": raw_ref,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "raw": [],
        "vision": [],
        "fleet": [],
    }

    for n in PROC_COUNTS:
        print(f"[N={n}] raw (PokeEnv only) ...")
        r_raw = _bench_procs(_raw_worker, n)
        results["raw"].append(r_raw)
        print(f"        {r_raw['aggregate_steps_per_s']:,.0f} steps/s")

        print(f"[N={n}] vision (PokeEnv + ObsBuilder, per-proc) ...")
        r_vis = _bench_procs(_vision_worker, n)
        results["vision"].append(r_vis)
        print(f"        {r_vis['aggregate_steps_per_s']:,.0f} steps/s")

        print(f"[N={n}] fleet (VecFleet shared-mem end-to-end) ...")
        r_fleet = _bench_fleet(n)
        results["fleet"].append(r_fleet)
        print(f"        {r_fleet['aggregate_steps_per_s']:,.0f} steps/s")

    # Attach overhead deltas.
    for i, n in enumerate(PROC_COUNTS):
        raw = results["raw"][i]["aggregate_steps_per_s"]
        vis = results["vision"][i]["aggregate_steps_per_s"]
        flt = results["fleet"][i]["aggregate_steps_per_s"]
        results["vision"][i]["overhead_vs_raw_pct"] = (raw - vis) / raw * 100.0
        results["fleet"][i]["overhead_vs_raw_pct"] = (raw - flt) / raw * 100.0

    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- summary table
    print("\n===== bench_fleet summary =====")
    print(f"machine: {results['meta']['physical_cpus']} phys / "
          f"{results['meta']['logical_cpus']} logical cpus")
    if raw_ref:
        print(f"bench0 raw @56: {raw_ref:,.0f} steps/s")
    hdr = f"{'N':>4} | {'variant':>7} | {'agg steps/s':>12} | {'per-proc':>9} | {'rss/inst MB':>11} | {'vs raw':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i, n in enumerate(PROC_COUNTS):
        for variant in ("raw", "vision", "fleet"):
            r = results[variant][i]
            oh = r.get("overhead_vs_raw_pct")
            oh_s = f"{-oh:+6.1f}%" if oh is not None else "   ref"
            print(f"{n:>4} | {variant:>7} | {r['aggregate_steps_per_s']:>12,.0f} | "
                  f"{r['per_proc_steps_per_s']:>9,.0f} | {r['rss_mean_mb']:>11.1f} | {oh_s:>8}")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
