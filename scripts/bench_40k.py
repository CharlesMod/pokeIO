#!/usr/bin/env python3
"""Phase 1.5 benchmark — path to >=40,000 agent-steps/sec at frame_skip=1.

Two modes:

  --profile   Decompose the per-frame cost (measure, don't guess):
                * render-on fps/core: tick(1, render=True)
                * compact framebuffer grab (channel-0 -> contiguous 144x160)
                * H2D transfer of a batch of framebuffers (pinned, async)
                * GPU batched vision (GPUVision.build)
                * stub batched inference (StubPolicy)
              Reports ms/frame and the implied single-core render-on ceiling.

  (default)   End-to-end AsyncFleet sweep over N in {28,56,128,256}, blocking
              (exact 1-frame latency) and overlapped (double-buffered) transport.
              Reports the agent-steps/sec throughput curve, the binding
              constraint at the top end, and whether >=40k is reached.
              Saves runs/bench_40k.json.

BLAS thread pools are pinned to 1 before numpy import (workers own one core).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pokeio.config import Config  # noqa: E402

CFG = Config()
ROM = os.path.join(ROOT, CFG.emu.rom_path)
STATE = os.path.join(ROOT, CFG.emu.reset_state)
OUT = os.path.join(ROOT, "runs", "bench_40k.json")
DEVICE = "cuda:1"
TARGET = 40_000


# ------------------------------------------------------------------- profiling
def profile(n_batch: int = 128, iters: int = 2000):
    import torch
    from pyboy import PyBoy

    from pokeio.pipeline.async_fleet import StubPolicy
    from pokeio.pipeline.gpu_vision import GPUVision

    res = {}

    # -- render-on tick + grab (single core, one live emulator) --------------
    pb = PyBoy(ROM, window="null", sound_emulated=False)
    with open(STATE, "rb") as fh:
        pb.load_state(fh)
    pb.tick(1, True)
    for _ in range(100):  # warm
        pb.button_press("right"); pb.tick(1, True); pb.button_release("right")

    t0 = time.perf_counter()
    for _ in range(iters):
        pb.button_press("right"); pb.tick(1, True); pb.button_release("right")
    t_tick = (time.perf_counter() - t0) / iters

    fb = np.ascontiguousarray(pb.screen.ndarray[:, :, 0])
    t0 = time.perf_counter()
    for _ in range(iters):
        fb = np.ascontiguousarray(pb.screen.ndarray[:, :, 0])
    t_grab = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        pb.button_press("right"); pb.tick(1, True); pb.button_release("right")
        _ = np.ascontiguousarray(pb.screen.ndarray[:, :, 0])
    t_step = (time.perf_counter() - t0) / iters
    pb.stop(save=False)

    res["render_on_ms_per_frame"] = t_tick * 1e3
    res["render_on_fps_per_core"] = 1.0 / t_tick
    res["framebuffer_grab_ms"] = t_grab * 1e3
    res["lean_step_ms_per_frame"] = t_step * 1e3
    res["lean_step_fps_per_core"] = 1.0 / t_step

    # -- GPU-side costs (batched) -------------------------------------------
    dev = torch.device(DEVICE)
    vis = GPUVision(CFG, device=DEVICE)
    policy = StubPolicy(vis.optical_dim, n_actions=8, device=DEVICE)
    frames_np = np.random.randint(0, 256, size=(n_batch, 144, 160), dtype=np.uint8)
    # snap to the 4 DMG shades so it's representative
    shades = np.array([24, 88, 184, 248], dtype=np.uint8)
    frames_np = shades[(frames_np.astype(np.int32) * 4 // 256)]
    pin = torch.empty((n_batch, 144, 160), dtype=torch.uint8, pin_memory=True)
    pin.copy_(torch.from_numpy(frames_np))

    def _sync():
        torch.cuda.synchronize(dev)

    # H2D
    for _ in range(50):
        g = pin.to(dev, non_blocking=True); _sync()
    t0 = time.perf_counter()
    for _ in range(500):
        g = pin.to(dev, non_blocking=True)
    _sync()
    t_h2d = (time.perf_counter() - t0) / 500

    gpu_frames = pin.to(dev)
    _sync()
    for _ in range(50):
        obs = vis.build(gpu_frames); _sync()
    t0 = time.perf_counter()
    for _ in range(500):
        obs = vis.build(gpu_frames)
    _sync()
    t_obs = (time.perf_counter() - t0) / 500

    flat = torch.cat([obs["coarse"].reshape(n_batch, -1),
                      obs["fovea"].reshape(n_batch, -1),
                      obs["motion"].reshape(n_batch, -1)], dim=1)
    for _ in range(50):
        a = policy(flat); _sync()
    t0 = time.perf_counter()
    for _ in range(500):
        a = policy(flat)
    _sync()
    t_inf = (time.perf_counter() - t0) / 500

    # full GPU pipeline incl D2H of actions
    for _ in range(50):
        gg = pin.to(dev, non_blocking=True)
        o = vis.build(gg)
        fl = torch.cat([o["coarse"].reshape(n_batch, -1), o["fovea"].reshape(n_batch, -1),
                        o["motion"].reshape(n_batch, -1)], dim=1)
        acts = policy(fl).cpu().numpy()
    t0 = time.perf_counter()
    for _ in range(300):
        gg = pin.to(dev, non_blocking=True)
        o = vis.build(gg)
        fl = torch.cat([o["coarse"].reshape(n_batch, -1), o["fovea"].reshape(n_batch, -1),
                        o["motion"].reshape(n_batch, -1)], dim=1)
        acts = policy(fl).cpu().numpy()
    t_full = (time.perf_counter() - t0) / 300

    res["gpu_batch"] = n_batch
    res["h2d_ms_per_batch"] = t_h2d * 1e3
    res["gpu_obs_ms_per_batch"] = t_obs * 1e3
    res["stub_infer_ms_per_batch"] = t_inf * 1e3
    res["gpu_full_pipeline_ms_per_batch"] = t_full * 1e3
    res["gpu_full_per_frame_us"] = t_full / n_batch * 1e6

    # -- implied ceilings ----------------------------------------------------
    phys = 28
    res["render_ceiling_28core_steps_per_s"] = res["render_on_fps_per_core"] * phys
    res["lean_ceiling_28core_steps_per_s"] = res["lean_step_fps_per_core"] * phys

    print("\n===== per-frame cost decomposition (profile) =====")
    print(f"render-on tick(1,True):     {res['render_on_ms_per_frame']:.4f} ms  "
          f"= {res['render_on_fps_per_core']:,.0f} fps/core")
    print(f"framebuffer grab (23KB):    {res['framebuffer_grab_ms']:.4f} ms")
    print(f"lean fs=1 step+grab:        {res['lean_step_ms_per_frame']:.4f} ms  "
          f"= {res['lean_step_fps_per_core']:,.0f} steps/s/core")
    print(f"H2D (batch {n_batch}):           {res['h2d_ms_per_batch']:.4f} ms  "
          f"({res['h2d_ms_per_batch']/n_batch*1e3:.2f} us/frame)")
    print(f"GPU obs (batch {n_batch}):       {res['gpu_obs_ms_per_batch']:.4f} ms  "
          f"({res['gpu_obs_ms_per_batch']/n_batch*1e3:.2f} us/frame)")
    print(f"stub infer (batch {n_batch}):    {res['stub_infer_ms_per_batch']:.4f} ms  "
          f"({res['stub_infer_ms_per_batch']/n_batch*1e3:.2f} us/frame)")
    print(f"GPU full pipeline (b{n_batch}):  {res['gpu_full_pipeline_ms_per_batch']:.4f} ms  "
          f"({res['gpu_full_per_frame_us']:.2f} us/frame)")
    print(f"implied render ceiling @28 cores: "
          f"{res['render_ceiling_28core_steps_per_s']:,.0f} steps/s")
    return res


# ------------------------------------------------------------------- end2end
def bench_e2e(n_list, frames=800, warmup=60, overlap_modes=(False, True)):
    from pokeio.pipeline.async_fleet import AsyncFleet
    rows = []
    for n in n_list:
        for overlap in overlap_modes:
            fleet = AsyncFleet(n, ROM, CFG, STATE, device=DEVICE,
                               gpu=True, overlap=overlap)
            fleet.start()
            try:
                st = fleet.run_frames(frames, warmup=warmup, timeout_s=120.0)
            finally:
                fleet.close()
            st.pop("actions", None)
            rows.append(st)
            tag = "overlap" if overlap else "blocking"
            print(f"[N={n:>4} {tag:>8}] {st['agent_steps_per_s']:>10,.0f} steps/s  "
                  f"(wait {st['mean_wait_ms']:.3f}ms  infer {st['mean_infer_ms']:.3f}ms)")
            time.sleep(0.5)  # let processes fully die before next spawn
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true", help="per-frame cost decomposition only")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--n", type=int, nargs="*", default=[28, 56, 128, 256])
    args = ap.parse_args()

    if not (os.path.exists(ROM) and os.path.exists(STATE)):
        sys.exit(f"missing rom/state ({ROM} / {STATE})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    import psutil
    import torch
    meta = {
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(1) if torch.cuda.is_available() else None,
        "frame_skip": CFG.emu.frame_skip,
        "obs_flat_dim": 6144,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_steps_per_s": TARGET,
    }

    prof = profile()

    if args.profile:
        payload = {"meta": meta, "profile": prof}
        with open(OUT, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nprofile -> {OUT}")
        return

    print("\n===== end-to-end AsyncFleet sweep (fs=1) =====")
    rows = bench_e2e(args.n, frames=args.frames, warmup=args.warmup)

    best = max(rows, key=lambda r: r["agent_steps_per_s"])
    hit = best["agent_steps_per_s"] >= TARGET

    payload = {"meta": meta, "profile": prof, "sweep": rows,
               "best": best, "target_hit": hit}
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n===== summary =====")
    print(f"{'N':>5} | {'mode':>8} | {'steps/s':>11} | {'wait ms':>8} | {'infer ms':>8}")
    print("-" * 52)
    for r in rows:
        print(f"{r['n_envs']:>5} | {'overlap' if r['overlap'] else 'blocking':>8} | "
              f"{r['agent_steps_per_s']:>11,.0f} | {r['mean_wait_ms']:>8.3f} | "
              f"{r['mean_infer_ms']:>8.3f}")
    print(f"\nBEST: {best['agent_steps_per_s']:,.0f} steps/s at N={best['n_envs']} "
          f"({'overlap' if best['overlap'] else 'blocking'}) — "
          f"{'>=40k HIT' if hit else 'below 40k'}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
