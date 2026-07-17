"""Audit micro-benchmarks: primitives on the hot path (no fleet).

Measures, in-process:
  1. time.sleep(5e-5) actual duration (the barrier nap)
  2. parent spin iteration cost: (wdone >= target).all() on n=32/64
  3. worker spin iteration cost: the lambda `ctl[0] >= local_round`
  4. ObsEncoder.encode_compact on a real screen
  5. NoveltyArchive.cell_key_compact on a real screen
  6. env.step_fast distribution (mean/p50/p90/p99/max) over 2000 steps
  7. pyboy save_state / load_state cost (Go-Explore capture/restore)
  8. Population.from_genomes + compile for 32/64 genomes
  9. one GPU forward round: h2d / forward / argmax+d2h split (n=32/64)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import io
import time
import numpy as np

ROOT = "/home/cmod/pokeIO"
ROM = f"{ROOT}/roms/pokemon_yellow.gb"
STATE = f"{ROOT}/roms/yellow_newgame.state"


def pct(a, q):
    return float(np.percentile(np.asarray(a), q))


def main():
    # 1. nap duration
    t = []
    for _ in range(2000):
        t0 = time.perf_counter()
        time.sleep(5e-5)
        t.append(time.perf_counter() - t0)
    print(f"[1] time.sleep(50us): mean={np.mean(t)*1e6:.1f}us p50={pct(t,50)*1e6:.1f}us "
          f"p99={pct(t,99)*1e6:.1f}us max={max(t)*1e6:.1f}us")

    # 2. parent spin iteration
    for n in (32, 64):
        wdone = np.zeros(2 + n, dtype=np.int64)[2:]
        target = 5
        t0 = time.perf_counter()
        N = 100000
        for _ in range(N):
            bool((wdone >= target).all())
        dt = time.perf_counter() - t0
        print(f"[2] parent spin iter n={n}: {dt/N*1e6:.2f} us/iter "
              f"(3000 spins = {dt/N*3000*1000:.2f} ms)")

    # 3. worker spin iteration
    ctl = np.zeros(66, dtype=np.int64)
    local_round = 5
    pred = lambda: ctl[0] >= local_round
    t0 = time.perf_counter()
    N = 300000
    for _ in range(N):
        pred()
    dt = time.perf_counter() - t0
    print(f"[3] worker spin iter: {dt/N*1e6:.2f} us/iter "
          f"(3000 spins = {dt/N*3000*1000:.2f} ms)")

    # boot a real env for realistic screens
    from pokeio.emu.env import PokeEnv
    from pokeio.emu.fleet import ObsEncoder
    from pokeio.reward.archive import NoveltyArchive

    env = PokeEnv(ROM, frame_skip=24, hold_frames=8)
    screen = env.reset(STATE)
    w64 = env.wram_strided(64)

    enc = ObsEncoder(24, 8)
    arch = NoveltyArchive()

    # warm
    for _ in range(50):
        enc.encode_compact(screen, w64)
        arch.cell_key_compact(screen, w64)

    N = 2000
    t0 = time.perf_counter()
    for _ in range(N):
        enc.encode_compact(screen, w64)
    print(f"[4] encode_compact: {(time.perf_counter()-t0)/N*1000:.3f} ms")

    t0 = time.perf_counter()
    for _ in range(N):
        arch.cell_key_compact(screen, w64)
    print(f"[5] cell_key_compact: {(time.perf_counter()-t0)/N*1000:.3f} ms")

    # screen shm-write cost proxy (23KB copy)
    dst = np.zeros((64, 144, 160), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(N):
        dst[3] = screen
    print(f"[5b] screen row copy: {(time.perf_counter()-t0)/N*1e6:.1f} us")

    # 6. step_fast distribution (random actions like training)
    rng = np.random.default_rng(0)
    times = []
    for _ in range(2000):
        a = int(rng.integers(0, 8))
        t0 = time.perf_counter()
        env.step_fast(a, 64)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"[6] step_fast (2000): mean={times.mean()*1000:.2f} p50={pct(times,50)*1000:.2f} "
          f"p90={pct(times,90)*1000:.2f} p99={pct(times,99)*1000:.2f} "
          f"max={times.max()*1000:.2f} ms")

    # 7. save/load state
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        buf = io.BytesIO()
        env.pyboy.save_state(buf)
        blob = buf.getvalue()
    print(f"[7] save_state: {(time.perf_counter()-t0)/N*1000:.2f} ms  (blob={len(blob)} B)")
    t0 = time.perf_counter()
    for _ in range(N):
        env.load_state(blob)
    print(f"[7] load_state: {(time.perf_counter()-t0)/N*1000:.2f} ms")
    # reset (canonical) for comparison
    t0 = time.perf_counter()
    for _ in range(20):
        env.reset(STATE)
    print(f"[7] env.reset(state file): {(time.perf_counter()-t0)/20*1000:.2f} ms")
    env.close()

    # 8. from_genomes + compile, 9. forward split
    import torch
    from pokeio.evo.genome import InnovationTracker, Population, make_genome
    from pokeio.evo.forward import population_forward_sparse

    dev = torch.device("cuda:1")
    enc_dim = enc.dim
    n_in = enc_dim
    tracker = InnovationTracker(n_in=n_in, n_out=8)
    grng = np.random.default_rng(0)
    max_conns = (n_in + 1) * 8 + 2048
    max_nodes = n_in + 1 + 8 + 256
    for n in (32, 64):
        genomes = [make_genome(n_in, 8, tracker, grng, connect="full", weight_scale=1.0)
                   for _ in range(n)]
        t0 = time.perf_counter()
        pop = Population.from_genomes(genomes, max_nodes=max_nodes, max_conns=max_conns)
        t1 = time.perf_counter()
        cp = pop.compile(dev)
        torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        print(f"[8] n={n}: from_genomes={t1-t0:.3f}s  compile={t2-t1:.3f}s")

        # 9. forward round split, after warmup
        X = np.random.rand(n, enc_dim).astype(np.float32)
        for _ in range(20):
            xt = torch.from_numpy(X).to(dev).unsqueeze(1)
            out = population_forward_sparse(cp, xt, steps=4)
            out[:, 0, :].argmax(dim=1).cpu().numpy()
        torch.cuda.synchronize(dev)
        h2d = fwd = d2h = 0.0
        N = 200
        for _ in range(N):
            t0 = time.perf_counter()
            xt = torch.from_numpy(X).to(dev).unsqueeze(1)
            torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            out = population_forward_sparse(cp, xt, steps=4)
            torch.cuda.synchronize(dev)
            t2 = time.perf_counter()
            acts = out[:, 0, :].argmax(dim=1).cpu().numpy()
            t3 = time.perf_counter()
            h2d += t1 - t0
            fwd += t2 - t1
            d2h += t3 - t2
        print(f"[9] n={n}: h2d={h2d/N*1000:.2f}  fwd={fwd/N*1000:.2f}  "
              f"argmax+d2h={d2h/N*1000:.2f}  total={(h2d+fwd+d2h)/N*1000:.2f} ms")


if __name__ == "__main__":
    main()
