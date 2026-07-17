"""Instrumented COPY of BarrierFleet + evaluate_wave_parallel (audit only).

Workers timestamp each phase (wake, capture, step, emit) into a shared tprof
block; the parent timestamps release/done and its own phases. perf_counter is
CLOCK_MONOTONIC on Linux, comparable across processes.

Modes:
  --noop            workers do nothing per round (pure barrier propagation)
  --players N       fleet size
  --epw N           envs per worker
  --goexplore 0|1   real Go-Explore capture/restore bookkeeping
  --rounds N        measured rounds
  --spin pure|nap   worker+parent wait strategy (nap = committed behaviour)
  --no-fwd          skip the GPU forward (parent does bookkeeping only)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import time
from multiprocessing import shared_memory

import numpy as np
import psutil

ROOT = "/home/cmod/pokeIO"
ROM = f"{ROOT}/roms/pokemon_yellow.gb"
STATE = f"{ROOT}/roms/yellow_newgame.state"

_SCREEN_H, _SCREEN_W = 144, 160
_OP_STEP, _OP_RESET, _OP_SHUTDOWN = 0, 1, 2
_MAX_STATE = 262144

NUMA_NODES = {
    0: list(range(0, 14)) + list(range(28, 42)),
    1: list(range(14, 28)) + list(range(42, 56)),
}


def _numa_core_order():
    n0, n1 = NUMA_NODES[0], NUMA_NODES[1]
    order = []
    for a, b in zip(n0, n1):
        order.append(a)
        order.append(b)
    order.extend(n0[len(n1):])
    order.extend(n1[len(n0):])
    return order


def _spin_wait(pred, mode, spin_budget=3000, nap=5e-5):
    i = 0
    while not pred():
        if mode == "nap":
            i += 1
            if i >= spin_budget:
                time.sleep(nap)


def _worker_main(slice_lo, slice_hi, noop, goexplore, spin_mode,
                 shm_names, n_envs, obs_dim, key_len, core, epw_first_only):
    import io as _io
    from pokeio.emu.env import PokeEnv
    from pokeio.emu.fleet import ObsEncoder
    from pokeio.reward.archive import NoveltyArchive

    try:
        if core is not None:
            psutil.Process().cpu_affinity([core])
    except Exception:
        pass

    encoder = ObsEncoder(24, 8)
    archive = NoveltyArchive()

    shms = []

    def reg(name, shape, dtype):
        shm = shared_memory.SharedMemory(name=shm_names[name])
        shms.append(shm)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    obs = reg("obs", (n_envs, obs_dim), np.float32)
    screens = reg("screens", (n_envs, _SCREEN_H, _SCREEN_W), np.uint8)
    keys = reg("keys", (n_envs, key_len), np.uint8)
    actions = reg("actions", (n_envs,), np.int32)
    dones = reg("dones", (n_envs,), np.uint8)
    cap_flag = reg("cap_flag", (n_envs,), np.uint8)
    cap_done = reg("cap_done", (n_envs,), np.uint8)
    cap_state = reg("cap_state", (n_envs, _MAX_STATE), np.uint8)
    cap_len = reg("cap_len", (n_envs,), np.int32)
    res_flag = reg("res_flag", (n_envs,), np.uint8)
    res_state = reg("res_state", (n_envs, _MAX_STATE), np.uint8)
    res_len = reg("res_len", (n_envs,), np.int32)
    ctl = reg("ctl", (2 + n_envs,), np.int64)
    # tprof[gi] = [t_wake, t_cap_done, t_step_done, t_emit_done] (abs perf_counter)
    tprof = reg("tprof", (n_envs, 4), np.float64)

    envs = []
    if not noop:
        envs = [PokeEnv(ROM, frame_skip=24, hold_frames=8)
                for _ in range(slice_lo, slice_hi)]

    def _emit(gi, screen, w64):
        screens[gi] = screen
        obs[gi] = encoder.encode_compact(screen, w64)
        k = archive.cell_key_compact(screen, w64)
        keys[gi] = np.frombuffer(k, dtype=np.uint8)

    local_round = 1
    try:
        while True:
            _spin_wait(lambda: ctl[0] >= local_round, spin_mode)
            t_wake = time.perf_counter()
            op = int(ctl[1])
            if op == _OP_SHUTDOWN:
                break
            if noop:
                for gi in range(slice_lo, slice_hi):
                    tprof[gi, 0] = t_wake
                    tprof[gi, 1] = tprof[gi, 2] = tprof[gi, 3] = time.perf_counter()
                    ctl[2 + gi] = local_round
                local_round += 1
                continue
            for li, gi in enumerate(range(slice_lo, slice_hi)):
                env = envs[li]
                if op == _OP_RESET:
                    if goexplore and res_flag[gi]:
                        env.load_state(bytes(res_state[gi, : int(res_len[gi])]))
                        env.pyboy.tick(1, True)
                        screen = env._obs()
                        w64 = env.wram_strided(64)
                    else:
                        screen = env.reset(STATE)
                        w64 = env.wram_strided(64)
                    dones[gi] = 0
                    tprof[gi, 0] = t_wake
                    tprof[gi, 1] = tprof[gi, 2] = time.perf_counter()
                    _emit(gi, screen, w64)
                    tprof[gi, 3] = time.perf_counter()
                else:
                    tprof[gi, 0] = t_wake
                    if goexplore and cap_flag[gi]:
                        buf = _io.BytesIO()
                        env.pyboy.save_state(buf)
                        blob = buf.getvalue()
                        nb = min(len(blob), _MAX_STATE)
                        cap_state[gi, :nb] = np.frombuffer(blob[:nb], dtype=np.uint8)
                        cap_len[gi] = nb
                        cap_done[gi] = 1
                    else:
                        cap_done[gi] = 0
                    tprof[gi, 1] = time.perf_counter()
                    screen, w64, done = env.step_fast(int(actions[gi]), 64)
                    tprof[gi, 2] = time.perf_counter()
                    dones[gi] = 1 if done else 0
                    _emit(gi, screen, w64)
                    tprof[gi, 3] = time.perf_counter()
            for gi in range(slice_lo, slice_hi):
                ctl[2 + gi] = local_round
            local_round += 1
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        for s in shms:
            try:
                s.close()
            except Exception:
                pass


class AuditFleet:
    def __init__(self, n_envs, obs_dim, key_len, noop, goexplore, spin_mode, epw,
                 parent_wait="spin"):
        self.n_envs = n_envs
        self.noop = noop
        self.goexplore = goexplore
        self.spin_mode = spin_mode
        self.parent_wait = parent_wait
        self._blocks = {}
        self.arr = {}

        def alloc(name, shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
            self._blocks[name] = shm
            self.arr[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        n = n_envs
        alloc("obs", (n, obs_dim), np.float32)
        alloc("screens", (n, _SCREEN_H, _SCREEN_W), np.uint8)
        alloc("keys", (n, key_len), np.uint8)
        alloc("actions", (n,), np.int32)
        alloc("dones", (n,), np.uint8)
        alloc("cap_flag", (n,), np.uint8)
        alloc("cap_done", (n,), np.uint8)
        alloc("cap_state", (n, _MAX_STATE), np.uint8)
        alloc("cap_len", (n,), np.int32)
        alloc("res_flag", (n,), np.uint8)
        alloc("res_state", (n, _MAX_STATE), np.uint8)
        alloc("res_len", (n,), np.int32)
        alloc("ctl", (2 + n,), np.int64)
        alloc("tprof", (n, 4), np.float64)
        self._ctl = self.arr["ctl"]
        self._ctl[:] = 0
        self._round = 0
        names = {k: v.name for k, v in self._blocks.items()}

        ctx = mp.get_context("spawn")
        order = _numa_core_order()
        self._procs = []
        wi = 0
        for lo in range(0, n, epw):
            hi = min(lo + epw, n)
            core = order[wi % len(order)]
            p = ctx.Process(target=_worker_main, args=(
                lo, hi, noop, goexplore, spin_mode, names, n, obs_dim,
                key_len, core, False), daemon=True)
            p.start()
            self._procs.append(p)
            wi += 1
        self.n_workers = len(self._procs)
        # last round's timing record, filled by _run_round
        self.t_rel = 0.0
        self.t_done = 0.0

    def _run_round(self, op):
        self._round += 1
        self._ctl[1] = op
        t_rel = time.perf_counter()
        self._ctl[0] = self._round
        target = self._round
        wdone = self._ctl[2:2 + self.n_envs]
        i = 0
        while not bool((wdone >= target).all()):
            if self.parent_wait == "sleep":
                time.sleep(5e-5)
            elif self.spin_mode == "nap":
                i += 1
                if i >= 3000:
                    time.sleep(5e-5)
        self.t_done = time.perf_counter()
        self.t_rel = t_rel

    def reset_all(self, restore=None):
        self.arr["res_flag"][:] = 0
        if restore and self.goexplore:
            for idx, blob in restore.items():
                b = blob[:_MAX_STATE]
                self.arr["res_state"][idx, : len(b)] = np.frombuffer(b, dtype=np.uint8)
                self.arr["res_len"][idx] = len(b)
                self.arr["res_flag"][idx] = 1
        self._run_round(_OP_RESET)
        return self.arr["obs"].copy()

    def step_all(self, actions, capture_flags=None):
        self.arr["actions"][: len(actions)] = np.asarray(actions, dtype=np.int32)
        if self.goexplore and capture_flags is not None:
            self.arr["cap_flag"][:] = capture_flags
        else:
            self.arr["cap_flag"][:] = 0
        self._run_round(_OP_STEP)
        obs = self.arr["obs"]
        keys = self.arr["keys"]
        dones = self.arr["dones"].astype(bool)
        captured = {}
        if self.goexplore:
            cd = self.arr["cap_done"]
            cl = self.arr["cap_len"]
            cs = self.arr["cap_state"]
            for i in range(self.n_envs):
                if cd[i]:
                    captured[i] = bytes(cs[i, : int(cl[i])])
        return obs, keys, dones, captured

    def close(self):
        try:
            self._ctl[1] = _OP_SHUTDOWN
            self._round += 1
            self._ctl[0] = self._round
        except Exception:
            pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for shm in self._blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--noop", action="store_true")
    ap.add_argument("--goexplore", type=int, default=1)
    ap.add_argument("--spin", choices=("nap", "pure"), default="nap")
    ap.add_argument("--epw", type=int, default=1)
    ap.add_argument("--no-fwd", action="store_true")
    ap.add_argument("--parent-wait", choices=("spin", "sleep"), default="spin")
    ap.add_argument("--pin-parent", type=int, default=-1)
    args = ap.parse_args()

    if args.pin_parent >= 0:
        psutil.Process().cpu_affinity([args.pin_parent])
    n = args.players
    from pokeio.emu.fleet import ObsEncoder, _archive_key_len
    from pokeio.reward.archive import NoveltyArchive
    from pokeio.reward.goexplore import GoExplore
    from pokeio.reward.novelty import WaveNovelty

    enc = ObsEncoder(24, 8)
    a0 = NoveltyArchive()
    ak = dict(screen_cells=a0.screen_cells, screen_levels=a0.screen_levels,
              wram_stride=a0.wram_stride, wram_levels=a0.wram_levels)
    key_len = _archive_key_len(ak, 64)

    use_fwd = not args.no_fwd and not args.noop
    if use_fwd:
        import torch
        from pokeio.evo.genome import InnovationTracker, Population, make_genome
        from pokeio.evo.forward import population_forward_sparse
        dev = torch.device("cuda:1")
        n_in = enc.dim
        tracker = InnovationTracker(n_in=n_in, n_out=8)
        grng = np.random.default_rng(0)
        genomes = [make_genome(n_in, 8, tracker, grng, connect="full",
                               weight_scale=1.0) for _ in range(n)]
        t0 = time.perf_counter()
        pop = Population.from_genomes(genomes, max_nodes=n_in + 1 + 8 + 256,
                                      max_conns=(n_in + 1) * 8 + 2048)
        t1 = time.perf_counter()
        cp = pop.compile(dev)
        torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        print(f"[wave-overhead] from_genomes={t1-t0:.3f}s compile={t2-t1:.3f}s")

    boot0 = time.perf_counter()
    fleet = AuditFleet(n, enc.dim, key_len, args.noop, bool(args.goexplore),
                       args.spin, args.epw, args.parent_wait)
    print(f"[fleet] {fleet.n_workers} workers booting...", flush=True)
    t0 = time.perf_counter()
    fleet.reset_all()
    print(f"[fleet] boot={time.perf_counter()-boot0:.1f}s "
          f"first reset={time.perf_counter()-t0:.2f}s", flush=True)
    # measure a warm reset round too
    t0 = time.perf_counter()
    obs = fleet.reset_all()
    reset_ms = (time.perf_counter() - t0) * 1000
    print(f"[fleet] warm reset_all: {reset_ms:.1f} ms")

    archive = NoveltyArchive()
    go = GoExplore(capacity=2048, rng=np.random.default_rng(1)) if args.goexplore else None
    wave = WaveNovelty(archive, n, mode="rarity", floor=0.1)

    R = args.rounds
    W = 30  # warmup
    cap_flags = np.zeros(n, dtype=np.uint8)
    pending = {}
    actions_full = np.zeros(n, dtype=np.int32)
    rng = np.random.default_rng(2)

    # per-round records
    rec_fwd = np.zeros(R)
    rec_actw = np.zeros(R)
    rec_bar = np.zeros(R)
    rec_book = np.zeros(R)
    rec_capbytes = np.zeros(R)
    rec_hascap = np.zeros(R, dtype=bool)
    # worker decomposition (abs times relative to t_rel)
    rec_wake_max = np.zeros(R); rec_wake_med = np.zeros(R)
    rec_step_max = np.zeros(R); rec_step_mean = np.zeros(R)
    rec_cap_max = np.zeros(R)
    rec_emit_max = np.zeros(R); rec_emit_mean = np.zeros(R)
    rec_lastflag = np.zeros(R)   # max t_emit_done - t_rel (last worker finish)
    rec_tail = np.zeros(R)       # t_done - last worker finish
    per_env_step = np.zeros(n)   # accumulated step time per env
    straggler_hist = np.zeros(n) # times env was the round straggler
    tp = fleet.arr["tprof"]

    for t in range(W + R):
        r = t - W
        c0 = time.perf_counter()
        if use_fwd:
            X = np.ascontiguousarray(obs[:n], dtype=np.float32)
            xt = torch.from_numpy(X).to(dev).unsqueeze(1)
            out = population_forward_sparse(cp, xt, steps=4)
            actions = out[:, 0, :].argmax(dim=1).cpu().numpy()
        else:
            actions = rng.integers(0, 8, size=n)
        c1 = time.perf_counter()
        actions_full[:n] = actions
        obs, keys, dones, captured = fleet.step_all(
            actions_full, cap_flags if go is not None else None)
        c2 = time.perf_counter()

        nbytes = 0
        if go is not None and pending:
            for idx, (key, depth) in pending.items():
                blob = captured.get(idx)
                if blob is not None:
                    nbytes += len(blob)
                    go.store_captured(key, blob, depth)
            pending = {}
        had_cap = cap_flags.any()
        cap_flags[:] = 0
        if not args.noop:
            for i in range(n):
                key = keys[i].tobytes()
                gnew = archive.add(key)
                prior = archive.visit(key)
                wave.observe_key(i, key, gnew, prior)
                if go is not None:
                    if not go.revisit(key) and gnew:
                        cap_flags[i] = 1
                        pending[i] = (key, t)
        c3 = time.perf_counter()

        if r >= 0:
            rec_fwd[r] = c1 - c0
            rec_bar[r] = fleet.t_done - fleet.t_rel
            rec_actw[r] = (c2 - c1) - rec_bar[r]  # step_all minus the barrier
            rec_book[r] = c3 - c2
            rec_capbytes[r] = nbytes
            rec_hascap[r] = had_cap
            trel = fleet.t_rel
            wake = tp[:n, 0] - trel
            capd = tp[:n, 1] - tp[:n, 0]
            stepd = tp[:n, 2] - tp[:n, 1]
            emitd = tp[:n, 3] - tp[:n, 2]
            fin = tp[:n, 3] - trel
            rec_wake_max[r] = wake.max(); rec_wake_med[r] = np.median(wake)
            rec_cap_max[r] = capd.max()
            rec_step_max[r] = stepd.max(); rec_step_mean[r] = stepd.mean()
            rec_emit_max[r] = emitd.max(); rec_emit_mean[r] = emitd.mean()
            rec_lastflag[r] = fin.max()
            rec_tail[r] = fleet.t_done - (trel + fin.max())
            per_env_step += stepd
            straggler_hist[int(np.argmax(fin))] += 1

    ms = 1000.0

    def s(a):
        return f"mean={a.mean()*ms:6.2f} p50={np.percentile(a,50)*ms:6.2f} p90={np.percentile(a,90)*ms:6.2f} p99={np.percentile(a,99)*ms:6.2f}"

    tot = rec_fwd + rec_actw + rec_bar + rec_book
    print(f"\n=== players={n} epw={args.epw} noop={args.noop} goexplore={args.goexplore} "
          f"spin={args.spin} fwd={'gpu' if use_fwd else 'off'} rounds={R} ===")
    print(f"round total   : {s(tot)}   -> {n/tot.mean():7.0f} steps/s")
    print(f" fwd          : {s(rec_fwd)}")
    print(f" actwrite     : {s(rec_actw)}")
    print(f" barrier      : {s(rec_bar)}")
    print(f"   wake max   : {s(rec_wake_max)}  (median worker {rec_wake_med.mean()*ms:.2f})")
    print(f"   cap max    : {s(rec_cap_max)}")
    print(f"   step max   : {s(rec_step_max)}  (mean worker {rec_step_mean.mean()*ms:.2f})")
    print(f"   emit max   : {s(rec_emit_max)}  (mean worker {rec_emit_mean.mean()*ms:.2f})")
    print(f"   lastfinish : {s(rec_lastflag)}")
    print(f"   detecttail : {s(rec_tail)}")
    print(f" book         : {s(rec_book)}")
    ncap = int(rec_hascap.sum())
    if ncap:
        print(f" capture rounds: {ncap}/{R}  barrier(cap)={rec_bar[rec_hascap].mean()*ms:.2f}ms "
              f"barrier(nocap)={rec_bar[~rec_hascap].mean()*ms:.2f}ms  "
              f"capbytes/round={rec_capbytes.mean()/1024:.0f}KB")
    print(f" archive cells={archive.size} go_states={go.size if go else 0}")
    order = _numa_core_order()
    pes = per_env_step / R * ms
    print(" per-env step ms (env@core):")
    line = "  "
    for i in range(n):
        line += f"{i}@{order[(i//args.epw) % len(order)]}:{pes[i]:.1f} "
        if (i + 1) % 8 == 0:
            print(line)
            line = "  "
    if line.strip():
        print(line)
    top = np.argsort(-straggler_hist)[:6]
    print(" straggler top6:", [(int(i), int(straggler_hist[i])) for i in top])
    fleet.close()


if __name__ == "__main__":
    main()
