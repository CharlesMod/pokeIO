"""Async free-running fleet prototype ("furnace" engine) — audit only.

No global step barrier. Each env has two shm sequence counters:

  obs_seq[i] = k   worker has published the observation after step k (k=0: reset)
  act_seq[i] = k   parent has published the action to apply to obs k

A worker steps env i as soon as ``act_seq[i] == obs_seq[i]`` (the action for
its newest obs has arrived). The parent loops continuously: snapshot obs_seq,
bookkeep every newly published obs (novelty/goexplore), run ONE GPU forward
over the whole population, and write actions for exactly the envs that were
ready in the snapshot. Store order on x86 (action row, then act_seq) makes the
handshake safe without locks.

Consequences vs the barrier fleet (bench_audit2_fleet):
  * straggler tax gone — a slow env/worker only slows itself, mean not max
  * the GPU forward overlaps worker stepping instead of stalling the fleet
  * a 47 ms Go-Explore save_state stalls one worker's 4 envs, not all 112

Capture protocol (semantics identical to the barrier fleet): the parent sets
cap_flag[i] BEFORE bumping act_seq[i]; the worker checks the flag when it
consumes that action, saves state first, then steps. cap_done[i] hands the
blob back; the parent collects and clears it each cycle.

Wave shape: every env runs exactly --rounds steps, then idles; the wave ends
when the slowest env finishes (tail paid once per wave, not every round).
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
_OP_RUN, _OP_RESET, _OP_SHUTDOWN = 0, 1, 2
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


def _worker_main(slice_lo, slice_hi, goexplore, shm_names, n_envs, obs_dim,
                 key_len, core, no_screens=False):
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
    obs_seq = reg("obs_seq", (n_envs,), np.int64)
    act_seq = reg("act_seq", (n_envs,), np.int64)
    cap_flag = reg("cap_flag", (n_envs,), np.uint8)
    cap_done = reg("cap_done", (n_envs,), np.uint8)
    cap_state = reg("cap_state", (n_envs, _MAX_STATE), np.uint8)
    cap_len = reg("cap_len", (n_envs,), np.int32)
    ctl = reg("ctl", (3 + n_envs,), np.int64)  # [round, op, target, acks...]
    busy = reg("busy", (n_envs,), np.float64)  # accumulated non-idle seconds

    envs = [PokeEnv(ROM, frame_skip=24, hold_frames=8)
            for _ in range(slice_lo, slice_hi)]
    my = list(range(slice_lo, slice_hi))

    def _emit(gi, screen, w64):
        if not no_screens:
            screens[gi] = screen
        obs[gi] = encoder.encode_compact(screen, w64)
        k = archive.cell_key_compact(screen, w64)
        keys[gi] = np.frombuffer(k, dtype=np.uint8)

    local_round = 1
    try:
        while True:
            while ctl[0] < local_round:
                time.sleep(5e-5)
            op = int(ctl[1])
            if op == _OP_SHUTDOWN:
                break
            if op == _OP_RESET:
                for li, gi in enumerate(my):
                    screen = envs[li].reset(STATE)
                    w64 = envs[li].wram_strided(64)
                    _emit(gi, screen, w64)
                    obs_seq[gi] = 0
                    ctl[3 + gi] = local_round
                local_round += 1
                continue
            # _OP_RUN: free-run until every owned env has done `target` steps
            target = int(ctl[2])
            spins = 0
            while True:
                progressed = False
                for li, gi in enumerate(my):
                    k = obs_seq[gi]
                    if k >= target or act_seq[gi] != k:
                        continue
                    t0 = time.perf_counter()
                    env = envs[li]
                    if goexplore and cap_flag[gi]:
                        buf = _io.BytesIO()
                        env.pyboy.save_state(buf)
                        blob = buf.getvalue()
                        nb = min(len(blob), _MAX_STATE)
                        cap_state[gi, :nb] = np.frombuffer(blob[:nb], dtype=np.uint8)
                        cap_len[gi] = nb
                        cap_flag[gi] = 0
                        cap_done[gi] = 1
                    screen, w64, _done = env.step_fast(int(actions[gi]), 64)
                    _emit(gi, screen, w64)
                    obs_seq[gi] = k + 1  # publish AFTER obs/keys are written
                    busy[gi] += time.perf_counter() - t0
                    progressed = True
                if not progressed:
                    if all(obs_seq[gi] >= target for gi in my):
                        break
                    # nap-spin: brief spin for reaction latency, then 50us naps.
                    # (Pure busy-spin measured 2x WORSE here: 28 spinning cores
                    # steal the package turbo/memory budget from stepping ones.)
                    spins += 1
                    if spins >= 200:
                        time.sleep(5e-5)
                else:
                    spins = 0
            for gi in my:
                ctl[3 + gi] = local_round
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


class AsyncFleet:
    def __init__(self, n_envs, obs_dim, key_len, goexplore, epw,
                 no_screens=False):
        self.n_envs = n_envs
        self.goexplore = goexplore
        self._blocks = {}
        self.arr = {}

        def alloc(name, shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
            self._blocks[name] = shm
            self.arr[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            self.arr[name][:] = np.zeros(1, dtype=dtype)

        n = n_envs
        alloc("obs", (n, obs_dim), np.float32)
        alloc("screens", (n, _SCREEN_H, _SCREEN_W), np.uint8)
        alloc("keys", (n, key_len), np.uint8)
        alloc("actions", (n,), np.int32)
        alloc("obs_seq", (n,), np.int64)
        alloc("act_seq", (n,), np.int64)
        alloc("cap_flag", (n,), np.uint8)
        alloc("cap_done", (n,), np.uint8)
        alloc("cap_state", (n, _MAX_STATE), np.uint8)
        alloc("cap_len", (n,), np.int32)
        alloc("ctl", (3 + n,), np.int64)
        alloc("busy", (n,), np.float64)
        self._ctl = self.arr["ctl"]
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
                lo, hi, goexplore, names, n, obs_dim, key_len, core,
                no_screens), daemon=True)
            p.start()
            self._procs.append(p)
            wi += 1
        self.n_workers = len(self._procs)

    def _release(self, op, target=0):
        self._round += 1
        self._ctl[1] = op
        self._ctl[2] = target
        self._ctl[0] = self._round

    def _wait_acks(self):
        acks = self._ctl[3:3 + self.n_envs]
        while not bool((acks >= self._round).all()):
            time.sleep(5e-5)

    def reset_all(self):
        self._release(_OP_RESET)
        self._wait_acks()
        self.arr["act_seq"][:] = -1
        return self.arr["obs"].copy()

    def close(self):
        try:
            self._release(_OP_SHUTDOWN)
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
    ap.add_argument("--players", type=int, default=112)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--goexplore", type=int, default=1)
    ap.add_argument("--epw", type=int, default=4)
    ap.add_argument("--pin-parent", type=int, default=55)
    ap.add_argument("--min-batch", type=int, default=1,
                    help="wait (up to --batch-timeout-us) until this many envs "
                         "are ready before paying the ~3ms fixed forward cost")
    ap.add_argument("--batch-timeout-us", type=int, default=2000)
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the 4-hop sparse forward + argmax as one CUDA "
                         "graph; collapses ~60 kernel launches into one replay")
    ap.add_argument("--fwd-interval-us", type=int, default=0,
                    help="fixed forward cadence: act on whatever is ready every "
                         "N us (replaces --min-batch gating when > 0)")
    ap.add_argument("--no-screens", action="store_true",
                    help="skip the 92KB/step screen copy into shm (dashboard "
                         "only needs screens for focused envs at ~10Hz)")
    args = ap.parse_args()

    if args.pin_parent >= 0:
        psutil.Process().cpu_affinity([args.pin_parent])
    n = args.players
    R = args.rounds

    import torch
    from pokeio.emu.fleet import ObsEncoder, _archive_key_len
    from pokeio.evo.forward import population_forward_sparse
    from pokeio.evo.genome import InnovationTracker, Population, make_genome
    from pokeio.reward.archive import NoveltyArchive
    from pokeio.reward.goexplore import GoExplore
    from pokeio.reward.novelty import WaveNovelty

    enc = ObsEncoder(24, 8)
    a0 = NoveltyArchive()
    ak = dict(screen_cells=a0.screen_cells, screen_levels=a0.screen_levels,
              wram_stride=a0.wram_stride, wram_levels=a0.wram_levels)
    key_len = _archive_key_len(ak, 64)

    dev = torch.device("cuda:1")
    tracker = InnovationTracker(n_in=enc.dim, n_out=8)
    grng = np.random.default_rng(0)
    genomes = [make_genome(enc.dim, 8, tracker, grng, connect="full",
                           weight_scale=1.0) for _ in range(n)]
    pop = Population.from_genomes(genomes, max_nodes=enc.dim + 1 + 8 + 256,
                                  max_conns=(enc.dim + 1) * 8 + 2048)
    cp = pop.compile(dev)
    torch.cuda.synchronize(dev)

    graph_fwd = None
    if args.cuda_graph:
        torch.cuda.set_device(dev)
        x_static = torch.zeros((n, 1, enc.dim), dtype=torch.float32, device=dev)
        warm = torch.cuda.Stream(dev)
        warm.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(warm):
            for _ in range(3):
                out = population_forward_sparse(cp, x_static, steps=4)
                _ = out[:, 0, :].argmax(dim=1)
        torch.cuda.current_stream(dev).wait_stream(warm)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = population_forward_sparse(cp, x_static, steps=4)
            acts_static = out[:, 0, :].argmax(dim=1).to(torch.int32)
        torch.cuda.synchronize(dev)

        def graph_fwd(X_np):
            x_static.copy_(torch.from_numpy(X_np).unsqueeze(1))
            graph.replay()
            return acts_static.cpu().numpy()
        # sanity: replay must agree with the eager path
        Xs = np.random.default_rng(3).random((n, enc.dim), dtype=np.float32)
        got = graph_fwd(Xs)
        ref = population_forward_sparse(
            cp, torch.from_numpy(Xs).unsqueeze(1).to(dev), steps=4
        )[:, 0, :].argmax(dim=1).cpu().numpy()
        assert (got == ref).all(), "CUDA graph forward diverges from eager"
        print("[graph] capture OK, replay matches eager")

    boot0 = time.perf_counter()
    fleet = AsyncFleet(n, enc.dim, key_len, bool(args.goexplore), args.epw,
                       no_screens=args.no_screens)
    print(f"[fleet] {fleet.n_workers} workers booting...", flush=True)
    fleet.reset_all()
    print(f"[fleet] boot+reset={time.perf_counter()-boot0:.1f}s", flush=True)

    archive = NoveltyArchive()
    go = GoExplore(capacity=2048, rng=np.random.default_rng(1)) if args.goexplore else None
    wave = WaveNovelty(archive, n, mode="rarity", floor=0.1)

    obs_seq = fleet.arr["obs_seq"]
    act_seq = fleet.arr["act_seq"]
    actions = fleet.arr["actions"]
    keys = fleet.arr["keys"]
    obs = fleet.arr["obs"]
    cap_done = fleet.arr["cap_done"]
    cap_len = fleet.arr["cap_len"]
    cap_state = fleet.arr["cap_state"]
    cap_flag = fleet.arr["cap_flag"]

    acted = np.full(n, -1, dtype=np.int64)   # last obs index we acted on
    booked = np.zeros(n, dtype=np.int64)     # obs index bookkept through
    pending = {}                             # env -> (key, depth) awaiting blob

    n_cycles = 0
    n_fwd = 0
    batch_sizes = []
    t_fwd = 0.0
    t_book = 0.0
    t_wall0 = time.perf_counter()
    t_last_fwd = t_wall0
    fleet._release(_OP_RUN, target=R)

    xt = torch.empty((n, 1, enc.dim), dtype=torch.float32, device=dev)

    while True:
        snap = obs_seq.copy()
        n_cycles += 1

        # ---- bookkeeping: every obs published since we last looked
        b0 = time.perf_counter()
        for i in np.nonzero(snap > booked)[0]:
            for k in range(int(booked[i]) + 1, int(snap[i]) + 1):
                key = keys[i].tobytes()  # newest obs only carries newest key;
                # intermediate keys are unobservable at this cadence — in
                # practice the parent laps every env each cycle (snap-booked==1)
                gnew = archive.add(key)
                prior = archive.visit(key)
                wave.observe_key(int(i), key, gnew, prior)
                if go is not None and not go.revisit(key) and gnew \
                        and k < R and not cap_flag[i] and not cap_done[i]:
                    cap_flag[i] = 1
                    pending[int(i)] = (key, int(snap[i]))
            booked[i] = snap[i]
        if go is not None:
            for i in np.nonzero(cap_done)[0]:
                ii = int(i)
                if ii in pending:
                    key, depth = pending.pop(ii)
                    go.store_captured(key, bytes(cap_state[ii, :int(cap_len[ii])]),
                                      depth)
                cap_done[ii] = 0
        t_book += time.perf_counter() - b0

        # ---- act on every env whose newest obs has no action yet
        ready = (snap > acted) & (snap < R)
        n_ready = int(ready.sum())
        remaining = int((acted < R - 1).sum())  # envs still owed any action
        if args.fwd_interval_us:
            # fixed-cadence clock: act on whatever is ready every interval
            if n_ready and (time.perf_counter() - t_last_fwd) * 1e6 \
                    < args.fwd_interval_us:
                continue  # busy-spin until the next tick
        elif n_ready and n_ready < min(args.min_batch, remaining) \
                and (time.perf_counter() - t_last_fwd) * 1e6 < args.batch_timeout_us:
            continue  # busy-spin; parent core must stay clocked up too
        if ready.any():
            f0 = time.perf_counter()
            X = np.ascontiguousarray(obs, dtype=np.float32)
            if graph_fwd is not None:
                acts = graph_fwd(X)
            else:
                xt.copy_(torch.from_numpy(X).unsqueeze(1).to(dev, non_blocking=True))
                out = population_forward_sparse(cp, xt, steps=4)
                acts = out[:, 0, :].argmax(dim=1).cpu().numpy().astype(np.int32)
            idx = np.nonzero(ready)[0]
            actions[idx] = acts[idx]
            act_seq[idx] = snap[idx]        # store AFTER actions (x86 TSO)
            acted[idx] = snap[idx]
            n_fwd += 1
            batch_sizes.append(len(idx))
            t_last_fwd = time.perf_counter()
            t_fwd += t_last_fwd - f0
        elif (snap >= R).all() and not pending:
            break
        # (no else-nap: busy-spin until the next obs lands)

    t_wall = time.perf_counter() - t_wall0
    fleet._wait_acks()
    busy = fleet.arr["busy"].copy()

    bs = np.array(batch_sizes)
    steps = n * R
    print(f"\n=== ASYNC players={n} epw={args.epw} goexplore={args.goexplore} "
          f"rounds={R} ===")
    print(f"wall={t_wall:.2f}s  ->  {steps / t_wall:7.0f} steps/s")
    print(f"parent: cycles={n_cycles} fwds={n_fwd} "
          f"fwd_mean={t_fwd / max(n_fwd, 1) * 1e3:.2f}ms fwd_total={t_fwd:.1f}s "
          f"book_total={t_book:.1f}s")
    print(f"batch size: mean={bs.mean():.1f} p50={np.percentile(bs, 50):.0f} "
          f"p10={np.percentile(bs, 10):.0f} p90={np.percentile(bs, 90):.0f}")
    print(f"env busy fraction: mean={busy.mean() / t_wall:.2f} "
          f"min={busy.min() / t_wall:.2f} max={busy.max() / t_wall:.2f} "
          f"(fraction of wall each env spent stepping+emitting)")
    print(f"archive cells={archive.size} go_states={go.size if go else 0} "
          f"fitness mean={wave.fitness.mean():.2f}")
    fleet.close()


if __name__ == "__main__":
    main()
