"""VecFleet — a NUMA-pinned, shared-memory vectorized fleet of VisionEnvs.

N worker processes each run a ``VisionEnv``. Observations are transferred to the
parent through POSIX shared memory (``multiprocessing.shared_memory``), not
pipes: each obs field is one big float32 block of shape ``(n_envs, *field)`` and
a worker writes only its own row. Small control messages (step/reset/shutdown +
acks carrying ``done``/``info``) go over per-worker duplex pipes.

Features
--------
* **Shared-memory obs** — coarse / fovea / motion / ram_aux, zero-copy per row.
* **NUMA pinning** — workers are round-robin bound across the two sockets via
  ``psutil.cpu_affinity`` so both memory controllers are exercised
  (node0 = cores 0-13,28-41; node1 = 14-27,42-55 on this box).
* **Dead-worker restart** — a crashed/hung worker is detected on step/reset,
  respawned, and restored to the fleet's reset state.
* **Clean shutdown** — context manager; unlinks the shared memory on close.

API
---
    fleet = VecFleet(n_envs, rom_path, config, state_path=...)
    obs = fleet.reset_all(state_path=None)          # -> batched obs dict
    obs, dones, infos = fleet.step_all(actions)     # actions: len-n_envs ints
    fleet.close()                                   # or use `with`
"""

from __future__ import annotations

import os

# Each worker is pinned to a single core, so multi-threaded BLAS is pure
# oversubscription: 56 procs x N BLAS threads collapses throughput ~5x. Cap the
# numeric thread pools to 1 BEFORE numpy imports (spawn children inherit these).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from multiprocessing import shared_memory  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from pokeio.config import Config
from pokeio.vision.preprocess import ObsBuilder

# NUMA topology of the target box (2x Xeon E5-2690 v4). Kept here so pinning is
# explicit; if a host differs, affinity simply falls back to all-cores.
NUMA_NODES = {
    0: list(range(0, 14)) + list(range(28, 42)),
    1: list(range(14, 28)) + list(range(42, 56)),
}

# Field dtype for all obs sheets.
_DTYPE = np.float32

# 0xC000..0xE000 == 8192 bytes of WRAM (see emu/env.py).
WRAM_END_LEN = 8192
_ACK_TIMEOUT = 30.0  # seconds to wait for a worker ack before declaring it dead


def _numa_core_order() -> list[int]:
    """Interleave the two NUMA nodes so consecutive workers alternate sockets."""
    n0, n1 = NUMA_NODES[0], NUMA_NODES[1]
    order: list[int] = []
    for a, b in zip(n0, n1):
        order.append(a)
        order.append(b)
    # In case the nodes are uneven, append any leftovers.
    order.extend(n0[len(n1):])
    order.extend(n1[len(n0):])
    return order


def _worker_main(widx, rom_path, cfg_dict, state_path, conn, shm_names, shapes,
                 aux_names, core):
    """Worker process: run a VisionEnv, write obs rows into shared memory.

    Per-step scalars (done, map_id) are written into small shared arrays so the
    ack sent back over the pipe can stay a bare sentinel (cheap to pickle).
    """
    # Late imports so the child starts clean under 'spawn'.
    from pokeio.config import Config as _Config
    from pokeio.vision.env_wrap import VisionEnv

    # NUMA / core pinning (best effort).
    try:
        if core is not None:
            psutil.Process().cpu_affinity([core])
    except Exception:
        pass

    cfg = _Config.from_dict(cfg_dict)
    # Attach shared-memory blocks as numpy views.
    shms = {k: shared_memory.SharedMemory(name=shm_names[k]) for k in shm_names}
    arrs = {
        k: np.ndarray(shapes[k], dtype=_DTYPE, buffer=shms[k].buf) for k in shm_names
    }
    dones_shm = shared_memory.SharedMemory(name=aux_names["dones"])
    mapids_shm = shared_memory.SharedMemory(name=aux_names["map_ids"])
    n = shapes[next(iter(shapes))][0]
    dones = np.ndarray((n,), dtype=np.uint8, buffer=dones_shm.buf)
    map_ids = np.ndarray((n,), dtype=np.int32, buffer=mapids_shm.buf)

    env = VisionEnv(rom_path, config=cfg)

    def _write(obs):
        for k in arrs:
            arrs[k][widx] = obs[k]

    try:
        while True:
            try:
                cmd = conn.recv()
            except EOFError:
                break
            op = cmd[0]
            if op == "shutdown":
                break
            elif op == "reset":
                sp = cmd[1] if cmd[1] is not None else state_path
                obs = env.reset(sp)
                _write(obs)
                dones[widx] = 0
                conn.send(True)
            elif op == "step":
                action = cmd[1]
                obs, _ram, done, info = env.step(action)
                _write(obs)
                dones[widx] = 1 if done else 0
                map_ids[widx] = int(info.get("map_id", -1))
                conn.send(True)
            else:  # pragma: no cover - defensive
                conn.send(False)
    finally:
        try:
            env.close()
        except Exception:
            pass
        for s in (*shms.values(), dones_shm, mapids_shm):
            s.close()


class VecFleet:
    """A vectorized fleet of NUMA-pinned VisionEnv workers with shared-mem obs."""

    def __init__(
        self,
        n_envs: int,
        rom_path: str | None = None,
        config: Config | None = None,
        state_path: str | None = None,
    ):
        self.n_envs = int(n_envs)
        self.config = config or Config()
        self.rom_path = rom_path or self.config.emu.rom_path
        self.state_path = state_path or self.config.emu.reset_state or None
        self._cfg_dict = self.config.to_dict()

        # Obs geometry comes straight from ObsBuilder so it can never drift.
        builder = ObsBuilder(self.config)
        self._field_shapes = builder.shapes  # per-env shapes
        self.obs_keys = list(self._field_shapes.keys())

        # Allocate one shared-memory block per field: (n_envs, *field_shape).
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._shm: dict[str, shared_memory.SharedMemory] = {}
        self._arr: dict[str, np.ndarray] = {}
        for k, fshape in self._field_shapes.items():
            shape = (self.n_envs, *fshape)
            nbytes = int(np.prod(shape)) * np.dtype(_DTYPE).itemsize
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self._shapes[k] = shape
            self._shm[k] = shm
            self._arr[k] = np.ndarray(shape, dtype=_DTYPE, buffer=shm.buf)
        self._shm_names = {k: self._shm[k].name for k in self.obs_keys}

        # Small shared aux arrays for per-step scalars (keeps acks tiny).
        self._dones_shm = shared_memory.SharedMemory(
            create=True, size=self.n_envs * np.dtype(np.uint8).itemsize
        )
        self._mapids_shm = shared_memory.SharedMemory(
            create=True, size=self.n_envs * np.dtype(np.int32).itemsize
        )
        self._dones = np.ndarray((self.n_envs,), dtype=np.uint8, buffer=self._dones_shm.buf)
        self._map_ids = np.ndarray((self.n_envs,), dtype=np.int32, buffer=self._mapids_shm.buf)
        self._aux_names = {"dones": self._dones_shm.name, "map_ids": self._mapids_shm.name}

        self._ctx = mp.get_context("spawn")
        self._core_order = _numa_core_order()
        self._parent_conn: list = [None] * self.n_envs
        self._procs: list = [None] * self.n_envs
        self._closed = False

        for i in range(self.n_envs):
            self._spawn_worker(i)

    # ------------------------------------------------------------------ workers
    def _core_for(self, widx: int) -> int | None:
        if not self._core_order:
            return None
        return self._core_order[widx % len(self._core_order)]

    def _spawn_worker(self, i: int) -> None:
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        p = self._ctx.Process(
            target=_worker_main,
            args=(
                i,
                self.rom_path,
                self._cfg_dict,
                self.state_path,
                child_conn,
                self._shm_names,
                self._shapes,
                self._aux_names,
                self._core_for(i),
            ),
            daemon=True,
        )
        p.start()
        child_conn.close()  # parent keeps only its end
        self._parent_conn[i] = parent_conn
        self._procs[i] = p

    def _restart_worker(self, i: int) -> None:
        """Respawn a dead/hung worker and restore it to the reset state."""
        p = self._procs[i]
        if p is not None and p.is_alive():
            p.terminate()
            p.join(timeout=5)
        try:
            if self._parent_conn[i] is not None:
                self._parent_conn[i].close()
        except Exception:
            pass
        self._spawn_worker(i)
        # Bring the fresh worker to the canonical state so the batch stays aligned.
        conn = self._parent_conn[i]
        conn.send(("reset", None))
        self._recv_or_die(i, allow_restart=False)

    def _send_or_restart(self, i: int, cmd) -> bool:
        """Send a command to worker i. On a broken pipe, restart it.

        Returns True if the command was delivered (expect a normal ack), or
        False if the worker was dead and got restarted (its shared-mem row now
        holds the reset obs; caller should skip the recv for this index).
        """
        try:
            self._parent_conn[i].send(cmd)
            return True
        except (BrokenPipeError, OSError, EOFError):
            self._restart_worker(i)
            return False

    def _recv_or_die(self, i: int, allow_restart: bool = True):
        """Receive one ack from worker i; restart it on timeout/EOF/crash.

        Returns True on a clean ack, or the "restarted" sentinel (not True) if
        the worker was dead/unresponsive and (optionally) got respawned. After a
        restart the shared-mem row holds the reset obs.
        """
        conn = self._parent_conn[i]
        try:
            if conn.poll(_ACK_TIMEOUT) and self._procs[i].is_alive():
                return conn.recv()
        except (EOFError, OSError):
            pass
        if allow_restart:
            self._restart_worker(i)
        return "restarted"

    # ------------------------------------------------------------------ control
    def reset_all(self, state_path: str | None = None) -> dict[str, np.ndarray]:
        """Reset every worker; return the batched obs dict."""
        self._check_open()
        delivered = [self._send_or_restart(i, ("reset", state_path)) for i in range(self.n_envs)]
        for i in range(self.n_envs):
            if delivered[i]:
                self._recv_or_die(i)
        return self._batched_obs()

    def step_all(self, actions):
        """Step every worker with its action; return (obs, dones, infos)."""
        self._check_open()
        acts = list(actions)
        if len(acts) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions, got {len(acts)}")
        delivered = [self._send_or_restart(i, ("step", int(acts[i]))) for i in range(self.n_envs)]
        restarted = np.zeros(self.n_envs, dtype=bool)
        for i in range(self.n_envs):
            if delivered[i]:
                ok = self._recv_or_die(i)
                if ok is not True:  # died/restarted during recv
                    restarted[i] = True
            else:
                restarted[i] = True
        # Scalars come from the shared aux arrays (workers wrote them directly).
        dones = self._dones.astype(bool)
        dones[restarted] = True  # a restarted env is treated as an episode end
        infos = [
            {"map_id": int(self._map_ids[i]), "restarted": bool(restarted[i])}
            for i in range(self.n_envs)
        ]
        return self._batched_obs(), dones, infos

    def _batched_obs(self) -> dict[str, np.ndarray]:
        """Copy each shared block out so callers are safe across the next step."""
        return {k: self._arr[k].copy() for k in self.obs_keys}

    # ------------------------------------------------------------------ shutdown
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("VecFleet is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for i in range(self.n_envs):
            conn = self._parent_conn[i]
            try:
                if conn is not None:
                    conn.send(("shutdown",))
            except Exception:
                pass
        for i in range(self.n_envs):
            p = self._procs[i]
            if p is not None:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            try:
                if self._parent_conn[i] is not None:
                    self._parent_conn[i].close()
            except Exception:
                pass
        for shm in (*self._shm.values(), self._dones_shm, self._mapids_shm):
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):  # best-effort safety net
        try:
            self.close()
        except Exception:
            pass


# ==========================================================================
# ObsEncoder — small screen+RAM observation encoder (numpy-only, torch-free)
# ==========================================================================
# Lives here (not train/loop.py) so the barrier workers can import it WITHOUT
# pulling torch into 32 spawned processes. loop.py re-imports it from here.
_SCREEN_H = 144
_SCREEN_W = 160


def _area_matrix(in_size: int, out_size: int) -> np.ndarray:
    """(out_size, in_size) row-normalized area-overlap resample matrix."""
    m = np.zeros((out_size, in_size), dtype=np.float64)
    scale = in_size / out_size
    for i in range(out_size):
        lo, hi = i * scale, (i + 1) * scale
        j0, j1 = int(np.floor(lo)), int(np.ceil(hi))
        for j in range(j0, min(j1, in_size)):
            overlap = min(hi, j + 1) - max(lo, j)
            if overlap > 0:
                m[i, j] = overlap
        s = m[i].sum()
        if s > 0:
            m[i] /= s
    return m


class ObsEncoder:
    """Raw (144,160) uint8 screen -> flat [obs_res*obs_res + n_ram] float32 obs."""

    def __init__(self, res: int, n_ram: int) -> None:
        self.res = int(res)
        self.n_ram = int(n_ram)
        self._row = _area_matrix(_SCREEN_H, self.res)  # (res, H)
        self._col = _area_matrix(_SCREEN_W, self.res).T  # (W, res)
        self.dim = self.res * self.res + self.n_ram

    def encode(self, screen: np.ndarray, wram: np.ndarray) -> np.ndarray:
        small = self._row @ screen.astype(np.float64) @ self._col  # (res,res) 0..255
        vis = (small / 255.0).astype(np.float32).ravel()
        if self.n_ram > 0:
            stride = max(1, wram.size // self.n_ram)
            ram = (wram[::stride][: self.n_ram].astype(np.float32)) / 255.0
            if ram.size < self.n_ram:  # pad if short
                ram = np.concatenate([ram, np.zeros(self.n_ram - ram.size, np.float32)])
            return np.concatenate([vis, ram])
        return vis

    def encode_compact(self, screen: np.ndarray, wram_strided: np.ndarray) -> np.ndarray:
        """Encode from a pre-strided WRAM slice (the worker hot path).

        Byte-identical to :meth:`encode` when ``wram_strided`` is
        ``raw_wram()[::wram_stride]`` and ``wram_stride`` divides the encoder's
        own RAM stride (true for the defaults: obs samples every 1024th byte,
        the archive every 64th, and 1024 % 64 == 0).
        """
        small = self._row @ screen.astype(np.float64) @ self._col
        vis = (small / 255.0).astype(np.float32).ravel()
        if self.n_ram > 0:
            stride = max(1, wram_strided.size // self.n_ram)
            ram = (wram_strided[::stride][: self.n_ram].astype(np.float32)) / 255.0
            if ram.size < self.n_ram:
                ram = np.concatenate([ram, np.zeros(self.n_ram - ram.size, np.float32)])
            return np.concatenate([vis, ram])
        return vis


# ==========================================================================
# BarrierFleet — shared-memory, spin-barrier, in-worker hashing/encoding
# ==========================================================================
# Ops signalled to workers via the shared control block.
_OP_STEP = 0
_OP_RESET = 1
_OP_SHUTDOWN = 2

# Per-env emulator save_state blob is ~200 KB; bound the shared capture/restore
# buffers a little above that. (Go-Explore state transport.)
_MAX_STATE = 262144  # 256 KiB


def _spin_wait(pred, spin_budget: int = 3000, nap: float = 5e-5) -> None:
    """Wait on ``pred`` (a cheap shm read): brief busy spin, then yield.

    The barrier flags live in shared memory, so the wait is a tight numpy read
    rather than a blocking pipe round-trip (which caps the pipe-based VecFleet at
    ~2.9k steps/s).  Lockstep means that whenever one side is waiting the OTHER
    side is doing the actual work (a ~2 ms step or forward), so a long busy spin
    would just steal cores from the active side — costly on a shared box (live1 +
    sibling agents).  A short spin keeps hand-off latency low, then we sleep a
    few tens of µs so a waiting process frees its core.
    """
    i = 0
    while not pred():
        i += 1
        if i >= spin_budget:
            time.sleep(nap)


def _barrier_worker_main(
    slice_lo, slice_hi, rom_path, frame_skip, hold_frames, reset_state,
    obs_res, obs_ram, wram_stride, arch_kwargs, goexplore,
    shm_names, n_envs, obs_dim, key_len, core,
):
    """Worker process: owns envs ``[slice_lo:slice_hi]``; hashes + encodes locally.

    Per barrier round the worker advances each of its envs one agent-step, writes
    the compact obs + the novelty cell key (both computed here, off the parent's
    critical path) into shared memory, then flips its barrier flag.  For
    Go-Explore it (a) restores a provided state at reset and (b) captures its
    current state on request (one round after the parent flags a globally-new
    cell, while the env still sits in that exact state).
    """
    # Late imports so the child starts clean under 'spawn' (torch is never
    # imported in a worker — only numpy / pyboy / the reward hashing).
    import io as _io

    from pokeio.emu.env import PokeEnv
    from pokeio.reward.archive import NoveltyArchive

    try:
        if core is not None:
            psutil.Process().cpu_affinity([core])
    except Exception:
        pass

    encoder = ObsEncoder(obs_res, obs_ram)
    archive = NoveltyArchive(**arch_kwargs)  # used only as a stateless key hasher

    def _attach(name, shape, dtype):
        shm = shared_memory.SharedMemory(name=name)
        return shm, np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    shms = []

    def reg(name, shape, dtype):
        shm, arr = _attach(shm_names[name], shape, dtype)
        shms.append(shm)
        return arr

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
    ctl = reg("ctl", (2 + n_envs,), np.int64)  # [0]=go_round [1]=op [2+i]=wdone_i

    envs = [
        PokeEnv(rom_path, frame_skip=frame_skip, hold_frames=hold_frames)
        for _ in range(slice_lo, slice_hi)
    ]

    def _emit(local_i, global_i, screen, w64):
        screens[global_i] = screen
        obs[global_i] = encoder.encode_compact(screen, w64)
        k = archive.cell_key_compact(screen, w64)
        keys[global_i] = np.frombuffer(k, dtype=np.uint8)

    local_round = 1
    try:
        while True:
            _spin_wait(lambda: ctl[0] >= local_round)
            op = int(ctl[1])
            if op == _OP_SHUTDOWN:
                break
            for li, gi in enumerate(range(slice_lo, slice_hi)):
                env = envs[li]
                if op == _OP_RESET:
                    if goexplore and res_flag[gi]:
                        env.load_state(bytes(res_state[gi, : int(res_len[gi])]))
                        env.pyboy.tick(1, True)
                        screen = env._obs()
                        w64 = env.wram_strided(wram_stride)
                    else:
                        screen = env.reset(reset_state)
                        w64 = env.wram_strided(wram_stride)
                    dones[gi] = 0
                    _emit(li, gi, screen, w64)
                else:  # _OP_STEP
                    # Deferred Go-Explore capture: save the state we are STILL in
                    # (from last round) before applying this round's action.
                    if goexplore and cap_flag[gi]:
                        buf = _io.BytesIO()
                        env.pyboy.save_state(buf)
                        blob = buf.getvalue()
                        n = min(len(blob), _MAX_STATE)
                        cap_state[gi, :n] = np.frombuffer(blob[:n], dtype=np.uint8)
                        cap_len[gi] = n
                        cap_done[gi] = 1
                    else:
                        cap_done[gi] = 0
                    screen, w64, done = env.step_fast(int(actions[gi]), wram_stride)
                    dones[gi] = 1 if done else 0
                    _emit(li, gi, screen, w64)
            # signal this round complete
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


class BarrierFleet:
    """Shared-memory, spin-barrier fleet of PokeEnv workers.

    Workers own contiguous slices of the ``n_envs`` emulators (``envs_per_worker``
    each), advance them in lockstep, and hash/encode observations locally so the
    parent's per-round work is only: one batched GPU forward, cheap archive
    dict-ops, and (optionally) Go-Explore state transport.  The parent drives the
    barrier with a monotonic round counter in shared memory — no per-step pickled
    pipe round-trips.
    """

    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        obs_res: int,
        obs_ram: int,
        rom_path: str,
        frame_skip: int,
        hold_frames: int,
        reset_state: str,
        archive_kwargs: dict,
        wram_stride: int = 64,
        goexplore: bool = False,
        envs_per_worker: int = 1,
    ):
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.goexplore = bool(goexplore)
        self.wram_stride = int(wram_stride)

        # Derive the fixed cell-key length from the archive's geometry.
        probe = _archive_key_len(archive_kwargs, wram_stride)
        self.key_len = probe

        # ------------------------------------------------------------ shm
        self._blocks: dict[str, shared_memory.SharedMemory] = {}
        self.arr: dict[str, np.ndarray] = {}

        def alloc(name, shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
            self._blocks[name] = shm
            self.arr[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        n = self.n_envs
        alloc("obs", (n, self.obs_dim), np.float32)
        alloc("screens", (n, _SCREEN_H, _SCREEN_W), np.uint8)
        alloc("keys", (n, self.key_len), np.uint8)
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

        self._ctl = self.arr["ctl"]
        self._ctl[:] = 0
        self._round = 0
        self._shm_names = {k: v.name for k, v in self._blocks.items()}

        # ------------------------------------------------------------ workers
        self._ctx = mp.get_context("spawn")
        core_order = _numa_core_order()
        epw = max(1, int(envs_per_worker))
        self._procs: list = []
        self._slices: list[tuple[int, int]] = []
        wi = 0
        for lo in range(0, n, epw):
            hi = min(lo + epw, n)
            core = core_order[wi % len(core_order)] if core_order else None
            p = self._ctx.Process(
                target=_barrier_worker_main,
                args=(
                    lo, hi, rom_path, frame_skip, hold_frames, reset_state,
                    obs_res, obs_ram, self.wram_stride, dict(archive_kwargs),
                    self.goexplore, self._shm_names, n, self.obs_dim,
                    self.key_len, core,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            self._slices.append((lo, hi))
            wi += 1
        self.n_workers = len(self._procs)
        self._closed = False

    # ------------------------------------------------------------------ barrier
    def _run_round(self, op: int) -> None:
        self._round += 1
        self._ctl[1] = op
        self._ctl[0] = self._round  # release workers
        target = self._round
        ctl = self._ctl
        n = self.n_envs
        wdone = ctl[2:2 + n]
        i = 0
        while not bool((wdone >= target).all()):
            i += 1
            if i >= 3000:
                # Liveness guard: a crashed worker can never flip its flag, so
                # spinning forever would wedge the whole run. Check periodically.
                if i % 20000 == 0 and any(not p.is_alive() for p in self._procs):
                    raise RuntimeError(
                        "BarrierFleet worker died mid-round; aborting (see stderr)"
                    )
                time.sleep(5e-5)

    # ------------------------------------------------------------------ control
    def reset_all(self, restore: dict[int, bytes] | None = None) -> np.ndarray:
        """Reset all envs (optionally restoring per-index Go-Explore states).

        ``restore`` maps env index -> save_state blob; those envs load the blob
        instead of the canonical reset state.  Returns the encoded obs block
        (a copy, safe across the next round)."""
        self.arr["res_flag"][:] = 0
        if restore and self.goexplore:
            for idx, blob in restore.items():
                b = blob[:_MAX_STATE]
                self.arr["res_state"][idx, : len(b)] = np.frombuffer(b, dtype=np.uint8)
                self.arr["res_len"][idx] = len(b)
                self.arr["res_flag"][idx] = 1
        self._run_round(_OP_RESET)
        return self.arr["obs"].copy()

    def step_all(self, actions: np.ndarray, capture_flags: np.ndarray | None = None):
        """Advance one agent-step; return ``(obs, keys, dones, captured)``.

        ``capture_flags`` (len n, uint8) requests a Go-Explore state capture for
        the flagged envs BEFORE they apply this step's action — i.e. it captures
        the state reported in the *previous* round.  ``captured`` maps env index
        -> state bytes for envs that captured this round.
        """
        self.arr["actions"][: len(actions)] = np.asarray(actions, dtype=np.int32)
        if self.goexplore and capture_flags is not None:
            self.arr["cap_flag"][:] = capture_flags
        else:
            self.arr["cap_flag"][:] = 0
        self._run_round(_OP_STEP)
        obs = self.arr["obs"]
        keys = self.arr["keys"]
        dones = self.arr["dones"].astype(bool)
        captured: dict[int, bytes] = {}
        if self.goexplore:
            cd = self.arr["cap_done"]
            cl = self.arr["cap_len"]
            cs = self.arr["cap_state"]
            for i in range(self.n_envs):
                if cd[i]:
                    captured[i] = bytes(cs[i, : int(cl[i])])
        return obs, keys, dones, captured

    @property
    def screens(self) -> np.ndarray:
        """Live view of the current per-env raw screens (for the live swarm)."""
        return self.arr["screens"]

    def key_bytes(self, i: int) -> bytes:
        return self.arr["keys"][i].tobytes()

    # ------------------------------------------------------------------ shutdown
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _archive_key_len(archive_kwargs: dict, wram_stride: int) -> int:
    """Exact byte length of a compact cell key for the given archive geometry."""
    from pokeio.reward.archive import NoveltyArchive

    a = NoveltyArchive(**archive_kwargs)
    dummy_screen = np.zeros((_SCREEN_H, _SCREEN_W), dtype=np.uint8)
    dummy_w = np.zeros((WRAM_END_LEN // wram_stride,), dtype=np.uint8)
    return len(a.cell_key_compact(dummy_screen, dummy_w))


__all__ = ["VecFleet", "BarrierFleet", "ObsEncoder", "NUMA_NODES"]
