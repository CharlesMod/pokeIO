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


__all__ = ["VecFleet", "NUMA_NODES"]
