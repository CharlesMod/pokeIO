"""Vectorized environments.

``ProcVec`` is an EnvPool/PufferLib-style asynchronous pool:
  * ``num_workers`` processes, each owning ``envs_per_worker`` GameEnvs;
  * observations, rewards, dones and actions live in shared memory (uint8-packed
    pixels, so a step moves a few KB per env, not float frames);
  * ``recv()`` returns the first ``batch_workers`` workers that finished, so while
    the GPU computes actions for one batch the other workers keep emulating.
    With batch_workers == num_workers it degrades to a synchronous pool.

Only small dicts (infos) and rare save-states travel over pipes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing.connection import wait
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from pokeio.env import GameEnv, ObsSpec
from pokeio.spec import GameSpec


class Batch:
    __slots__ = ("env_ids", "obs", "rewards", "dones", "infos")

    def __init__(self, env_ids, obs, rewards, dones, infos):
        self.env_ids: np.ndarray = env_ids
        self.obs: dict[str, np.ndarray] = obs
        self.rewards: np.ndarray = rewards
        # uint8 per env: 0 = continuing, 1 = episode ended (obs is a fresh start),
        # 2 = state was swapped by the swarm (the sent action was NOT executed;
        # the trainer masks that transition out of the loss).
        self.dones: np.ndarray = dones
        self.infos: list[dict[str, Any]] = infos  # each has "env_id"


def probe_obs_spec(spec: GameSpec) -> ObsSpec:
    env = GameEnv(spec, env_id=0)
    try:
        return env.obs_spec
    finally:
        env.close()


class SerialVec:
    """All envs in-process. For debugging, evaluation and tests."""

    def __init__(self, spec: GameSpec, num_envs: int, seed: int = 0, headless: bool = True):
        self.envs = [GameEnv(spec, env_id=i, seed=seed, headless=headless) for i in range(num_envs)]
        self.num_envs = num_envs
        self.obs_spec = self.envs[0].obs_spec
        self.batch_size = num_envs
        self._obs: list[dict] = []
        self._pending: dict[int, bytes] = {}
        self._last = None

    def reset(self) -> None:
        self._obs = [e.reset() for e in self.envs]
        self._last = (np.zeros(self.num_envs, np.float32), np.zeros(self.num_envs, np.uint8), [])

    def recv(self) -> Batch:
        rewards, dones, infos = self._last
        obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
        return Batch(np.arange(self.num_envs), obs, rewards, dones, infos)

    def send(self, actions: np.ndarray, env_ids: np.ndarray | None = None) -> None:
        rewards = np.zeros(self.num_envs, np.float32)
        dones = np.zeros(self.num_envs, np.uint8)
        infos = []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            if i in self._pending:
                self._obs[i] = env.load_state(self._pending.pop(i))
                dones[i] = 2
                continue
            self._obs[i], rewards[i], dones[i], info = env.step(int(a))
            if info:
                info["env_id"] = i
                infos.append(info)
        self._last = (rewards, dones, infos)

    def load_states(self, assignments: dict[int, bytes]) -> None:
        self._pending.update(assignments)

    def close(self) -> None:
        for e in self.envs:
            e.close()


# ---------------------------------------------------------------------------
# multiprocessing


def _shm_arrays(shm: SharedMemory, layout: dict[str, tuple[tuple[int, ...], np.dtype, int]]):
    return {
        k: np.ndarray(shape, dtype=dt, buffer=shm.buf, offset=off) for k, (shape, dt, off) in layout.items()
    }


def _layout(obs_spec: ObsSpec, n: int):
    items = {f"obs.{k}": ((n, *shape), dt) for k, (shape, dt) in obs_spec.arrays().items()}
    items["rewards"] = ((n,), np.dtype(np.float32))
    items["dones"] = ((n,), np.dtype(np.uint8))
    items["actions"] = ((n,), np.dtype(np.int32))
    layout, off = {}, 0
    for k, (shape, dt) in items.items():
        off = (off + 63) // 64 * 64
        layout[k] = (shape, dt, off)
        off += int(np.prod(shape)) * dt.itemsize
    return layout, off


def _worker(wid, spec, env_ids, seed, shm_name, layout, conn, cpu):
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError:
            pass
    shm = SharedMemory(name=shm_name)
    arr = _shm_arrays(shm, layout)
    envs = [GameEnv(spec, env_id=i, seed=seed) for i in env_ids]
    keys = [k[4:] for k in layout if k.startswith("obs.")]

    def write(i, obs):
        for k in keys:
            arr["obs." + k][i] = obs[k]

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "step":
                infos = []
                for j, i in enumerate(env_ids):
                    st = payload.get(i) if payload else None
                    if st is not None:
                        obs = envs[j].load_state(st)
                        arr["rewards"][i] = 0.0
                        arr["dones"][i] = 2
                        write(i, obs)
                        continue
                    obs, r, d, info = envs[j].step(int(arr["actions"][i]))
                    write(i, obs)
                    arr["rewards"][i] = r
                    arr["dones"][i] = d
                    if info:
                        info["env_id"] = i
                        infos.append(info)
                conn.send(infos)
            elif cmd == "reset":
                for j, i in enumerate(env_ids):
                    write(i, envs[j].reset())
                    arr["rewards"][i] = 0.0
                    arr["dones"][i] = 0
                conn.send([])
            elif cmd == "call":  # (method, kwargs) on every env -> list of results
                method, kwargs = payload
                conn.send([getattr(e, method)(**kwargs) for e in envs])
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        for e in envs:
            e.close()
        del arr
        shm.close()


class ProcVec:
    def __init__(
        self,
        spec: GameSpec,
        num_workers: int,
        envs_per_worker: int,
        batch_workers: int | None = None,
        seed: int = 0,
        pin_cpus: list[int] | None = None,
    ):
        self.spec = spec
        self.num_workers = num_workers
        self.k = envs_per_worker
        self.num_envs = num_workers * envs_per_worker
        self.batch_workers = batch_workers or num_workers
        if self.num_workers % self.batch_workers:
            raise ValueError("num_workers must be divisible by batch_workers")
        self.batch_size = self.batch_workers * self.k
        self.obs_spec = probe_obs_spec(spec)
        self.layout, nbytes = _layout(self.obs_spec, self.num_envs)
        self.shm = SharedMemory(create=True, size=max(nbytes, 1))
        self.arr = _shm_arrays(self.shm, self.layout)
        self.obs_keys = [k[4:] for k in self.layout if k.startswith("obs.")]

        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(num_workers):
            parent, child = ctx.Pipe()
            ids = list(range(w * self.k, (w + 1) * self.k))
            cpu = pin_cpus[w % len(pin_cpus)] if pin_cpus else None
            p = ctx.Process(
                target=_worker,
                args=(w, spec, ids, seed, self.shm.name, self.layout, child, cpu),
                daemon=True,
            )
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self._conn_to_worker = {id(c): w for w, c in enumerate(self.conns)}
        self.inflight: set[int] = set()
        self.ready: list[tuple[int, list]] = []
        self.current: list[int] = []
        self.pending: dict[int, bytes] = {}

    def _env_ids(self, workers: list[int]) -> np.ndarray:
        return np.concatenate([np.arange(w * self.k, (w + 1) * self.k) for w in workers])

    def reset(self) -> None:
        for c in self.conns:
            c.send(("reset", None))
        self.inflight = set(range(self.num_workers))

    def recv(self) -> Batch:
        """Next ready batch: up to ``batch_workers`` workers (fewer when the caller
        is holding some workers back, e.g. at the end of a rollout)."""
        target = min(self.batch_workers, len(self.inflight) + len(self.ready))
        if target == 0:
            raise RuntimeError("recv() with no workers in flight")
        while len(self.ready) < target:
            conns = [self.conns[w] for w in self.inflight]
            for c in wait(conns):
                w = self._conn_to_worker[id(c)]
                self.ready.append((w, c.recv()))
                self.inflight.discard(w)
        batch, self.ready = self.ready[:target], self.ready[target:]
        self.current = [w for w, _ in batch]
        ids = self._env_ids(self.current)
        obs = {k: self.arr["obs." + k][ids] for k in self.obs_keys}  # fancy index = copy
        infos = [i for _, infos in batch for i in infos]
        return Batch(ids, obs, self.arr["rewards"][ids].copy(), self.arr["dones"][ids].copy(), infos)

    def send(self, actions: np.ndarray, env_ids: np.ndarray | None = None) -> None:
        """Step the envs of the last ``recv`` (or the whole-worker subset ``env_ids``)."""
        if env_ids is None:
            env_ids = self._env_ids(self.current)
        workers = sorted({int(i) // self.k for i in env_ids})
        self.arr["actions"][env_ids] = actions
        for w in workers:
            loads = {}
            for i in range(w * self.k, (w + 1) * self.k):
                if i in self.pending:
                    loads[i] = self.pending.pop(i)
            self.conns[w].send(("step", loads))
            self.inflight.add(w)
        self.current = []

    def load_states(self, assignments: dict[int, bytes]) -> None:
        """Queue save-states to load; applied on each env's next step."""
        self.pending.update(assignments)

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        deadline = time.time() + 5
        for p in self.procs:
            p.join(timeout=max(0.1, deadline - time.time()))
            if p.is_alive():
                p.terminate()
        del self.arr
        self.shm.close()
        self.shm.unlink()


def make_vec(spec: GameSpec, cfg, seed: int = 0):
    """Build a vec env from a ``VecConfig``."""
    if cfg.num_workers <= 0:
        return SerialVec(spec, cfg.envs_per_worker, seed=seed)
    pin = list(range(cfg.pin_offset, cfg.pin_offset + cfg.num_workers)) if cfg.pin_cpus else None
    return ProcVec(spec, cfg.num_workers, cfg.envs_per_worker, cfg.batch_workers, seed=seed, pin_cpus=pin)
