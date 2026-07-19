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

from pokeio.config import Config, VisionConfig
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
# Retina Nature-CNN input side (== evo.retina.IN_SIDE). Kept as a torch-free
# literal so the workers can size/detect the retina obs WITHOUT importing torch;
# only a retina-mode FovealEncoder lazily imports evo.retina (see below).
_RETINA_SIDE = 84


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
        self.tap_addrs: list[int] = []  # mined counter taps (see set_taps)

    def encode(self, screen: np.ndarray, wram: np.ndarray) -> np.ndarray:
        small = self._row @ screen.astype(np.float64) @ self._col  # (res,res) 0..255
        vis = (small / 255.0).astype(np.float32).ravel()
        if self.n_ram > 0:
            stride = max(1, wram.size // self.n_ram)
            ram = (wram[::stride][: self.n_ram].astype(np.float32)) / 255.0
            if ram.size < self.n_ram:  # pad if short
                ram = np.concatenate([ram, np.zeros(self.n_ram - ram.size, np.float32)])
            # mined progress-counter taps override the blind stride sample
            # (set via set_taps; workers apply the same override in _emit so
            # parent-side showcase/replay obs stay identical to training obs).
            if self.tap_addrs:
                for j, a in enumerate(self.tap_addrs[: self.n_ram]):
                    idx = a - 0xC000
                    if 0 <= idx < wram.size:
                        ram[j] = wram[idx] / 255.0
            return np.concatenate([vis, ram])
        return vis

    def set_taps(self, addrs: list[int] | None) -> None:
        """Point the RAM tail at mined counter addresses (None = legacy stride)."""
        self.tap_addrs = list(addrs) if addrs else []

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
# FovealEncoder — active-vision obs (periphery + movable fovea + motion +
# proprioception + connect-protected RAM taps).  See
# docs/specs/active-vision-spine.md §2 (obs tensor) and §3.3-§3.4 (saccade).
# ==========================================================================
# Drop-in replacement for ObsEncoder.  Torch-free (numpy only) so the spawned
# barrier/async workers can import it without pulling torch into 32 processes;
# loop.py (the parent) imports the SAME class so the parent encode and BOTH
# worker _emit paths call one code path and emit byte-identical vectors.
#
# DIM = 2*G^2 + FG^2 + 14 + n_ram float32 layout (G = periph_grid = 12,
# F = fovea_native_px = 48, FG = fovea_grid, n_ram = 8), all blocks contiguous.
# With FG = G (fovea_grid 0 or == periph_grid) this is the legacy 454-d obs; the
# generic offsets are shown with block sizes (periph G^2, fovea FG^2, motion G^2):
#     periphery [0 : G^2]           144x160 -> area-resample to GxG          [0,1]
#     fovea     [G^2 : G^2+FG^2]    native FxF crop @ gaze -> resample F->FG  [0,1]
#     motion    [G^2+FG^2 : +G^2]   (periph_t - periph_{t-1} + 1)/2; 0.5 init [0,1]
#     proprio   [.. : +14/16]       14-d efference copy (see _proprio); +2 = the
#                                   reflex TARGET when reflex_gaze on (§4)       [-1,1]
#     ram       [.. : +n_ram]       mined tap bytes (connect-protected)      [0,1]
# Legacy default (G=FG=12, n_ram=8): 144|144|144|14|8 = 454, block bounds
# 0,144,288,432,446,454.  Sharp-fovea example (G=12, FG=32): 144|1024|144|14|8
# = 1334 (fovea now 1 px/cell over a 32px window instead of a 4 px/cell smudge).
#
# Trans-saccadic foveal memory (optical-frontend-v2 §1a; foveal_memory=True) adds
# TWO M^2 blocks (M = mem_grid) AFTER motion, so the layout becomes
#     .. motion [.. : +G^2]   | buffer [.. : +M^2] | staleness [.. : +M^2] | proprio ..
# and DIM grows to 2*G^2 + FG^2 + 2*M^2 + 14 + n_ram (e.g. G=12,FG=32,M=24,n_ram=8:
# 144|1024|144|576|576|14|8 = 2486).  The buffer is a persistent per-env scene the
# fovea stamps into; staleness is its per-cell confidence (0 fresh .. 1 stale).
# OFF (default) => byte-identical to the sharp-fovea obs above (no new blocks; the
# motion block still immediately precedes proprio, so o_motion_hi == o_proprio).
#
# Reflex gaze + top-down modulation (optical-frontend-v2 §3/§4; reflex_gaze=True)
# grows PROPRIO 14->16 (the trailing 2 dims carry the bottom-up reflex TARGET —
# a soft-argmax of motion x staleness — so the controller conditions its evolved
# additive correction on where the reflex pulls).  No new visual block, no N_OUT
# change (it re-purposes the existing saccade outputs).  OFF => proprio stays 14.
#
# The fovea's F->FG resample is DECOUPLED from the periphery's ->G downsample so
# the fovea can be native-sharp while the periphery stays biomimetically coarse
# (low-acuity periphery); see optical-frontend-v2 §2.  Sensor seam (#15/§9a): the
# encoder consumes an (H, W, C) frame — screen_h/screen_w are params and C is
# ``channels`` (grayscale C=1 now via ObsBuilder.normalize_shades; C=3 RGB is the
# future console/webcam drop-in, no layout change above C).
#
# Stateful PER ENV (indexed by env id): gaze (gy,gx), last saccade (dx,dy),
# previous periphery (motion), a per-episode step counter (step_frac), — when
# foveal_memory=True — the persistent scene buffer + staleness map + the per-region
# divergence EMA that self-calibrates invalidation, and — when reflex_gaze=True —
# the per-env reflex TARGET (surfaced in proprio) (all reset per episode).
class FovealEncoder:
    """Stateful periphery+fovea+motion+proprio+ram encoder (spec §2/§3).

    Usage (identical in the parent and in both worker _emit paths)::

        enc = FovealEncoder(n_envs, periph_grid=12, fovea_native_px=48,
                            n_ram=8, saccade_gain=32, saccade_every_k=1,
                            episode_steps=<wave/episode length>)
        enc.reset()                              # gaze -> center, motion -> 0.5
        # per agent-step, for env i:
        enc.update_gaze(i, dx, dy)               # §3.3 saccade dynamics + store
        vec = enc.encode(i, screen, wram, button=applied_button)

    ``.dim`` is 454 for the committed foveal defaults (G=FG=12); a sharp fovea
    (``fovea_grid`` > 0) grows it to ``2*G^2 + FG^2 + 14 + n_ram``.  ``.reset(i)``
    resets one env; ``.reset()`` resets all.  Gaze resets to centre ``(72,80)``.

    With ``mode="retina"`` (Phase-1; select via ``config.vision.mode=="retina"``
    or by requesting ``obs_dim==14134``) the encoder keeps the IDENTICAL gaze /
    saccade / proprio / ram machinery but ships the retina's PIXEL input instead
    of the 12x12 vectors, a fixed **14134-dim** float32 layout (n_ram=8):

        periph84 [0:7056]      full raw screen -> 84x84 (retina.downscale)   [0,1]
        fovea84  [7056:14112]  48px native gaze crop -> 84x84 (crop_fovea)   [0,1]
        proprio  [14112:14126] SAME 14-d efference copy as foveal           [-1,1]
        ram      [14126:14134] SAME 8 mined tap bytes as foveal              [0,1]

    Retina mode emits SINGLE frames; the 4-frame temporal stack is built
    parent-side by the loop (per-env rings), so nothing is stacked here.
    """

    def __init__(
        self,
        n_envs: int = 1,
        *,
        periph_grid: int = 12,
        fovea_native_px: int = 48,
        fovea_grid: int = 0,
        n_ram: int = 8,
        saccade_gain: float = 32.0,
        saccade_every_k: int = 1,
        screen_h: int = _SCREEN_H,
        screen_w: int = _SCREEN_W,
        shades: int = 4,
        channels: int = 1,
        episode_steps: int = 1024,
        mode: str = "foveal",
        foveal_memory: bool = False,
        mem_grid: int = 24,
        mem_ema_decay: float = 0.99,
        mem_stale_z: float = 1.5,
        mem_stale_warmup: int = 16,
        reflex_gaze: bool = False,
        reflex_gain: float = 1.0,
        reflex_ema_decay: float = 0.99,
        reflex_beta: float = 4.0,
    ) -> None:
        self.n_envs = int(n_envs)
        self.mode = str(mode)
        self.G = int(periph_grid)
        self.F = int(fovea_native_px)
        # Sharp-fovea resample side (optical-frontend-v2 §2): FG decouples fovea
        # acuity from the coarse periphery.  0 (or == G) => FG=G => byte-identical
        # legacy; >0 makes the fovea FG*FG at F/FG px/cell.
        self.FG = int(fovea_grid) if int(fovea_grid) > 0 else self.G
        # Trans-saccadic foveal memory (optical-frontend-v2 §1a).  Foveal-only:
        # retina mode has no periphery/motion vectors to stamp against, so the
        # keystone is a foveal feature (§1a).  M = mem_grid is the buffer side;
        # mem_ema_decay/mem_stale_z/mem_stale_warmup self-calibrate invalidation
        # (dimensionless, mirror ac.salience_*; no fixed change magnitude — §11b).
        self.foveal_memory = bool(foveal_memory) and self.mode != "retina"
        self.M = int(mem_grid) if int(mem_grid) > 0 else 24
        self.mem_ema_decay = float(mem_ema_decay)
        self.mem_stale_z = float(mem_stale_z)
        self.mem_stale_warmup = int(mem_stale_warmup)
        self.n_mem = self.M * self.M
        # Reflex gaze + top-down modulation (optical-frontend-v2 §3/§4).  Foveal-
        # only: the reflex reads the motion sheet (retina has none), so it never
        # arms in retina mode.  ON exposes the reflex TARGET as 2 extra proprio
        # efference dims (n_proprio 14->16 below), so the sparse genome can
        # condition its top-down correction on where the reflex pulls (§4).  The
        # blend (gaze_delta = reflex + learned) + the self-calibrated reflex_gain
        # live PARENT-SIDE (loop.ReflexGaze / live.py); the encoder just computes
        # the target here + surfaces it in proprio.  reflex_beta is the soft-argmax
        # sharpness over a max-normalized salience map (dimensionless — §11b).
        self.reflex_gaze = bool(reflex_gaze) and self.mode != "retina"
        self.reflex_gain = float(reflex_gain)
        self.reflex_ema_decay = float(reflex_ema_decay)
        self.reflex_beta = float(reflex_beta)
        # Sensor seam (#15/§9a): (H,W,C) frame channels.  C=1 grayscale now (the
        # ObsBuilder shade path assumes a single luma plane); C=3 RGB is the
        # future console/webcam drop-in.  Kept as an attribute, not yet wired
        # through the resample (that is the color increment, §9b task #17).
        self.C = int(channels)
        self.n_ram = int(n_ram)
        self.gain = float(saccade_gain)
        self.every_k = max(1, int(saccade_every_k))
        self.H = int(screen_h)
        self.W = int(screen_w)
        self.shades = int(shades)
        self.episode_steps = max(1, int(episode_steps))

        g = self.G
        self.n_periph = g * g
        self.n_fovea = self.FG * self.FG   # sharp fovea: FGxFG (legacy FG==G => g*g)
        self.n_motion = g * g              # periphery/motion stay coarse at GxG
        # 14-d efference copy; +2 (=> 16) for the reflex TARGET when reflex_gaze is
        # on (optical-frontend-v2 §4).  Grows n_in by 2 => a fresh run; OFF keeps 14
        # so the obs is byte-identical to Increment B (retina is always reflex-off).
        self.n_proprio = 16 if self.reflex_gaze else 14

        if self.mode == "retina":
            # Phase-1 retina obs: ship the retina encoder's PIXEL input, not the
            # 12x12 foveal vector.  Single 84x84 periphery (full screen area-
            # downscaled) + single 84x84 fovea (48px native gaze crop upsampled)
            # + the SAME 14-d proprio efference copy + 8-d mined RAM tail. NO
            # motion sheet (the 4-frame temporal STACK is built PARENT-SIDE by
            # the loop's per-env rings — this emits current single frames only).
            self._side = _RETINA_SIDE                    # 84 (== retina.IN_SIDE)
            self.n_periph84 = self._side * self._side    # 7056
            self.n_fovea84 = self._side * self._side     # 7056
            self.dim = (self.n_periph84 + self.n_fovea84
                        + self.n_proprio + self.n_ram)   # 14134 @ n_ram=8
            self._o_periph = 0                           # periph84 [0:7056]
            self._o_fovea = self.n_periph84              # fovea84  [7056:14112]
            self._o_proprio = self._o_fovea + self.n_fovea84   # proprio [14112:14126]
            self._o_ram = self._o_proprio + self.n_proprio     # ram     [14126:14134]
            # No motion sheet in retina obs (spec [AC] §3.4: retina salience is
            # the deferred controller-latent L1). Public None => AC salience off.
            self.o_motion = None
            self.o_motion_hi = None       # no motion block (§1a memory is foveal-only)
            self.o_proprio = self._o_proprio
            self.foveal_memory = False    # keystone is foveal-only; never on in retina
            # Lazy import: evo.retina pulls torch, which the torch-free foveal
            # workers must never import. Only a retina-mode encoder touches it,
            # so the default (foveal 454) path stays torch-free across all 56
            # spawned workers. Reuse retina.downscale / retina.crop_fovea so the
            # 84x84 tensors are byte-exactly what the parent-side retina expects.
            from pokeio.evo import retina as _retina
            self._retina = _retina
            assert _retina.IN_SIDE == self._side, (
                f"retina.IN_SIDE={_retina.IN_SIDE} != fleet _RETINA_SIDE={self._side}"
            )
        else:
            # Block offsets (contiguous).  Trans-saccadic memory (§1a) inserts the
            # buffer + staleness blocks AFTER motion, BEFORE proprio — both are
            # visual channels, so the E3 blind-ablation gate ([0:o_proprio]) zeroes
            # them along with periph/fovea/motion.  OFF => no blocks, and the motion
            # block still immediately precedes proprio (o_motion_hi == o_proprio),
            # so the obs is byte-identical to the sharp-fovea (Increment A) layout.
            self._o_periph = 0
            self._o_fovea = self.n_periph
            self._o_motion = self._o_fovea + self.n_fovea
            o = self._o_motion + self.n_motion
            # [AC] salience slice HI: END of the motion block (NOT o_proprio, which
            # now moves past the memory blocks).  == o_proprio when memory is off.
            self.o_motion_hi = o
            if self.foveal_memory:
                self._o_buffer = o
                self._o_stale = self._o_buffer + self.n_mem
                o = self._o_stale + self.n_mem
            else:
                self._o_buffer = None
                self._o_stale = None
            self._o_proprio = o
            self._o_ram = self._o_proprio + self.n_proprio
            self.dim = self._o_ram + self.n_ram
            # Public offsets for the [AC] parent-side salience slice (spec §3.4):
            # the motion block [o_motion:o_motion_hi] is the gaze-invariant frame
            # difference already in the obs — the surprise signal, zero extra work.
            self.o_motion = self._o_motion
            self.o_proprio = self._o_proprio

        # Fovea centre clamp: keep the FxF window fully on-screen (spec §3.3).
        self._half = self.F // 2
        self._gx_lo, self._gx_hi = float(self._half), float(self.W - self._half)
        self._gy_lo, self._gy_hi = float(self._half), float(self.H - self._half)
        self._cy = self.H // 2  # 72 (row centre)
        self._cx = self.W // 2  # 80 (col centre)

        # Area-resample matrices (built once).
        self._prow = _area_matrix(self.H, g)        # (G, H) periphery rows
        self._pcol = _area_matrix(self.W, g).T      # (W, G) periphery cols
        self._frow = _area_matrix(self.F, self.FG)  # (FG, F) fovea rows (F->FG)
        self._fcol = _area_matrix(self.F, self.FG).T  # (F, FG) fovea cols

        # Trans-saccadic memory geometry (§1a; built once, foveal_memory only).
        if self.foveal_memory:
            M = self.M
            # Fixed-size sharp-fovea stamp: the FxF native crop warps to an
            # (mf_rows, mf_cols) buffer block that SLIDES with gaze (gaze is clamped
            # so the FxF window is fully on-screen, so the block always fits).
            self._mf_rows = min(M, max(1, int(round(self.F * M / self.H))))
            self._mf_cols = min(M, max(1, int(round(self.F * M / self.W))))
            self._sfrow = _area_matrix(self.F, self._mf_rows)      # (mf_rows, F)
            self._sfcol = _area_matrix(self.F, self._mf_cols).T    # (F, mf_cols)
            # Nearest-cell upsample G->M (each buffer cell -> its periphery region),
            # for spreading a per-region invalidation over the covered buffer cells.
            self._g2m_row = np.minimum((np.arange(M) * g) // M, g - 1)
            self._g2m_col = np.minimum((np.arange(M) * g) // M, g - 1)
            # Same G->M upsample as ONE flat (M*M,) gather index into a C-order
            # (G,G) array, so ``sal[g2m_row][:, g2m_col]`` (two fancy indexes) is
            # one ``np.take(...).reshape(M,M)`` (bit-identical, fewer/cheaper ops).
            self._g2m_flat = (self._g2m_row[:, None] * g
                              + self._g2m_col[None, :]).ravel()

        # Reuse the EXACT per-frame shade ranking used by the rest of the
        # pipeline (ObsBuilder.normalize_shades) so grayscale is consistent.
        self._shade = ObsBuilder(
            VisionConfig(shades=self.shades, screen_height=self.H, screen_width=self.W)
        )

        # Per-env state.
        self._gy = np.empty(self.n_envs, np.float64)
        self._gx = np.empty(self.n_envs, np.float64)
        self._last_dx = np.zeros(self.n_envs, np.float64)
        self._last_dy = np.zeros(self.n_envs, np.float64)
        self._nstep = np.zeros(self.n_envs, np.int64)
        self._prev_periph: list = [None] * self.n_envs
        # Reflex-gaze target per env (§3/§4): the soft-argmax of motion x staleness
        # in normalized screen coords [-1,1], recomputed each encode + surfaced in
        # proprio[14:16].  Seeded to screen-centre (0,0), which is also the current
        # gaze after reset, so the reset-frame reflex pull is exactly zero.
        if self.reflex_gaze:
            self._reflex_tx = np.zeros(self.n_envs, np.float64)
            self._reflex_ty = np.zeros(self.n_envs, np.float64)
            # Centroid index vector for the soft-argmax (constant; == np.arange(P)
            # where P = M with foveal memory on, else G).  Precomputed so the hot
            # path skips a per-encode np.arange.
            self._reflex_idx = np.arange(self.M if self.foveal_memory else self.G)
        # Trans-saccadic memory per-env state (§1a): the persistent scene buffer,
        # its staleness map, and the per-region divergence EMA that self-calibrates
        # invalidation.  PER-ENV is load-bearing for engine-parity — each env's
        # buffer/EMA depend ONLY on its own frame sequence, so serial and furnace
        # produce bit-identical state.  All reset per episode (buffer is a
        # per-episode percept; its change baseline resets WITH it, or a post-reset
        # buffer of 0.5 would spuriously fire against the pre-reset baseline).
        if self.foveal_memory:
            self._mem = np.empty((self.n_envs, self.M, self.M), np.float32)
            self._stale = np.empty((self.n_envs, self.M, self.M), np.float32)
            # _periph_ref: the low-res periphery per G-region AT THE TIME that region
            # was last refreshed (stamped/invalidated) — the change signal is
            # |periph_now - _periph_ref| (has the periphery moved since we committed
            # this region), NOT sharp-buffer-vs-coarse-periph (which always carries a
            # resample residual and would spuriously fire on a static screen).
            self._periph_ref = np.zeros((self.n_envs, g, g), np.float32)
            self._chg_mu = np.zeros((self.n_envs, g, g), np.float64)
            self._chg_var = np.zeros((self.n_envs, g, g), np.float64)
            self._mem_steps = np.zeros(self.n_envs, np.int64)
            self._mem_last_fire = np.zeros((self.n_envs, g, g), bool)
        self.tap_addrs: list[int] = []  # parent-path mined taps (see set_taps)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, env_idx: int | None = None) -> None:
        """Reset gaze to centre + clear motion/efference for one env (or all).

        With ``foveal_memory`` the persistent scene buffer + staleness map + the
        per-region divergence EMA are ALSO cleared here (§1a: the buffer is a
        per-episode percept — 0.5 neutral, maximally stale until the fovea stamps
        it in; clearing the EMA with it keeps a post-reset buffer from firing
        invalidations against a pre-reset change baseline)."""
        idxs = range(self.n_envs) if env_idx is None else (int(env_idx),)
        for i in idxs:
            self._gy[i] = float(self._cy)
            self._gx[i] = float(self._cx)
            self._last_dx[i] = 0.0
            self._last_dy[i] = 0.0
            self._nstep[i] = 0
            self._prev_periph[i] = None
            if self.reflex_gaze:
                self._reflex_tx[i] = 0.0   # screen-centre == gaze after reset
                self._reflex_ty[i] = 0.0
            if self.foveal_memory:
                self._mem[i] = 0.5      # neutral "no percept yet"
                self._stale[i] = 1.0    # fully stale until first stamp
                self._periph_ref[i] = 0.0  # seeded to the first frame's periphery
                self._chg_mu[i] = 0.0
                self._chg_var[i] = 0.0
                self._mem_steps[i] = 0
                self._mem_last_fire[i] = False

    # -------------------------------------------------------------- saccade
    def update_gaze(self, env_idx: int, dx: float, dy: float) -> tuple[float, float]:
        """Integrate one saccade command into the gaze centre (spec §3.3).

        ``gx <- clip(gx + GAIN*tanh(dx), F/2, W-F/2)`` = clip(.., 24, 136);
        ``gy <- clip(gy + GAIN*tanh(dy), F/2, H-F/2)`` = clip(.., 24, 120).
        Applied every ``saccade_every_k`` steps; the raw command ``(dx, dy)`` is
        always stored as the efference copy and the per-episode step counter is
        advanced.  Returns the new ``(gy, gx)``.

        Gaze-actuator seam (#16 / §9a): this IS the gaze-actuator interface — it
        consumes a gaze command ``(dpan, dtilt)`` (``dx, dy``; a future ``dzoom``
        is the natural 3rd DOF for foveal scale) and returns the realized gaze.
        The command is already a VELOCITY (``GAIN*tanh``), so it maps to a PTZ slew
        rate directly: SOFTWARE-CROP now (the fovea can teleport, clamp is the only
        limit); PTZ-later swaps this body for a velocity-limited, latency-bearing
        physical pan/tilt with no change to the caller (the reflex + top-down blend
        both emit into this same ``(dx, dy)`` command).  The incoming ``(dx, dy)``
        is the parent-blended ``reflex_delta + learned_delta`` when reflex gaze is
        on (§4); this integrator is agnostic to how the command was formed.
        """
        i = int(env_idx)
        self._last_dx[i] = float(dx)
        self._last_dy[i] = float(dy)
        if int(self._nstep[i]) % self.every_k == 0:
            # Scalar clamp via builtin min/max (== np.clip for finite scalars, the
            # gain*tanh term is bounded): ~6x cheaper than the np.clip wrapper.
            nx = self._gx[i] + self.gain * np.tanh(float(dx))
            self._gx[i] = min(max(nx, self._gx_lo), self._gx_hi)
            ny = self._gy[i] + self.gain * np.tanh(float(dy))
            self._gy[i] = min(max(ny, self._gy_lo), self._gy_hi)
        self._nstep[i] += 1
        return float(self._gy[i]), float(self._gx[i])

    def gaze(self, env_idx: int) -> tuple[float, float]:
        """Current ``(gy, gx)`` fovea centre for an env (for telemetry/tests)."""
        i = int(env_idx)
        return float(self._gy[i]), float(self._gx[i])

    def set_taps(self, addrs: list[int] | None) -> None:
        """Point the RAM tail at mined counter addresses (parent path).

        When set, :meth:`encode` overlays ``wram[addr-0xC000]`` onto the ram
        block from the FULL wram it is passed (mirrors ObsEncoder.set_taps).
        Workers leave this empty and overlay from live emulator memory in
        ``_emit`` instead (they only hold a strided wram slice)."""
        self.tap_addrs = list(addrs) if addrs else []

    # ------------------------------------------------ trans-saccadic memory (§1a)
    def mem_buffer(self, env_idx: int) -> np.ndarray:
        """Current persistent scene buffer ``(M,M)`` for an env (copy; telemetry
        / rendering / tests).  Empty ``(0,0)`` when ``foveal_memory`` is off."""
        if not self.foveal_memory:
            return np.zeros((0, 0), np.float32)
        return self._mem[int(env_idx)].copy()

    def mem_staleness(self, env_idx: int) -> np.ndarray:
        """Current staleness map ``(M,M)`` for an env (0 fresh .. 1 stale; copy)."""
        if not self.foveal_memory:
            return np.zeros((0, 0), np.float32)
        return self._stale[int(env_idx)].copy()

    def mem_last_invalidated(self, env_idx: int) -> np.ndarray:
        """The ``(G,G)`` boolean regions invalidated on this env's LAST encode
        (telemetry / tests; the change-blindness witness)."""
        if not self.foveal_memory:
            return np.zeros((0, 0), bool)
        return self._mem_last_fire[int(env_idx)].copy()

    # ---------------------------------------------- reflex gaze target (§3/§4)
    def reflex_target(self, env_idx: int) -> tuple[float, float]:
        """Current reflex gaze target ``(tx, ty)`` for an env, in normalized screen
        coords ``[-1,1]`` (telemetry / rendering / tests) — the same value written
        into ``proprio[14:16]``.  ``(0.0, 0.0)`` (== screen centre) when
        ``reflex_gaze`` is off."""
        if not self.reflex_gaze:
            return (0.0, 0.0)
        i = int(env_idx)
        return (float(self._reflex_tx[i]), float(self._reflex_ty[i]))

    def _reflex_target(self, i: int, motion: np.ndarray) -> tuple[float, float]:
        """[§3] Bottom-up reflex gaze target for env ``i`` from ``motion`` (× the
        staleness map when ``foveal_memory`` is on).

        The salience map is the motion MAGNITUDE ``|motion - 0.5|`` (0.5 == no
        motion), multiplied by the per-cell staleness when the trans-saccadic
        memory is on (orient to what CHANGED **or** hasn't been refreshed lately —
        §1a's ``motion x staleness``; motion alone otherwise).  A **soft-argmax**
        (center-of-mass over a softmax of the max-normalized salience — so the
        sharpness ``reflex_beta`` is scale-free, no fixed magnitude, §11b) gives the
        target in grid coords, converted to normalized screen coords ``[-1,1]``.

        A flat field (no motion) has an all-zero salience map => the soft-argmax is
        undefined, so we return the CURRENT gaze (a zero pull — the harmless
        "reflex ~ no pull" fallback, robust to any gaze position).  Zero rng,
        per-env => engine-parity safe."""
        sal = np.abs(motion.astype(np.float64) - 0.5)            # (G,G) motion mag
        if self.foveal_memory:
            # upsample motion G->M (nearest, reuse the §1a mapping) so it aligns
            # with the M x M staleness map, then weight by staleness.  One flat
            # np.take gather == the old sal[g2m_row][:, g2m_col] (bit-identical).
            sal = np.take(sal, self._g2m_flat).reshape(self.M, self.M) * self._stale[i]
            P = self.M
        else:
            P = self.G
        flat = sal.ravel()
        smax = float(flat.max())
        if smax <= 1e-9:  # flat / no salient change: aim at current gaze (zero pull)
            return (self._gx[i] / self.W * 2.0 - 1.0, self._gy[i] / self.H * 2.0 - 1.0)
        # soft-argmax: softmax over the max-normalized salience, then center of mass.
        w = np.exp(self.reflex_beta * (flat / smax - 1.0))
        w /= w.sum()
        wm = w.reshape(P, P)
        idx = self._reflex_idx                                   # == np.arange(P)
        r_star = float((wm.sum(axis=1) * idx).sum())             # row centroid
        c_star = float((wm.sum(axis=0) * idx).sum())             # col centroid
        tgt_x = (c_star + 0.5) / P * 2.0 - 1.0
        tgt_y = (r_star + 0.5) / P * 2.0 - 1.0
        return (tgt_x, tgt_y)

    def _calibrate_invalidation(self, i: int, chg: np.ndarray) -> np.ndarray:
        """[§1a / §11b] Self-calibrated peripheral-change invalidation for env ``i``.

        Fire (mark stale) a region when its peripheral change ``chg`` (G,G — how
        far the live periphery has moved from its last-refreshed reference) is
        ``mem_stale_z`` std ABOVE THIS env's OWN recent change for that region —
        a per-region EMA-z, a dimensionless surprise with NO fixed change
        magnitude (the no-tuned-knobs mandate), mirroring the [AC] MotorClock
        salience reflex.  Compares against the PRE-update baseline,
        then folds this sample into the per-region EMA (Welford-style, μ seeded on
        the first sample so a constant divergence never spuriously fires through
        warm-up).  Zero rng, per-env => engine-parity safe.  Returns the (G,G)
        boolean fire mask (also stored for :meth:`mem_last_invalidated`)."""
        c64 = np.asarray(chg, dtype=np.float64)
        mu = self._chg_mu[i]
        var = self._chg_var[i]
        steps = int(self._mem_steps[i])
        dev = c64 - mu  # deviation from the running mean: reused by fire + the EMA
        # warm: enough per-env history AND a live per-region variance to scale by.
        # Short-circuit the sqrt+compare during warm-up (the pre-warm mask is all-
        # False, exactly what ``warm & ...`` produced).
        if steps >= self.mem_stale_warmup:
            fire = (var > 0.0) & (dev > self.mem_stale_z * np.sqrt(var))
        else:
            fire = np.zeros_like(var, dtype=bool)
        d = self.mem_ema_decay
        if steps == 0:  # seed μ with the first sample (baseline never lags up from 0)
            mu_new = c64.copy()
            var_new = np.zeros_like(var)
        else:
            mu_new = mu + (1.0 - d) * dev
            var_new = d * var + (1.0 - d) * dev * (c64 - mu_new)
        self._chg_mu[i] = mu_new
        self._chg_var[i] = var_new
        self._mem_steps[i] = steps + 1
        self._mem_last_fire[i] = fire
        return fire

    def _stamp_origin(self, i: int) -> tuple[int, int]:
        """Top-left ``(r0,c0)`` buffer cell of the fovea's screen footprint, so the
        fixed ``mf_rows x mf_cols`` stamp lands under the current gaze (clamped
        into ``[0, M-mf]`` — gaze clamping keeps the FxF window on-screen)."""
        r0 = int(round((self._gy[i] - self._half) * self.M / self.H))
        c0 = int(round((self._gx[i] - self._half) * self.M / self.W))
        r0 = min(max(r0, 0), self.M - self._mf_rows)
        c0 = min(max(c0, 0), self.M - self._mf_cols)
        return r0, c0

    def _mem_step(self, i: int, periph: np.ndarray, crop: np.ndarray) -> None:
        """One trans-saccadic memory update for env ``i`` (§1a); mutates the
        buffer + staleness in place.

        Order: (1) change signal = |live periphery - the periphery each region
        showed when last refreshed|; (2) self-calibrated per-region invalidation;
        (3) age every cell; (4) decay invalidated regions toward the live low-res
        value + mark them stale + re-reference; (5) stamp the sharp fovea into its
        footprint (fresh) + re-reference its G-footprint — stamp LAST so a
        freshly-glimpsed region always wins over ageing/invalidation.
        """
        g = self.G
        # 1. change signal: has the low-res periphery MOVED since we last committed
        #    each region?  On the episode's first frame, seed the reference to the
        #    current periphery so the scene starts un-surprising (chg == 0).
        if int(self._mem_steps[i]) == 0:
            self._periph_ref[i] = periph
        chg = np.abs(periph.astype(np.float64) - self._periph_ref[i])          # (G,G)
        # 2. self-calibrated invalidation (per-region EMA-z; no fixed magnitude).
        fire = self._calibrate_invalidation(i, chg)                            # (G,G)
        # 3. age every cell (steps-since-refresh confidence decay, horizon-
        #    normalized like proprio's step_frac; read live so the async
        #    episode_steps override tracks — NOT a hand-tuned magnitude).
        st = self._stale[i]
        st += 1.0 / float(self.episode_steps)
        st.clip(0.0, 1.0, out=st)  # method form: skips the np.clip dispatch wrapper
        # 4. invalidate fired regions: decay the buffer toward the live low-res
        #    periphery (fall back to what the periphery now shows), mark stale, and
        #    re-reference (we've accepted the new low-res state; watch for the NEXT
        #    change from here).
        if fire.any():
            # G->M nearest upsample via one flat np.take each (== the old
            # fire[g2m_row][:, g2m_col] double fancy-index, bit-identical).
            fire_m = np.take(fire, self._g2m_flat).reshape(self.M, self.M)  # (M,M) bool
            live_m = np.take(periph, self._g2m_flat).reshape(self.M, self.M)  # live low-res
            self._mem[i][fire_m] = live_m[fire_m].astype(np.float32)
            self._stale[i][fire_m] = 1.0
            self._periph_ref[i][fire] = periph[fire]
        # 5. stamp the sharp fovea crop into its footprint (fresh) + re-reference
        #    the G-regions the fovea now covers (they match the live periphery).
        r0, c0 = self._stamp_origin(i)
        mr, mc = self._mf_rows, self._mf_cols
        block = (self._sfrow @ crop @ self._sfcol).astype(np.float32)  # (mf_rows,mf_cols)
        self._mem[i][r0 : r0 + mr, c0 : c0 + mc] = block
        self._stale[i][r0 : r0 + mr, c0 : c0 + mc] = 0.0
        half = self._half
        gr0 = max(0, int(np.floor((self._gy[i] - half) * g / self.H)))
        gr1 = min(g, int(np.ceil((self._gy[i] + half) * g / self.H)))
        gc0 = max(0, int(np.floor((self._gx[i] - half) * g / self.W)))
        gc1 = min(g, int(np.ceil((self._gx[i] + half) * g / self.W)))
        self._periph_ref[i][gr0:gr1, gc0:gc1] = periph[gr0:gr1, gc0:gc1]

    # --------------------------------------------------------------- helpers
    def _crop(self, norm: np.ndarray, gy: float, gx: float) -> np.ndarray:
        """Native FxF crop centred at (gy,gx); zero-padded on edge overhang."""
        f, half = self.F, self._half
        out = np.zeros((f, f), np.float64)
        top = int(np.floor(gy + 0.5)) - half   # round gaze to nearest pixel
        left = int(np.floor(gx + 0.5)) - half
        sr0, sr1 = max(0, top), min(self.H, top + f)
        sc0, sc1 = max(0, left), min(self.W, left + f)
        if sr1 > sr0 and sc1 > sc0:
            dr0, dc0 = sr0 - top, sc0 - left
            out[dr0 : dr0 + (sr1 - sr0), dc0 : dc0 + (sc1 - sc0)] = norm[sr0:sr1, sc0:sc1]
        return out

    def _proprio(self, i: int, button: int) -> np.ndarray:
        """14-d efference copy (spec §2.1), all in [-1,1]; 16-d with reflex gaze.

        [gx*2/W-1, gy*2/H-1, dx_prev, dy_prev,
         up,down,left,right,A,B,START,SELECT,NOOP one-hot, step_frac
         (, reflex_target_x, reflex_target_y)].

        With ``reflex_gaze`` the trailing 2 dims carry the bottom-up reflex TARGET
        (§4) so the controller conditions its top-down correction on where the
        reflex pulls (the top-down/bottom-up handshake).  The parent
        (``loop.ReflexGaze`` / live.py) reads these back to form the additive
        reflex delta, so this is the single source of truth for the target."""
        p = np.zeros(self.n_proprio, np.float32)
        p[0] = self._gx[i] / self.W * 2.0 - 1.0
        p[1] = self._gy[i] / self.H * 2.0 - 1.0
        p[2] = np.float32(self._last_dx[i])
        p[3] = np.float32(self._last_dy[i])
        b = int(button)
        if 0 <= b < 9:  # button ids match emu.env.ACTIONS ordering
            p[4 + b] = 1.0
        p[13] = min(float(self._nstep[i]) / float(self.episode_steps), 1.0)
        if self.reflex_gaze:  # 2 efference dims: where the reflex is pulling (§4)
            p[14] = np.float32(self._reflex_tx[i])
            p[15] = np.float32(self._reflex_ty[i])
        return p

    def _ram(self, wram: np.ndarray | None) -> np.ndarray:
        """Blind stride sample of WRAM into the ram block (taps overlaid after).

        Byte-identical whether given the full 8 KB wram (parent) or the strided
        slice the workers hold, because ``stride = size//n_ram`` composes: e.g.
        raw[::1024][:8] == raw[::64][::16][:8] (1024 % 64 == 0)."""
        ram = np.zeros(self.n_ram, np.float32)
        if self.n_ram > 0 and wram is not None:
            w = np.asarray(wram)
            if w.size:
                stride = max(1, w.size // self.n_ram)
                seg = (w[::stride][: self.n_ram].astype(np.float32)) / 255.0
                ram[: seg.size] = seg
        return ram

    # ------------------------------------------------------------- ram overlay
    def _ram_block(
        self, wram: np.ndarray | None, taps: list[tuple[int, float]] | None
    ) -> np.ndarray:
        """Blind stride sample + connect-protected tap overlay for the ram tail.

        Byte-identical to the inline overlay the foveal ``encode`` does (parent
        path overlays mined taps from full ``wram``; worker path leaves
        ``tap_addrs`` empty and overlays from live emulator memory in ``_emit``).
        Shared so the retina path produces the IDENTICAL 8-d ram block."""
        ram = self._ram(wram)
        if self.tap_addrs and wram is not None:  # parent path: overlay from wram
            for j, a in enumerate(self.tap_addrs[: self.n_ram]):
                idx = a - 0xC000
                if 0 <= idx < wram.size:
                    ram[j] = wram[idx] / 255.0
        if taps:  # explicit (slot, value01) overlay
            for slot, v in taps:
                if 0 <= slot < self.n_ram:
                    ram[slot] = np.float32(v)
        return ram

    # -------------------------------------------------------------- retina encode
    def _encode_retina(
        self,
        i: int,
        screen: np.ndarray,
        wram: np.ndarray | None,
        *,
        button: int,
        taps: list[tuple[int, float]] | None,
    ) -> np.ndarray:
        """Build the 14134-d retina obs for env ``i`` (mode=="retina").

        ``periph84`` = full raw screen area-downscaled to 84x84 (retina.downscale);
        ``fovea84``  = 48px native crop at the CURRENT gaze, upsampled to 84x84
        (retina.crop_fovea). Gaze is rounded to the nearest pixel with the SAME
        rule the foveal ``_crop`` uses (``int(floor(g+0.5))``), so both modes crop
        at an identical centre. ``proprio``/``ram`` are the Phase-0 blocks."""
        r = self._retina
        side = self._side
        scr = np.asarray(screen)
        periph84 = r.downscale(scr, side)                       # (84,84) f32 [0,1]
        # Round the float gaze to a pixel index the same way _crop does; the
        # native FxF crop is upsampled to 84x84 (retina resize) and flattened.
        gy_px = int(np.floor(float(self._gy[i]) + 0.5))
        gx_px = int(np.floor(float(self._gx[i]) + 0.5))
        fovea84 = r.crop_fovea(scr, gy_px, gx_px, self.F, side)  # (84,84) f32 [0,1]

        proprio = self._proprio(i, button)
        ram = self._ram_block(wram, taps)

        vec = np.empty(self.dim, np.float32)
        vec[self._o_periph : self._o_fovea] = periph84.ravel()
        vec[self._o_fovea : self._o_proprio] = fovea84.ravel()
        vec[self._o_proprio : self._o_ram] = proprio
        vec[self._o_ram : self.dim] = ram
        return vec

    # ------------------------------------------------------------------ encode
    def encode(
        self,
        env_idx: int,
        screen: np.ndarray,
        wram: np.ndarray | None = None,
        *,
        button: int = 8,
        taps: list[tuple[int, float]] | None = None,
    ) -> np.ndarray:
        """Build the obs vector for env ``env_idx`` from its current state.

        Foveal mode -> 454-d (periphery/fovea/motion/proprio/ram); retina mode
        -> 14134-d (periph84/fovea84/proprio/ram). Details below cover foveal.

        ``screen`` is a raw (H,W) uint8 frame; ``wram`` is the full 8 KB block
        (parent) or the strided slice (worker).  ``button`` is the button id
        just applied (0..8, defaults to NOOP for the reset obs).  ``taps`` is an
        optional list of ``(ram_slot, value01)`` overlays.  Call
        :meth:`update_gaze` first each step so the fovea crop and proprio see
        the freshly-integrated gaze (the workers do exactly this in ``_emit``).
        """
        i = int(env_idx)
        if self.mode == "retina":
            return self._encode_retina(i, screen, wram, button=button, taps=taps)
        # Normalize straight to float64 (the dtype the resample matmuls + crop
        # consume): one np.take gather instead of a float32 build + float64 copy.
        normd = self._shade.normalize_shades_f64(screen)     # (H,W) float64 [0,1]

        periph = (self._prow @ normd @ self._pcol).astype(np.float32)   # (G,G)
        crop = self._crop(normd, self._gy[i], self._gx[i])             # (F,F)
        fov = (self._frow @ crop @ self._fcol).astype(np.float32)      # (FG,FG)

        prev = self._prev_periph[i]
        if prev is None:
            motion = np.full((self.G, self.G), 0.5, np.float32)
        else:
            motion = (((periph - prev) + 1.0) * 0.5).astype(np.float32)
        self._prev_periph[i] = periph

        # Trans-saccadic foveal memory (§1a): stamp the sharp fovea into the
        # persistent scene buffer, age it, and self-calibrate peripheral-change
        # invalidation.  Mutates _mem/_stale in place; both enter the obs below.
        # OFF => this block + the buffer/staleness writes are skipped and the obs
        # is byte-identical to Increment A (o_motion_hi == o_proprio).
        if self.foveal_memory:
            self._mem_step(i, periph, crop)

        # Reflex gaze target (§3/§4): soft-argmax of motion x staleness -> proprio.
        # Computed AFTER _mem_step so it reads the freshly-updated staleness (a
        # just-glimpsed region is now fresh => low salience there, so the reflex
        # orients AWAY from what we just refreshed).  OFF => proprio stays 14-d and
        # this is skipped (obs byte-identical to Increment B).  Zero rng, per-env.
        if self.reflex_gaze:
            self._reflex_tx[i], self._reflex_ty[i] = self._reflex_target(i, motion)

        proprio = self._proprio(i, button)
        ram = self._ram(wram)
        if self.tap_addrs and wram is not None:  # parent path: overlay from wram
            for j, a in enumerate(self.tap_addrs[: self.n_ram]):
                idx = a - 0xC000
                if 0 <= idx < wram.size:
                    ram[j] = wram[idx] / 255.0
        if taps:  # explicit (slot, value01) overlay
            for slot, v in taps:
                if 0 <= slot < self.n_ram:
                    ram[slot] = np.float32(v)

        vec = np.empty(self.dim, np.float32)
        vec[self._o_periph : self._o_fovea] = periph.ravel()
        vec[self._o_fovea : self._o_motion] = fov.ravel()
        vec[self._o_motion : self.o_motion_hi] = motion.ravel()
        if self.foveal_memory:  # buffer + staleness blocks (§1a)
            vec[self._o_buffer : self._o_stale] = self._mem[i].ravel()
            vec[self._o_stale : self._o_proprio] = self._stale[i].ravel()
        vec[self._o_proprio : self._o_ram] = proprio
        vec[self._o_ram : self.dim] = ram
        return vec


# ==========================================================================
# ReflexGaze — parent-side reflex-gaze blend + self-calibrated gain (§3/§4)
# ==========================================================================
# Lives here (not train/loop.py) so BOTH the trainer (loop.py) and the live
# showcase (live.py) import it from one place WITHOUT a circular import
# (loop imports live), and it sits next to the FovealEncoder it pairs with (the
# encoder computes the reflex TARGET; this consumes it).  Torch-free (numpy only).
class ReflexGaze:
    """[optical-frontend-v2 §3/§4] Parent-side reflex-gaze blend + self-calibrated gain.

    Closes the active-vision loop.  :class:`FovealEncoder` computes the bottom-up
    reflex TARGET (soft-argmax of motion x staleness) and surfaces it in
    ``proprio[14:16]`` (the single source of truth).  This latch turns the
    ``target - current_gaze`` pull into a reflex saccade COMMAND and ADDS it to the
    controller's learned saccade (§4)::

        gaze_delta = reflex_delta + learned_delta

    so the net can FOLLOW the reflex (learned≈0), NUDGE it, or OVERRIDE it (large
    learned; the ``tanh`` in :meth:`FovealEncoder.update_gaze` saturates on the
    learned term).  No ``N_OUT`` change — it re-purposes the existing saccade
    outputs, so ``fast_reproduce`` / the genome are untouched.

    ``reflex_gain`` is SELF-CALIBRATED per §11b — NOT a fixed pixel step.  The raw
    pull is normalized by a per-env EMA of its OWN recent magnitude (seeded on the
    first sample so step 0 is bounded), so the emitted command is a dimensionless
    velocity that self-scales to the env's pull distribution and ``reflex_gain`` is
    a pure multiplier.  Reuses the exact [AC]/MotorClock per-env EMA pattern.

    **Zero rng, per-env** => engine-parity: env ``i``'s reflex depends only on its
    own obs (its proprio target + gaze) and its own EMA, which ticks only when that
    env is ready — identical serial-vs-furnace (mirrors ``MotorClock.decide``'s
    ``ready_idx`` handling).
    """

    _EPS = 1e-6  # pull-scale floor (guards the seed div; foveal pulls are ~O(1))

    def __init__(self, n: int, gain: float, ema_decay: float = 0.99):
        self.n = int(n)
        self.gain = float(gain)
        self.decay = float(ema_decay)
        # per-env EMA of the reflex pull magnitude + a step counter (game-level
        # calibration, NOT per-episode; like MotorClock's salience EMA it is not
        # cleared by reset — keeping it per-env is what preserves engine-parity).
        self._ema = np.zeros(self.n, dtype=np.float64)
        self._steps = np.zeros(self.n, dtype=np.int64)

    def command(self, pull_x, pull_y, ready_idx=None):
        """Self-calibrated reflex delta for the pull ``(pull_x, pull_y)`` (full-
        length ``(n,)`` arrays).  Returns full-length ``(rdx, rdy)`` that are ZERO
        off the ready set, so a caller can add them straight onto ``gdx``/``gdy``.
        Only the ``ready_idx`` rows advance their EMA (a non-ready env didn't step,
        so its calibration must not tick)."""
        rdx = np.zeros(self.n, dtype=np.float32)
        rdy = np.zeros(self.n, dtype=np.float32)
        idx = (
            np.arange(self.n) if ready_idx is None
            else np.asarray(ready_idx, dtype=np.intp)
        )
        if idx.size == 0:
            return rdx, rdy
        px = np.asarray(pull_x, dtype=np.float64)[idx]
        py = np.asarray(pull_y, dtype=np.float64)[idx]
        mag = np.sqrt(px * px + py * py)
        ema_old = self._ema[idx]
        # The scale-EMA calibrates on MEANINGFUL pulls only. A zero pull (e.g. the
        # reset frame, gaze == target) must NOT seed it — else the scale collapses to
        # ~0 and the next real pull divides by ~EPS, saturating the reflex and drowning
        # the learned saccade (top-down override goes inert). So emit + seed only when
        # mag > EPS: the first meaningful pull self-scales to a bounded ~gain step;
        # later pulls are EMA-normalized (above the env's baseline -> stronger orienting
        # jerk, below -> gentler). ``_steps`` counts MEANINGFUL pulls (the seeded flag).
        seeded = self._steps[idx] > 0
        meaningful = mag > self._EPS
        scale = np.where(seeded, ema_old, mag)            # unseeded -> self-scale
        inv = np.where(meaningful, self.gain / (scale + self._EPS), 0.0)
        rdx[idx] = (px * inv).astype(np.float32)
        rdy[idx] = (py * inv).astype(np.float32)
        d = self.decay
        upd = np.where(seeded, d * ema_old + (1.0 - d) * mag, mag)  # decay or seed
        self._ema[idx] = np.where(meaningful, upd, ema_old)        # zero pull -> hold
        self._steps[idx] = self._steps[idx] + meaningful.astype(np.int64)
        return rdx, rdy

    def blend(self, X, o_proprio, gdx, gdy, ready_idx=None):
        """Add the self-calibrated reflex delta to the learned saccade (§4).

        Reads the reflex TARGET (``proprio[o_proprio+14:+16]``) and the current
        gaze (``proprio[o_proprio:+2]``) straight from the obs matrix ``X`` — the
        SAME source in the parent-encoder and the worker paths, so the blend is
        engine-agnostic.  Returns the new ``(gdx, gdy)`` (reflex + learned)."""
        cur_x = X[:, o_proprio + 0]
        cur_y = X[:, o_proprio + 1]
        tgt_x = X[:, o_proprio + 14]
        tgt_y = X[:, o_proprio + 15]
        rdx, rdy = self.command(tgt_x - cur_x, tgt_y - cur_y, ready_idx=ready_idx)
        return gdx + rdx, gdy + rdy

    @staticmethod
    def maybe(encoder, n: int) -> "ReflexGaze | None":
        """Build a per-wave latch of width ``n`` when ``encoder.reflex_gaze`` is on,
        else ``None`` (the off-switch => the blend is skipped, gaze == legacy)."""
        if encoder is None or not getattr(encoder, "reflex_gaze", False):
            return None
        return ReflexGaze(
            n, float(getattr(encoder, "reflex_gain", 1.0)),
            float(getattr(encoder, "reflex_ema_decay", 0.99)),
        )


# ==========================================================================
# BarrierFleet — shared-memory, spin-barrier, in-worker hashing/encoding
# ==========================================================================
# Ops signalled to workers via the shared control block.
_OP_STEP = 0
_OP_RESET = 1
_OP_SHUTDOWN = 2

# Per-env emulator save_state blob is ~200 KB; this is only a FALLBACK bound for
# the shared capture/restore buffers when the real size can't be probed (A5).
# The live buffers are sized from an actual save_state measured at fleet init
# (see ``_probe_state_len`` / ``_state_capacity``), so a larger-state game gets
# buffers that fit instead of silently truncating every Go-Explore capture.
_MAX_STATE = 262144  # 256 KiB (fallback only)

# Headroom added over the probed save_state size so minor per-state variation
# (RTC banks, mapper quirks) can never overflow the buffer and force a skip.
_STATE_HEADROOM_MIN = 65536  # 64 KiB

# --- A8: hung-worker watchdog -------------------------------------------------
# The parent's barrier/ack hot-wait used to detect only a CRASHED worker
# (is_alive()==False); a worker wedged in a pathological tick or a never-
# returning save_state stays alive and busy-spins a parent core forever with no
# diagnostic. A generous per-round wall-clock deadline turns that invisible hang
# into a fail-fast RuntimeError naming the offending env index(es) + pid. The
# budget is derived from the round's work (env count x steps) with a large
# per-env-step margin so it can never false-trip a slow-but-legit round; it can
# also be pinned via POKEIO_ROUND_DEADLINE_S (or the constructor) for tests.
_PER_ENVSTEP_BUDGET_S = 0.5   # per env-step; ~40x the ~12 ms worst-case real step
_ROUND_DEADLINE_FLOOR_S = 60.0
try:
    _ROUND_DEADLINE_ENV = float(os.environ.get("POKEIO_ROUND_DEADLINE_S", "") or 0.0)
except ValueError:
    _ROUND_DEADLINE_ENV = 0.0


def _probe_state_len(rom_path, frame_skip, hold_frames, reset_state) -> int | None:
    """Actual byte length of a PyBoy ``save_state`` for this ROM.

    Mirrors :func:`_archive_key_len`: measure the real geometry once at fleet
    init rather than trusting a hardcoded cap. Returns ``None`` if the probe
    fails (caller falls back to :data:`_MAX_STATE`)."""
    try:
        import io as _io

        from pokeio.emu.env import PokeEnv

        env = PokeEnv(rom_path, frame_skip=frame_skip, hold_frames=hold_frames)
        try:
            env.reset(reset_state)
            buf = _io.BytesIO()
            env.pyboy.save_state(buf)  # exact call the workers make
            return len(buf.getvalue())
        finally:
            env.close()
    except Exception:
        return None


def _state_capacity(probe: int | None) -> int:
    """Buffer size for a save_state blob: probed size + generous headroom."""
    if not probe or probe <= 0:
        return _MAX_STATE
    return int(probe) + max(_STATE_HEADROOM_MIN, int(probe) // 4)


def _store_capture(cap_state_row: np.ndarray, blob: bytes, cap: int) -> int:
    """Write ``blob`` into ``cap_state_row`` iff it fits; NEVER truncate.

    Returns the stored length, or ``-1`` when the blob is larger than the buffer
    (the caller must then SKIP the capture — storing a truncated blob would
    corrupt the archive entry and every restore made from it). This replaces the
    old silent ``min(len(blob), _MAX_STATE)`` truncation (A5)."""
    n = len(blob)
    if n > cap:
        return -1
    cap_state_row[:n] = np.frombuffer(blob, dtype=np.uint8)
    return n


# Busy-spin by default. Counterintuitive but measured (2026-07 audit): on this
# Broadwell-EP box a core that naps between barrier hand-offs is clamped by the
# hardware p-state logic to its MINIMUM frequency (1.2 GHz) even at ~70% duty —
# and the clamp survives governor=performance, min_freq pinning, EPB=0 and
# C-state disabling. Every ~2 ms emulator step then takes ~5 ms, and the barrier
# waits on the slowest of N such steps. Keeping waiting cores hot is worth 3.1x
# end-to-end (P32: 1,638 -> 5,057 sps). Set POKEIO_NAP=1 to restore yielding
# waits when sharing the box with other workloads.
_NAP = os.environ.get("POKEIO_NAP", "") not in ("", "0")


def _spin_wait(pred, spin_budget: int = 3000, nap: float = 5e-5) -> None:
    """Wait on ``pred`` (a cheap shm read): busy spin (default) or spin-then-nap.

    The barrier flags live in shared memory, so the wait is a tight numpy read
    rather than a blocking pipe round-trip (which caps the pipe-based VecFleet at
    ~2.9k steps/s).  A pure busy spin looks wasteful — the waiting side is idle
    while the other side works — but napping instead down-clocks the core to
    1.2 GHz and slows the WORK phases 2.6x (see ``_NAP`` above), which costs far
    more than the spin burns.  ``POKEIO_NAP=1`` opts into the polite behaviour.
    """
    if not _NAP:
        i = 0
        while not pred():
            i += 1
            if i % 1_000_000 == 0 and os.getppid() == 1:
                # Parent died and we were reparented to init: a hot spin would
                # otherwise burn this core forever (the old nap version merely
                # leaked a sleeping process). Exit instead of orphan-spinning.
                raise SystemExit(1)
        return
    i = 0
    while not pred():
        i += 1
        if i >= spin_budget:
            time.sleep(nap)


def _paced_wait(pred, nap: float = 2e-3) -> None:
    """Sleep-wait used ONLY while the parent's shm pace flag is set (realtime
    spectate mode).  At the paced ~2.5 rounds/s a busy spin would burn every
    worker core ~99% idle; sleeping instead lets the box go quiet.  The 1.2 GHz
    p-state clamp that makes napping catastrophic in max mode (see ``_NAP``) is
    IRRELEVANT here: a clamped ~5 ms emulator step inside a 400 ms round budget
    changes nothing.  Keeps the parent-death getppid escape of the spin path.
    """
    i = 0
    while not pred():
        time.sleep(nap)
        i += 1
        if i % 512 == 0 and os.getppid() == 1:
            raise SystemExit(1)  # orphaned (parent died): exit, don't leak


def _barrier_worker_main(
    slice_lo, slice_hi, rom_path, frame_skip, hold_frames, reset_state,
    obs_res, obs_ram, wram_stride, arch_kwargs, goexplore, expose_wram,
    shm_names, n_envs, obs_dim, key_len, core, state_cap,
    periph_grid, fovea_native_px, fovea_grid, saccade_gain, saccade_every_k,
    episode_steps, foveal_memory, mem_grid, mem_ema_decay, mem_stale_z,
    mem_stale_warmup, reflex_gaze, reflex_gain, reflex_ema_decay, reflex_beta,
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

    # Active-vision encoder (stateful per env; §2/§3), auto-selected from obs_dim:
    #   obs_dim == 2*G^2+FG^2+[2*M^2]+14/16+n_ram -> foveal obs (Phase-0 pixel
    #                                    vectors; 454 when FG==G, larger for a sharp
    #                                    fovea / +2*M^2 for the §1a trans-saccadic
    #                                    memory / +2 proprio for §4 reflex gaze)
    #   obs_dim == 2*84^2+14+n_ram      -> retina 14134 obs (Phase-1 pixel input)
    # else fall back to the legacy flat ObsEncoder (back-compat res^2+ram obs).
    # n_envs-wide so it can be indexed by the GLOBAL env id (= shm obs rows).
    _fg = int(fovea_grid) if int(fovea_grid) > 0 else int(periph_grid)
    _mem_extra = (2 * int(mem_grid) * int(mem_grid)) if foveal_memory else 0
    _npro = 16 if reflex_gaze else 14  # reflex gaze adds 2 proprio dims (§4)
    _foveal_dim = 2 * periph_grid * periph_grid + _fg * _fg + _mem_extra + _npro + obs_ram
    _retina_dim = 2 * _RETINA_SIDE * _RETINA_SIDE + 14 + obs_ram
    use_retina = int(obs_dim) == int(_retina_dim)
    use_foveal = int(obs_dim) == int(_foveal_dim)
    use_active = use_foveal or use_retina  # stateful FovealEncoder (gaze/saccade)
    if use_active:
        encoder = FovealEncoder(
            n_envs, periph_grid=periph_grid, fovea_native_px=fovea_native_px,
            fovea_grid=fovea_grid, n_ram=obs_ram, saccade_gain=saccade_gain,
            saccade_every_k=saccade_every_k, episode_steps=episode_steps,
            mode=("retina" if use_retina else "foveal"),
            foveal_memory=bool(foveal_memory), mem_grid=int(mem_grid),
            mem_ema_decay=float(mem_ema_decay), mem_stale_z=float(mem_stale_z),
            mem_stale_warmup=int(mem_stale_warmup), reflex_gaze=bool(reflex_gaze),
            reflex_gain=float(reflex_gain), reflex_ema_decay=float(reflex_ema_decay),
            reflex_beta=float(reflex_beta),
        )
    else:
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
    gaze_dx = reg("gaze_dx", (n_envs,), np.float32)  # saccade cmd (exact float)
    gaze_dy = reg("gaze_dy", (n_envs,), np.float32)
    dones = reg("dones", (n_envs,), np.uint8)
    cap_flag = reg("cap_flag", (n_envs,), np.uint8)
    cap_done = reg("cap_done", (n_envs,), np.uint8)
    cap_state = reg("cap_state", (n_envs, state_cap), np.uint8)
    cap_len = reg("cap_len", (n_envs,), np.int32)
    cap_trunc = reg("cap_trunc", (n_envs,), np.int64)  # skipped (oversize) captures
    res_flag = reg("res_flag", (n_envs,), np.uint8)
    res_state = reg("res_state", (n_envs, state_cap), np.uint8)
    res_len = reg("res_len", (n_envs,), np.int32)
    # [0]=go_round [1]=op [2]=pace(1=realtime sleep-waits) [3+i]=wdone_i
    ctl = reg("ctl", (3 + n_envs,), np.int64)
    # mined progress-counter taps: [0]=version, [1:]=GB addresses (0=unset).
    tapcfg = reg("tapcfg", (1 + obs_ram,), np.int64)
    # OPT-IN full-WRAM export (default OFF): when set, the parent allocated a
    # (n_envs, 8192) uint8 block so an out-of-band reward loop can read raw game
    # state. Only attach it when the flag is set — the NEAT/furnace path never
    # allocates it, so this stays byte-identical when disabled.
    wram_full = reg("wram", (n_envs, WRAM_END_LEN), np.uint8) if expose_wram else None

    envs = [
        PokeEnv(rom_path, frame_skip=frame_skip, hold_frames=hold_frames)
        for _ in range(slice_lo, slice_hi)
    ]

    tap_ver = 0
    taps: list[tuple[int, int]] = []  # (obs column, GB address)
    _tap_base = obs_dim - obs_ram

    def _emit(local_i, global_i, screen, w64, button):
        env = envs[local_i]
        screens[global_i] = screen
        if use_active:
            obs[global_i] = encoder.encode(global_i, screen, w64, button=button)
        else:
            obs[global_i] = encoder.encode_compact(screen, w64)
        # Overlay mined counter taps identically to the async path (was MISSING
        # here — connect-protected taps otherwise saw the blind stride sample).
        if taps:
            mem = env.pyboy.memory
            for col, a in taps:
                obs[global_i, col] = mem[a] / 255.0
        # Opt-in: publish the full 8 KB WRAM snapshot for the parent reward loop.
        if expose_wram:
            wram_full[global_i] = env.raw_wram()
        k = archive.cell_key_compact(screen, w64)
        keys[global_i] = np.frombuffer(k, dtype=np.uint8)

    local_round = 1
    warned_trunc = False
    try:
        while True:
            # Pace flag checked ONCE per round (not per spin iteration): when
            # clear the wait below is the exact busy-spin hot path; when set
            # (realtime spectate) the worker sleep-waits between rounds.  A
            # mid-wait mode flip takes effect on the next round (<= 1 round).
            if ctl[2]:
                _paced_wait(lambda: ctl[0] >= local_round)
            else:
                _spin_wait(lambda: ctl[0] >= local_round)
            op = int(ctl[1])
            if op == _OP_SHUTDOWN:
                break
            if op == _OP_RESET and int(tapcfg[0]) != tap_ver:  # new mined taps
                tap_ver = int(tapcfg[0])
                taps = [
                    (_tap_base + j, int(tapcfg[1 + j]))
                    for j in range(obs_ram)
                    if int(tapcfg[1 + j]) > 0
                ]
            for li, gi in enumerate(range(slice_lo, slice_hi)):
                env = envs[li]
                if op == _OP_RESET:
                    if use_active:
                        encoder.reset(gi)  # gaze -> centre, motion -> 0.5
                    if goexplore and res_flag[gi]:
                        env.load_state(bytes(res_state[gi, : int(res_len[gi])]))
                        env.pyboy.tick(1, True)
                        screen = env._obs()
                        w64 = env.wram_strided(wram_stride)
                    else:
                        screen = env.reset(reset_state)
                        w64 = env.wram_strided(wram_stride)
                    dones[gi] = 0
                    _emit(li, gi, screen, w64, 8)  # reset obs: last button = NOOP
                else:  # _OP_STEP
                    # Integrate this step's saccade command BEFORE _emit builds
                    # the obs, so the fovea crop + proprio see the new gaze (§3.4).
                    if use_active:
                        encoder.update_gaze(gi, float(gaze_dx[gi]), float(gaze_dy[gi]))
                    # Deferred Go-Explore capture: save the state we are STILL in
                    # (from last round) before applying this round's action.
                    if goexplore and cap_flag[gi]:
                        buf = _io.BytesIO()
                        env.pyboy.save_state(buf)
                        blob = buf.getvalue()
                        n = _store_capture(cap_state[gi], blob, state_cap)
                        if n < 0:
                            # Too big for the buffer: SKIP, never truncate (a
                            # truncated blob would corrupt every restore made
                            # from this cell). Count it so the parent can flag it.
                            cap_done[gi] = 0
                            cap_trunc[gi] += 1
                            if not warned_trunc:
                                import sys as _sys
                                print(
                                    f"[fleet pid={os.getpid()}] save_state "
                                    f"{len(blob)}B > cap {state_cap}B; SKIPPING "
                                    f"Go-Explore capture (env {gi}) — raise "
                                    f"buffer headroom", file=_sys.stderr, flush=True,
                                )
                                warned_trunc = True
                        else:
                            cap_len[gi] = n
                            cap_done[gi] = 1
                    else:
                        cap_done[gi] = 0
                    screen, w64, done = env.step_fast(int(actions[gi]), wram_stride)
                    dones[gi] = 1 if done else 0
                    _emit(li, gi, screen, w64, int(actions[gi]))
            # signal this round complete
            for gi in range(slice_lo, slice_hi):
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
        expose_wram: bool = False,
        envs_per_worker: int = 1,
        round_deadline_s: float | None = None,
        periph_grid: int = 12,
        fovea_native_px: int = 48,
        fovea_grid: int = 0,
        saccade_gain: float = 32.0,
        saccade_every_k: int = 1,
        episode_steps: int = 1024,
        foveal_memory: bool = False,
        mem_grid: int = 24,
        mem_ema_decay: float = 0.99,
        mem_stale_z: float = 1.5,
        mem_stale_warmup: int = 16,
        reflex_gaze: bool = False,
        reflex_gain: float = 1.0,
        reflex_ema_decay: float = 0.99,
        reflex_beta: float = 4.0,
    ):
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.obs_ram = int(obs_ram)
        self.goexplore = bool(goexplore)
        # OPT-IN: expose full per-env WRAM (8 KB) to the parent (default OFF, so
        # the NEAT/furnace path is byte-identical — no extra alloc/reg/read).
        self.expose_wram = bool(expose_wram)
        self.wram_stride = int(wram_stride)
        # Splatted (order-critical) into the worker main after the fixed args;
        # keep in sync with the _barrier_worker_main signature.
        self._foveal = (
            int(periph_grid), int(fovea_native_px), int(fovea_grid),
            float(saccade_gain), int(saccade_every_k), int(episode_steps),
            bool(foveal_memory), int(mem_grid), float(mem_ema_decay),
            float(mem_stale_z), int(mem_stale_warmup),
            bool(reflex_gaze), float(reflex_gain), float(reflex_ema_decay),
            float(reflex_beta),
        )

        # Derive the fixed cell-key length from the archive's geometry.
        probe = _archive_key_len(archive_kwargs, wram_stride)
        self.key_len = probe

        # A5: size the Go-Explore state buffers from an ACTUAL save_state (probed
        # once here, mirroring the key-length probe) instead of a hardcoded cap,
        # so a larger-state game never silently truncates a capture. Only worth
        # the ~1 emulator boot when go-explore is on; otherwise the (unused)
        # buffers stay at the fallback size.
        self.state_cap = (
            _state_capacity(
                _probe_state_len(rom_path, frame_skip, hold_frames, reset_state)
            )
            if self.goexplore
            else _MAX_STATE
        )

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
        alloc("gaze_dx", (n,), np.float32)  # saccade cmd dx (exact, per env)
        alloc("gaze_dy", (n,), np.float32)  # saccade cmd dy (exact, per env)
        alloc("dones", (n,), np.uint8)
        alloc("cap_flag", (n,), np.uint8)
        alloc("cap_done", (n,), np.uint8)
        alloc("cap_state", (n, self.state_cap), np.uint8)
        alloc("cap_len", (n,), np.int32)
        alloc("cap_trunc", (n,), np.int64)  # A5: skipped (oversize) captures
        alloc("res_flag", (n,), np.uint8)
        alloc("res_state", (n, self.state_cap), np.uint8)
        alloc("res_len", (n,), np.int32)
        alloc("ctl", (3 + n,), np.int64)
        # mined progress-counter taps: [0]=version, [1:]=GB addresses (0=unset).
        alloc("tapcfg", (1 + self.obs_ram,), np.int64)
        # OPT-IN full-WRAM export (default OFF): only alloc'd when requested, so
        # the production NEAT/furnace path never pays for it (no shm block, and
        # its name never enters _shm_names -> workers never attach it).
        if self.expose_wram:
            alloc("wram", (n, WRAM_END_LEN), np.uint8)

        self._ctl = self.arr["ctl"]
        self._ctl[:] = 0
        self._round = 0
        self._paced = False  # parent-side mirror of ctl[2] (realtime spectate)
        self._shm_names = {k: v.name for k, v in self._blocks.items()}

        # ------------------------------------------------------------ workers
        self._ctx = mp.get_context("spawn")
        core_order = _numa_core_order()
        epw = max(1, int(envs_per_worker))
        self._epw = epw
        # A8: generous per-round wall-clock deadline (see module constants).
        self._round_budget = _ROUND_DEADLINE_ENV or round_deadline_s or max(
            _ROUND_DEADLINE_FLOOR_S, epw * _PER_ENVSTEP_BUDGET_S
        )
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
                    self.goexplore, self.expose_wram, self._shm_names, n,
                    self.obs_dim, self.key_len, core, self.state_cap,
                    *self._foveal,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            self._slices.append((lo, hi))
            wi += 1
        self.n_workers = len(self._procs)
        self._closed = False

    # ------------------------------------------------------------------ pacing
    def set_pace(self, realtime: bool) -> None:
        """Flip the shm pace flag: realtime -> workers (and the parent's
        round-wait) use sleep-waits between rounds; max -> pure busy-spin,
        bit-for-bit the pre-pace behaviour.  Safe to call any time; workers
        pick it up at their next round boundary."""
        self._paced = bool(realtime)
        self._ctl[2] = 1 if realtime else 0

    # ------------------------------------------------------------------ barrier
    def _release_round(self, op: int) -> None:
        """Release the workers into a new round (returns immediately)."""
        self._round += 1
        self._ctl[1] = op
        self._ctl[0] = self._round  # release workers

    def _run_round(self, op: int) -> None:
        self._release_round(op)
        self._await_round()

    def _round_timeout_error(self, target: int) -> RuntimeError:
        """Build a diagnostic for a round that blew its wall-clock deadline (A8).

        A crashed worker never flips its flag AND fails ``is_alive`` (caught by
        the liveness guard); a HUNG-but-alive worker never flips its flag yet
        stays alive, so only the deadline catches it. Names the stuck env
        index(es) and the owning worker pid so the wedge is actionable."""
        wdone = self._ctl[3:3 + self.n_envs]
        stuck = [i for i in range(self.n_envs) if int(wdone[i]) < target]
        parts = []
        for (lo, hi), p in zip(self._slices, self._procs):
            owned = [i for i in stuck if lo <= i < hi]
            if owned:
                parts.append(f"envs{owned}@pid{p.pid}(alive={p.is_alive()})")
        return RuntimeError(
            f"BarrierFleet round exceeded {self._round_budget:.1f}s deadline; "
            f"hung worker(s): {'; '.join(parts) or 'unknown'} — aborting"
        )

    def _await_round(self) -> None:
        target = self._round
        ctl = self._ctl
        n = self.n_envs
        wdone = ctl[3:3 + n]
        deadline = time.monotonic() + self._round_budget
        i = 0
        if self._paced:
            # Realtime spectate: the parent sleeps too (the p-state clamp is
            # irrelevant inside a 400 ms round budget — see _paced_wait).
            while not bool((wdone >= target).all()):
                time.sleep(1e-3)
                i += 1
                if i % 1000 == 0:
                    if any(not p.is_alive() for p in self._procs):
                        raise RuntimeError(
                            "BarrierFleet worker died mid-round; aborting (see stderr)"
                        )
                    if time.monotonic() > deadline:
                        raise self._round_timeout_error(target)
            return
        if not _NAP:
            # Hot wait (see _NAP): never sleep, or this core drops to 1.2 GHz and
            # the next forward/bookkeeping phase runs 2.6x slow. Liveness +
            # deadline guards kept on a coarse period.
            while not bool((wdone >= target).all()):
                i += 1
                if i % 200000 == 0:
                    if any(not p.is_alive() for p in self._procs):
                        raise RuntimeError(
                            "BarrierFleet worker died mid-round; aborting (see stderr)"
                        )
                    if time.monotonic() > deadline:
                        raise self._round_timeout_error(target)
            return
        while not bool((wdone >= target).all()):
            i += 1
            if i >= 3000:
                # Liveness + deadline guards: a crashed worker can never flip its
                # flag (caught by is_alive), a hung-but-alive one only by the
                # deadline. Checked periodically so the spin stays cheap.
                if i % 20000 == 0:
                    if any(not p.is_alive() for p in self._procs):
                        raise RuntimeError(
                            "BarrierFleet worker died mid-round; aborting (see stderr)"
                        )
                    if time.monotonic() > deadline:
                        raise self._round_timeout_error(target)
                time.sleep(5e-5)

    # ------------------------------------------------------------------ control
    def reset_all_begin(self, restore: dict[int, bytes] | None = None) -> None:
        """Ship restore blobs + release the reset round WITHOUT waiting.

        The caller may do unrelated work (e.g. pack/compile the next wave's
        genomes) while the workers reset, then call :meth:`reset_all_end`."""
        self.arr["res_flag"][:] = 0
        if restore and self.goexplore:
            for idx, blob in restore.items():
                if len(blob) > self.state_cap:
                    # Never restore a truncated state (A5): captures are never
                    # stored truncated, so this should not happen — skip rather
                    # than load a corrupt partial blob.
                    continue
                self.arr["res_state"][idx, : len(blob)] = np.frombuffer(
                    blob, dtype=np.uint8
                )
                self.arr["res_len"][idx] = len(blob)
                self.arr["res_flag"][idx] = 1
        self._release_round(_OP_RESET)

    @property
    def state_truncations(self) -> int:
        """Total Go-Explore captures SKIPPED because the blob outgrew the buffer
        (A5). Non-zero means the probed buffer headroom is too small — captures
        are dropped, never truncated, so the archive stays uncorrupted."""
        return int(self.arr["cap_trunc"].sum()) if "cap_trunc" in self.arr else 0

    def reset_all_end(self) -> np.ndarray:
        """Wait for the reset round released by :meth:`reset_all_begin`."""
        self._await_round()
        return self.arr["obs"].copy()

    def reset_all(self, restore: dict[int, bytes] | None = None) -> np.ndarray:
        """Reset all envs (optionally restoring per-index Go-Explore states).

        ``restore`` maps env index -> save_state blob; those envs load the blob
        instead of the canonical reset state.  Returns the encoded obs block
        (a copy, safe across the next round)."""
        self.reset_all_begin(restore)
        return self.reset_all_end()

    def step_all(
        self,
        actions: np.ndarray,
        capture_flags: np.ndarray | None = None,
        gaze_dx: np.ndarray | None = None,
        gaze_dy: np.ndarray | None = None,
    ):
        """Advance one agent-step; return ``(obs, keys, dones, captured)``.

        ``actions`` (len n int) is the per-env button id. ``gaze_dx``/``gaze_dy``
        (len n float, exact — no quantization) are the per-env saccade commands;
        each worker integrates them into its gaze (§3.3) BEFORE building the obs.
        Omit them (None) to hold the gaze fixed this step (dx=dy=0).
        ``capture_flags`` (len n, uint8) requests a Go-Explore state capture for
        the flagged envs BEFORE they apply this step's action — i.e. it captures
        the state reported in the *previous* round.  ``captured`` maps env index
        -> state bytes for envs that captured this round.

        ``capture_flags`` stays the 2nd positional arg for back-compat with the
        pre-saccade call ``step_all(actions, capture_flags)``; pass the saccade
        by keyword: ``step_all(actions, capture_flags=cf, gaze_dx=..., gaze_dy=...)``.
        """
        self.arr["actions"][: len(actions)] = np.asarray(actions, dtype=np.int32)
        if gaze_dx is not None:
            self.arr["gaze_dx"][: len(gaze_dx)] = np.asarray(gaze_dx, dtype=np.float32)
        else:
            self.arr["gaze_dx"][:] = 0.0
        if gaze_dy is not None:
            self.arr["gaze_dy"][: len(gaze_dy)] = np.asarray(gaze_dy, dtype=np.float32)
        else:
            self.arr["gaze_dy"][:] = 0.0
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

    def set_tap_addrs(self, addrs: list[int]) -> None:
        """Point the obs RAM tail at mined progress-counter addresses.

        Workers re-read the tap config at the next reset round (generation
        boundary), so call this BEFORE :meth:`reset_all_begin`. Up to
        ``obs_ram`` GB addresses; unset slots keep the blind stride sample.
        Mirrors :meth:`AsyncFleet.set_tap_addrs` so the barrier path overlays
        the same connect-protected taps (previously it never did)."""
        cfg = self.arr["tapcfg"]
        k = min(len(addrs), self.obs_ram)
        cfg[1 : 1 + k] = np.asarray(addrs[:k], dtype=np.int64)
        cfg[1 + k :] = 0
        cfg[0] += 1  # version bump LAST (x86 TSO: workers see addrs first)

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


# ==========================================================================
# AsyncFleet — free-running workers, no per-step barrier ("furnace" engine)
# ==========================================================================
# Measured motivation (2026-07 audit2/audit3, quiet box, 112 players / 28
# workers, goexplore on): the barrier round spends ~34 ms waiting for the
# SLOWEST worker (mean worker does ~19 ms of work — emulation cost is game-
# state dependent, 3.7 ms typical vs 9-12 ms in scroll/dialog scenes) plus
# ~5 ms of GPU forward all workers sit through. Removing the lockstep turns
# max-of-workers into mean-of-workers and overlaps the forward with stepping:
# 2,797 -> 8,129 steps/s (2.9x) with identical novelty/Go-Explore semantics.
#
# Protocol: two per-env sequence counters in shared memory.
#
#   obs_seq[i] = k   worker published the obs after step k   (k=0: reset obs)
#   act_seq[i] = k   parent published the action to apply to obs k
#
# A worker steps env i exactly when ``act_seq[i] == obs_seq[i]`` (the action
# for its newest obs has arrived) and ``obs_seq[i] < target``. The parent
# loops: snapshot obs_seq, bookkeep every newly published obs, forward the
# population once, write actions for the envs that were ready, bump their
# act_seq. Ordinary x86 store order (payload row first, sequence counter
# second) makes the handshake safe without locks; each env advances at most
# one step per parent cycle, so the parent never misses an obs.
#
# Consequences:
#   * a slow env only slows itself (no straggler tax),
#   * a ~47 ms Go-Explore save_state stalls one worker's slice, not the fleet,
#   * per-env trajectories are BIT-IDENTICAL to the barrier engine (an env's
#     action k depends only on its own obs k), but the ORDER envs hit the
#     novelty archive within a generation is timing-dependent, so rarity
#     credit and capture ownership can differ run-to-run.

_OP_RUN = 3  # barrier ops: 0=step 1=reset 2=shutdown (shared numbering)


def _async_worker_main(
    slice_lo, slice_hi, rom_path, frame_skip, hold_frames, reset_state,
    obs_res, obs_ram, wram_stride, arch_kwargs, goexplore, expose_wram,
    shm_names, n_envs, obs_dim, key_len, core, state_cap,
    periph_grid, fovea_native_px, fovea_grid, saccade_gain, saccade_every_k,
    episode_steps, foveal_memory, mem_grid, mem_ema_decay, mem_stale_z,
    mem_stale_warmup, reflex_gaze, reflex_gain, reflex_ema_decay, reflex_beta,
):
    """Free-running worker: owns envs ``[slice_lo:slice_hi]``.

    Round-robins its envs, stepping any env whose next action has arrived;
    hashes + encodes locally (same as the barrier worker). Between actions it
    nap-spins (200 hot polls then 50 us sleeps) — the p-state clamp that makes
    napping catastrophic for the barrier's sub-ms hand-offs is a non-issue at
    the multi-ms cadence of a free-running env slice, and the validated 8.1k
    steps/s configuration ran exactly this wait.
    """
    import io as _io

    from pokeio.emu.env import PokeEnv
    from pokeio.reward.archive import NoveltyArchive

    try:
        if core is not None:
            psutil.Process().cpu_affinity([core])
    except Exception:
        pass

    # Active-vision encoder (stateful per env; §2/§3), auto-selected from obs_dim:
    #   obs_dim == 2*G^2+FG^2+[2*M^2]+14/16+n_ram -> foveal obs (454 when FG==G,
    #                                    larger for a sharp fovea / +2*M^2 for §1a
    #                                    memory / +2 proprio for §4 reflex gaze)
    #   obs_dim == 2*84^2+14+n_ram      -> retina 14134 obs (Phase-1 pixel input)
    # else the legacy flat ObsEncoder (back-compat). Indexed by GLOBAL env id to
    # match the shm obs rows.
    _fg = int(fovea_grid) if int(fovea_grid) > 0 else int(periph_grid)
    _mem_extra = (2 * int(mem_grid) * int(mem_grid)) if foveal_memory else 0
    _npro = 16 if reflex_gaze else 14  # reflex gaze adds 2 proprio dims (§4)
    _foveal_dim = 2 * periph_grid * periph_grid + _fg * _fg + _mem_extra + _npro + obs_ram
    _retina_dim = 2 * _RETINA_SIDE * _RETINA_SIDE + 14 + obs_ram
    use_retina = int(obs_dim) == int(_retina_dim)
    use_foveal = int(obs_dim) == int(_foveal_dim)
    use_active = use_foveal or use_retina  # stateful FovealEncoder (gaze/saccade)
    if use_active:
        encoder = FovealEncoder(
            n_envs, periph_grid=periph_grid, fovea_native_px=fovea_native_px,
            fovea_grid=fovea_grid, n_ram=obs_ram, saccade_gain=saccade_gain,
            saccade_every_k=saccade_every_k, episode_steps=episode_steps,
            mode=("retina" if use_retina else "foveal"),
            foveal_memory=bool(foveal_memory), mem_grid=int(mem_grid),
            mem_ema_decay=float(mem_ema_decay), mem_stale_z=float(mem_stale_z),
            mem_stale_warmup=int(mem_stale_warmup), reflex_gaze=bool(reflex_gaze),
            reflex_gain=float(reflex_gain), reflex_ema_decay=float(reflex_ema_decay),
            reflex_beta=float(reflex_beta),
        )
    else:
        encoder = ObsEncoder(obs_res, obs_ram)
    archive = NoveltyArchive(**arch_kwargs)  # stateless key hasher

    shms = []

    def reg(name, shape, dtype):
        shm = shared_memory.SharedMemory(name=shm_names[name])
        shms.append(shm)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    obs = reg("obs", (n_envs, obs_dim), np.float32)
    screens = reg("screens", (n_envs, _SCREEN_H, _SCREEN_W), np.uint8)
    keys = reg("keys", (n_envs, key_len), np.uint8)
    actions = reg("actions", (n_envs,), np.int32)
    gaze_dx = reg("gaze_dx", (n_envs,), np.float32)  # saccade cmd (exact float)
    gaze_dy = reg("gaze_dy", (n_envs,), np.float32)
    dones = reg("dones", (n_envs,), np.uint8)
    obs_seq = reg("obs_seq", (n_envs,), np.int64)
    act_seq = reg("act_seq", (n_envs,), np.int64)
    cap_flag = reg("cap_flag", (n_envs,), np.uint8)
    cap_done = reg("cap_done", (n_envs,), np.uint8)
    cap_state = reg("cap_state", (n_envs, state_cap), np.uint8)
    cap_len = reg("cap_len", (n_envs,), np.int32)
    cap_trunc = reg("cap_trunc", (n_envs,), np.int64)  # skipped (oversize) captures
    res_flag = reg("res_flag", (n_envs,), np.uint8)
    res_state = reg("res_state", (n_envs, state_cap), np.uint8)
    res_len = reg("res_len", (n_envs,), np.int32)
    # [0]=round [1]=op [2]=pace [3]=target_steps [4+i]=ack_i
    ctl = reg("ctl", (4 + n_envs,), np.int64)
    # mined progress-counter taps: [0]=version, [1:]=GB addresses (0=unset)
    tapcfg = reg("tapcfg", (1 + obs_ram,), np.int64)
    # OPT-IN full-WRAM export (default OFF): only attach when the parent asked
    # for it (and hence allocated the (n_envs, 8192) uint8 block). Disabled ->
    # this reg never runs, keeping the furnace path byte-identical.
    wram_full = reg("wram", (n_envs, WRAM_END_LEN), np.uint8) if expose_wram else None

    envs = [
        PokeEnv(rom_path, frame_skip=frame_skip, hold_frames=hold_frames)
        for _ in range(slice_lo, slice_hi)
    ]
    my = list(range(slice_lo, slice_hi))

    tap_ver = 0
    taps: list[tuple[int, int]] = []  # (obs column, GB address)
    warned_trunc = False

    def _emit(gi, env, screen, w64, button):
        screens[gi] = screen
        if use_active:
            obs[gi] = encoder.encode(gi, screen, w64, button=button)
        else:
            obs[gi] = encoder.encode_compact(screen, w64)
        # Mined counter taps replace the blind stride sample in the RAM tail:
        # the agent SEES its progress counters, and the parent reads them from
        # the obs rows it already books to score progress fitness.
        if taps:
            mem = env.pyboy.memory
            for col, a in taps:
                obs[gi, col] = mem[a] / 255.0
        # Opt-in: publish the full 8 KB WRAM snapshot for the parent reward loop.
        if expose_wram:
            wram_full[gi] = env.raw_wram()
        k = archive.cell_key_compact(screen, w64)
        keys[gi] = np.frombuffer(k, dtype=np.uint8)

    local_round = 1
    try:
        while True:
            # ---- wait for the next round release (reset / run / shutdown)
            i = 0
            while ctl[0] < local_round:
                time.sleep(2e-3 if ctl[2] else 5e-5)
                i += 1
                if i % 512 == 0 and os.getppid() == 1:
                    raise SystemExit(1)  # orphaned: exit, don't leak
            op = int(ctl[1])
            if op == _OP_SHUTDOWN:
                break
            if op == _OP_RESET:
                if int(tapcfg[0]) != tap_ver:  # new mined taps this gen
                    tap_ver = int(tapcfg[0])
                    base = obs_dim - obs_ram
                    taps = [
                        (base + j, int(tapcfg[1 + j]))
                        for j in range(obs_ram)
                        if int(tapcfg[1 + j]) > 0
                    ]
                for li, gi in enumerate(my):
                    env = envs[li]
                    if use_active:
                        encoder.reset(gi)  # gaze -> centre, motion -> 0.5
                    if goexplore and res_flag[gi]:
                        env.load_state(bytes(res_state[gi, : int(res_len[gi])]))
                        env.pyboy.tick(1, True)
                        screen = env._obs()
                        w64 = env.wram_strided(wram_stride)
                    else:
                        screen = env.reset(reset_state)
                        w64 = env.wram_strided(wram_stride)
                    dones[gi] = 0
                    _emit(gi, env, screen, w64, 8)  # reset obs: last button = NOOP
                    obs_seq[gi] = 0
                    ctl[4 + gi] = local_round
                local_round += 1
                continue
            # ---- _OP_RUN: free-run until every owned env reaches the target
            target = int(ctl[3])
            if use_active:
                encoder.episode_steps = max(1, target)  # step_frac denominator
            spins = 0
            while True:
                progressed = False
                for li, gi in enumerate(my):
                    k = int(obs_seq[gi])
                    if k >= target or act_seq[gi] != k:
                        continue
                    env = envs[li]
                    # Integrate obs k's saccade command into the gaze BEFORE the
                    # step + _emit build obs k+1 (§3.4; one-tick efference delay).
                    if use_active:
                        encoder.update_gaze(gi, float(gaze_dx[gi]), float(gaze_dy[gi]))
                    # Deferred Go-Explore capture: save the state we are STILL
                    # in (obs k) before applying obs k's action — same semantics
                    # as the barrier engine, minus the fleet-wide stall.
                    if goexplore and cap_flag[gi]:
                        buf = _io.BytesIO()
                        env.pyboy.save_state(buf)
                        blob = buf.getvalue()
                        nb = _store_capture(cap_state[gi], blob, state_cap)
                        cap_flag[gi] = 0
                        if nb < 0:
                            # Too big: SKIP, never truncate (A5) — a truncated
                            # blob would corrupt every restore from this cell.
                            cap_done[gi] = 0
                            cap_trunc[gi] += 1
                            if not warned_trunc:
                                import sys as _sys
                                print(
                                    f"[fleet pid={os.getpid()}] save_state "
                                    f"{len(blob)}B > cap {state_cap}B; SKIPPING "
                                    f"Go-Explore capture (env {gi}) — raise "
                                    f"buffer headroom", file=_sys.stderr, flush=True,
                                )
                                warned_trunc = True
                        else:
                            cap_len[gi] = nb
                            cap_done[gi] = 1
                    screen, w64, done = env.step_fast(
                        int(actions[gi]), wram_stride
                    )
                    dones[gi] = 1 if done else 0
                    _emit(gi, env, screen, w64, int(actions[gi]))
                    obs_seq[gi] = k + 1  # publish AFTER the payload rows
                    progressed = True
                if progressed:
                    spins = 0
                    continue
                if all(obs_seq[gi] >= target for gi in my):
                    break
                spins += 1
                if ctl[2]:
                    time.sleep(2e-3)  # realtime spectate: box goes quiet
                elif spins >= 200:
                    time.sleep(5e-5)
                    if spins % 4096 == 0 and os.getppid() == 1:
                        raise SystemExit(1)
            for gi in my:
                ctl[4 + gi] = local_round
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
    """Free-running shared-memory fleet (the "furnace" engine).

    Construction args match :class:`BarrierFleet` so the trainer can swap
    engines with a flag. Per-wave driving differs: the parent calls
    ``reset_all_begin/_end`` (identical), then ``begin_wave(n, steps)`` and
    runs its own act/observe loop against the ``obs_seq``/``act_seq`` counters
    (see ``evaluate_wave_async``), then ``end_wave()``.
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
        expose_wram: bool = False,
        envs_per_worker: int = 1,
        round_deadline_s: float | None = None,
        periph_grid: int = 12,
        fovea_native_px: int = 48,
        fovea_grid: int = 0,
        saccade_gain: float = 32.0,
        saccade_every_k: int = 1,
        episode_steps: int = 1024,
        foveal_memory: bool = False,
        mem_grid: int = 24,
        mem_ema_decay: float = 0.99,
        mem_stale_z: float = 1.5,
        mem_stale_warmup: int = 16,
        reflex_gaze: bool = False,
        reflex_gain: float = 1.0,
        reflex_ema_decay: float = 0.99,
        reflex_beta: float = 4.0,
    ):
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.obs_ram = int(obs_ram)
        self.goexplore = bool(goexplore)
        # OPT-IN: expose full per-env WRAM (8 KB) to the parent (default OFF, so
        # the furnace/NEAT path is byte-identical — no extra alloc/reg/read).
        self.expose_wram = bool(expose_wram)
        self.wram_stride = int(wram_stride)
        # Splatted (order-critical) into the worker main after the fixed args;
        # keep in sync with the _async_worker_main signature.
        self._foveal = (
            int(periph_grid), int(fovea_native_px), int(fovea_grid),
            float(saccade_gain), int(saccade_every_k), int(episode_steps),
            bool(foveal_memory), int(mem_grid), float(mem_ema_decay),
            float(mem_stale_z), int(mem_stale_warmup),
            bool(reflex_gaze), float(reflex_gain), float(reflex_ema_decay),
            float(reflex_beta),
        )
        self.key_len = _archive_key_len(archive_kwargs, wram_stride)
        # A5: probe the real save_state size to size the transport buffers.
        self.state_cap = (
            _state_capacity(
                _probe_state_len(rom_path, frame_skip, hold_frames, reset_state)
            )
            if self.goexplore
            else _MAX_STATE
        )

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
        alloc("gaze_dx", (n,), np.float32)  # saccade cmd dx (exact, per env)
        alloc("gaze_dy", (n,), np.float32)  # saccade cmd dy (exact, per env)
        alloc("dones", (n,), np.uint8)
        alloc("obs_seq", (n,), np.int64)
        alloc("act_seq", (n,), np.int64)
        alloc("cap_flag", (n,), np.uint8)
        alloc("cap_done", (n,), np.uint8)
        alloc("cap_state", (n, self.state_cap), np.uint8)
        alloc("cap_len", (n,), np.int32)
        alloc("cap_trunc", (n,), np.int64)  # A5: skipped (oversize) captures
        alloc("res_flag", (n,), np.uint8)
        alloc("res_state", (n, self.state_cap), np.uint8)
        alloc("res_len", (n,), np.int32)
        alloc("ctl", (4 + n,), np.int64)
        # mined progress-counter taps: [0]=version, [1:]=GB addresses (0=unset).
        # When set, workers overwrite the obs RAM tail with these bytes instead
        # of the blind stride sample (see _async_worker_main / set_tap_addrs).
        alloc("tapcfg", (1 + self.obs_ram,), np.int64)
        # OPT-IN full-WRAM export (default OFF): only alloc'd when requested, so
        # the production furnace path never pays for it (no shm block, and its
        # name never enters _shm_names -> workers never attach it).
        if self.expose_wram:
            alloc("wram", (n, WRAM_END_LEN), np.uint8)

        self._ctl = self.arr["ctl"]
        self._ctl[:] = 0
        self._round = 0
        self._paced = False
        self._shm_names = {k: v.name for k, v in self._blocks.items()}

        self._ctx = mp.get_context("spawn")
        core_order = _numa_core_order()
        epw = max(1, int(envs_per_worker))
        self._epw = epw
        # A8: per-round/-wave wall-clock deadline. None -> derive generously in
        # _await_acks from the wave's step budget (see _await_budget_for).
        self._deadline_override = _ROUND_DEADLINE_ENV or round_deadline_s or None
        self._await_budget = _ROUND_DEADLINE_FLOOR_S
        self._procs: list = []
        self._slices: list[tuple[int, int]] = []
        wi = 0
        for lo in range(0, n, epw):
            hi = min(lo + epw, n)
            core = core_order[wi % len(core_order)] if core_order else None
            p = self._ctx.Process(
                target=_async_worker_main,
                args=(
                    lo, hi, rom_path, frame_skip, hold_frames, reset_state,
                    obs_res, obs_ram, self.wram_stride, dict(archive_kwargs),
                    self.goexplore, self.expose_wram, self._shm_names, n,
                    self.obs_dim, self.key_len, core, self.state_cap,
                    *self._foveal,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            self._slices.append((lo, hi))
            wi += 1
        self.n_workers = len(self._procs)
        self._closed = False

    # ------------------------------------------------------------------ pacing
    def set_pace(self, realtime: bool) -> None:
        """Realtime -> workers sleep-wait between polls; max -> nap-spin."""
        self._paced = bool(realtime)
        self._ctl[2] = 1 if realtime else 0

    # ------------------------------------------------------------------ taps
    def set_tap_addrs(self, addrs: list[int]) -> None:
        """Point the obs RAM tail at mined progress-counter addresses.

        Workers re-read the tap config at the next reset round (generation
        boundary), so call this BEFORE ``reset_all_begin``. Up to ``obs_ram``
        GB addresses; unset slots keep the legacy blind stride sample.
        """
        cfg = self.arr["tapcfg"]
        k = min(len(addrs), self.obs_ram)
        cfg[1 : 1 + k] = np.asarray(addrs[:k], dtype=np.int64)
        cfg[1 + k :] = 0
        cfg[0] += 1  # version bump LAST (x86 TSO: workers see addrs first)

    # ------------------------------------------------------------------ actions
    def submit_actions(self, idx, buttons, gaze_dx, gaze_dy, obs_idx) -> None:
        """Publish per-env button + saccade for a batch of ready envs (§3.4).

        Replaces the parent's inline ``actions[idx]=..; act_seq[idx]=..`` in the
        free-run act/observe loop. ``idx`` is the int array of env indices whose
        newest obs is being answered; ``buttons`` the argmax button ids for those
        envs; ``gaze_dx``/``gaze_dy`` the EXACT saccade floats (no quantization);
        ``obs_idx`` the obs-sequence index each action answers (the value to
        write into ``act_seq``). Payload rows (gaze + actions) are written FIRST
        and ``act_seq`` LAST, so a worker reading ``act_seq==k`` is guaranteed to
        see this step's gaze/action (x86 TSO), never a stale one.

        The arrays may be full-length (indexed by ``idx``) or already sliced to
        ``idx`` — both are accepted, matching ``actions_shm[idx] = acts[idx]``.
        """
        self.arr["gaze_dx"][idx] = gaze_dx
        self.arr["gaze_dy"][idx] = gaze_dy
        self.arr["actions"][idx] = buttons
        self.arr["act_seq"][idx] = obs_idx  # publish LAST (x86 TSO)

    # ------------------------------------------------------------------ rounds
    def _await_budget_for(self, op: int, target: int) -> float:
        """Generous wall-clock deadline for the round just released (A8).

        A free-run wave lets each owned env run ``target`` steps, so the budget
        scales with ``envs_per_worker * target`` at a large per-env-step margin;
        reset rounds get the floor. An explicit override (env/ctor) wins."""
        if self._deadline_override:
            return float(self._deadline_override)
        if op == _OP_RUN:
            return max(
                _ROUND_DEADLINE_FLOOR_S,
                self._epw * max(1, int(target)) * _PER_ENVSTEP_BUDGET_S,
            )
        return max(_ROUND_DEADLINE_FLOOR_S, self._epw * _PER_ENVSTEP_BUDGET_S)

    def _release(self, op: int, target: int = 0) -> None:
        self._await_budget = self._await_budget_for(op, target)
        self._round += 1
        self._ctl[1] = op
        self._ctl[3] = target
        self._ctl[0] = self._round  # release LAST

    def _acks_timeout_error(self, target: int) -> RuntimeError:
        """Diagnostic for a wave/reset that blew its deadline: a hung-but-alive
        worker never acks yet passes is_alive, so only the deadline catches it."""
        acks = self._ctl[4:4 + self.n_envs]
        stuck = [i for i in range(self.n_envs) if int(acks[i]) < target]
        parts = []
        for (lo, hi), p in zip(self._slices, self._procs):
            owned = [i for i in stuck if lo <= i < hi]
            if owned:
                parts.append(f"envs{owned}@pid{p.pid}(alive={p.is_alive()})")
        return RuntimeError(
            f"AsyncFleet round exceeded {self._await_budget:.1f}s deadline; "
            f"hung worker(s): {'; '.join(parts) or 'unknown'} — aborting"
        )

    def _await_acks(self) -> None:
        acks = self._ctl[4:4 + self.n_envs]
        target = self._round
        deadline = time.monotonic() + self._await_budget
        i = 0
        while not bool((acks >= target).all()):
            time.sleep(1e-3 if self._paced else 5e-5)
            i += 1
            if i % 2000 == 0:
                self.check_alive()
                if time.monotonic() > deadline:
                    raise self._acks_timeout_error(target)

    def check_alive(self) -> None:
        if any(not p.is_alive() for p in self._procs):
            raise RuntimeError(
                "AsyncFleet worker died mid-wave; aborting (see stderr)"
            )

    @property
    def state_truncations(self) -> int:
        """Total Go-Explore captures SKIPPED because the blob outgrew the buffer
        (A5) — dropped, never truncated, so the archive stays uncorrupted."""
        return int(self.arr["cap_trunc"].sum()) if "cap_trunc" in self.arr else 0

    # ------------------------------------------------------------------ control
    def reset_all_begin(self, restore: dict[int, bytes] | None = None) -> None:
        """Ship restore blobs + release the reset round WITHOUT waiting."""
        self.arr["res_flag"][:] = 0
        if restore and self.goexplore:
            for idx, blob in restore.items():
                if len(blob) > self.state_cap:
                    # Never restore a truncated state (A5); captures are never
                    # stored truncated, so this is a defensive skip.
                    continue
                self.arr["res_state"][idx, : len(blob)] = np.frombuffer(
                    blob, dtype=np.uint8
                )
                self.arr["res_len"][idx] = len(blob)
                self.arr["res_flag"][idx] = 1
        # No worker reads act_seq during a reset round; park every env before
        # the wave so nothing steps until the parent issues its first action.
        self.arr["act_seq"][:] = -1
        self.arr["cap_flag"][:] = 0
        self.arr["cap_done"][:] = 0
        self.arr["gaze_dx"][:] = 0.0  # no saccade until the first action lands
        self.arr["gaze_dy"][:] = 0.0
        self._release(_OP_RESET)

    def reset_all_end(self) -> np.ndarray:
        self._await_acks()
        return self.arr["obs"].copy()

    def reset_all(self, restore: dict[int, bytes] | None = None) -> np.ndarray:
        self.reset_all_begin(restore)
        return self.reset_all_end()

    def begin_wave(self, n_active: int, target_steps: int) -> None:
        """Release the free-run round: envs ``[0:n_active]`` will each run
        ``target_steps`` steps; any tail envs are parked as already-done."""
        if n_active < self.n_envs:
            # Workers are idle between rounds, so parking the unused tail by
            # advancing their obs_seq to the target is race-free here.
            self.arr["obs_seq"][n_active:] = target_steps
        self._release(_OP_RUN, target=target_steps)

    def end_wave(self) -> None:
        """Wait for every worker to ack wave completion."""
        self._await_acks()

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


__all__ = [
    "VecFleet", "BarrierFleet", "AsyncFleet",
    "ObsEncoder", "FovealEncoder", "ReflexGaze", "NUMA_NODES",
]
