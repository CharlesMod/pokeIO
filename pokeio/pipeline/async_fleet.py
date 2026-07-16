"""AsyncFleet — decoupled, shared-memory, barrier-synchronized render-on fleet.

The Phase-1.5 transport. N worker processes each drive a bare PyBoy at
``frame_skip=1`` (reflex control): apply the action published for the previous
frame, ``tick(1, render=True)``, grab a compact 144x160 grayscale framebuffer,
and write it straight into a shared-memory ring. Synchronization is a
**sense-reversing barrier in shared memory** (monotone generation counters, no
locks, no pipes) between the N workers and a single controller — NOT the 2N
blocking pipe round-trips of ``emu.fleet.VecFleet`` (which caps ~2.9k steps/s).

The controller (driven in the parent by :meth:`AsyncFleet.run_frames`) waits for
all N framebuffers of a generation, hands the batch to an ``infer`` callable
(the GPU vision+inference path by default), and publishes one action per worker.

Pipeline latency
----------------
Exactly **1 frame**: the action a worker applies to produce frame ``g`` is the
one the controller inferred from frame ``g-1`` (:meth:`_worker_main`). The
worker blocks on the controller's release counter only for the (tiny) inference
gap; the CPU render of the whole fleet and the GPU batch do not overlap in this
reference scheme. An overlapped double-buffered variant is provided by
``overlap=True`` (see :meth:`run_frames`), which hides the GPU batch under the
next render at the cost of the action being up to 2 frames stale.

BLAS thread pools are pinned to 1 *before numpy import* (workers own one core
each; multi-threaded BLAS is pure oversubscription). Workers are NUMA-pinned.
Shared memory is unlinked on :meth:`close` (no leaked ``/dev/shm``).
"""

from __future__ import annotations

import os

# Pin numeric thread pools to 1 BEFORE numpy import anywhere (spawn children
# inherit these). One core per worker -> multi-threaded BLAS collapses throughput.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from multiprocessing import shared_memory  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402

from pokeio.config import Config  # noqa: E402

# NUMA topology of the target box (2x Xeon E5-2690 v4); consecutive workers
# alternate sockets so both memory controllers stay busy.
NUMA_NODES = {
    0: list(range(0, 14)) + list(range(28, 42)),
    1: list(range(14, 28)) + list(range(42, 56)),
}

ACTIONS = ("up", "down", "left", "right", "a", "b", "start", "select")
_DEFAULT_ACTION = 4  # 'a' — a harmless default before the first inference


def _numa_core_order() -> list[int]:
    n0, n1 = NUMA_NODES[0], NUMA_NODES[1]
    order: list[int] = []
    for a, b in zip(n0, n1):
        order += [a, b]
    order.extend(n0[len(n1):])
    order.extend(n1[len(n0):])
    return order


def _spin_wait(pred: Callable[[], bool], deadline: float | None = None) -> bool:
    """Adaptive spin: busy-loop briefly, then yield the scheduler slice.

    Returns True when ``pred()`` holds, False if ``deadline`` (perf_counter)
    passes first. Yielding (``sleep(0)``) keeps oversubscribed fleets (N > cores)
    from live-locking on the barrier while staying sub-render-frame responsive.
    """
    spins = 0
    while not pred():
        spins += 1
        if spins & 0x3F:  # ~63 tight checks between scheduler yields
            continue
        if deadline is not None and time.perf_counter() > deadline:
            return False
        time.sleep(0)
    return True


# --------------------------------------------------------------------- worker
def _worker_main(widx, rom_path, state_path, frame_skip, hold_frames,
                 core, shm_names, n, H, W, overlap):
    """Worker: render-on emulate at fs=1, write framebuffer, barrier-sync.

    Shared control (generation counters, monotone int64):
      w_gen[widx]  worker sets = g after writing frames[buf][widx] for gen g
      r_gen[0]     controller sets = g after publishing actions for gen g
      stop[0]      1 -> shut down
    """
    # The parent owns creation + unlink of every segment; this child only
    # ATTACHES. Stop the child's resource_tracker from also tracking (and later
    # trying to unlink) them — the classic 3.12 double-unlink footgun that
    # otherwise spams "KeyError"/"leaked shared_memory" at exit.
    try:
        from multiprocessing import resource_tracker as _rt
        _orig_reg = _rt.register
        _rt.register = (lambda name, rtype: None if rtype == "shared_memory"
                        else _orig_reg(name, rtype))
        _rt.unregister = (lambda name, rtype: None if rtype == "shared_memory"
                          else _orig_reg(name, rtype))
    except Exception:
        pass

    import psutil
    from pyboy import PyBoy

    try:
        if core is not None:
            psutil.Process().cpu_affinity([core])
    except Exception:
        pass

    nbuf = 2 if overlap else 1
    shm = {k: shared_memory.SharedMemory(name=shm_names[k]) for k in shm_names}
    frames = np.ndarray((nbuf, n, H, W), dtype=np.uint8, buffer=shm["frames"].buf)
    actions = np.ndarray((n,), dtype=np.int32, buffer=shm["actions"].buf)
    w_gen = np.ndarray((n,), dtype=np.int64, buffer=shm["w_gen"].buf)
    r_gen = np.ndarray((1,), dtype=np.int64, buffer=shm["r_gen"].buf)
    c_gen = np.ndarray((1,), dtype=np.int64, buffer=shm["c_gen"].buf)
    stop = np.ndarray((1,), dtype=np.uint8, buffer=shm["stop"].buf)

    pyboy = PyBoy(rom_path, window="null", sound_emulated=False)
    if state_path is not None:
        with open(state_path, "rb") as fh:
            pyboy.load_state(fh)
    pyboy.tick(1, True)  # settle

    held = None
    hold_frames = int(hold_frames)
    frame_skip = int(frame_skip)
    try:
        g = 0
        while not stop[0]:
            buf = g % nbuf
            if overlap and g >= nbuf:
                # Don't overwrite a buffer the controller may still be reading:
                # wait until it has consumed generation g-nbuf.
                if not _spin_wait(lambda: c_gen[0] >= g - nbuf or stop[0]):
                    break
                if stop[0]:
                    break

            a = int(actions[widx])
            name = ACTIONS[a] if 0 <= a < len(ACTIONS) else ACTIONS[_DEFAULT_ACTION]

            if frame_skip <= 1:
                # Lean reflex step: press, one rendered tick, release. True
                # consecutive-frame motion; no action-hold.
                pyboy.button_press(name)
                pyboy.tick(1, True)
                pyboy.button_release(name)
            else:
                pyboy.button_press(name)
                pyboy.tick(hold_frames, False)
                pyboy.button_release(name)
                rem = frame_skip - hold_frames
                if rem > 1:
                    pyboy.tick(rem - 1, False)
                pyboy.tick(1, True)

            # Compact grayscale grab (channel 0), ~23 KB, minimal copy.
            np.copyto(frames[buf, widx], pyboy.screen.ndarray[:, :, 0])
            w_gen[widx] = g

            if not overlap:
                # Blocking 1-frame latency: wait for actions of gen g.
                if not _spin_wait(lambda: r_gen[0] >= g or stop[0]):
                    break
            g += 1
    finally:
        try:
            pyboy.stop(save=False)
        except Exception:
            pass
        for s in shm.values():
            s.close()


# --------------------------------------------------------------------- policy
class StubPolicy:
    """Placeholder whole-population forward pass on the GPU.

    A single fixed (seeded) linear projection ``optical_dim -> n_actions`` over
    the flattened obs, argmax'd to a per-agent action. Stands in for the real
    evo/forward.py batched network eval so the end-to-end loop is realistic.
    """

    def __init__(self, in_dim: int, n_actions: int = 8,
                 device: str = "cuda:1", seed: int = 0):
        import torch
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        w = torch.randn(in_dim, n_actions, generator=g) / (in_dim ** 0.5)
        b = torch.randn(n_actions, generator=g)
        self.W = w.to(self.device)
        self.b = b.to(self.device)
        self.n_actions = n_actions

    def __call__(self, flat):  # flat: (N, in_dim) torch tensor on device
        import torch
        with torch.no_grad():
            logits = flat @ self.W + self.b
            return torch.argmax(logits, dim=1).to(torch.int32)


# --------------------------------------------------------------------- fleet
class AsyncFleet:
    """N NUMA-pinned render-on workers + a shared-memory barrier controller."""

    def __init__(
        self,
        n_envs: int,
        rom_path: str | None = None,
        config: Config | None = None,
        state_path: str | None = None,
        device: str = "cuda:1",
        infer: Callable | None = None,
        gpu: bool = True,
        overlap: bool = False,
    ):
        self.n = int(n_envs)
        self.config = config or Config()
        self.rom_path = rom_path or self.config.emu.rom_path
        self.state_path = state_path or self.config.emu.reset_state or None
        self.frame_skip = int(self.config.emu.frame_skip)
        self.hold_frames = int(self.config.emu.button_hold_frames)
        self.device = device
        self.overlap = bool(overlap)
        self.H = int(self.config.vision.screen_height)
        self.W = int(self.config.vision.screen_width)
        self._nbuf = 2 if self.overlap else 1
        self._closed = False

        # --- shared memory blocks
        self._shm: dict[str, shared_memory.SharedMemory] = {}

        def _mk(name, shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            s = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            self._shm[name] = s
            return np.ndarray(shape, dtype=dtype, buffer=s.buf)

        self._frames = _mk("frames", (self._nbuf, self.n, self.H, self.W), np.uint8)
        self._actions = _mk("actions", (self.n,), np.int32)
        self._w_gen = _mk("w_gen", (self.n,), np.int64)
        self._r_gen = _mk("r_gen", (1,), np.int64)
        self._c_gen = _mk("c_gen", (1,), np.int64)
        self._stop = _mk("stop", (1,), np.uint8)
        self._frames[:] = 0
        self._actions[:] = _DEFAULT_ACTION
        self._w_gen[:] = -1
        self._r_gen[0] = -1
        self._c_gen[0] = -1
        self._stop[0] = 0
        self._shm_names = {k: v.name for k, v in self._shm.items()}

        self._ctx = mp.get_context("spawn")
        self._core_order = _numa_core_order()
        self._procs: list = []

        # --- inference: an explicit callback (tests) takes precedence; otherwise
        # the built-in CUDA-stream-pipelined GPU controller (benchmark path).
        self._gpu = False
        if infer is not None:
            self._infer = infer
        elif gpu:
            self._infer = None
            self._gpu = True
            self._setup_gpu()
        else:
            self._infer = None  # must be provided

    # ---------------------------------------------------------------- gpu setup
    def _setup_gpu(self):
        import torch
        from pokeio.pipeline.gpu_vision import GPUVision
        self._torch = torch
        self._dev = torch.device(self.device)
        self._gpu_vision = GPUVision(self.config, device=self.device)
        self._policy = StubPolicy(self._gpu_vision.optical_dim,
                                  n_actions=len(ACTIONS), device=self.device)
        # Double-buffered pinned host staging + device action buffers so the
        # controller pipelines: launch gen g's H2D+vision+forward on a stream and
        # publish gen g-1's (already-computed) actions — the only per-gen sync is
        # on a tiny (N,) int tensor, hiding the ~0.9 ms GPU round-trip.
        self._pin = [torch.empty((self.n, self.H, self.W), dtype=torch.uint8,
                                 pin_memory=True) for _ in range(2)]
        self._act_dev = [torch.zeros(self.n, dtype=torch.int32, device=self._dev)
                         for _ in range(2)]
        self._act_host = [torch.empty(self.n, dtype=torch.int32, pin_memory=True)
                          for _ in range(2)]
        self._stream = torch.cuda.Stream(self._dev)
        # One event per pipeline slot, recorded after each gen's action D2H copy,
        # so the controller can wait for gen g-1 *only* (not the just-launched
        # gen g still queued behind it on the in-order stream).
        self._done_evt = [torch.cuda.Event() for _ in range(2)]

    def _gpu_launch(self, g: int, frames_view: np.ndarray, release_cb):
        """Copy gen g's frames to pinned host (then release the buffer) and
        enqueue the whole H2D->vision->forward on the CUDA stream -> act_dev[g%2]."""
        torch = self._torch
        slot = g % 2
        self._pin[slot].copy_(torch.from_numpy(frames_view))
        if release_cb is not None:
            release_cb()  # buffer is safe to reuse: host copy is done
        with torch.cuda.stream(self._stream):
            gpu = self._pin[slot].to(self._dev, non_blocking=True)
            obs = self._gpu_vision.build(gpu)
            flat = torch.cat([obs["coarse"].reshape(self.n, -1),
                              obs["fovea"].reshape(self.n, -1),
                              obs["motion"].reshape(self.n, -1)], dim=1)
            logits = flat @ self._policy.W + self._policy.b
            self._act_dev[slot] = torch.argmax(logits, dim=1).to(torch.int32)
            self._act_host[slot].copy_(self._act_dev[slot], non_blocking=True)
            self._done_evt[slot].record(self._stream)

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        for i in range(self.n):
            core = self._core_order[i % len(self._core_order)] if self._core_order else None
            p = self._ctx.Process(
                target=_worker_main,
                args=(i, self.rom_path, self.state_path, self.frame_skip,
                      self.hold_frames, core, self._shm_names, self.n,
                      self.H, self.W, self.overlap),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def _pin_controller(self) -> None:
        """Pin the controller to a core the workers don't own (best effort)."""
        try:
            import psutil
            used = set(self._core_order[i % len(self._core_order)]
                       for i in range(self.n))
            free = [c for c in self._core_order if c not in used]
            core = free[-1] if free else self._core_order[-1]
            psutil.Process().cpu_affinity([core])
        except Exception:
            pass

    # ------------------------------------------------------------------ run
    def run_frames(self, n_frames: int, warmup: int = 0,
                   collect_actions: bool = False, timeout_s: float = 60.0):
        if self._gpu:
            return self._run_frames_gpu(n_frames, warmup, timeout_s)
        return self._run_frames_cb(n_frames, warmup, collect_actions, timeout_s)

    # ---------------------------------------------------- pipelined GPU loop
    def _run_frames_gpu(self, n_frames: int, warmup: int = 0, timeout_s: float = 60.0):
        """CUDA-stream-pipelined controller. Publishes gen g-1's actions while
        gen g's GPU work is in flight; workers double-buffer and never block on
        the GPU. Effective action latency ~2 frames (still ~33 ms)."""
        torch = self._torch
        self._pin_controller()
        t_wait = t_ctrl = 0.0
        t0 = None
        counted = 0
        total = n_frames + warmup
        deadline0 = time.perf_counter()
        for g in range(total):
            deadline = time.perf_counter() + timeout_s
            tw = time.perf_counter()
            if not _spin_wait(lambda: int(self._w_gen.min()) >= g, deadline):
                raise TimeoutError(f"workers stalled at gen {g}")
            t_wait_g = time.perf_counter() - tw
            if g == warmup:
                t0 = time.perf_counter()

            tc = time.perf_counter()
            buf = g % self._nbuf
            frames_view = self._frames[buf]
            release_cb = (lambda gg=g: self._c_gen.__setitem__(0, gg)) if self.overlap else None
            self._gpu_launch(g, frames_view, release_cb)
            if not self.overlap:
                # 1-frame latency: publish THIS gen's actions before releasing.
                self._done_evt[g % 2].synchronize()
                self._actions[:] = self._act_host[g % 2].numpy()
                self._r_gen[0] = g
            else:
                # Publish the PREVIOUS gen's (already-finished) actions; wait only
                # on gen g-1's event, so gen g's GPU work overlaps the next render.
                if g > 0:
                    self._done_evt[(g - 1) % 2].synchronize()
                    self._actions[:] = self._act_host[(g - 1) % 2].numpy()
                self._r_gen[0] = g
            t_ctrl_g = time.perf_counter() - tc

            if g >= warmup:
                t_wait += t_wait_g
                t_ctrl += t_ctrl_g
                counted += 1

        dt = time.perf_counter() - t0 if t0 is not None else 0.0
        agent_steps = counted * self.n
        return {
            "n_envs": self.n, "frames": counted, "agent_steps": agent_steps,
            "wall_s": dt,
            "agent_steps_per_s": agent_steps / dt if dt > 0 else 0.0,
            "mean_wait_ms": (t_wait / counted * 1e3) if counted else 0.0,
            "mean_infer_ms": (t_ctrl / counted * 1e3) if counted else 0.0,
            "overlap": self.overlap, "frame_skip": self.frame_skip,
        }

    # ------------------------------------------------------ callback loop
    def _run_frames_cb(self, n_frames: int, warmup: int = 0,
                       collect_actions: bool = False, timeout_s: float = 60.0):
        """Drive the controller for ``n_frames`` generations; return stats.

        Blocking (overlap=False): per gen, wait all frames -> infer -> publish
        actions -> release. Exactly 1-frame latency.
        Overlapped (overlap=True): per gen, wait all frames -> mark consumed
        (release the buffer) -> infer -> publish. Workers free-run into the other
        buffer, hiding the GPU batch under the next render (action up to 2 frames
        stale).
        """
        if self._infer is None:
            raise RuntimeError("no inference callable configured")

        collected = [] if collect_actions else None
        t_wait = t_infer = 0.0
        t0 = None
        counted = 0
        deadline0 = time.perf_counter() + timeout_s
        for g in range(n_frames + warmup):
            deadline = max(deadline0, time.perf_counter() + timeout_s)
            tw = time.perf_counter()
            ok = _spin_wait(lambda: int(self._w_gen.min()) >= g, deadline)
            t_wait_g = time.perf_counter() - tw
            if not ok:
                raise TimeoutError(f"workers stalled at gen {g}")

            if g == warmup:
                t0 = time.perf_counter()

            buf = g % self._nbuf
            frames_view = self._frames[buf]

            if self.overlap:
                # Copy out so workers may reuse the buffer immediately.
                frames_batch = frames_view.copy()
                self._c_gen[0] = g  # release buffer `buf` for reuse (gen g+nbuf)
            else:
                frames_batch = frames_view

            ti = time.perf_counter()
            acts = self._infer(frames_batch)
            t_infer_g = time.perf_counter() - ti

            self._actions[:] = np.asarray(acts, dtype=np.int32)
            if collected is not None and g >= warmup:
                collected.append(np.asarray(acts, dtype=np.int32).copy())
            self._r_gen[0] = g  # publish + release (blocking mode)

            if g >= warmup:
                t_wait += t_wait_g
                t_infer += t_infer_g
                counted += 1

        dt = time.perf_counter() - t0 if t0 is not None else 0.0
        agent_steps = counted * self.n
        stats = {
            "n_envs": self.n,
            "frames": counted,
            "agent_steps": agent_steps,
            "wall_s": dt,
            "agent_steps_per_s": agent_steps / dt if dt > 0 else 0.0,
            "mean_wait_ms": (t_wait / counted * 1e3) if counted else 0.0,
            "mean_infer_ms": (t_infer / counted * 1e3) if counted else 0.0,
            "overlap": self.overlap,
            "frame_skip": self.frame_skip,
        }
        if collected is not None:
            stats["actions"] = collected
        return stats

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop[0] = 1
        # Unblock any worker waiting on release/consume counters.
        self._r_gen[0] = np.iinfo(np.int64).max
        self._c_gen[0] = np.iinfo(np.int64).max
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for s in self._shm.values():
            try:
                s.close()
                s.unlink()
            except Exception:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["AsyncFleet", "StubPolicy", "NUMA_NODES", "ACTIONS"]
