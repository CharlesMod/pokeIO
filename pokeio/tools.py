"""Operational tools: probe a spec against a ROM, benchmark env throughput,
build a start state from a boot macro, record a policy rollout."""

from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

import numpy as np

from pokeio.env import GameEnv
from pokeio.platforms import make_platform
from pokeio.screen import unpack
from pokeio.spec import GameSpec


def save_png(path: str | Path, img: np.ndarray) -> None:
    """Minimal dependency-free PNG writer for (H, W) or (H, W, 3) uint8."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        color, rows = 0, img
    else:
        color, rows = 2, img.reshape(img.shape[0], -1)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def obs_to_image(env: GameEnv, obs: dict) -> np.ndarray:
    """Side-by-side grayscale view of every pixel channel the policy sees."""
    os_ = env.obs_spec
    chans = [unpack(obs["pixels"][c], os_.pixel_bpp) for c in range(os_.pixels[0])]
    scale = 255 // max(os_.levels - 1, 1)
    return np.concatenate([255 - (c.astype(np.int32) * scale).clip(0, 255) for c in chans], axis=1).astype(np.uint8)


# ---------------------------------------------------------------------------


def probe(spec: GameSpec, out_dir: str | Path = "probe", steps: int = 200, seed: int = 0) -> None:
    """Print every memory field at the start state, check that directions move the
    player, and dump screen/visited-mask overlays to PNG for calibration."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = GameEnv(spec, env_id=0, seed=seed)
    try:
        obs = env.reset()
        print(f"spec {spec.name}: platform={spec.platform} buttons={spec.controls.buttons}")
        print(f"obs spec: {env.obs_spec}")
        if env.mem.unresolved:
            print(f"UNRESOLVED fields (missing symbols): {env.mem.unresolved}")
        if env.progress.disabled:
            print(f"disabled terms/milestones: {env.progress.disabled}")
        print("memory at start:")
        for name, f in spec.memory.items():
            if not env.mem.has(name):
                continue
            v = env.mem.block(name) if f.is_block else env.mem.array(name)
            shown = f"{len(v)} bytes, popcount={int(np.unpackbits(v.astype(np.uint8)).sum())}" if f.is_block else v.tolist()
            print(f"  {name:22s} @0x{env.mem.addr[name]:04X}  {shown}")
        print(f"position: {env._position()}  paused={env._paused() if env.has_position else None}")
        print(f"frontier score: {env._score():.2f}   milestones reached: {list(env.progress.reached)}")
        save_png(out / "start_screen.png", env.render_rgb())
        save_png(out / "start_obs.png", obs_to_image(env, obs))

        if env.has_position:
            for b in ("up", "down", "left", "right"):
                if b not in env.buttons:
                    continue
                before = env._position()
                obs, *_ = env.step(env.buttons.index(b))
                print(f"  press {b:5s}: {before} -> {env._position()}")
            save_png(out / "after_moves_obs.png", obs_to_image(env, obs))

        rng = np.random.default_rng(seed)
        t0 = time.time()
        total = 0.0
        for i in range(steps):
            obs, r, d, info = env.step(int(rng.integers(len(env.buttons))))
            total += r
            if info.get("milestones"):
                print(f"  step {i}: milestones {info['milestones']}")
        dt = time.time() - t0
        print(f"{steps} random steps: {steps/dt:.0f} steps/s, return {total:.3f}, "
              f"cells {env.cells.unique}, rooms {len(env.cells.rooms)}")
        save_png(out / "end_screen.png", env.render_rgb())
        save_png(out / "end_obs.png", obs_to_image(env, obs))
        print(f"wrote PNGs to {out}/")
    finally:
        env.close()


def bench(spec: GameSpec, workers: int, envs_per_worker: int, batch_workers: int, seconds: float = 30.0) -> float:
    """Env-only throughput with random actions (no learner)."""
    from pokeio.config import VecConfig
    from pokeio.vec import make_vec

    vec = make_vec(spec, VecConfig(num_workers=workers, envs_per_worker=envs_per_worker,
                                   batch_workers=batch_workers or workers))
    rng = np.random.default_rng(0)
    n_act = vec.obs_spec.n_actions
    vec.reset()
    steps, t0 = 0, time.time()
    try:
        while time.time() - t0 < seconds:
            b = vec.recv()
            vec.send(rng.integers(n_act, size=len(b.env_ids)).astype(np.int32), b.env_ids)
            steps += len(b.env_ids)
    finally:
        vec.close()
    sps = steps / (time.time() - t0)
    print(f"{workers} workers x {envs_per_worker} envs (batch {batch_workers}): {sps:,.0f} env-steps/s")
    return sps


def _run_macro(p, items: list, hold: int, gap: int) -> None:
    for item in items:
        op = item[0]
        if op == "tick":
            p.tick(int(item[1]), False)
        elif op == "press":
            btn, rep = item[1], int(item[2]) if len(item) > 2 else 1
            for _ in range(rep):
                p.press(btn)
                p.tick(hold, False)
                p.release(btn)
                p.tick(gap, False)
        elif op == "seq":
            for _ in range(int(item[2]) if len(item) > 2 else 1):
                _run_macro(p, item[1], hold, gap)
        else:
            raise ValueError(f"unknown macro op {op!r}")


def make_state(spec: GameSpec, out: str | Path | None = None, hold: int = 6, gap: int = 8) -> Path:
    """Power on, play ``start.boot_macro``, save the state to ``out`` (default:
    the spec's first start state)."""
    if not spec.boot_macro:
        raise SystemExit("spec has no start.boot_macro")
    out = Path(out) if out else spec.start_states[0]
    p = make_platform(spec)
    try:
        _run_macro(p, spec.boot_macro, hold, gap)
        for b in ("up", "down", "left", "right", "a", "b", "start", "select"):
            p.release(b)
        p.tick(1, True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(p.save_state())
        save_png(out.with_suffix(".png"), p.screen())
        print(f"saved {out} ({out.stat().st_size} bytes); screenshot {out.with_suffix('.png')}")
    finally:
        p.close()
    return out


def record(spec: GameSpec, checkpoint: str | Path, out_dir: str | Path, steps: int = 2000,
           every: int = 1, greedy: bool = False, device: str = "cpu") -> None:
    """Roll out one env with a trained policy; write frames (PNG) + a trace JSONL."""
    import json

    import torch

    from pokeio.train import load_policy

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy, _ = load_policy(checkpoint, device)
    env = GameEnv(spec, env_id=0, seed=0)
    dev = torch.device(device)
    h, c = policy.initial_state(1, dev)
    start = torch.ones(1, device=dev)
    obs = env.reset()
    with open(out / "trace.jsonl", "w") as trace, torch.no_grad():
        for t in range(steps):
            o = {k: torch.from_numpy(obs[k][None]).to(dev) for k in ("pixels", "bits", "scalars", "cats")}
            logits, value, (h, c) = policy.step(o, (h, c), start)
            start = torch.zeros(1, device=dev)
            a = int(logits.argmax(-1)) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
            obs, r, d, info = env.step(a)
            if d:
                start = torch.ones(1, device=dev)
            if t % every == 0:
                save_png(out / f"frame_{t:06d}.png", env.render_rgb())
            trace.write(json.dumps({"t": t, "action": env.buttons[a], "reward": r, "value": float(value),
                                    "pos": env._position(), "info": {k: v for k, v in info.items()
                                                                     if k not in ("frontier",)}}) + "\n")
    env.close()
    print(f"wrote {steps} steps to {out}/ (ffmpeg -i {out}/frame_%06d.png out.mp4)")
