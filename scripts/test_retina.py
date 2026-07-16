"""Verify Lane C — the learned-encoder retina.

Collect frames from a random-policy rollout on the canonical newgame state,
build temporal (K-frame + motion) stacks, train the conv autoencoder on cuda:1,
show the reconstruction MSE dropping, and dump a before/after PNG grid.

Run:  PYTHONPATH=/home/cmod/pokeIO python scripts/test_retina.py
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
import torch

from pokeio.emu.env import PokeEnv
from pokeio.evo.retina import (
    DEFAULT_HISTORY,
    Retina,
    build_dataset,
    preferred_device,
)

ROM = "roms/pokemon_yellow.gb"
STATE = "roms/yellow_newgame.state"
N_FRAMES = 400
TRAIN_STEPS = 400
OUT_PNG = "runs/retina_recon.png"


def collect_frames(n: int, seed: int = 0) -> list[np.ndarray]:
    """Random-policy rollout; return n raw (144,160) uint8 frames."""
    rng = np.random.default_rng(seed)
    env = PokeEnv(ROM)
    frames = [env.reset(STATE)]
    for _ in range(n - 1):
        obs, _ram, _done, _info = env.step(int(rng.integers(0, 8)))
        frames.append(obs)
    env.close()
    return frames


def _write_gray_png(path: str, img: np.ndarray) -> None:
    """Dependency-free 8-bit grayscale PNG writer (stdlib zlib only)."""
    img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape
    raw = bytearray()
    for row in img:  # each scanline prefixed with filter byte 0
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


def save_png(model: Retina, data: np.ndarray, device, path: str, n: int = 6) -> bool:
    """Save a before/after grid (top row orig, bottom row recon) as a PNG.

    Uses a stdlib-only PNG writer so it works without matplotlib/Pillow, which
    are not installed in this env. Also prints an ASCII thumbnail of one pair.
    """
    model.eval()
    idx = np.linspace(0, len(data) - 1, n).astype(int)
    batch = torch.as_tensor(data[idx], dtype=torch.float32).to(device)
    with torch.no_grad():
        recon, _ = model(batch)
    cur = model.history - 1  # index of the newest grayscale channel
    orig = (batch[:, cur].cpu().numpy() * 255.0)
    rec = (recon[:, cur].cpu().numpy() * 255.0)

    h, w = orig.shape[1:]
    pad = 2
    grid = np.zeros((2 * h + pad, n * w + (n - 1) * pad), dtype=np.float32)
    for j in range(n):
        x0 = j * (w + pad)
        grid[:h, x0:x0 + w] = orig[j]
        grid[h + pad:, x0:x0 + w] = rec[j]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_gray_png(path, grid)
    print(f"[png] saved before/after grid (top=orig, bottom=recon) "
          f"-> {os.path.abspath(path)}")

    # ASCII thumbnail of the first pair so it's visible in the log too.
    ramp = " .:-=+*#%@"
    def ascii_of(arr: np.ndarray, rows: int = 12, cols: int = 24) -> list[str]:
        small = arr[::max(1, arr.shape[0] // rows), ::max(1, arr.shape[1] // cols)]
        return ["".join(ramp[min(len(ramp) - 1, int(v / 256 * len(ramp)))] for v in r)
                for r in small]
    print("[ascii] orig (left) vs recon (right), frame 0:")
    for lo, lr in zip(ascii_of(orig[0]), ascii_of(rec[0])):
        print(f"    {lo}   |   {lr}")
    return True


def main() -> None:
    device = preferred_device()
    print(f"[device] using {device}")
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(device.index)} "
              f"(sm_{'.'.join(map(str, torch.cuda.get_device_capability(device.index)))})")

    print(f"[data] collecting {N_FRAMES} random-policy frames from {STATE} ...")
    frames = collect_frames(N_FRAMES)
    data = build_dataset(frames, history=DEFAULT_HISTORY)
    print(f"[data] built temporal stacks: {data.shape} "
          f"(N, K+1 channels, H, W); dtype={data.dtype}")

    model = Retina(history=DEFAULT_HISTORY)
    print(f"[model] Retina z_dim={model.z_dim} params={model.num_params():,} "
          f"channels_in={model.history + 1}")

    # Warm baseline loss for reference.
    model.to(device)
    with torch.no_grad():
        b = torch.as_tensor(data[:64], dtype=torch.float32).to(device)
        recon, z = model(b)
        base = torch.nn.functional.mse_loss(recon, b).item()
    print(f"[train] initial recon MSE = {base:.5f}; z shape = {tuple(z.shape)}")

    history = model.fit(data, steps=TRAIN_STEPS, batch_size=64, lr=1e-3,
                        device=device, log_every=25)
    print("[train] loss trajectory (every 25 steps):")
    for i, l in enumerate(history):
        print(f"    step {i * 25:4d}: MSE = {l:.5f}")
    print(f"[train] first->last: {history[0]:.5f} -> {history[-1]:.5f} "
          f"({100 * (1 - history[-1] / history[0]):.1f}% reduction)")

    save_png(model, data, device, OUT_PNG)
    print(f"[ok] retina trained on {device} with no sm_60/tensor-core error")


if __name__ == "__main__":
    main()
