r"""Lane C — the learned-encoder "retina".

A small self-supervised convolutional AUTOENCODER trained by gradient descent on
the game's OWN frames (no external data, no labels). It compresses a short window
of grayscale screen frames into a compact bottleneck vector ``z``, which an
evolved recurrent controller consumes as its observation (the ERL-Re² pattern:
gradient-trained representation, evolution-trained policy on top).

Why this lane exists (the console-ladder / 3D path)
---------------------------------------------------
Lanes A/B in this project bake in 2D-tile geometric assumptions that are cheap
and accurate on a Game Boy but do NOT survive the climb up the console ladder
(Mode-7 GBA, N64 perspective/3D). Lane C makes **zero geometric assumptions**:
it is just convolutions over pixels + reconstruction loss, so the exact same
substrate carries forward to 3D consoles with no code change — only the input
resolution / channel count changes.

TEMPORAL input — inferring 3D-from-2D
-------------------------------------
The crucial design choice for forward-compatibility: the encoder does not see a
single still frame. Its input is a **stack of the K most-recent frames plus an
explicit motion (frame-difference) channel**:

    input channels = [ f_{t-K+1}, ..., f_{t-1}, f_t,  (f_t - f_{t-1}) ]
                       \______ K grayscale frames ______/  \__ motion __/

A single 2D projection is depth-ambiguous, but *motion parallax* across a short
history disambiguates it: nearer surfaces sweep across the projection faster
than farther ones, and the frame-difference channel hands that signal to the
first conv layer directly. This is exactly the cue a network needs to recover
depth / perspective / egomotion from a flat image sequence — and it is the SAME
mechanism whether the pixels come from a 2D Game Boy scroll or a 3D N64 camera.
So temporal stacking is not a Game-Boy convenience; it is the substrate feature
that lets Lane C generalize UP the ladder with no change to the encoder itself.

The bottleneck ``z`` as the controller interface
-------------------------------------------------
``encode(frames) -> z`` is the whole public contract for evolution. An evolved
recurrent controller (NEAT genome with recurrent connections; see ``evo/``)
takes ``z`` (dim ``z_dim``) as its input vector each tick and emits the 8 action
logits. Concretely, per env step:

    stack = RetinaStacker.push(obs)     # maintain rolling K-frame history
    z     = retina.encode(stack)        # (z_dim,) fp32 features, no grad
    logits = controller.forward(z, h)   # evolved recurrent net -> Discrete(8)

The retina is trained by gradient (reconstruction MSE) either offline on
collected frames or periodically alongside evolution; the controller is trained
by evolution against ``z``. They share no gradients — ``encode`` runs under
``torch.no_grad()`` for the controller. Because ``z`` is a fixed-width vector,
the controller's input size is stable even as the pixel front-end grows for a
bigger console; only ``retina`` is swapped.

P100 note: fp32 throughout, no AMP/bf16 — the P100 (sm_60) has no tensor cores,
so mixed precision buys nothing and can error. Kept lean on purpose.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Native Game Boy screen; we downscale by 2 for a tractable, P100-lean tensor.
RAW_H, RAW_W = 144, 160
IN_H, IN_W = RAW_H // 2, RAW_W // 2  # 72 x 80

DEFAULT_HISTORY = 3   # K most-recent frames stacked as channels
Z_DIM = 128           # bottleneck feature width handed to the evolved controller


def preferred_device() -> torch.device:
    """cuda:1 if present (card 0 hosts the GLM server), else cpu."""
    if torch.cuda.is_available():
        idx = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


# --------------------------------------------------------------------------- io
def downscale(frame: np.ndarray) -> np.ndarray:
    """Raw (144,160) uint8 -> (72,80) float32 in [0,1] via 2x2 average pool."""
    f = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
    f = f.view(1, 1, RAW_H, RAW_W)
    f = F.avg_pool2d(f, kernel_size=2)
    return f.view(IN_H, IN_W).numpy()


def build_stack(frames: list[np.ndarray], t: int, history: int = DEFAULT_HISTORY) -> np.ndarray:
    """Build one temporal input tensor centred on frame ``t``.

    Returns a (history+1, IN_H, IN_W) float32 array: ``history`` downscaled
    grayscale frames [t-history+1 .. t] followed by a motion channel
    (f_t - f_{t-1}), remapped from [-1,1] to [0,1] so all channels share a scale.
    Requires ``t >= history`` so both the history and the motion diff exist.
    """
    small = [downscale(frames[i]) for i in range(t - history + 1, t + 1)]
    # motion = newest frame minus the frame immediately preceding the window.
    motion = small[-1] - downscale(frames[t - 1])
    motion = (motion + 1.0) * 0.5  # [-1,1] -> [0,1]
    chans = small + [motion]
    return np.stack(chans, axis=0).astype(np.float32)


def build_dataset(frames: list[np.ndarray], history: int = DEFAULT_HISTORY) -> np.ndarray:
    """Vectorized: turn a frame rollout into an (N, history+1, IN_H, IN_W) array."""
    stacks = [build_stack(frames, t, history) for t in range(history, len(frames))]
    return np.stack(stacks, axis=0).astype(np.float32)


class RetinaStacker:
    """Rolling K-frame history for online use inside the env loop.

    Feed raw (144,160) observations via :meth:`push`; get back the current
    (1, history+1, IN_H, IN_W) stack ready for :meth:`Retina.encode`.
    """

    def __init__(self, history: int = DEFAULT_HISTORY):
        self.history = history
        self.buf: deque[np.ndarray] = deque(maxlen=history + 1)

    def reset(self, obs: np.ndarray) -> None:
        self.buf.clear()
        small = downscale(obs)
        for _ in range(self.history + 1):
            self.buf.append(small)

    def push(self, obs: np.ndarray) -> np.ndarray:
        if not self.buf:
            self.reset(obs)
        else:
            self.buf.append(downscale(obs))
        frames = list(self.buf)          # newest last; len == history+1
        window = frames[-self.history:]  # K newest frames
        motion = (window[-1] - frames[-self.history - 1] + 1.0) * 0.5
        chans = window + [motion]
        return np.stack(chans, axis=0)[None].astype(np.float32)


# ------------------------------------------------------------------ the network
class Retina(nn.Module):
    """Conv autoencoder: (K+1)-channel temporal stack -> z -> reconstruction.

    Encoder downsamples 72x80 -> 9x10 over three stride-2 convs, then a linear
    layer projects to the ``z_dim`` bottleneck. The decoder mirrors it with
    transposed convs back to the full (K+1)-channel input for the MSE loss.
    """

    def __init__(self, history: int = DEFAULT_HISTORY, z_dim: int = Z_DIM):
        super().__init__()
        self.history = history
        self.z_dim = z_dim
        c_in = history + 1  # K frames + motion channel

        # 72x80 -> 36x40 -> 18x20 -> 9x10
        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self._feat_hw = (IN_H // 8, IN_W // 8)  # (9, 10)
        flat = 128 * self._feat_hw[0] * self._feat_hw[1]
        self.to_z = nn.Linear(flat, z_dim)
        self.from_z = nn.Linear(z_dim, flat)

        # 9x10 -> 18x20 -> 36x40 -> 72x80
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, c_in, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    # -- core ---------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K+1, H, W) -> (B, z_dim). Accepts a numpy stack too."""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(next(self.parameters()).device, dtype=torch.float32)
        h = self.enc(x).flatten(1)
        return self.to_z(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_z(z).view(-1, 128, *self._feat_hw)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    # -- training ------------------------------------------------------------
    def train_step(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """One gradient step of reconstruction MSE. Returns the scalar loss."""
        self.train()
        batch = batch.to(next(self.parameters()).device, dtype=torch.float32)
        recon, _ = self.forward(batch)
        loss = F.mse_loss(recon, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def fit(self, data: np.ndarray, steps: int = 300, batch_size: int = 64,
            lr: float = 1e-3, device: torch.device | None = None,
            log_every: int = 25) -> list[float]:
        """Train on an (N, K+1, H, W) array; return the per-log loss history."""
        device = device or preferred_device()
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        data_t = torch.as_tensor(data, dtype=torch.float32)
        n = data_t.shape[0]
        history: list[float] = []
        for step in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),))
            loss = self.train_step(data_t[idx], opt)
            if step % log_every == 0 or step == steps - 1:
                history.append(loss)
        return history

    @torch.no_grad()
    def encode_np(self, stack: np.ndarray) -> np.ndarray:
        """Inference helper for the controller: numpy stack -> numpy z (no grad)."""
        self.eval()
        return self.encode(stack).cpu().numpy()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
