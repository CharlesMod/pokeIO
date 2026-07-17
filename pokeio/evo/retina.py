r"""Lane C — the learned-encoder "retina" (decoder-free, self-supervised).

A small convolutional encoder trained by gradient descent on the game's OWN
frames (no external data, no labels). It compresses short windows of grayscale
screen frames into a compact latent ``z = [z_periph | z_fovea]`` that an evolved
recurrent controller consumes as its observation (the ERL-Re² pattern: a
gradient-trained representation, an evolution-trained policy on top).

Why decoder-free (this is a rewrite of the old MSE autoencoder)
--------------------------------------------------------------
Pokémon screens are mostly static tiles and text. A pixel-reconstruction loss
spends all its capacity re-drawing the HUD / dialog boxes and learns almost
nothing about the tiny, agent-*controllable* part of the frame — the exact trap
that made the old ``F.mse_loss(recon, batch)`` encoder useless for control. This
module removes the decoder entirely and replaces reconstruction with two
control-relevant self-supervised objectives (docs/specs/active-vision-spine.md
§4):

  * **SPR** (self-predictive representations): roll a latent *transition model*
    forward ``K`` steps conditioned on the actions taken, and pull the predicted
    future latent toward the (stop-gradient, EMA-encoded, augmented) latent of
    the frame that actually occurred. Normalized **cosine** loss — never L2.
    Collapse is prevented by an asymmetric online predictor + stop-gradient
    (BYOL/SimSiam discipline), *not* by a reconstruction anchor.
  * **Inverse dynamics** (``q_psi``): predict the button pressed between two
    consecutive latents. This forces the features onto agent-controllable
    content and away from HUD/animation. The same net is reused for the
    empowerment reward term (§6.4), so it is exposed as ``self.q_psi``.

The Go-Explore cell code is an **FSQ** quantization of ``z_periph`` (§4.4): a
finite scalar quantizer that, unlike VQ, cannot codebook-collapse and needs no
commitment loss / EMA codebook / dead-code reseed. The integer 5-tuple *is* the
cell code, computed on the periphery so cells are invariant to gaze.

Two-stream, one encoder
-----------------------
The encoder is a single shared-weight Nature-CNN applied to two 84×84 4-frame
stacks: the *periphery* (whole screen) and the *fovea* (a native 48×48 gaze
crop upsampled to 84×84). Global-pooled linear heads produce ``z_periph`` (48)
and ``z_fovea`` (32); the policy latent is their concatenation (80).

P100 / sm_60 note: fp32 throughout — no AMP / bf16 / fp16 autocast. The Tesla
P100 has no tensor cores, so mixed precision buys nothing here and can error.
At ~0.3M params the encoder is never the bottleneck (the ~10k env-step/s CPU
wall is), so fp32 is free. ``grid_sample`` and ``BatchNorm`` used below are all
fine on Pascal.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Native Game Boy screen.
RAW_H, RAW_W = 144, 160

# Nature-CNN input geometry: each stream is a K-frame stack at IN_SIDE×IN_SIDE.
IN_SIDE = 84
FRAME_STACK = 4
FOVEA_NATIVE = 48        # native fovea crop side (upsampled to IN_SIDE)
DEFAULT_HISTORY = FRAME_STACK  # kept for backward-compat imports

# Default latent widths (docs/specs/active-vision-spine.md §4.2).
Z_PERIPH = 48
Z_FOVEA = 32
Z_DIM = Z_PERIPH + Z_FOVEA  # policy latent width = 80


def preferred_device() -> torch.device:
    """cuda:1 if present (card 0 hosts the GLM server / retina learner), else cpu."""
    if torch.cuda.is_available():
        idx = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


# --------------------------------------------------------------------------- io
def _to_chw(frame: np.ndarray) -> torch.Tensor:
    """(H,W) uint8/float -> (1,1,H,W) float32 in [0,1]."""
    f = torch.as_tensor(np.ascontiguousarray(frame))
    if not torch.is_floating_point(f):
        f = f.float().div_(255.0)
    else:
        f = f.float()
    return f.view(1, 1, *f.shape[-2:])


def _resize_to(f: torch.Tensor, size: int) -> torch.Tensor:
    """(1,1,H,W) -> (1,1,size,size). Area-average when shrinking, bilinear up."""
    h, w = f.shape[-2:]
    if h >= size and w >= size:
        return F.adaptive_avg_pool2d(f, (size, size))
    return F.interpolate(f, size=(size, size), mode="bilinear", align_corners=False)


def downscale(frame: np.ndarray, size: int = IN_SIDE) -> np.ndarray:
    """Raw (144,160) uint8 -> (size,size) float32 in [0,1] via area-average."""
    return _resize_to(_to_chw(frame), size).view(size, size).numpy()


def crop_fovea(frame: np.ndarray, gy: int, gx: int, native: int = FOVEA_NATIVE,
               size: int = IN_SIDE) -> np.ndarray:
    """Native ``native``×``native`` crop centered at gaze ``(gy,gx)``, upsampled to
    ``size``×``size`` (zero-padded where the crop overhangs an edge).

    ``(gy,gx)`` are pixel coordinates on the raw (144,160) screen.
    """
    f = _to_chw(frame)[0, 0]                       # (H,W) float32
    half = native // 2
    top, left = gy - half, gx - half
    crop = f.new_zeros((native, native))
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(RAW_H, top + native), min(RAW_W, left + native)
    if y1 > y0 and x1 > x0:
        crop[y0 - top:y1 - top, x0 - left:x1 - left] = f[y0:y1, x0:x1]
    return _resize_to(crop.view(1, 1, native, native), size).view(size, size).numpy()


def build_stack(frames: list[np.ndarray], t: int, history: int = FRAME_STACK,
                size: int = IN_SIDE) -> np.ndarray:
    """One periphery input: (history, size, size) float32 = frames [t-history+1 .. t].

    Frames before index 0 are clamped to frame 0 (episode start), so ``t`` may be
    anything in ``[0, len(frames)-1]``.
    """
    idx = [max(0, t - history + 1 + k) for k in range(history)]
    chans = [downscale(frames[i], size) for i in idx]
    return np.stack(chans, axis=0).astype(np.float32)


def build_fovea_stack(frames: list[np.ndarray], t: int, gaze: list | None = None,
                      history: int = FRAME_STACK, native: int = FOVEA_NATIVE,
                      size: int = IN_SIDE) -> np.ndarray:
    """One fovea input: (history, size, size) float32 of gaze crops [t-history+1 .. t].

    ``gaze`` is an optional list of ``(gy,gx)`` per frame; defaults to screen center.
    """
    idx = [max(0, t - history + 1 + k) for k in range(history)]
    if gaze is None:
        gy, gx = RAW_H // 2, RAW_W // 2
        chans = [crop_fovea(frames[i], gy, gx, native, size) for i in idx]
    else:
        chans = [crop_fovea(frames[i], gaze[i][0], gaze[i][1], native, size) for i in idx]
    return np.stack(chans, axis=0).astype(np.float32)


def build_dataset(frames: list[np.ndarray], history: int = FRAME_STACK,
                  size: int = IN_SIDE) -> np.ndarray:
    """Turn a frame rollout into an (N, history, size, size) periphery-stack array."""
    stacks = [build_stack(frames, t, history, size) for t in range(len(frames))]
    return np.stack(stacks, axis=0).astype(np.float32)


class RetinaStacker:
    """Rolling K-frame history for online use inside the env loop (single stream).

    Feed raw (144,160) observations via :meth:`push`; get back the current
    ``(1, history, IN_SIDE, IN_SIDE)`` stack. Use one stacker for the periphery
    (push the full screen) and a second for the fovea (push gaze crops).
    """

    def __init__(self, history: int = FRAME_STACK, size: int = IN_SIDE):
        self.history = history
        self.size = size
        self.buf: deque[np.ndarray] = deque(maxlen=history)

    def reset(self, small: np.ndarray) -> None:
        self.buf.clear()
        for _ in range(self.history):
            self.buf.append(small)

    def push(self, frame: np.ndarray) -> np.ndarray:
        """``frame`` is a raw (144,160) screen (periphery) or a raw crop (fovea)."""
        small = downscale(frame, self.size)
        if not self.buf:
            self.reset(small)
        else:
            self.buf.append(small)
        return np.stack(list(self.buf), axis=0)[None].astype(np.float32)


# ------------------------------------------------------------------ sub-modules
class NatureCNN(nn.Module):
    """Nature-DQN conv trunk: [B,K,84,84] -> [B,64,7,7] (shared over both streams)."""

    def __init__(self, c_in: int = FRAME_STACK):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, stride=4), nn.ReLU(inplace=True),  # 84 -> 20
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),    # 20 -> 9
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(inplace=True),    # 9  -> 7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transition(nn.Module):
    """SPR latent transition: 2× 64-ch 3×3 conv on the 7×7 map, BN after conv1 only.

    The action is broadcast (one value per channel) to every spatial cell.
    """

    def __init__(self, ch: int = 64, act_dim: int = 11):
        super().__init__()
        self.c1 = nn.Conv2d(ch + act_dim, ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a = a[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
        x = torch.cat([h, a], dim=1)
        x = F.relu(self.bn(self.c1(x)))
        x = F.relu(self.c2(x))
        return x


class Projection(nn.Module):
    """SPR projection head: 64 -> 256 (BN, ReLU) -> 256."""

    def __init__(self, c_in: int = 64, dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(c_in, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.bn(self.fc1(x))))


# ------------------------------------------------------------------ the network
class Retina(nn.Module):
    """Decoder-free SSL encoder (SPR + inverse dynamics + FSQ cell code).

    See :mod:`docs/specs/active-vision-spine.md` §4. Two 84×84 4-frame streams
    (periphery, fovea) share one Nature-CNN; pooled linear heads emit
    ``z_periph`` and ``z_fovea``. Trained by SPR (cosine, horizon ``spr_k``) plus
    an inverse-dynamics head; the FSQ quantization of ``z_periph`` is the
    Go-Explore cell code.
    """

    def __init__(self, z_periph: int = Z_PERIPH, z_fovea: int = Z_FOVEA,
                 spr_k: int = 5, ema_tau: float = 0.0, inverse_dynamics: bool = True,
                 fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5), n_act: int = 9,
                 c_in: int = FRAME_STACK, proj_dim: int = 256,
                 aug_shift_px: int = 4, aug_jitter: float = 0.05,
                 lambda_spr: float = 2.0):
        super().__init__()
        self.z_periph = int(z_periph)
        self.z_fovea = int(z_fovea)
        self.z_dim = self.z_periph + self.z_fovea
        self.spr_k = int(spr_k)
        self.ema_tau = float(ema_tau)
        self.inverse_dynamics = bool(inverse_dynamics)
        self.n_act = int(n_act)
        self.fsq_levels = list(fsq_levels)
        self.aug_shift_px = int(aug_shift_px)
        self.aug_jitter = float(aug_jitter)
        self.lambda_spr = float(lambda_spr)
        # Transition conditions on the full action: n_act button one-hot + (dx,dy).
        self.act_cond_dim = self.n_act + 2

        # -- shared encoder + pooled latent heads --
        self.encoder = NatureCNN(c_in)
        self.periph_head = nn.Linear(64, self.z_periph)
        self.fovea_head = nn.Linear(64, self.z_fovea)

        # -- SPR: transition + online projection + asymmetric predictor --
        self.transition = Transition(64, self.act_cond_dim)
        self.proj = Projection(64, proj_dim)
        self.predictor = nn.Linear(proj_dim, proj_dim)  # asymmetric collapse guard

        # -- SPR target branch (EMA of encoder+proj; no predictor, no grad) --
        self.target_encoder = NatureCNN(c_in)
        self.target_proj = Projection(64, proj_dim)
        self._sync_target(hard=True)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_proj.parameters():
            p.requires_grad_(False)

        # -- inverse-dynamics head q_psi (reused for empowerment §6.4) --
        self.q_psi = nn.Sequential(
            nn.Linear(2 * self.z_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.n_act),
        )

        # -- FSQ projection (z_periph -> len(levels) dims); no codebook needed --
        self.fsq_proj = nn.Linear(self.z_periph, len(self.fsq_levels))
        self.register_buffer(
            "fsq_levels_t", torch.tensor(self.fsq_levels, dtype=torch.float32))

    # -- device / tensor helpers -------------------------------------------
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _t(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        return x.to(self._device(), dtype=torch.float32)

    @staticmethod
    def _pool(m: torch.Tensor) -> torch.Tensor:
        return m.mean(dim=(2, 3))  # global average pool -> [B,64]

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # [B,64,7,7]

    # -- core encode --------------------------------------------------------
    def encode(self, periph: torch.Tensor, fovea: torch.Tensor) -> torch.Tensor:
        """(B,K,84,84)×2 -> (B, z_periph+z_fovea). Accepts numpy too."""
        periph = self._t(periph)
        fovea = self._t(fovea)
        zp = self.periph_head(self._pool(self._feat(periph)))
        zf = self.fovea_head(self._pool(self._feat(fovea)))
        return torch.cat([zp, zf], dim=-1)

    def z_periph_of(self, periph: torch.Tensor) -> torch.Tensor:
        return self.periph_head(self._pool(self._feat(self._t(periph))))

    def inverse_logits(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        """q_psi(a | z_t, z_{t+1}) button logits — exposed for empowerment reuse."""
        return self.q_psi(torch.cat([z_t, z_t1], dim=-1))

    # -- augmentation (§4.3: ±shift px + intensity jitter; NO horizontal flip) --
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        s = self.aug_shift_px
        if s > 0:
            tx = torch.randint(-s, s + 1, (b,), device=x.device).float() * (2.0 / w)
            ty = torch.randint(-s, s + 1, (b,), device=x.device).float() * (2.0 / h)
            theta = torch.zeros(b, 2, 3, device=x.device)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            theta[:, 0, 2] = tx
            theta[:, 1, 2] = ty
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, mode="nearest", padding_mode="border",
                              align_corners=False)
        j = self.aug_jitter
        if j > 0:
            scale = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * j
            x = (x * scale).clamp(0.0, 1.0)
        return x

    # -- SPR action conditioning -------------------------------------------
    def _act_cond(self, buttons: torch.Tensor, saccade: torch.Tensor | None) -> torch.Tensor:
        onehot = F.one_hot(buttons.clamp(0, self.n_act - 1), self.n_act).float()
        if saccade is None:
            saccade = onehot.new_zeros((onehot.shape[0], 2))
        return torch.cat([onehot, saccade], dim=-1)  # [B, n_act+2]

    # -- EMA target sync ----------------------------------------------------
    @torch.no_grad()
    def _sync_target(self, hard: bool = False) -> None:
        tau = 0.0 if hard else self.ema_tau
        pairs = ((self.encoder, self.target_encoder), (self.proj, self.target_proj))
        for online, target in pairs:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(tau).add_(po.detach(), alpha=1.0 - tau)
            for bo, bt in zip(online.buffers(), target.buffers()):
                bt.copy_(bo)  # BN running stats: hard-copy either way

    # -- training -----------------------------------------------------------
    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer) -> dict:
        """One SPR + inverse-dynamics gradient step.

        ``batch`` keys:
          ``periph``  : (B, T, K, 84, 84) periphery stacks, T = spr_k + 1
          ``fovea``   : (B, T, K, 84, 84) fovea stacks (optional; falls back to periph)
          ``actions`` : (B, T-1) int64 button ids
          ``saccade`` : (B, T-1, 2) float32 (dx,dy), optional

        Returns a dict of scalar losses + inverse-dynamics accuracy.
        """
        self.train()
        dev = self._device()
        periph = self._t(batch["periph"])                       # [B,T,K,84,84]
        fovea = batch.get("fovea")
        fovea = periph if fovea is None else self._t(fovea)
        actions = torch.as_tensor(batch["actions"]).to(dev).long()  # [B,T-1]
        saccade = batch.get("saccade")
        if saccade is not None:
            saccade = self._t(saccade)
        b, tt = periph.shape[0], periph.shape[1]
        k = min(self.spr_k, tt - 1, actions.shape[1])

        # SimSiam/BYOL target refresh (ema_tau=0.0 -> hard copy of online).
        self._sync_target(hard=(self.ema_tau == 0.0))

        # ---- SPR: roll the transition model K steps in latent-map space ----
        h = self._feat(self._augment(periph[:, 0]))             # [B,64,7,7]
        spr_terms = []
        for step in range(k):
            sacc_k = None if saccade is None else saccade[:, step]
            a = self._act_cond(actions[:, step], sacc_k)
            h = self.transition(h, a)
            pred = self.predictor(self.proj(self._pool(h)))     # [B,256]
            with torch.no_grad():
                tgt_map = self.target_encoder(self._augment(periph[:, step + 1]))
                tgt = self.target_proj(self._pool(tgt_map))     # [B,256]
            pred = F.normalize(pred, dim=-1)
            tgt = F.normalize(tgt, dim=-1)
            spr_terms.append(-(pred * tgt).sum(-1))             # cosine, [B]
        spr_loss = torch.stack(spr_terms, dim=0).mean()

        # ---- inverse dynamics on clean (un-augmented) policy latents ----
        z_seq = [self.encode(periph[:, t], fovea[:, t]) for t in range(k + 1)]
        inv_terms = []
        correct = 0
        total = 0
        if self.inverse_dynamics:
            for step in range(k):
                logits = self.inverse_logits(z_seq[step], z_seq[step + 1])
                inv_terms.append(F.cross_entropy(logits, actions[:, step]))
                correct += int((logits.argmax(-1) == actions[:, step]).sum().item())
                total += b
            inv_loss = torch.stack(inv_terms, dim=0).mean()
        else:
            inv_loss = spr_loss.new_zeros(())

        loss = self.lambda_spr * spr_loss + inv_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {
            "loss": float(loss.detach()),
            "spr": float(spr_loss.detach()),
            "inv": float(inv_loss.detach()),
            "inv_acc": (correct / total) if total else float("nan"),
        }

    def fit(self, frames: list[np.ndarray], actions=None, *, steps: int = 300,
            batch_size: int = 32, lr: float = 1e-3, device: torch.device | None = None,
            gaze: list | None = None, log_every: int = 25) -> list[dict]:
        """Standalone warm-up harness over an in-memory replay buffer of frames.

        ``frames`` is a contiguous rollout of raw (144,160) screens; ``actions`` is
        the per-transition button id (len ``len(frames)-1``). Builds periphery +
        fovea 4-stacks once, then samples length-(spr_k+1) windows and steps.

        NOTE: this materializes the full stack tensors, so it is for MODEST
        buffers (a few thousand frames). The real 200k-frame warm-up should stream
        minibatches straight into :meth:`train_step` from the swarm replay buffer.
        """
        device = device or preferred_device()
        self.to(device)
        n = len(frames)
        if actions is None:
            actions = np.zeros(n - 1, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)

        periph = torch.from_numpy(build_dataset(frames))              # [N,K,84,84]
        fov = torch.from_numpy(np.stack(
            [build_fovea_stack(frames, t, gaze) for t in range(n)], axis=0))
        acts = torch.from_numpy(actions)                             # [N-1]

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        k = self.spr_k
        max_start = n - 1 - k
        if max_start < 0:
            raise ValueError(f"need at least spr_k+1={k + 1} frames, got {n}")
        hist: list[dict] = []
        for step in range(steps):
            starts = torch.randint(0, max_start + 1, (min(batch_size, max_start + 1),))
            p = torch.stack([periph[s:s + k + 1] for s in starts.tolist()])   # [B,K+1,4,84,84]
            f = torch.stack([fov[s:s + k + 1] for s in starts.tolist()])
            a = torch.stack([acts[s:s + k] for s in starts.tolist()])          # [B,K]
            out = self.train_step({"periph": p, "fovea": f, "actions": a}, opt)
            if step % log_every == 0 or step == steps - 1:
                hist.append(out)
        return hist

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def encode_np(self, periph_stack: np.ndarray, fovea_stack: np.ndarray) -> np.ndarray:
        """Controller inference: numpy stacks -> numpy z (no grad, eval).

        Accepts a single stack ``(K,84,84)`` -> ``(z_dim,)`` or a batch
        ``(B,K,84,84)`` -> ``(B,z_dim)``.
        """
        self.eval()
        p = self._t(periph_stack)
        f = self._t(fovea_stack)
        squeeze = p.dim() == 3
        if squeeze:
            p = p[None]
            f = f[None]
        z = self.encode(p, f).cpu().numpy().astype(np.float32)
        return z[0] if squeeze else z

    def _fsq_quantize(self, z5: torch.Tensor) -> torch.Tensor:
        """[B,L] real -> [B,L] int levels in [0, level_i-1] (bounded, no codebook)."""
        levels = self.fsq_levels_t.to(z5.device)
        zt = torch.tanh(z5)                                   # (-1,1)
        return torch.round((zt + 1.0) * 0.5 * (levels - 1.0)).long()

    @torch.no_grad()
    def fsq_code(self, periph_stack: np.ndarray) -> np.ndarray:
        """Periphery -> the integer FSQ 5-tuple that IS the Go-Explore cell code."""
        self.eval()
        p = self._t(periph_stack)
        squeeze = p.dim() == 3
        if squeeze:
            p = p[None]
        z5 = self.fsq_proj(self.z_periph_of(p))
        codes = self._fsq_quantize(z5).cpu().numpy().astype(np.int64)
        return codes[0] if squeeze else codes

    def snapshot(self, device: torch.device | None = None) -> "Retina":
        """Frozen (eval, no-grad) deep copy for population inference on ``infer_card``."""
        self.eval()
        snap = copy.deepcopy(self)
        if device is not None:
            snap.to(device)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        return snap

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
