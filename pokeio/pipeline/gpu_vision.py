"""GPUVision — batched, on-GPU replica of ``vision.preprocess.ObsBuilder``.

Takes a batch of raw Game Boy grayscale framebuffers ``(N, H, W)`` uint8 living
on ``cuda:1`` and produces the same named observation sheets ObsBuilder makes on
the CPU, but as a single batched set of torch tensors and with **zero per-frame
``np.unique``**:

    coarse   (N, S, S)   whole screen area-resampled to S x S, in [0, 1]
    fovea    (N, F, F)   F x F center crop of the native screen, [0, 1]
    motion   (N, S, S)   (coarse_t - coarse_{t-1} + 1) / 2  (0.5 on first frame)
    ram_aux  (N, K)      placeholder zeros (kept for shape-parity with ObsBuilder)

Exactness vs the CPU reference
------------------------------
* **coarse** uses the *identical* area-overlap resample matrices ObsBuilder
  builds (ported from ``preprocess._area_matrix``), so it is an exact
  area-average downscale — not ``F.interpolate``, whose non-integer-ratio 'area'
  mode diverges slightly. Computed in float64 then cast, matching the CPU path.
* **fovea** is the identical center crop.
* **motion** keeps the previous coarse batch resident on the GPU (never round
  trips to the CPU) and applies the same ``(d+1)/2`` mapping.
* **shade normalization** uses a cached 256-entry LUT instead of ``np.unique``.
  The LUT maps the (fixed, hardware) DMG palette shade values onto the evenly
  spaced levels ObsBuilder assigns when all shades are present. This is exact
  for any frame whose distinct shades are the full palette (the overwhelmingly
  common case, incl. all real gameplay frames). It only diverges from
  ObsBuilder's *per-frame re-ranking* on a frame that contains a strict interior
  subset of the palette (e.g. only the two middle shades) — a degenerate case
  that does not occur in normal play. See ``normalize_shades`` for the fallback.

Kernel strategy: the whole thing is a handful of large batched torch ops
(indexing gather + two matmuls + a slice + a subtract). For batches of a few
hundred frames on a P100 this is comfortably GPU-bound-trivial, so the pure-torch
path is used; a fused raw-CUDA kernel was profiled but not required (see
``scripts/bench_40k.py --profile``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (kept for optional interpolate path)

from pokeio.config import Config, VisionConfig
from pokeio.vision.preprocess import _area_matrix

# The four DMG (monochrome) shade values PyBoy emits in channel 0 of its RGBA
# framebuffer, ascending (measured on this ROM/state). ObsBuilder ranks the
# distinct values present and maps them onto k/(n-1); with all four present the
# ranks are exactly {0, 1/3, 2/3, 1}. The LUT encodes precisely that.
DMG_SHADES = (24, 88, 184, 248)


def build_shade_lut(shades=DMG_SHADES, n_levels: int = 4) -> np.ndarray:
    """(256,) float32 LUT: palette value -> normalized level; others -> v/255.

    Replicates ObsBuilder.normalize_shades for the full-palette case without a
    per-frame ``np.unique``. Non-palette bytes fall back to ``v/255`` so a stray
    value never crashes (it just won't be bit-exact — irrelevant for DMG).
    """
    lut = (np.arange(256, dtype=np.float32) / 255.0)
    shades = sorted(set(int(s) for s in shades))
    n = len(shades)
    if n > 1:
        levels = np.arange(n, dtype=np.float32) / (n - 1)
        for v, lvl in zip(shades, levels):
            lut[v] = lvl
    elif n == 1:
        lut[shades[0]] = 0.0
    return lut


class GPUVision:
    """Batched on-GPU obs builder. Stateful: holds previous coarse for motion."""

    KEYS = ("coarse", "fovea", "motion", "ram_aux")

    def __init__(
        self,
        config: Config | VisionConfig | None = None,
        device: str | torch.device = "cuda:1",
        shades=DMG_SHADES,
    ):
        if config is None:
            vcfg = VisionConfig()
        elif isinstance(config, Config):
            vcfg = config.vision
        elif isinstance(config, VisionConfig):
            vcfg = config
        else:
            raise TypeError(f"unsupported config type: {type(config)!r}")
        self.vcfg = vcfg
        self.device = torch.device(device)

        self.H = int(vcfg.screen_height)
        self.W = int(vcfg.screen_width)
        self.shades = int(vcfg.shades)
        self.coarse_size = int(vcfg.coarse_size)
        self.fovea_size = int(vcfg.fovea_size)
        self.ram_aux_dim = int(vcfg.ram_aux_dim)
        self.use_motion = bool(vcfg.motion_channel)

        # Identical area-resample matrices as ObsBuilder (float64 -> cast).
        row = _area_matrix(self.H, self.coarse_size)          # (S, H)
        col = _area_matrix(self.W, self.coarse_size).T        # (W, S)
        self._row = torch.from_numpy(row).to(self.device, torch.float64)
        self._col = torch.from_numpy(col).to(self.device, torch.float64)

        lut = build_shade_lut(shades, self.shades)
        self._lut = torch.from_numpy(lut).to(self.device)     # (256,) float32

        # Fovea crop geometry (may overhang; we zero-pad exactly like ObsBuilder).
        f = self.fovea_size
        self._ftop = (self.H - f) // 2
        self._fleft = (self.W - f) // 2

        self._prev_coarse: torch.Tensor | None = None

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Clear the stored previous coarse batch (motion -> 0.5 next build)."""
        self._prev_coarse = None

    # -------------------------------------------------------------- normalize
    def normalize_shades(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """(N,H,W) uint8 -> (N,H,W) float32 [0,1] via the cached LUT."""
        return self._lut[frames_u8.long()]

    # -------------------------------------------------------------- sub-sheets
    def _coarse(self, norm: torch.Tensor) -> torch.Tensor:
        """Batched exact area-average downscale to (N, S, S) (float64 -> f32)."""
        x = norm.to(torch.float64)                              # (N,H,W)
        # (S,H) x (N,H,W) -> (N,S,W) ; then (N,S,W) x (W,S) -> (N,S,S)
        tmp = torch.einsum("sh,nhw->nsw", self._row, x)
        out = torch.einsum("nsw,wc->nsc", tmp, self._col)
        return out.to(torch.float32)

    def _fovea(self, norm: torch.Tensor) -> torch.Tensor:
        """(N,F,F) center crop of the native screen, zero-padded if it overhangs."""
        f = self.fovea_size
        n = norm.shape[0]
        top, left = self._ftop, self._fleft
        sr0, sr1 = max(0, top), min(self.H, top + f)
        sc0, sc1 = max(0, left), min(self.W, left + f)
        if sr0 == top and sc0 == left and sr1 == top + f and sc1 == left + f:
            return norm[:, top:top + f, left:left + f].contiguous()
        out = torch.zeros((n, f, f), dtype=torch.float32, device=self.device)
        dr0, dc0 = sr0 - top, sc0 - left
        out[:, dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = norm[:, sr0:sr1, sc0:sc1]
        return out

    def _motion(self, coarse: torch.Tensor) -> torch.Tensor:
        if self._prev_coarse is None or self._prev_coarse.shape != coarse.shape:
            return torch.full_like(coarse, 0.5)
        return (coarse - self._prev_coarse + 1.0) * 0.5

    # ------------------------------------------------------------------ build
    @torch.no_grad()
    def build(self, frames_u8: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build the batched obs dict from (N,H,W) uint8 frames on-device."""
        if frames_u8.device != self.device:
            frames_u8 = frames_u8.to(self.device, non_blocking=True)
        norm = self.normalize_shades(frames_u8)
        coarse = self._coarse(norm)
        fovea = self._fovea(norm)
        motion = (
            self._motion(coarse)
            if self.use_motion
            else torch.full_like(coarse, 0.5)
        )
        self._prev_coarse = coarse
        n = frames_u8.shape[0]
        ram_aux = torch.zeros((n, self.ram_aux_dim), dtype=torch.float32,
                              device=self.device)
        return {"coarse": coarse, "fovea": fovea, "motion": motion, "ram_aux": ram_aux}

    @torch.no_grad()
    def flatten(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """(N, flat_dim) concatenation in ObsBuilder order: coarse,fovea,motion,ram_aux."""
        n = obs["coarse"].shape[0]
        return torch.cat(
            [obs["coarse"].reshape(n, -1), obs["fovea"].reshape(n, -1),
             obs["motion"].reshape(n, -1), obs["ram_aux"].reshape(n, -1)],
            dim=1,
        )

    # --------------------------------------------------------------- geometry
    @property
    def optical_dim(self) -> int:
        s, f = self.coarse_size, self.fovea_size
        return s * s + f * f + s * s

    @property
    def flat_dim(self) -> int:
        return self.optical_dim + self.ram_aux_dim


__all__ = ["GPUVision", "build_shade_lut", "DMG_SHADES"]
