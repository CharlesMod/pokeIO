"""ObsBuilder — turns a raw Game Boy grayscale frame into the observation dict
the evolving network consumes (Phase 1 vision spec, see TODO.md).

The observation is a dict of named float32 fields:

    coarse   (S, S)   whole 144x160 screen area-resampled to SxS, in [0,1]
    fovea    (F, F)   FxF center crop of the NATIVE screen (no resample), [0,1]
    motion   (S, S)   coarse_t - coarse_{t-1} mapped to [0,1] via (d+1)/2
    ram_aux  (K,)     placeholder RAM aux vector (zeros for now; Phase 3 fills it)

Design notes
------------
* Game-agnostic. The fovea is a fixed screen-centered crop: the player sprite is
  screen-centered in GB overworld games, and it stays an informative center crop
  in menus/battles. It is padded (with 0.0) if the crop exceeds screen bounds.
* Grayscale normalization is robust to the DMG's <=4 shade values: detect the
  unique values, rank them, and map ascending shades onto {0, 1/3, 2/3, 1}
  (n distinct -> k/(n-1)). Falls back to /255 if more than `shades` distinct
  values appear (e.g. a GBC/color buffer).
* ``ram_aux`` is ALWAYS a separate named field. It is never blended into the
  visual sheets. This keeps the substrate's RAM-input coordinate region distinct
  (Phase 2) and lets the HyperNEAT substrate address the sheets geometrically.
* The sheets remain individually accessible (for HyperNEAT); ``flatten`` gives a
  single concatenated 1D vector (for the fixed-topology GA baseline).

Sizes are read from ``config.vision``; nothing here is game-specific.
"""

from __future__ import annotations

import numpy as np

from pokeio.config import Config, VisionConfig


def _area_matrix(in_size: int, out_size: int) -> np.ndarray:
    """(out_size, in_size) row-normalized area-overlap resample matrix.

    Output pixel i covers the input span [i*in/out, (i+1)*in/out); each entry is
    the fractional overlap of that span with input pixel j. Rows sum to 1, so
    ``M @ signal`` is an exact area-average downscale (and a sane upscale too).
    Deterministic and vectorizable via a single matmul.
    """
    m = np.zeros((out_size, in_size), dtype=np.float64)
    scale = in_size / out_size
    for i in range(out_size):
        lo = i * scale
        hi = (i + 1) * scale
        j0 = int(np.floor(lo))
        j1 = int(np.ceil(hi))
        for j in range(j0, min(j1, in_size)):
            overlap = min(hi, j + 1) - max(lo, j)
            if overlap > 0:
                m[i, j] = overlap
        s = m[i].sum()
        if s > 0:
            m[i] /= s
    return m


class ObsBuilder:
    """Builds the observation dict from a raw grayscale screen.

    Stateful: it stores the previous coarse frame internally to compute the
    motion sheet. Call ``reset()`` at episode boundaries to clear it.
    """

    KEYS = ("coarse", "fovea", "motion", "ram_aux")

    def __init__(self, config: Config | VisionConfig | None = None):
        if config is None:
            vcfg = VisionConfig()
        elif isinstance(config, Config):
            vcfg = config.vision
        elif isinstance(config, VisionConfig):
            vcfg = config
        else:
            raise TypeError(f"unsupported config type: {type(config)!r}")
        self.vcfg = vcfg

        self.H = int(vcfg.screen_height)
        self.W = int(vcfg.screen_width)
        self.shades = int(vcfg.shades)
        self.coarse_size = int(vcfg.coarse_size)
        self.fovea_size = int(vcfg.fovea_size)
        self.ram_aux_dim = int(vcfg.ram_aux_dim)
        self.use_motion = bool(vcfg.motion_channel)

        # Precompute the area-resample matrices for the coarse sheet.
        self._row_mat = _area_matrix(self.H, self.coarse_size)  # (S, H)
        self._col_mat = _area_matrix(self.W, self.coarse_size).T  # (W, S)

        self._prev_coarse: np.ndarray | None = None

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Clear the stored previous frame (motion is 0.5 on the next build)."""
        self._prev_coarse = None

    # -------------------------------------------------------------- normalize
    def normalize_shades(self, screen_gray: np.ndarray) -> np.ndarray:
        """Map a uint8 grayscale frame to float32 [0,1] robustly.

        <=`shades` distinct values -> rank map onto evenly spaced levels
        (4 shades -> {0, 1/3, 2/3, 1}); otherwise fall back to /255.
        """
        g = np.asarray(screen_gray)
        uniq = np.unique(g)
        n = uniq.size
        if 1 < n <= self.shades:
            levels = (np.arange(n, dtype=np.float32) / (n - 1)).astype(np.float32)
            idx = np.searchsorted(uniq, g)
            return levels[idx].astype(np.float32, copy=False)
        if n == 1:
            # A single flat shade is ambiguous; treat it as darkest.
            return np.zeros(g.shape, dtype=np.float32)
        return (g.astype(np.float32) / 255.0)

    # -------------------------------------------------------------- sub-sheets
    def _coarse(self, norm: np.ndarray) -> np.ndarray:
        """Area-resample the normalized full screen to coarse_size x coarse_size."""
        out = self._row_mat @ norm.astype(np.float64) @ self._col_mat
        return out.astype(np.float32, copy=False)

    def _fovea(self, norm: np.ndarray) -> np.ndarray:
        """FxF center crop of the native screen; zero-padded if out of bounds."""
        f = self.fovea_size
        out = np.zeros((f, f), dtype=np.float32)
        # Center of the screen; crop [top, top+f) x [left, left+f).
        top = (self.H - f) // 2
        left = (self.W - f) // 2
        # Source region clipped to the screen.
        sr0, sr1 = max(0, top), min(self.H, top + f)
        sc0, sc1 = max(0, left), min(self.W, left + f)
        # Destination offset when the crop overhangs a screen edge.
        dr0 = sr0 - top
        dc0 = sc0 - left
        out[dr0 : dr0 + (sr1 - sr0), dc0 : dc0 + (sc1 - sc0)] = norm[sr0:sr1, sc0:sc1]
        return out

    def _motion(self, coarse: np.ndarray) -> np.ndarray:
        """(coarse_t - coarse_{t-1} + 1) / 2 in [0,1]; 0.5 when no prev frame."""
        if self._prev_coarse is None:
            return np.full_like(coarse, 0.5, dtype=np.float32)
        diff = coarse - self._prev_coarse  # in [-1, 1]
        return ((diff + 1.0) * 0.5).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ build
    def build(
        self, screen_gray: np.ndarray, ram_vector: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Build the observation dict from a raw (H, W) uint8 grayscale frame."""
        norm = self.normalize_shades(screen_gray)
        coarse = self._coarse(norm)
        fovea = self._fovea(norm)
        motion = (
            self._motion(coarse)
            if self.use_motion
            else np.full_like(coarse, 0.5, dtype=np.float32)
        )
        self._prev_coarse = coarse

        ram_aux = self._ram_aux(ram_vector)

        return {
            "coarse": coarse,
            "fovea": fovea,
            "motion": motion,
            "ram_aux": ram_aux,
        }

    def _ram_aux(self, ram_vector: np.ndarray | None) -> np.ndarray:
        """Length-K float32 RAM aux vector (PLACEHOLDER until Phase 3).

        If a raw vector is provided we take the first K bytes, normalize /255,
        and zero-pad; otherwise return zeros. Never mixed into the visual sheets.
        """
        k = self.ram_aux_dim
        out = np.zeros(k, dtype=np.float32)
        if ram_vector is not None:
            v = np.asarray(ram_vector, dtype=np.float32).ravel()
            if v.size:
                take = min(k, v.size)
                seg = v[:take]
                # Normalize raw bytes to [0,1] if they look like uint8 reads.
                if seg.max(initial=0.0) > 1.0:
                    seg = seg / 255.0
                out[:take] = seg
        return out

    # ---------------------------------------------------------------- helpers
    def flatten(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all fields into one 1D float32 vector (GA baseline input).

        Order: coarse, fovea, motion, ram_aux. ``flat_dim`` gives the length.
        """
        parts = [
            obs["coarse"].ravel(),
            obs["fovea"].ravel(),
            obs["motion"].ravel(),
            obs["ram_aux"].ravel(),
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)

    # --------------------------------------------------------------- geometry
    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        s, f, k = self.coarse_size, self.fovea_size, self.ram_aux_dim
        return {
            "coarse": (s, s),
            "fovea": (f, f),
            "motion": (s, s),
            "ram_aux": (k,),
        }

    @property
    def optical_dim(self) -> int:
        """Flattened length of the visual sheets only (coarse+fovea+motion)."""
        s, f = self.coarse_size, self.fovea_size
        return s * s + f * f + s * s

    @property
    def flat_dim(self) -> int:
        """Flattened length of the full obs (optical + ram_aux)."""
        return self.optical_dim + self.ram_aux_dim


__all__ = ["ObsBuilder"]
