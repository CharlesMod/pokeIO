"""Basal ganglia — the gradient actor-critic (#32), the ONE load-bearing region.

Replaces the direct-encoded NEAT genome with a gradient-trained actor-critic on the
foveal percept. Small and Pascal-friendly (sm_60: float32, conv + MLP, NO attention).

Percept -> gradient (occipital, #33), MULTI-RESOLUTION. A foveal obs is
``periph(G^2) | fovea(FG^2) | motion(G^2) | proprio | ram``. Two regimes:

  * **Sharp fovea (FG > G)** — the fovea is a high-acuity native crop (e.g. FG=48,
    the 48px active region at 1 px/cell, legible) while the periphery/motion stay
    coarse (G=12). The trunk runs a dedicated strided CONV over the FGxFG fovea
    (downsampling it to ~GxG features) alongside a shared conv over the stacked
    periphery+motion GxG sheets — biomimetic acuity gradient, and the sharp detail
    reaches the gradient.
  * **Uniform (FG == G)** — legacy: the three equal GxG sheets are one 3-channel
    conv (byte-compatible with earlier FG=12 policies/champions).

Falls back to a flat MLP if the layout doesn't parse (defensive).

Action space (v1): the 9-way button (Discrete PPO); gaze is driven outside the RL
distribution (centered, or reflex — the active-vision layer). Actor emits button
logits + a state value.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

N_BUTTONS = 9  # up down left right A B START SELECT NOOP (ACTIONS order)


def _orthogonal(module: nn.Module, gain: float) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class ActorCritic(nn.Module):
    """obs -> (button logits, value). Multi-resolution foveal trunk.

    ``obs_dim`` = flat foveal obs width (= ``FovealEncoder.dim`` = ``fleet.obs_dim``).
    ``periph_grid`` (G) = periphery/motion sheet side; ``fovea_grid`` (FG) = the sharp
    fovea side (FG>G => sharp native fovea; FG==G => legacy uniform).
    """

    def __init__(self, obs_dim: int, periph_grid: int = 12, fovea_grid: int | None = None,
                 hidden: int = 256, n_buttons: int = N_BUTTONS, grid: int | None = None) -> None:
        super().__init__()
        if grid is not None:            # back-compat alias for periph_grid
            periph_grid = grid
        if fovea_grid is None:
            fovea_grid = periph_grid
        self.obs_dim = int(obs_dim)
        self.G = int(periph_grid)
        self.FG = int(fovea_grid)
        self.grid = self.G              # legacy attr
        self.n_buttons = int(n_buttons)

        G, FG = self.G, self.FG
        self.sharp = FG > G
        n_spatial = (2 * G * G + FG * FG) if self.sharp else (3 * G * G)
        self.n_extra = self.obs_dim - n_spatial
        self.use_conv = self.n_extra >= 0 and G > 0 and FG > 0

        if self.use_conv and self.sharp:
            # periphery + motion: shared 2-channel GxG conv
            self.pm_conv = nn.Sequential(
                _orthogonal(nn.Conv2d(2, 16, 3, padding=1), np.sqrt(2)), nn.ReLU(),
                _orthogonal(nn.Conv2d(16, 32, 3, padding=1), np.sqrt(2)), nn.ReLU())
            # sharp fovea: dedicated conv, stride-2 twice to bring FG -> ~FG/4 features
            self.fov_conv = nn.Sequential(
                _orthogonal(nn.Conv2d(1, 16, 3, stride=2, padding=1), np.sqrt(2)), nn.ReLU(),
                _orthogonal(nn.Conv2d(16, 32, 3, stride=2, padding=1), np.sqrt(2)), nn.ReLU())
            fov_side = (FG + 1) // 2
            fov_side = (fov_side + 1) // 2
            trunk_in = 32 * G * G + 32 * fov_side * fov_side + self.n_extra
        elif self.use_conv:
            # legacy uniform: single 3-channel GxG conv
            self.conv = nn.Sequential(
                _orthogonal(nn.Conv2d(3, 16, 3, padding=1), np.sqrt(2)), nn.ReLU(),
                _orthogonal(nn.Conv2d(16, 32, 3, padding=1), np.sqrt(2)), nn.ReLU())
            trunk_in = 32 * G * G + self.n_extra
        else:
            trunk_in = self.obs_dim

        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(trunk_in, hidden), np.sqrt(2)), nn.ReLU(),
            _orthogonal(nn.Linear(hidden, hidden), np.sqrt(2)), nn.ReLU())
        self.pi = _orthogonal(nn.Linear(hidden, self.n_buttons), 0.01)
        self.vf = _orthogonal(nn.Linear(hidden, 1), 1.0)

    # -- forward -----------------------------------------------------------
    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.use_conv:
            return self.trunk(obs)
        G, FG = self.G, self.FG
        if self.sharp:
            g2, f2 = G * G, FG * FG
            periph = obs[:, :g2]
            fovea = obs[:, g2:g2 + f2]
            motion = obs[:, g2 + f2:2 * g2 + f2]
            extra = obs[:, 2 * g2 + f2:]
            pm = torch.stack([periph, motion], dim=1).reshape(-1, 2, G, G)
            fov = fovea.reshape(-1, 1, FG, FG)
            h = torch.cat([self.pm_conv(pm).flatten(1), self.fov_conv(fov).flatten(1)], dim=1)
            h = torch.cat([h, extra], dim=1) if extra.shape[1] else h
            return self.trunk(h)
        n = 3 * G * G
        sheets = obs[:, :n].reshape(-1, 3, G, G)
        extra = obs[:, n:]
        h = self.conv(sheets).flatten(1)
        h = torch.cat([h, extra], dim=1) if extra.shape[1] else h
        return self.trunk(h)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._features(obs)
        return self.pi(h), self.vf(h).squeeze(-1)

    # -- rollout / update API ---------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, greedy: bool = False) -> dict:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        buttons = logits.argmax(-1) if greedy else dist.sample()
        return {"buttons": buttons, "logp": dist.log_prob(buttons),
                "value": value, "entropy": dist.entropy()}

    def evaluate_actions(self, obs: torch.Tensor, buttons: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(buttons), dist.entropy(), value

    # -- coupling adapter --------------------------------------------------
    def numpy_policy_fn(self, device=None):
        dev = device or next(self.parameters()).device

        @torch.no_grad()
        def _fn(obs_np):
            obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=dev)
            logits, _ = self.forward(obs)
            return logits.detach().cpu().numpy()

        return _fn


__all__ = ["ActorCritic", "N_BUTTONS"]
