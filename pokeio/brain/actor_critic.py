"""Basal ganglia — the gradient actor-critic (#32), the ONE load-bearing region.

The RED-TEAM REVISION's single load-bearing mapping: replace the direct-encoded
NEAT genome (which after 90 gens was ~random-init, its fitness harvested from
Go-Explore restores) with a gradient-trained actor-critic on the EXISTING foveal
percept. Small and Pascal-friendly (sm_60: float32, conv + MLP, NO attention).

Percept -> gradient (occipital, #33). A foveal obs is
``periph(G^2) | fovea(FG^2) | motion(G^2) | proprio | ram`` (see the fleet
contract / FovealEncoder). For the default config (G=FG=12, memory off) the first
three blocks are 12x12 sheets, so the trunk runs a tiny 2-layer CONV over them as
a 3-channel image — this is what forces motion + fovea to structurally enter the
gradient (the audit found the old policies were screen-blind; a conv over the
motion sheet cannot ignore it). The proprio+ram tail is concatenated after the
conv. If the spatial blocks are not equal-sized (a sharp fovea FG!=G, or memory
blocks present) the trunk falls back to a plain MLP over the flat obs.

Action space (v1). The RL action is the **9-way button** (Discrete, ACTIONS
order) — the gate (reach party>0 from boot) rides on button competence. Gaze is
held centered / reflex-driven for v1; LEARNED active-vision gaze is the explicit
post-gate stage (#11), added back once the reflex core learns. So the actor emits
button logits + a state value; no continuous gaze head in the RL distribution.
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
    """obs -> (button logits, value). Shared trunk (conv-over-sheets or MLP).

    ``obs_dim`` is the flat foveal obs width (= ``FovealEncoder.dim`` =
    ``fleet.obs_dim``). ``grid`` is the periph/fovea/motion sheet side (G); when
    the first ``3*grid*grid`` obs entries are the three equal sheets, the conv
    trunk is used, else a flat MLP. ``hidden`` sizes the MLP trunk/heads.
    """

    def __init__(self, obs_dim: int, grid: int = 12, hidden: int = 256,
                 n_buttons: int = N_BUTTONS) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.grid = int(grid)
        self.n_buttons = int(n_buttons)
        n_spatial = 3 * self.grid * self.grid
        # conv trunk only when the three equal sheets fit at the front of the obs
        self.use_conv = n_spatial <= self.obs_dim and self.grid > 0
        self.n_spatial = n_spatial if self.use_conv else 0
        self.n_extra = self.obs_dim - self.n_spatial

        if self.use_conv:
            self.conv = nn.Sequential(
                _orthogonal(nn.Conv2d(3, 16, 3, padding=1), np.sqrt(2)), nn.ReLU(),
                _orthogonal(nn.Conv2d(16, 32, 3, padding=1), np.sqrt(2)), nn.ReLU(),
            )
            trunk_in = 32 * self.grid * self.grid + self.n_extra
        else:
            self.conv = None
            trunk_in = self.obs_dim

        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(trunk_in, hidden), np.sqrt(2)), nn.ReLU(),
            _orthogonal(nn.Linear(hidden, hidden), np.sqrt(2)), nn.ReLU(),
        )
        # small policy-head gain => near-uniform initial actions (avoids the
        # constant-action collapse the audit found at init).
        self.pi = _orthogonal(nn.Linear(hidden, self.n_buttons), 0.01)
        self.vf = _orthogonal(nn.Linear(hidden, 1), 1.0)

    # -- forward -----------------------------------------------------------
    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            g = self.grid
            sheets = obs[:, : self.n_spatial].reshape(-1, 3, g, g)
            extra = obs[:, self.n_spatial:]
            h = self.conv(sheets).flatten(1)
            h = torch.cat([h, extra], dim=1) if extra.shape[1] else h
            return self.trunk(h)
        return self.trunk(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(button_logits (B, n_buttons), value (B,))``."""
        h = self._features(obs)
        return self.pi(h), self.vf(h).squeeze(-1)

    # -- rollout / update API ---------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, greedy: bool = False) -> dict:
        """Sample an action for rollout. Returns buttons/logp/value/entropy tensors."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        buttons = logits.argmax(-1) if greedy else dist.sample()
        return {
            "buttons": buttons,
            "logp": dist.log_prob(buttons),
            "value": value,
            "entropy": dist.entropy(),
        }

    def evaluate_actions(self, obs: torch.Tensor, buttons: torch.Tensor):
        """PPO update: log-prob + entropy of ``buttons`` under current policy + value."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(buttons), dist.entropy(), value

    # -- coupling adapter --------------------------------------------------
    def numpy_policy_fn(self, device=None):
        """Return a ``obs_batch (np) -> logits (np)`` callable for
        :mod:`pokeio.brain.coupling` (blind_delta / ram_ablation_delta probes)."""
        dev = device or next(self.parameters()).device

        @torch.no_grad()
        def _fn(obs_np):
            obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=dev)
            logits, _ = self.forward(obs)
            return logits.detach().cpu().numpy()

        return _fn


__all__ = ["ActorCritic", "N_BUTTONS"]
