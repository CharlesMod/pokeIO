"""Recurrent actor-critic built from an ObsSpec.

    pixels (screen [+ visited mask], bit-packed) -> unpack on device -> CNN
    bits (packed binary features)               -> unpack -> Linear
    scalars                                      -> as-is
    cats                                         -> embeddings (shared per spec feature)
    concat -> Linear(hidden) [+LayerNorm] -> LSTMCell(hidden) -> actor / critic

The critic predicts a *normalized* value (see ValueNorm in train.py).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pokeio.config import PolicyConfig
from pokeio.env import ObsSpec


def _init(m: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)
    return m


class Policy(nn.Module):
    def __init__(self, obs_spec: ObsSpec, cfg: PolicyConfig):
        super().__init__()
        self.obs_spec = obs_spec
        c, h, _pw = obs_spec.pixels
        self.pixel_shape = (c, h, obs_spec.pixel_width)
        bpp = obs_spec.pixel_bpp
        self.bpp = bpp
        if bpp < 8:
            per = 8 // bpp
            self.register_buffer(
                "pix_shift", torch.arange(per - 1, -1, -1, dtype=torch.uint8) * bpp, persistent=False
            )
        self.pix_mask = (1 << bpp) - 1 if bpp < 8 else 255
        self.pix_scale = 1.0 / max(obs_spec.levels - 1, 1)
        self.register_buffer("bit_shift", torch.arange(7, -1, -1, dtype=torch.uint8), persistent=False)

        layers: list[nn.Module] = []
        in_c = c
        for out_c, k, s in cfg.conv:
            layers += [_init(nn.Conv2d(in_c, out_c, k, s)), nn.ReLU()]
            in_c = out_c
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_cnn = self.cnn(torch.zeros(1, *self.pixel_shape)).shape[1]

        self.n_bits = obs_spec.bits * 8
        self.bits_net = (
            nn.Sequential(_init(nn.Linear(self.n_bits, cfg.bits_hidden)), nn.ReLU())
            if self.n_bits
            else None
        )
        self.n_scalars = obs_spec.scalars

        # one embedding table per spec feature group
        groups: dict[int, tuple[int, int]] = {}
        self.cat_groups: list[int] = []
        for num, emb, g in obs_spec.cats:
            groups[g] = (num, emb)
            self.cat_groups.append(g)
        self.embeds = nn.ModuleDict({str(g): nn.Embedding(n, e) for g, (n, e) in groups.items()})
        n_cats = sum(groups[g][1] for g in self.cat_groups)

        n_in = n_cnn + (cfg.bits_hidden if self.bits_net else 0) + self.n_scalars + n_cats
        enc: list[nn.Module] = [_init(nn.Linear(n_in, cfg.hidden))]
        if cfg.layer_norm:
            enc.append(nn.LayerNorm(cfg.hidden))
        enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        self.lstm = nn.LSTMCell(cfg.hidden, cfg.hidden)
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            else:
                nn.init.zeros_(p)
        self.actor = _init(nn.Linear(cfg.hidden, obs_spec.n_actions), 0.01)
        self.critic = _init(nn.Linear(cfg.hidden, 1), 1.0)
        self.hidden = cfg.hidden

    # ------------------------------------------------------------------ obs
    def unpack_pixels(self, packed: torch.Tensor) -> torch.Tensor:
        if self.bpp == 8:
            x = packed
        else:
            x = (packed.unsqueeze(-1) >> self.pix_shift) & self.pix_mask
            x = x.reshape(packed.shape[0], *self.pixel_shape)
        return x.float() * self.pix_scale

    def encode(self, obs: dict[str, torch.Tensor], pixels: torch.Tensor | None = None) -> torch.Tensor:
        """``pixels`` (float, unpacked) overrides obs["pixels"]: used for saliency."""
        if pixels is None:
            pixels = self.unpack_pixels(obs["pixels"])
        feats = [self.cnn(pixels)]
        if self.bits_net is not None:
            b = obs["bits"][:, : self.obs_spec.bits]
            bits = ((b.unsqueeze(-1) >> self.bit_shift) & 1).reshape(b.shape[0], -1).float()
            feats.append(self.bits_net(bits))
        if self.n_scalars:
            feats.append(obs["scalars"][:, : self.n_scalars].float())
        if self.cat_groups:
            cats = obs["cats"].long()
            feats += [self.embeds[str(g)](cats[:, i]) for i, g in enumerate(self.cat_groups)]
        return self.encoder(torch.cat(feats, dim=1))

    # ------------------------------------------------------------------ recurrent core
    def initial_state(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(n, self.hidden, device=device)
        return z, z.clone()

    def step(self, obs, state, starts: torch.Tensor):
        """One step for a batch. ``starts`` (float, 1 = new episode) zeroes the state."""
        x = self.encode(obs)
        keep = (1.0 - starts).unsqueeze(1)
        h, c = self.lstm(x, (state[0] * keep, state[1] * keep))
        return self.actor(h), self.critic(h).squeeze(1), (h, c)

    def saliency(self, obs, state, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """|d logit(argmax) / d pixel| for a batch: where the policy's decision is
        sensitive to the screen. Returns (saliency [B, C, H, W], probs [B, A])."""
        pix = self.unpack_pixels(obs["pixels"]).detach().requires_grad_(True)
        x = self.encode(obs, pixels=pix)
        keep = (1.0 - starts).unsqueeze(1)
        h, _ = self.lstm(x, (state[0] * keep, state[1] * keep))
        logits = self.actor(h)
        top = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
        (grad,) = torch.autograd.grad(top, pix)
        return grad.abs(), torch.softmax(logits.detach(), -1)

    def sequence(self, obs, state, starts: torch.Tensor):
        """Unroll over time. ``obs`` tensors are [T, B, ...]; returns [T, B] outputs."""
        T, B = starts.shape
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
        x = self.encode(flat).reshape(T, B, -1)
        h, c = state
        outs = []
        for t in range(T):
            keep = (1.0 - starts[t]).unsqueeze(1)
            h, c = self.lstm(x[t], (h * keep, c * keep))
            outs.append(h)
        hs = torch.stack(outs)
        return self.actor(hs), self.critic(hs).squeeze(-1)
