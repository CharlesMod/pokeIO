"""Phase-1 learned-retina tests (docs/specs/active-vision-spine.md §11 test 8).

The retina is a decoder-free SSL encoder (SPR + inverse dynamics + FSQ). To make
"the SPR loss decreases" a *meaningful* claim rather than a trivial one, the
synthetic task has real, action-dependent structure: a bright square drifts on a
torus, and each step's displacement is fully determined by the button pressed.
So both objectives are genuinely learnable —

  * SPR: the next-frame latent is predictable from the current latent + action;
  * inverse dynamics: the button is recoverable from two consecutive latents.

Everything is kept small and fast (a few hundred synthetic frames, a couple
hundred steps, a ~0.5M-param net) — it runs in seconds on a P100 and still on
CPU. It trains once (module-scoped fixture) and shares the trained model across
the individual assertions.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pokeio.evo.retina import (
    Retina,
    build_dataset,
    build_fovea_stack,
)

# --- action -> (dy,dx) displacement on the torus (button id == index) ---
_MV = 16
_MOVES = {
    0: (-_MV, 0),   # up
    1: (_MV, 0),    # down
    2: (0, -_MV),   # left
    3: (0, _MV),    # right
    4: (_MV, _MV),  # A
    5: (-_MV, -_MV),  # B
    6: (_MV, -_MV),  # START
    7: (-_MV, _MV),  # SELECT
    8: (0, 0),      # NOOP
}

_H, _W = 144, 160
_SQ = 16  # square side


def _render(cy: int, cx: int) -> np.ndarray:
    """Bright square (wrapped on a torus) on a dark 144x160 screen."""
    img = np.zeros((_H, _W), dtype=np.uint8)
    ys = np.arange(cy - _SQ // 2, cy + _SQ // 2) % _H
    xs = np.arange(cx - _SQ // 2, cx + _SQ // 2) % _W
    img[np.ix_(ys, xs)] = 255
    return img


def _make_rollout(t: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """A length-``t`` rollout of frames + the per-transition button ids."""
    rng = np.random.default_rng(seed)
    cy, cx = _H // 2, _W // 2
    frames = [_render(cy, cx)]
    actions: list[int] = []
    for _ in range(t - 1):
        a = int(rng.integers(0, 9))
        dy, dx = _MOVES[a]
        cy = (cy + dy) % _H
        cx = (cx + dx) % _W
        frames.append(_render(cy, cx))
        actions.append(a)
    return frames, np.asarray(actions, dtype=np.int64)


def _device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _encode_held(trained) -> np.ndarray:
    """Encode the cached held-out periph+fovea stacks -> (N, z_dim) latent matrix."""
    return trained["model"].encode_np(trained["held_periph"], trained["held_fovea"])


# Fast-but-meaningful sizes; a touch more budget when a GPU is available.
_CUDA = torch.cuda.is_available()
SPR_K = 3
TRAIN_T = 400
HELD_T = 160
STEPS = 200 if _CUDA else 90
BATCH = 32 if _CUDA else 12


import pytest


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = _device()
    frames, actions = _make_rollout(TRAIN_T, seed=0)
    model = Retina(spr_k=SPR_K, aug_shift_px=4, aug_jitter=0.05, n_act=9).to(dev)
    hist = model.fit(frames, actions, steps=STEPS, batch_size=BATCH, lr=1e-3,
                     device=dev, log_every=1)
    held_frames, held_actions = _make_rollout(HELD_T, seed=1)
    # Precompute held-out stacks once (the CPU stack-building is the wall-time cost).
    held_periph = build_dataset(held_frames)
    held_fovea = np.stack(
        [build_fovea_stack(held_frames, t) for t in range(len(held_frames))], axis=0)
    return {
        "model": model,
        "hist": hist,
        "dev": dev,
        "held_frames": held_frames,
        "held_actions": held_actions,
        "held_periph": held_periph,
        "held_fovea": held_fovea,
    }


# --------------------------------------------------------------------------
# SPR: the self-predictive cosine loss must actually go down.
# --------------------------------------------------------------------------
def test_spr_cosine_loss_decreases(trained):
    spr = [h["spr"] for h in trained["hist"]]
    assert len(spr) >= 20
    n = max(3, len(spr) // 10)
    start = float(np.mean(spr[:n]))
    end = float(np.mean(spr[-n:]))
    # cosine loss is -cos in [-1, 0]; a real decrease (>0.05 gain in alignment).
    assert end < start - 0.05, f"SPR loss did not decrease: {start:.3f} -> {end:.3f}"
    # and it should be clearly negative (predictions aligned with targets).
    assert end < -0.1, f"SPR alignment too weak: end={end:.3f}"


# --------------------------------------------------------------------------
# Inverse dynamics: recover the button from consecutive latents, on HELD-OUT data.
# --------------------------------------------------------------------------
def test_inverse_dynamics_beats_chance(trained):
    model = trained["model"]
    actions = trained["held_actions"]
    model.eval()
    with torch.no_grad():
        z = torch.as_tensor(_encode_held(trained)).to(trained["dev"])
        logits = model.inverse_logits(z[:-1], z[1:])
        pred = logits.argmax(-1).cpu().numpy()
    acc = float((pred == actions).mean())
    # chance is 1/9 ≈ 0.111; a learnable task should clear it comfortably.
    assert acc > 1.0 / 9.0, f"inverse-dyn accuracy {acc:.3f} at/below chance"
    assert acc > 0.20, f"inverse-dyn accuracy {acc:.3f} not meaningfully above chance"


# --------------------------------------------------------------------------
# Latent must NOT collapse: per-dim variance survives on a held-out batch.
# --------------------------------------------------------------------------
def test_latent_does_not_collapse(trained):
    model = trained["model"]
    z = _encode_held(trained)  # (N, 80)
    std = z.std(axis=0)
    assert z.shape[1] == model.z_dim
    # most dims carry real variance (not a single constant point).
    frac_alive = float((std > 1e-3).mean())
    assert frac_alive > 0.5, f"only {frac_alive:.2f} of latent dims are alive"
    assert float(z.std()) > 1e-2, "overall latent variance collapsed"


# --------------------------------------------------------------------------
# FSQ cell code: no dead-code domination / entropy stays high over the batch.
# --------------------------------------------------------------------------
def test_fsq_code_entropy_high(trained):
    model = trained["model"]
    periph = trained["held_periph"]
    codes = model.fsq_code(periph)  # (N, 5) int
    assert codes.shape == (len(trained["held_frames"]), 5)
    assert codes.dtype == np.int64
    # levels are respected: each dim within [0, level-1].
    for d, lvl in enumerate(model.fsq_levels):
        assert codes[:, d].min() >= 0 and codes[:, d].max() <= lvl - 1

    keys = [tuple(int(v) for v in row) for row in codes]
    n = len(keys)
    counts = np.array([keys.count(k) for k in set(keys)], dtype=np.float64)
    freqs = counts / n
    # no single cell swallows the whole batch (the VQ dead-code failure mode)...
    assert freqs.max() < 0.9, f"one FSQ code dominates ({freqs.max():.2f})"
    # ...and the code distribution carries real entropy.
    entropy_bits = float(-(freqs * np.log2(freqs)).sum())
    assert len(set(keys)) >= 3, f"only {len(set(keys))} distinct FSQ codes"
    assert entropy_bits > 0.5, f"FSQ entropy too low ({entropy_bits:.2f} bits)"


# --------------------------------------------------------------------------
# encode_np contract: single -> (z_dim,), batch -> (B, z_dim), fp32.
# --------------------------------------------------------------------------
def test_encode_np_shapes(trained):
    model = trained["model"]
    periph = trained["held_periph"]
    fovea = trained["held_fovea"]

    z1 = model.encode_np(periph[0], fovea[0])
    assert z1.shape == (model.z_dim,) and z1.dtype == np.float32

    zb = model.encode_np(periph[:5], fovea[:5])
    assert zb.shape == (5, model.z_dim) and zb.dtype == np.float32


# --------------------------------------------------------------------------
# snapshot(): a frozen, no-grad, value-identical copy for the population.
# --------------------------------------------------------------------------
def test_snapshot_is_frozen(trained):
    model = trained["model"]
    snap = model.snapshot()

    # no parameter requires grad
    assert all(not p.requires_grad for p in snap.parameters())

    # parameters are value-identical to the source
    src = dict(model.named_parameters())
    for name, p in snap.named_parameters():
        assert torch.equal(p.detach().cpu(), src[name].detach().cpu()), f"param {name} diverged"

    # it is a distinct object (deep copy), not an alias
    assert snap is not model

    # snapshot inference produces the same latent as the (eval) source
    model.eval()
    periph = trained["held_periph"][:4]
    fovea = trained["held_fovea"][:4]
    z_src = model.encode_np(periph, fovea)
    z_snap = snap.encode_np(periph, fovea)
    assert np.allclose(z_src, z_snap, atol=1e-5)
