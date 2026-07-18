"""Phase-1 retina-in-loop integration (docs/specs/retina-in-loop.md).

Covers the learned SPR/FSQ retina wired into ``train/loop.py`` as the perceptual
spine — everything gated on ``config.vision.mode == "retina"``; the Phase-0
foveal path (n_in=454) is untouched.

  * ``RetinaObsPipe`` maps a batch of raw 14134-d worker obs to the 102-d
    controller latent, with the correct block slots
    ``[z_periph|z_fovea|proprio|ram]`` and per-env 4-frame ring stacking;
  * the rings RESET on an episode/restore boundary (no frame bleed across
    ``reset``);
  * a tiny end-to-end retina-mode run (``--engine barrier``, pop 8, players 4,
    warmup tiny, gens 2, swap_gens 1) does not crash: the warm-up trains (SPR
    loss logged), the genome n_in is 102, the frozen snapshot swaps, and
    telemetry is written;
  * foveal mode still builds n_in=454 and runs unchanged.

The end-to-end tests spawn a real BarrierFleet; they need the Yellow ROM and
skip cleanly if it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pokeio.config import Config
from pokeio.evo.retina import IN_SIDE, Z_DIM, Retina
from pokeio.train.loop import RetinaObsPipe, train

# Retina worker-obs geometry (config defaults: n_ram=8).
SIDE = IN_SIDE                          # 84
N_PIX = SIDE * SIDE                     # 7056 per stream
N_RAM = 8
RETINA_DIM = 2 * N_PIX + 14 + N_RAM     # 14134
OFFSETS = (0, N_PIX, 2 * N_PIX, 2 * N_PIX + 14, RETINA_DIM)  # periph/fovea/proprio/ram/dim

ROM = "roms/pokemon_yellow.gb"
STATE = "roms/yellow_newgame.state"
_HAVE_ROM = Path(ROM).exists() and Path(STATE).exists()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _frame(rng) -> np.ndarray:
    """A random 84x84 grayscale frame in [0,1] (one periph/fovea stream)."""
    return rng.random((SIDE, SIDE), dtype=np.float32)


def _raw_row(periph84, fovea84, proprio, ram) -> np.ndarray:
    """Assemble one 14134-d worker obs from its four blocks."""
    v = np.empty(RETINA_DIM, np.float32)
    v[0:N_PIX] = periph84.ravel()
    v[N_PIX:2 * N_PIX] = fovea84.ravel()
    v[2 * N_PIX:2 * N_PIX + 14] = proprio
    v[2 * N_PIX + 14:RETINA_DIM] = ram
    return v


def _small_retina() -> Retina:
    """A retina at the committed widths, CPU, deterministic weights."""
    torch.manual_seed(0)
    return Retina(z_periph=48, z_fovea=32, spr_k=5, ema_tau=0.0,
                  inverse_dynamics=True, fsq_levels=(8, 8, 8, 5, 5), n_act=9)


# --------------------------------------------------------------------------
# 1. obs transform: block slots + ring stacking
# --------------------------------------------------------------------------
def test_pipe_maps_14134_to_102_with_correct_blocks():
    rng = np.random.default_rng(1)
    snap = _small_retina().snapshot(device=torch.device("cpu"))
    pipe = RetinaObsPipe(snap, n_envs=3, offsets=OFFSETS, side=SIDE, stack=4)

    periph = np.stack([_frame(rng) for _ in range(3)])
    fovea = np.stack([_frame(rng) for _ in range(3)])
    proprio = rng.uniform(-1, 1, (3, 14)).astype(np.float32)
    ram = rng.random((3, N_RAM), dtype=np.float32)
    raw = np.stack([_raw_row(periph[i], fovea[i], proprio[i], ram[i]) for i in range(3)])

    out = pipe.transform(raw)
    assert out.shape == (3, Z_DIM + 14 + N_RAM) == (3, 102)
    assert out.dtype == np.float32

    # proprio + ram blocks pass through byte-exactly ([80:94] and [94:102]).
    np.testing.assert_array_equal(out[:, Z_DIM:Z_DIM + 14], proprio)
    np.testing.assert_array_equal(out[:, Z_DIM + 14:], ram)

    # z block = the frozen snapshot's encode of the (first-frame) 4-copy stacks.
    ps = np.stack([np.stack([periph[i]] * 4) for i in range(3)])
    fs = np.stack([np.stack([fovea[i]] * 4) for i in range(3)])
    z_expected = snap.encode_np(ps, fs)
    np.testing.assert_allclose(out[:, :Z_DIM], z_expected, rtol=1e-5, atol=1e-5)


def test_pipe_ring_stacks_across_steps():
    """Second frame pushes into the ring: stack is [f0,f0,f0,f1], not [f1]*4."""
    rng = np.random.default_rng(2)
    snap = _small_retina().snapshot(device=torch.device("cpu"))
    pipe = RetinaObsPipe(snap, n_envs=1, offsets=OFFSETS, side=SIDE, stack=4)

    p0, f0 = _frame(rng), _frame(rng)
    p1, f1 = _frame(rng), _frame(rng)
    z = np.zeros(14, np.float32)
    r = np.zeros(N_RAM, np.float32)

    pipe.transform(_raw_row(p0, f0, z, r)[None])           # fills ring with 4x frame0
    out1 = pipe.transform(_raw_row(p1, f1, z, r)[None])     # pushes frame1

    ps = np.stack([p0, p0, p0, p1])[None]
    fs = np.stack([f0, f0, f0, f1])[None]
    z_expected = snap.encode_np(ps, fs)
    np.testing.assert_allclose(out1[:, :Z_DIM], z_expected, rtol=1e-5, atol=1e-5)

    # ...and it must NOT equal the naive [f1]*4 stacking (rings actually carry).
    z_naive = snap.encode_np(np.stack([p1] * 4)[None], np.stack([f1] * 4)[None])
    assert not np.allclose(out1[:, :Z_DIM], z_naive, rtol=1e-4, atol=1e-4)


def test_pipe_rings_reset_on_restore():
    """reset() clears the ring: the next frame refills 4 copies (no bleed)."""
    rng = np.random.default_rng(3)
    snap = _small_retina().snapshot(device=torch.device("cpu"))
    pipe = RetinaObsPipe(snap, n_envs=1, offsets=OFFSETS, side=SIDE, stack=4)
    z = np.zeros(14, np.float32)
    r = np.zeros(N_RAM, np.float32)

    p0, f0 = _frame(rng), _frame(rng)
    pipe.transform(_raw_row(p0, f0, z, r)[None])     # ring = [f0]*4
    pipe.reset(0)                                    # episode/restore boundary

    pa, fa = _frame(rng), _frame(rng)
    out = pipe.transform(_raw_row(pa, fa, z, r)[None])
    z_fresh = snap.encode_np(np.stack([pa] * 4)[None], np.stack([fa] * 4)[None])
    np.testing.assert_allclose(out[:, :Z_DIM], z_fresh, rtol=1e-5, atol=1e-5)

    z_bleed = snap.encode_np(np.stack([p0, p0, p0, pa])[None],
                             np.stack([f0, f0, f0, fa])[None])
    assert not np.allclose(out[:, :Z_DIM], z_bleed, rtol=1e-4, atol=1e-4)


def test_snapshot_swap_changes_latent():
    """set_snapshot swaps the frozen encoder; a distinct snapshot -> distinct z."""
    rng = np.random.default_rng(4)
    snap_a = _small_retina().snapshot(device=torch.device("cpu"))
    torch.manual_seed(999)
    snap_b = Retina(z_periph=48, z_fovea=32).snapshot(device=torch.device("cpu"))
    pipe = RetinaObsPipe(snap_a, n_envs=1, offsets=OFFSETS, side=SIDE, stack=4)

    p, f = _frame(rng), _frame(rng)
    z0 = np.zeros(14, np.float32)
    r0 = np.zeros(N_RAM, np.float32)
    out_a = pipe.transform(_raw_row(p, f, z0, r0)[None]).copy()
    pipe.reset(0)
    pipe.set_snapshot(snap_b)
    out_b = pipe.transform(_raw_row(p, f, z0, r0)[None]).copy()
    assert not np.allclose(out_a[:, :Z_DIM], out_b[:, :Z_DIM], atol=1e-4)


# --------------------------------------------------------------------------
# 2. end-to-end: retina-mode barrier run (no crash, warm-up, swap, telemetry)
# --------------------------------------------------------------------------
def _tiny_config(run_id: str, runs_dir: Path, mode: str) -> Config:
    cfg = Config()
    cfg.run.run_id = run_id
    cfg.run.seed = 0
    cfg.run.runs_dir = str(runs_dir)
    cfg.evo.pop_size = 8
    cfg.emu.n_players = 4
    cfg.emu.frame_skip = 24
    cfg.emu.button_hold_frames = 8
    cfg.vision.mode = mode
    if mode == "retina":
        cfg.retina.enable = True
        cfg.retina.warmup_frames = 200   # tiny warm-up for the smoke
        cfg.retina.swap_gens = 1         # swap every gen so the test sees it
    return cfg


@pytest.mark.skipif(not _HAVE_ROM, reason="Yellow ROM / reset-state not available")
def test_retina_end_to_end_barrier(tmp_path, capsys):
    cfg = _tiny_config("p1_test", tmp_path, mode="retina")
    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    run_dir = train(
        gens=2,
        pop_size=cfg.evo.pop_size,
        players=cfg.emu.n_players,
        episode_steps=16,
        obs_res=24,
        run_id=cfg.run.run_id,
        config=cfg,
        device_str=dev,
        live=False,
        novelty_mode="rarity",
        goexplore=True,
        restore_prob=0.5,
        parallel=True,
        engine="barrier",
        init_connect="full",
        boot_gauntlet_every=0,
        recurrent_memory=True,
        checkpoint_every=0,
        retina_train_steps=4,
    )
    out = capsys.readouterr().out

    # controller n_in is the 102-d latent, N_OUT=11 (not the 14134 fleet obs).
    assert "n_in=102 N_OUT=11" in out
    assert "fleet obs 14134" in out

    # warm-up trained the retina and logged SPR loss.
    assert "[retina/warmup]" in out
    assert "warm-up done" in out
    assert "snapshot #0 frozen" in out

    # SPR cosine loss dropped across the warm-up (never L2; cosine loss < 0).
    line = next(ln for ln in out.splitlines() if "warm-up done" in ln)
    spr0 = float(line.split("SPR ")[1].split(" -> ")[0])
    spr1 = float(line.split(" -> ")[1].split(",")[0])
    assert spr1 <= spr0 + 1e-3, f"SPR loss did not drop: {spr0} -> {spr1}"

    # the frozen snapshot swapped at least once (swap_gens=1).
    assert "snapshot swapped" in out

    # telemetry written with real GenerationRecords.
    tel = Path(run_dir) / "telemetry.jsonl"
    assert tel.exists()
    gens = [json.loads(ln) for ln in tel.read_text().splitlines() if ln.strip()]
    assert any(r.get("type") == "generation" for r in gens)

    # resolved-run.json records the retina knobs for reproducibility.
    resolved = json.loads((Path(run_dir) / "resolved-run.json").read_text())
    assert resolved["config"]["vision"]["mode"] == "retina"
    assert resolved["config"]["retina"]["warmup_frames"] == 200


@pytest.mark.skipif(not _HAVE_ROM, reason="Yellow ROM / reset-state not available")
def test_retina_pipe_env_id_routing():
    """Under FURNACE async batching the parent gets an arbitrary, out-of-order
    subset of ready envs per cycle. ``env_ids`` must route each row into its OWN
    ring so every env's temporal 4-stack is its own last-4 frames in order,
    independent of cross-env interleaving. (No ROM/GPU needed — pure ring logic.)"""
    from collections import deque

    pipe = RetinaObsPipe(None, 3, OFFSETS, side=SIDE, stack=4)

    def _row(env, t):
        v = np.float32((env * 10 + t) / 100.0)  # unique per (env,t), in [0,1]
        f = np.full((SIDE, SIDE), v, np.float32)
        return _raw_row(f, f, np.zeros(14, np.float32), np.zeros(N_RAM, np.float32)), v

    # interleaved async batches of (env, frame_t) — out of order, partial subsets
    batches = [[(0, 0), (2, 0)], [(1, 0)], [(0, 1), (1, 1), (2, 1)], [(2, 2), (0, 2)]]
    exp = {e: deque(maxlen=4) for e in range(3)}
    for batch in batches:
        rows, ids, vals = [], [], []
        for (e, t) in batch:
            r, v = _row(e, t)
            rows.append(r); ids.append(e); vals.append((e, v))
        ps, fs, _, _ = pipe._stacks(np.stack(rows), env_ids=np.array(ids, dtype=int))
        for (e, v) in vals:  # mirror the pipe's clamp-to-frame-0 fill on first frame
            if not exp[e]:
                for _ in range(4):
                    exp[e].append(v)
            else:
                exp[e].append(v)
        for i, (e, _t) in enumerate(batch):
            got = ps[i][:, 0, 0]  # (4,) per-frame scalar of env e's stack
            assert np.allclose(got, list(exp[e])), (
                f"env {e}: stack {got} != own last-4 {list(exp[e])}")
            assert np.allclose(fs[i][:, 0, 0], list(exp[e]))  # fovea ring too


@pytest.mark.skipif(not _HAVE_ROM, reason="Yellow ROM / reset-state not available")
def test_retina_furnace_engine_runs(tmp_path):
    """retina + FURNACE now runs end-to-end (the async port): no stall/crash, the
    warm-up trains, evolution runs on the 102-d latent, telemetry is written."""
    cfg = _tiny_config("pf_run", tmp_path, mode="retina")
    run_dir = train(
        gens=1, pop_size=8, players=4, episode_steps=16, obs_res=24,
        run_id=cfg.run.run_id, config=cfg,
        device_str="cuda:1" if torch.cuda.is_available() else "cpu",
        live=False, parallel=True, engine="furnace",
        goexplore=True, boot_gauntlet_every=0, checkpoint_every=0,
    )
    assert (Path(run_dir) / "telemetry.jsonl").exists()


# --------------------------------------------------------------------------
# 3. foveal (Phase-0) path unchanged
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_ROM, reason="Yellow ROM / reset-state not available")
def test_foveal_mode_unchanged(tmp_path, capsys):
    cfg = _tiny_config("p0_test", tmp_path, mode="foveal")
    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    run_dir = train(
        gens=2, pop_size=8, players=4, episode_steps=16, obs_res=24,
        run_id=cfg.run.run_id, config=cfg, device_str=dev,
        live=False, goexplore=True, parallel=True, engine="barrier",
        init_connect="full", boot_gauntlet_every=0,
        recurrent_memory=True, checkpoint_every=0,
    )
    out = capsys.readouterr().out
    # Foveal controller keeps the 454-d obs; no retina spine is constructed.
    assert "n_in=454 N_OUT=11" in out
    assert "[retina/warmup]" not in out
    assert "Phase-1 spine" not in out
    tel = Path(run_dir) / "telemetry.jsonl"
    assert tel.exists()
    gens = [json.loads(ln) for ln in tel.read_text().splitlines() if ln.strip()]
    assert any(r.get("type") == "generation" for r in gens)
