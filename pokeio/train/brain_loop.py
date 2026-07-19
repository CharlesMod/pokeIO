"""PokeIO v2.0 System-1 training entrypoint (parallels train/loop.py).

Constructs the furnace (BarrierFleet, expose_wram + goexplore), the gradient
actor-critic, and the backward-robustification curriculum, then runs the RL loop
toward the gate (#35): the policy reaches party>0 from a cold boot.

Additive: this is a NEW entrypoint alongside the NEAT loop; it touches none of the
neuroevolution path. Run:

    PYTHONPATH=. .venv/bin/python -m pokeio.train.brain_loop \
        --n-envs 64 --iterations 2000 --device cuda:1 --run-id brain1

Writes ``runs/<run_id>/brain_metrics.jsonl`` (one record per iteration) and
checkpoints ``runs/<run_id>/brain.pt`` (policy + optimizer + curriculum) every
``--checkpoint-every`` iterations. Card 0 is the LLM (deferred); training defaults
to ``cuda:1`` (Pascal-honesty: ~1 P100 of training compute).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pokeio.brain.replay import DemoTrajectory
from pokeio.brain.rl_loop import BrainConfig, BrainTrainer
from pokeio.emu.fleet import BarrierFleet, FovealEncoder

_ROM = "roms/pokemon_yellow.gb"
_STATE = "roms/yellow_newgame.state"


def build_fleet(n_envs: int, *, rom=_ROM, state=_STATE, obs_ram=8,
                frame_skip=24, hold_frames=8, periph_grid=12,
                fovea_native_px=48, fovea_grid=0) -> BarrierFleet:
    """A BarrierFleet sized to a matching FovealEncoder, with the reward WRAM
    channel + Go-Explore restore both ON (backward-robustification needs restore)."""
    enc = FovealEncoder(n_envs, periph_grid=periph_grid,
                        fovea_native_px=fovea_native_px, fovea_grid=fovea_grid,
                        n_ram=obs_ram)
    fleet = BarrierFleet(
        n_envs, enc.dim, periph_grid, obs_ram, rom, frame_skip, hold_frames, state, {},
        wram_stride=64, goexplore=True, expose_wram=True,
        periph_grid=periph_grid, fovea_native_px=fovea_native_px, fovea_grid=fovea_grid,
    )
    return fleet


def _checkpoint(path: Path, tr: BrainTrainer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": tr.policy.state_dict(),
        "opt": tr.opt.state_dict(),
        "iter": tr.iter,
        "curriculum": tr.curriculum.state(),
        "frontier": tr.curriculum.frontier,
        "success_ema": tr.curriculum.success_ema,
    }, str(path))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="PokeIO v2.0 System-1 RL trainer")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--run-id", default="brain1")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=BrainConfig.lr)
    ap.add_argument("--gamma", type=float, default=BrainConfig.gamma)
    ap.add_argument("--horizon-max", type=int, default=BrainConfig.horizon_max)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path("runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "brain_metrics.jsonl"
    ckpt_path = run_dir / "brain.pt"

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[brain] CUDA unavailable -> falling back to cpu")
        device = "cpu"

    print(f"[brain] building fleet: n_envs={args.n_envs}")
    fleet = build_fleet(args.n_envs)
    cfg = BrainConfig(n_envs=args.n_envs, lr=args.lr, gamma=args.gamma,
                      horizon_max=args.horizon_max, eval_every=args.eval_every,
                      seed=args.seed)
    tr = BrainTrainer(fleet, device=device, demo=DemoTrajectory(), cfg=cfg)
    # Start the backward-robustification frontier AT the demo's milestone depth
    # (game-agnostic: found by replaying the demo), so the run begins where real
    # learning is — not wasting iterations restoring past an already-won milestone.
    md = tr.demo.milestone_depth()
    tr.curriculum.frontier = float(min(len(tr.demo), md))
    print(f"[brain] trainer ready: obs_dim={tr.obs_dim} device={device} "
          f"demo_len={len(tr.demo)} milestone_depth={md} "
          f"frontier={tr.curriculum.frontier:.0f}")

    t0 = time.monotonic()

    def _log(rec: dict) -> None:
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        parts = [f"it={rec['iter']}", f"H={rec['H']}", f"reach={rec['reached']:.2f}",
                 f"front={rec['frontier']:.0f}", f"ema={rec['success_ema']:.2f}",
                 f"rew={rec['rew_sum_mean']:.2f}", f"ent={rec['entropy']:.2f}"]
        if "eval" in rec:
            e = rec["eval"]
            parts.append(f"| GATE stoch={e['gate_reached_frac']:.2f} "
                         f"greedy={e.get('gate_reached_greedy', 0.0):.2f} "
                         f"prog={e['progress_mean']:.1f} blind={e['blind_delta']:.3f}")
        print("[brain] " + " ".join(parts), flush=True)
        if rec["iter"] % args.checkpoint_every == 0:
            _checkpoint(ckpt_path, tr)

    try:
        tr.train(args.iterations, on_log=_log)
        _checkpoint(ckpt_path, tr)
        _export_champion(run_dir, tr, args)
    finally:
        fleet.close()
    dt = time.monotonic() - t0
    print(f"[brain] done: {args.iterations} iters in {dt:.0f}s -> {ckpt_path}")


def _export_champion(run_dir: Path, tr, args) -> None:
    """Export the trained champion as separate brain (phenotype) + genome (genotype)
    files for replay / breeding / cross-game seeding (pokeio.brain.champion)."""
    from pokeio.brain.champion import export_champion

    best = max((r["eval"]["gate_reached_frac"] for r in tr.history if "eval" in r),
               default=0.0)
    cdir = run_dir / "champions"
    cdir.mkdir(parents=True, exist_ok=True)
    obs_spec = {"encoder": "FovealEncoder", "dim": tr.obs_dim, "grid": 12,
                "periph_grid": 12, "fovea_native_px": 48, "fovea_grid": 0, "n_ram": 8}
    meta = {"game": "Pokemon Yellow", "console": "gb", "backend": "pyboy",
            "run_id": args.run_id, "iter": tr.iter, "gate_stochastic": best,
            "id": f"{args.run_id}:{tr.iter}"}
    export_champion(tr.policy, str(cdir / "champion_brain.pt"),
                    str(cdir / "champion_genome.npz"), obs_spec=obs_spec, meta=meta)
    print(f"[brain] champion exported -> {cdir}/ (brain + genome; gate={best:.2f})")


if __name__ == "__main__":
    main()
