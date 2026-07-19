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


CANVAS_GRID = 96  # M: trans-saccadic memory buffer side (the persisted-vision canvas)
GB_FPS = 59.7275  # Game Boy DMG refresh — the domain's own time base for cadence


def build_fleet(n_envs: int, *, rom=_ROM, state=_STATE, obs_ram=8,
                frame_skip=24, hold_frames=8, periph_grid=12,
                fovea_native_px=48, fovea_grid=0, reflex_gaze=True,
                foveal_memory=True, mem_grid=CANVAS_GRID,
                saccade_substeps: int | None = None) -> BarrierFleet:
    """A BarrierFleet sized to a matching FovealEncoder, with the reward WRAM
    channel + Go-Explore restore both ON (backward-robustification needs restore).

    BIOMIMETIC PERSISTED-VISION CANVAS (foveal_memory): the encoder maintains an MxM
    buffer that is blurry (low-res periphery) everywhere EXCEPT where the reflex
    saccade recently stamped the sharp native fovea; those clear patches age/decay and
    are invalidated by peripheral change (motion/contrast) — self-calibrated. The
    reflex gaze fills it in over saccades. The AI acts on this CANVAS alone (see
    ActorCritic canvas mode), so the raw fovea block is redundant (fovea_grid=0)."""
    enc = FovealEncoder(n_envs, periph_grid=periph_grid,
                        fovea_native_px=fovea_native_px, fovea_grid=fovea_grid,
                        n_ram=obs_ram, reflex_gaze=reflex_gaze,
                        foveal_memory=foveal_memory, mem_grid=mem_grid)
    # Saccade cadence: DERIVE the intra-action sub-steps from (GB fps, frame_skip,
    # actuator ceiling) so the fovea saccades at >= human rate in game-time, capped
    # at what a real gimbal can settle (no hand-tuned frame count). Override for A/B.
    from pokeio.emu.saccade_cadence import derive_cadence
    cad = derive_cadence(GB_FPS, frame_skip)
    sub_steps = cad.sub_steps if saccade_substeps is None else max(1, int(saccade_substeps))
    print(f"[brain] saccade cadence: action={cad.action_hz:.2f}Hz  gaze={sub_steps * cad.action_hz:.2f}Hz "
          f"(S={sub_steps}, >=human={sub_steps * cad.action_hz >= 4.0 - 1e-9}, "
          f"actuator<= {cad.actuator_ceiling_hz:.0f}Hz)")
    fleet = BarrierFleet(
        n_envs, enc.dim, periph_grid, obs_ram, rom, frame_skip, hold_frames, state, {},
        wram_stride=64, goexplore=True, expose_wram=True,
        periph_grid=periph_grid, fovea_native_px=fovea_native_px, fovea_grid=fovea_grid,
        reflex_gaze=reflex_gaze, foveal_memory=foveal_memory, mem_grid=mem_grid,
        sub_steps=sub_steps,
    )
    return fleet


def _trim_metrics(path: Path, keep_through_iter: int) -> None:
    """Drop metric records with ``iter > keep_through_iter`` — a crash/power loss can
    log a few iterations PAST the last saved checkpoint; on resume those would duplicate
    the re-run iters, so trim the jsonl back to the checkpoint for a clean, continuous log."""
    if not path.exists():
        return
    kept = []
    with open(path) as fh:
        for line in fh:
            try:
                if int(json.loads(line).get("iter", 0)) <= keep_through_iter:
                    kept.append(line)
            except Exception:
                continue
    with open(path, "w") as fh:
        fh.writelines(kept)


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
    ap.add_argument("--dashboard", action="store_true",
                    help="stream a live GUI feed to runs/<id>/live.json (serve with "
                         "python -m pokeio.dash.serve --port 8600, view /brain?run=<id>)")
    ap.add_argument("--warmstart", default=None,
                    help="champion brain.pt to load (skip re-learning the first milestone); "
                         "starts the curriculum AT boot to explore deeper into the game")
    ap.add_argument("--resume", default=None,
                    help="brain.pt checkpoint to CONTINUE (restores policy+optimizer+curriculum"
                         "+iter and trains to --iterations); use after a crash/power loss to pick "
                         "up where the run left off. Distinct from --warmstart (which resets the "
                         "frontier to boot for deeper exploration).")
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
    tr = BrainTrainer(fleet, device=device, mem_grid=CANVAS_GRID,
                      demo=DemoTrajectory(), cfg=cfg)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        tr.policy.load_state_dict(ckpt["policy"])
        try:
            tr.opt.load_state_dict(ckpt["opt"])
        except Exception as e:                          # optimizer shape drift -> fresh moments
            print(f"[brain] resume: optimizer state skipped ({e})")
        tr.iter = int(ckpt.get("iter", 0))
        tr.curriculum.frontier = float(ckpt.get("frontier", tr.curriculum.frontier))
        tr.curriculum.success_ema = float(ckpt.get("success_ema", 0.0))
        cst = ckpt.get("curriculum") or {}
        if isinstance(cst, dict) and "seen" in cst:
            tr.curriculum._seen = int(cst["seen"])
        _trim_metrics(metrics_path, tr.iter)            # drop iters logged past the checkpoint
        print(f"[brain] RESUME from {args.resume}; iter={tr.iter} "
              f"frontier={tr.curriculum.frontier:.0f} ema={tr.curriculum.success_ema:.2f} "
              f"obs_dim={tr.obs_dim} -> continuing to iter {args.iterations}")
    elif args.warmstart:
        state = torch.load(args.warmstart, map_location=device, weights_only=False)
        sd = state.get("state_dict", state.get("policy", state))
        tr.policy.load_state_dict(sd)
        tr.curriculum.frontier = 0.0   # already solve the milestone -> explore from boot
        print(f"[brain] WARMSTART from {args.warmstart}; frontier=0 (explore deeper), "
              f"obs_dim={tr.obs_dim}")
    else:
        # Start the backward-robustification frontier AT the demo's milestone depth
        # (game-agnostic: found by replaying the demo), so the run begins where real
        # learning is — not restoring past an already-won milestone.
        md = tr.demo.milestone_depth()
        tr.curriculum.frontier = float(min(len(tr.demo), md))
        print(f"[brain] trainer ready: obs_dim={tr.obs_dim} device={device} "
              f"demo_len={len(tr.demo)} milestone_depth={md} "
              f"frontier={tr.curriculum.frontier:.0f}")

    streamer = None
    if args.dashboard:
        from pokeio.brain.showcase import BrainStreamer
        streamer = BrainStreamer(run_dir, fleet, rom_path=_ROM, reset_state=_STATE,
                                 obs_dim=tr.obs_dim, fovea_grid=tr.fovea_grid,
                                 mem_grid=tr.mem_grid, run_id=args.run_id)
        print(f"[brain] dashboard live -> runs/{args.run_id}/live.json "
              f"(serve :8600, view /brain?run={args.run_id})")

    t0 = time.monotonic()
    live_metrics: dict = {}

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
        if streamer is not None:
            live_metrics.update({
                "frontier": rec["frontier"], "frontier_frac": rec["frontier_frac"],
                "reach": rec["reached"], "success_ema": rec["success_ema"],
                "entropy": rec["entropy"], "reward": rec["rew_sum_mean"],
            })
            if "eval" in rec:
                e = rec["eval"]
                live_metrics.update({"gate_stochastic": e["gate_reached_frac"],
                                     "gate_greedy": e.get("gate_reached_greedy", 0.0),
                                     "progress_mean": e["progress_mean"]})
            streamer.sync(tr.policy, live_metrics, rec["iter"])
        if rec["iter"] % args.checkpoint_every == 0:
            _checkpoint(ckpt_path, tr)

    remaining = max(0, args.iterations - tr.iter)   # resume continues to --iterations, not past it
    try:
        tr.train(remaining, on_log=_log)
        _checkpoint(ckpt_path, tr)
        _export_champion(run_dir, tr, args)
    finally:
        if streamer is not None:
            streamer.close()
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
