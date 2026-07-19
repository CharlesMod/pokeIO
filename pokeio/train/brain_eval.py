"""Evaluate a trained System-1 checkpoint from a cold boot (#35/#44).

Loads a brain.pt checkpoint and runs the from-boot gate eval BOTH stochastically
(the honest 'can the policy do it' measure) and greedily (argmax), with the
greedy-stall map histogram — the definitive read on policy quality + the
stochastic/greedy gap. Standalone so it can run after training frees the fleet.

    PYTHONPATH=. .venv/bin/python -m pokeio.train.brain_eval \
        --run-id brain1 --device cuda:1 --horizon 768
"""

from __future__ import annotations

import argparse
import json

import torch

from pokeio.brain.replay import DemoTrajectory
from pokeio.brain.rl_loop import BrainConfig, BrainTrainer
from pokeio.train.brain_loop import build_fleet


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a System-1 brain checkpoint")
    ap.add_argument("--run-id", default="brain1")
    ap.add_argument("--ckpt", default=None, help="path to brain.pt (default runs/<run-id>/brain.pt)")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--horizon", type=int, default=768)
    args = ap.parse_args(argv)

    ckpt = args.ckpt or f"runs/{args.run_id}/brain.pt"
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    fleet = build_fleet(args.n_envs)
    try:
        cfg = BrainConfig(n_envs=args.n_envs, eval_horizon=args.horizon)
        tr = BrainTrainer(fleet, device=device, demo=DemoTrajectory(), cfg=cfg)
        state = torch.load(ckpt, map_location=device)
        tr.policy.load_state_dict(state["policy"])
        tr.policy.eval()
        print(f"[eval] loaded {ckpt} (trained iters={state.get('iter', '?')})")
        rep = tr.evaluate_gate(horizon=args.horizon)
        print("[eval] from-boot gate report:")
        print(json.dumps({
            "gate_reached_stochastic": rep["gate_reached_frac"],
            "gate_reached_greedy": rep["gate_reached_greedy"],
            "progress_mean": rep["progress_mean"],
            "progress_max": rep["progress_max"],
            "greedy_stall_maps": rep["greedy_stall_maps"],
            "blind_delta": rep["blind_delta"],
            "action_diversity": rep["action_diversity"],
            "action_entropy": rep["action_entropy"],
        }, indent=2))
    finally:
        fleet.close()


if __name__ == "__main__":
    main()
