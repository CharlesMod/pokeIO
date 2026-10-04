"""pokeIO command line.

    python -m pokeio train  --spec games/pokemon_yellow/spec.yaml --config configs/p100.yaml
    python -m pokeio eval   --spec ... --ckpt runs/<run>/ckpt/latest.pt --episodes 32 --steps 20000
    python -m pokeio probe  --spec ...            # check RAM fields / alignment on a real ROM
    python -m pokeio bench  --spec ... --workers 48 --envs 6
    python -m pokeio make-state --spec ...        # build the start state from start.boot_macro
    python -m pokeio record --spec ... --ckpt ... --out rec/
    python -m pokeio dash   --run runs/<name>        # browse a run's wall offline

While training, the live wall is served on http://<host>:8600/ (config: dash.*).
"""

from __future__ import annotations

import argparse
import sys

from pokeio.config import load_config
from pokeio.spec import load_spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pokeio")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_spec(p):
        p.add_argument("--spec", default=None, help="game spec YAML")
        p.add_argument("--root", default=".", help="repo root for relative paths in the spec")

    p = sub.add_parser("train")
    add_spec(p)
    p.add_argument("--config", action="append", default=[], help="config YAML (repeatable)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.lr=1e-4")

    p = sub.add_parser("eval")
    add_spec(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--blind", action="store_true", help="zero the screen channel (vision-reliance check)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="write the report JSON here")

    p = sub.add_parser("probe")
    add_spec(p)
    p.add_argument("--out", default="probe")
    p.add_argument("--steps", type=int, default=200)

    p = sub.add_parser("bench")
    add_spec(p)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--envs", type=int, default=6)
    p.add_argument("--batch-workers", type=int, default=0)
    p.add_argument("--seconds", type=float, default=30.0)

    p = sub.add_parser("make-state")
    add_spec(p)
    p.add_argument("--out", default=None)

    p = sub.add_parser("dash", help="serve the training wall for a run directory (offline view)")
    p.add_argument("--run", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8600)

    p = sub.add_parser("record")
    add_spec(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="recording")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--greedy", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "train":
        cfg = load_config(args.config, args.set)
        spec = load_spec(args.spec or cfg.spec, args.root)
        cfg.spec = args.spec or cfg.spec
        from pokeio.train import Trainer

        Trainer(cfg, spec).train()
        return 0

    if args.cmd == "dash":
        from pokeio.dash import serve_offline

        serve_offline(args.run, args.host, args.port)
        return 0

    spec = load_spec(args.spec or load_config().spec, args.root)
    if args.cmd == "eval":
        from pokeio.evaluate import evaluate, print_report, save_report

        r = evaluate(spec, args.ckpt, args.episodes, args.steps, args.greedy, args.workers,
                     args.blind, args.device)
        print_report(r)
        if args.out:
            save_report(r, args.out)
    elif args.cmd == "probe":
        from pokeio.tools import probe

        probe(spec, args.out, args.steps)
    elif args.cmd == "bench":
        from pokeio.tools import bench

        bench(spec, args.workers, args.envs, args.batch_workers, args.seconds)
    elif args.cmd == "make-state":
        from pokeio.tools import make_state

        make_state(spec, args.out)
    elif args.cmd == "record":
        from pokeio.tools import record

        record(spec, args.ckpt, args.out, args.steps, args.every, args.greedy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
