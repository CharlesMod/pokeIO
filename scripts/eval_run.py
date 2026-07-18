#!/usr/bin/env python3
"""Run pokeIO's standing credibility controls on a run's champion + telemetry.

    python scripts/eval_run.py --run <run_id> [--checkpoint path]

Runs, on the run's checkpointed champion, all three audit controls and prints a
clean report:

  [1] noise / blank-frame ablation  — is it seeing, or timing? (flags screen-blind)
  [2] random-weight-search baseline — does it beat best-of-K random weights?
  [3] milestone geo-mean (HEADLINE) — Crafter-style geo-mean, IQM + bootstrap CI

Dependency-light (numpy + torch + repo modules). ROM-guarded: a real emulator
probe is used for foveal runs when the ROM is present, otherwise a synthetic
probe (the report says which). Never runs training or a full eval.

Feeding a manifest (later): pass ``--manifest runs/<id>/manifest.json``. Once the
loop logs per-dimension progress as ``reward_terms[<dimension id>]`` per
generation, milestone [3] becomes manifest-driven automatically; until then a
manifest just relabels the headline while the generic mined-progress ladder
supplies the rates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a bare script (python scripts/eval_run.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pokeio.config import CONFIG_FILENAME, Config  # noqa: E402
from pokeio.eval import harness  # noqa: E402


def _resolve_run_dir(run: str, runs_dir: str) -> Path:
    """Accept ``--run`` as a direct path or as a run id under ``--runs-dir``."""
    p = Path(run)
    if p.exists() and (p / "checkpoint.pkl").exists():
        return p
    cand = Path(runs_dir) / run
    if cand.exists():
        return cand
    # fall back to the literal path so the error message is about the real target
    return p if p.exists() else cand


def _load_config(run_dir: Path):
    cfg_path = run_dir / CONFIG_FILENAME
    if cfg_path.exists():
        try:
            return Config.load(cfg_path)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] warning: could not load {cfg_path} ({e}); using defaults")
    return None


def _load_manifest(path: str | None):
    if not path:
        return None
    from pokeio.manifest.schema import Manifest

    return Manifest.load_json(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pokeIO standing credibility eval")
    ap.add_argument("--run", required=True, help="run id (under --runs-dir) or a run directory path")
    ap.add_argument("--checkpoint", default=None, help="explicit path to a checkpoint.pkl (overrides --run)")
    ap.add_argument("--runs-dir", default="runs", help="root dir that holds run folders (default: runs)")
    ap.add_argument("--manifest", default=None, help="optional manifest.json for named milestones")
    ap.add_argument("--genome-index", type=int, default=None, help="pick this genome instead of the elite")
    ap.add_argument("--k-random", type=int, default=64, help="K random genomes for the baseline")
    ap.add_argument("--probe-size", type=int, default=200, help="probe observations")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cpu | cuda:1 | ...")
    ap.add_argument("--n-boot", type=int, default=10_000, help="bootstrap resamples for the CI")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--milestone-signal", default=None, help="force a telemetry signal for generic milestones")
    ap.add_argument("--no-rom", action="store_true", help="skip the real emulator probe (synthetic only)")
    args = ap.parse_args(argv)

    run_dir = (
        Path(args.checkpoint).parent if args.checkpoint else _resolve_run_dir(args.run, args.runs_dir)
    )
    config = _load_config(run_dir)
    manifest = _load_manifest(args.manifest)

    try:
        rep = harness.run_all(
            run_dir,
            checkpoint_path=args.checkpoint,
            genome_index=args.genome_index,
            config=config,
            manifest=manifest,
            k_random=args.k_random,
            probe_size=args.probe_size,
            probe_seed=args.seed,
            use_rom=not args.no_rom,
            device=args.device,
            n_boot=args.n_boot,
            ci=args.ci,
            milestone_signal=args.milestone_signal,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[eval] {e}", file=sys.stderr)
        return 2

    print(harness.format_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
