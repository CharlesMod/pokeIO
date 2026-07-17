"""``python -m pokeio.analytics <run-id>`` — a trainer's-eye progress report.

Reads runs/<id>/telemetry.jsonl (agent stats) and, when present, the run's
checkpoint (to decode furthest game progress). Prints a compact terminal
dashboard with sparkline trends. Read-only; safe to run against a live run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pokeio.analytics.trends import (
    fmt_int,
    humanize_secs,
    sparkline,
    trend_arrow,
)
from pokeio.telemetry.schema import read_generations


def _col(series, key, default=0.0):
    out = []
    for r in series:
        v = getattr(r, key, None)
        out.append(default if v is None else v)
    return out


def _rt(series, key, default=None):
    """Pull a reward_terms value per generation (None where absent)."""
    return [r.reward_terms.get(key, default) for r in series]


def _line(label: str, spark: str, tail: str = "") -> str:
    return f"  {label:<16}{spark}  {tail}"


def build_report(run_dir: Path) -> str:
    recs = read_generations(run_dir)
    if not recs:
        return f"no telemetry in {run_dir} (has the run produced a generation yet?)"
    recs.sort(key=lambda r: r.gen)
    last = recs[-1]
    n = len(recs)
    L: list[str] = []

    # -- header -----------------------------------------------------------
    wall = _col(recs, "wall_time")
    span = wall[-1] - wall[0] if len(wall) > 1 else wall[-1]
    L.append("═" * 70)
    L.append(f"  pokeIO run '{run_dir.name}'  ·  gen {last.gen}  ·  "
             f"{n} records  ·  {humanize_secs(span)} wall")
    L.append("═" * 70)

    # -- fitness ----------------------------------------------------------
    best = _col(recs, "fitness_best")
    med = _col(recs, "fitness_median")
    L.append("FITNESS  (novelty + mined progress, selection scale)")
    L.append(_line("best", sparkline(best),
                   f"{best[-1]:8.1f}  {trend_arrow(best)}"))
    L.append(_line("median", sparkline(med),
                   f"{med[-1]:8.2f}  {trend_arrow(med)}"))

    # -- exploration ------------------------------------------------------
    cells = _col(recs, "archive_cells")
    depth = [d if d else 0 for d in _rt(recs, "goexplore_max_depth", 0)]
    L.append("")
    L.append("EXPLORATION  (Go-Explore frontier)")
    L.append(_line("archive cells", sparkline(cells),
                   f"{fmt_int(cells[-1]):>8}  {trend_arrow(cells)}"))
    L.append(_line("frontier depth", sparkline(depth),
                   f"{fmt_int(depth[-1]):>8}  {trend_arrow(depth)}"))
    # boot gauntlet: only gens where it ran (>=0)
    bg = [(r.gen, r.boot_depth, r.boot_depth_frac) for r in recs
          if getattr(r, "boot_depth", -1.0) >= 0]
    if bg:
        fracs = [f for _, _, f in bg]
        g_last, d_last, f_last = bg[-1]
        L.append(_line("boot-gauntlet", sparkline(fracs),
                       f"{100 * f_last:6.1f}%  depth {int(d_last)} @gen{g_last}  "
                       f"{trend_arrow(fracs)}"))
        L.append("                  ↑ how deep the champion re-reaches the "
                 "frontier FROM BOOT (the real competence signal)")

    # -- population -------------------------------------------------------
    spec = _col(recs, "n_species")
    killed = [k or 0 for k in _rt(recs, "species_stagnant_killed", 0)]
    L.append("")
    L.append("POPULATION")
    L.append(_line("species", sparkline(spec),
                   f"{int(spec[-1]):>8}  {trend_arrow(spec)}"))
    if any(killed):
        L.append(_line("stagnant kills", sparkline(killed),
                       f"{int(sum(killed)):>8} total"))

    # -- mined progress ---------------------------------------------------
    taps = [t or 0 for t in _rt(recs, "progress_taps", 0)]
    if any(taps):
        pbest = [p or 0 for p in _rt(recs, "progress_best", 0)]
        L.append("")
        L.append("MINED PROGRESS  (game-agnostic RAM counters wired to fitness)")
        L.append(_line("progress signal", sparkline(pbest),
                       f"{pbest[-1]:8.1f}  {trend_arrow(pbest)}"))
        L.append(_line("active taps", sparkline(taps),
                       f"{int(taps[-1]):>8} counters"))

    # -- throughput -------------------------------------------------------
    sps = _col(recs, "throughput_sps")
    L.append("")
    L.append("THROUGHPUT")
    L.append(_line("agent-steps/s", sparkline(sps),
                   f"{fmt_int(sps[-1]):>8}  {trend_arrow(sps)}"))
    gpu = last.gpu or []
    if gpu:
        util = " ".join(f"card{c.get('index')}={c.get('util', 0):.0f}%"
                        for c in gpu)
        L.append(f"  gpu               {util}   cpu={last.cpu_pct:.0f}%")

    # -- game progress (optional; decoded from the checkpoint) ------------
    try:
        from pokeio.analytics.gameprogress import checkpoint_progress_summary
        gp = checkpoint_progress_summary(run_dir)
        if gp:
            L.append("")
            L.append(gp)
    except Exception:
        pass  # decoder optional / checkpoint absent — agent stats still print

    L.append("═" * 70)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="pokeIO trainer progress report")
    ap.add_argument("run_id", nargs="?", default=None,
                    help="run id under runs/ (default: newest)")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.run_id:
        run_dir = runs_dir / args.run_id
    else:  # newest by telemetry mtime
        cands = [p.parent for p in runs_dir.glob("*/telemetry.jsonl")]
        if not cands:
            print(f"no runs with telemetry under {runs_dir}")
            return
        run_dir = max(cands, key=lambda p: (p / "telemetry.jsonl").stat().st_mtime)

    print(build_report(run_dir))


if __name__ == "__main__":
    main()
