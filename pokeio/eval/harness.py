"""Orchestration: load a run's champion, run the three controls, format a report.

Glue only — the science lives in :mod:`pokeio.eval.controls` and
:mod:`pokeio.eval.metrics`. This module:

* recovers the champion genome from a run's checkpoint (the elite, i.e.
  ``argmax(.fitness)`` over the checkpointed population — the closest recoverable
  proxy for the ``fits.argmax()`` champion the loop showcased),
* infers the obs layout from the champion's I/O widths,
* builds a probe — real emulator observations when a ROM is available (foveal
  runs), else a synthetic stand-in — and clearly labels which,
* runs noise ablation + random-weight baseline + the milestone geo-mean metric,
* renders a clean text report.

ROM use is fully guarded: nothing here imports PyBoy unless a real probe is
requested and a ROM is present, so the harness (and its tests) run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pokeio.eval import controls, metrics
from pokeio.evo.forward import TANH, CompiledPopulation
from pokeio.evo.genome import Population
from pokeio.train.checkpoint import load_checkpoint_status, LOAD_OK

_DEFAULT_N_RAM = 8


# --------------------------------------------------------------------------
# champion recovery
# --------------------------------------------------------------------------
@dataclass
class Champion:
    genome: Any
    index: int
    fitness: float
    n_in: int
    n_out: int
    is_retina: bool
    gen: int
    n_nodes: int
    n_conns: int

    def compile(self, device="cpu") -> CompiledPopulation:
        g = self.genome
        pop = Population.from_genomes(
            [g], max_nodes=len(g.nodes) + 16, max_conns=max(1, len(g.conns)) + 16
        )
        return pop.compile(device)


def load_champion(
    run_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    index: int | None = None,
) -> Champion:
    """Recover the champion genome from a run's checkpoint.

    ``index`` overrides the default elite pick (``argmax(.fitness)``). Raises if
    no usable checkpoint is present (a credibility eval must never silently
    fabricate a champion).
    """
    src = Path(checkpoint_path).parent if checkpoint_path else Path(run_dir)
    data, status = load_checkpoint_status(src)
    if status != LOAD_OK or not data:
        raise FileNotFoundError(
            f"no usable checkpoint under {src} (status={status}); "
            "point --checkpoint at a run's checkpoint.pkl"
        )
    genomes = data.get("genomes") or []
    if not genomes:
        raise ValueError(f"checkpoint at {src} has no genomes")
    if index is None:
        fits = [float(getattr(g, "fitness", 0.0)) for g in genomes]
        index = int(np.argmax(fits))
    g = genomes[index]
    return Champion(
        genome=g,
        index=index,
        fitness=float(getattr(g, "fitness", 0.0)),
        n_in=int(g.n_in),
        n_out=int(g.n_out),
        is_retina=bool(data.get("retina") is not None),
        gen=int(data.get("gen", -1)),
        n_nodes=len(g.nodes),
        n_conns=len(g.conns),
    )


# --------------------------------------------------------------------------
# probe construction
# --------------------------------------------------------------------------
def build_probe(
    layout: controls.ObsLayout,
    *,
    run_dir: str | Path | None = None,
    config=None,
    b: int = 200,
    seed: int = 1,
    use_rom: bool = True,
) -> tuple[np.ndarray, str]:
    """Build a probe observation buffer; return ``(probe, source_label)``.

    Tries a REAL emulator probe first (foveal runs with a ROM present), then
    falls back to a synthetic stand-in. The label is surfaced in the report so a
    reader always knows whether the numbers came from real or synthetic obs.
    """
    if use_rom and config is not None:
        real = _try_emulator_probe(layout, config, b=b, seed=seed)
        if real is not None:
            return real, "emulator(real)"
    return controls.synthetic_probe(layout, b=b, seed=seed), "synthetic"


def _try_emulator_probe(layout, config, *, b: int, seed: int) -> np.ndarray | None:
    """Best-effort real foveal probe from the newgame state; ``None`` on any miss.

    Guarded so a missing ROM / non-foveal controller / any import failure simply
    falls back to synthetic rather than erroring the whole eval.
    """
    try:
        vis = config.vision
        # Only the foveal active-vision controller is reconstructable here; the
        # retina latent needs the learned encoder snapshot, the legacy flat obs a
        # different encoder — both fall back to synthetic.
        if getattr(vis, "mode", "foveal") != "foveal" or layout.n_proprio != 14:
            return None
        rom = Path(config.emu.rom_path)
        state = config.emu.reset_state
        if not rom.exists():
            return None
        from pokeio.emu.env import PokeEnv
        from pokeio.emu.fleet import FovealEncoder

        enc = FovealEncoder(
            1,
            periph_grid=vis.periph_grid,
            fovea_native_px=vis.fovea_native_px,
            n_ram=layout.n_ram,
            saccade_gain=vis.saccade_gain,
            saccade_every_k=vis.saccade_every_k,
            episode_steps=max(b, 256),
            mode="foveal",
        )
        if enc.dim != layout.n_in:
            return None
        rng = np.random.default_rng(seed)
        obs: list[np.ndarray] = []
        with PokeEnv(str(rom), frame_skip=config.emu.frame_skip,
                     sticky_input=True) as env:
            screen = env.reset(state if state and Path(state).exists() else None)
            wram = env.raw_wram()
            enc.reset(0)
            button = 8
            for _ in range(b):
                dx, dy = rng.uniform(-1, 1), rng.uniform(-1, 1)
                enc.update_gaze(0, dx, dy)
                obs.append(enc.encode(0, screen, wram, button=button).copy())
                button = int(rng.integers(0, 9))
                screen, wram, _ = env.step_fast(button, wram_stride=1)
        return np.asarray(obs, dtype=np.float32)
    except Exception:
        return None


# --------------------------------------------------------------------------
# full run
# --------------------------------------------------------------------------
@dataclass
class EvalReport:
    run_id: str
    champion: Champion
    layout: controls.ObsLayout
    probe_source: str
    probe_size: int
    noise: controls.NoiseAblationResult
    random_baseline: controls.RandomBaselineResult
    milestones: metrics.MilestoneReport


def run_all(
    run_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    genome_index: int | None = None,
    config=None,
    manifest=None,
    k_random: int = 64,
    probe_size: int = 200,
    probe_seed: int = 1,
    use_rom: bool = True,
    device: str = "cpu",
    n_boot: int = 10_000,
    ci: float = 0.95,
    milestone_signal: str | None = None,
) -> EvalReport:
    """Run all three standing controls on a run's champion + telemetry."""
    run_dir = Path(run_dir)
    champ = load_champion(run_dir, checkpoint_path=checkpoint_path, index=genome_index)

    n_ram = _DEFAULT_N_RAM
    connect, sparse_k = "full", 32
    if config is not None:
        n_ram = int(getattr(config.vision, "obs_ram_bytes", n_ram))
        connect = str(getattr(config.evo, "init_connect", connect))
        sparse_k = int(getattr(config.evo, "init_k", sparse_k))

    layout = controls.infer_obs_layout(champ.n_in, champ.n_out, n_ram=n_ram)
    probe, probe_source = build_probe(
        layout, run_dir=run_dir, config=config, b=probe_size,
        seed=probe_seed, use_rom=use_rom,
    )

    cp = champ.compile(device)

    # Control 1 — noise/blank ablation on the champion.
    noise = controls.noise_ablation(
        cp, probe, layout.optical_hi, device=device,
        n_buttons=min(champ.n_out, 9),
    )

    # Control 2 — random-weight baseline, same scorer as the champion's score.
    scorer = controls.vision_dependence_scorer(probe, layout.optical_hi, device=device)
    champ_score = float(scorer(cp)[0])
    rand = controls.random_weight_baseline(
        n_in=champ.n_in, n_out=champ.n_out, k=k_random, score_fn=scorer,
        champion_score=champ_score, connect=connect, sparse_k=sparse_k,
        n_ram=layout.n_ram, n_proprio=layout.n_proprio, device=device,
        seed=probe_seed,
    )

    # Control 3 — milestone geo-mean from telemetry (+ optional manifest).
    rates, meta = metrics.milestone_rates_from_telemetry(
        run_dir, manifest=manifest, signal=milestone_signal
    )
    milestones = metrics.milestone_geomean_report(
        rates, n_boot=n_boot, ci=ci, seed=probe_seed,
        signal=meta.get("signal", ""), meta=meta,
    )

    return EvalReport(
        run_id=run_dir.name,
        champion=champ,
        layout=layout,
        probe_source=probe_source,
        probe_size=len(probe),
        noise=noise,
        random_baseline=rand,
        milestones=milestones,
    )


def format_report(rep: EvalReport) -> str:
    """Render an :class:`EvalReport` as a clean, aligned text block."""
    c = rep.champion
    lay = rep.layout
    ms = rep.milestones
    lo, hi = ms.ci
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"  pokeIO standing credibility eval  —  run '{rep.run_id}'")
    lines.append("=" * 72)
    lines.append(
        f"champion : gen {c.gen}  idx {c.index}  fitness {c.fitness:.4f}  "
        f"({'retina' if c.is_retina else 'foveal/flat'}; "
        f"n_in={c.n_in} n_out={c.n_out}, {c.n_nodes} nodes / {c.n_conns} conns)"
    )
    lines.append(
        f"obs      : optical[0:{lay.optical_hi}] proprio={lay.n_proprio} "
        f"ram={lay.n_ram}   probe={rep.probe_size} obs [{rep.probe_source}]"
    )
    lines.append("")

    lines.append("-- [1] noise / blank-frame ablation " + "-" * 36)
    lines.append("   " + rep.noise.summary(0))
    if bool(rep.noise.blind[0]):
        lines.append("   !! FLAG: champion is screen-blind — fitness is action-timing / spawn luck.")
    else:
        lines.append("   ok: action depends on the optical input (survives ablation).")
    lines.append("")

    lines.append("-- [2] random-weight-search baseline " + "-" * 35)
    lines.append("   " + rep.random_baseline.summary())
    if not rep.random_baseline.beats_random:
        lines.append("   !! FLAG: structured search did NOT beat best-of-K random weights.")
    lines.append("")

    lines.append("-- [3] milestone geometric-mean (HEADLINE) " + "-" * 29)
    lines.append(
        f"   geo-mean = {ms.geomean:.4f}   IQM(rates) = {ms.iqm:.4f}   "
        f"{int(ms.ci_level * 100)}% CI = [{lo:.4f}, {hi:.4f}]"
    )
    mode = ms.meta.get("mode", "?")
    if mode == "generic":
        lines.append(
            f"   ({ms.n_milestones} generic milestones from '{ms.signal}', "
            f"{ms.meta.get('n_gens', 0)} gens; peak={ms.meta.get('peak', 0):.3g})"
        )
    elif mode == "manifest":
        lines.append(
            f"   ({ms.n_milestones} manifest milestones "
            f"'{ms.meta.get('milestone_label', '')}', {ms.meta.get('n_gens', 0)} gens)"
        )
    else:
        lines.append(f"   (no milestones extracted — telemetry mode='{mode}')")
    lines.append("=" * 72)
    return "\n".join(lines)


__all__ = [
    "Champion",
    "load_champion",
    "build_probe",
    "EvalReport",
    "run_all",
    "format_report",
]
