"""pokeIO training loop — a runnable evolutionary (NEAT) loop that actually plays
Pokemon Yellow and emits real telemetry.

Pipeline per generation
------------------------
1. Split the population into **waves** of ``--players`` genomes.
2. For each wave: reset ``players`` :class:`PokeEnv` instances to the canonical
   new-game state, then for ``--episode-steps`` steps gather every player's
   small observation, run **one batched sparse forward** on cuda:1, argmax to a
   button, step every env, and score Go-Explore novelty (new archive cells).
3. Each genome's fitness = the number of cells it was first to discover.
4. Evolve with the NEAT operators (speciation, elitism, tournament, crossover,
   mutation) from :mod:`pokeio.evo.ops`.
5. Emit a :class:`GenerationRecord` (+ a short champion replay to champion.jsonl)
   via :class:`TelemetryWriter` to ``runs/<run_id>/``.

This is deliberately **single-process** for v1: one Python process drives a
modest number of PyBoy instances in a plain loop and batches all inference on the
GPU.  That is enough to prove the loop end-to-end; multiprocessing the emulators
is a later throughput optimization (the GPU forward already batches the whole
wave).  Everything imported from ``evo/``, ``emu/``, ``telemetry/`` is used
as-is — nothing in those packages is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from pokeio.config import Config
from pokeio.emu.env import PokeEnv
from pokeio.evo.forward import population_forward_sparse
from pokeio.evo.genome import InnovationTracker, Population, make_genome
from pokeio.evo.ops import MutationRates, Speciation, compatibility_distance, reproduce
from pokeio.reward.archive import NoveltyArchive
from pokeio.reward.goexplore import GoExplore
from pokeio.reward.novelty import WaveNovelty
from pokeio.train.live import ChampionShowcase, LiveStreamer
from pokeio.telemetry.schema import (
    ChampionStep,
    GenerationRecord,
    TelemetryWriter,
)

_SCREEN_H = 144
_SCREEN_W = 160
N_OUT = 8  # Discrete(8): up down left right A B START SELECT
FORWARD_STEPS = 4  # propagation hops per inference (covers evolved depth)


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------
def pick_device(prefer: str = "cuda:1") -> torch.device:
    """cuda:1 is reserved for training inference (card 0 hosts the GLM)."""
    if torch.cuda.is_available():
        idx = int(prefer.split(":")[1]) if ":" in prefer else 0
        if idx < torch.cuda.device_count():
            return torch.device(prefer)
        return torch.device("cuda:0")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# small observation encoder (obs_res x obs_res grayscale + a few RAM bytes)
# --------------------------------------------------------------------------
def _area_matrix(in_size: int, out_size: int) -> np.ndarray:
    m = np.zeros((out_size, in_size), dtype=np.float64)
    scale = in_size / out_size
    for i in range(out_size):
        lo, hi = i * scale, (i + 1) * scale
        j0, j1 = int(np.floor(lo)), int(np.ceil(hi))
        for j in range(j0, min(j1, in_size)):
            overlap = min(hi, j + 1) - max(lo, j)
            if overlap > 0:
                m[i, j] = overlap
        s = m[i].sum()
        if s > 0:
            m[i] /= s
    return m


class ObsEncoder:
    """Raw (144,160) uint8 screen -> flat [obs_res*obs_res + n_ram] float32 obs."""

    def __init__(self, res: int, n_ram: int) -> None:
        self.res = int(res)
        self.n_ram = int(n_ram)
        self._row = _area_matrix(_SCREEN_H, self.res)  # (res, H)
        self._col = _area_matrix(_SCREEN_W, self.res).T  # (W, res)
        self.dim = self.res * self.res + self.n_ram

    def encode(self, screen: np.ndarray, wram: np.ndarray) -> np.ndarray:
        small = self._row @ screen.astype(np.float64) @ self._col  # (res,res) 0..255
        vis = (small / 255.0).astype(np.float32).ravel()
        if self.n_ram > 0:
            stride = max(1, wram.size // self.n_ram)
            ram = (wram[::stride][: self.n_ram].astype(np.float32)) / 255.0
            if ram.size < self.n_ram:  # pad if short
                ram = np.concatenate([ram, np.zeros(self.n_ram - ram.size, np.float32)])
            return np.concatenate([vis, ram])
        return vis


# --------------------------------------------------------------------------
# telemetry helpers (real measurements)
# --------------------------------------------------------------------------
def query_gpu() -> list[dict]:
    """Parse ``nvidia-smi`` for per-card utilization / memory (best-effort)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return []
    cards = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            cards.append(
                {
                    "index": int(parts[0]),
                    "util": float(parts[1]),
                    "mem_used": float(parts[2]),
                    "mem_total": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return cards


def _screen_ref(screen: np.ndarray) -> str:
    return "blake2b:" + hashlib.blake2b(screen.tobytes(), digest_size=8).hexdigest()


# --------------------------------------------------------------------------
# core evaluation
# --------------------------------------------------------------------------
def evaluate_wave(
    genomes,
    envs,
    encoder: ObsEncoder,
    archive: NoveltyArchive,
    device: torch.device,
    episode_steps: int,
    max_nodes: int,
    max_conns: int,
    reset_state: str,
    streamer: LiveStreamer | None = None,
    gen: int = 0,
    novelty_mode: str = "rarity",
    novelty_floor: float = 0.1,
    goexplore: GoExplore | None = None,
    restore_prob: float = 0.5,
) -> int:
    """Evaluate one wave of genomes in lockstep; sets genome.fitness in place.

    With ``goexplore`` supplied, a fraction (``restore_prob``) of players start
    the episode from a sampled frontier cell (Go-Explore cell-restore) instead of
    the fixed new-game state, and the emulator state of every newly-discovered
    cell is captured for future restarts.

    Returns the number of agent-steps performed (n_genomes * episode_steps)."""
    n = len(genomes)
    wave = WaveNovelty(archive, n, mode=novelty_mode, floor=novelty_floor)

    # reset each player: either restore from a promising frontier cell (go-explore)
    # or fall back to the canonical new-game state.
    screens = []
    wrams = []
    for i in range(n):
        entry = None
        if (
            goexplore is not None
            and goexplore.size > 0
            and goexplore.rng.random() < restore_prob
        ):
            entry = goexplore.sample()
        if entry is not None:
            obs = goexplore.restore(envs[i], entry)
        else:
            obs = envs[i].reset(reset_state)
        screens.append(obs)
        wrams.append(envs[i].raw_wram())
    dead = [False] * n

    pop = Population.from_genomes(genomes, max_nodes=max_nodes, max_conns=max_conns)
    cp = pop.compile(device)

    for t in range(episode_steps):
        X = np.empty((n, encoder.dim), dtype=np.float32)
        for i in range(n):
            X[i] = encoder.encode(screens[i], wrams[i])
        xt = torch.from_numpy(X).to(device).unsqueeze(1)  # (n, 1, dim)
        out = population_forward_sparse(cp, xt, steps=FORWARD_STEPS)  # (n,1,N_OUT)
        actions = out[:, 0, :].argmax(dim=1).cpu().numpy()

        for i in range(n):
            screen, wram, done, _info = envs[i].step(int(actions[i]))
            screens[i] = screen
            wrams[i] = wram
            dead[i] = dead[i] or bool(done)
            wave.observe(i, screen, wram)
            # Capture the state of every fresh frontier cell for later restarts.
            if goexplore is not None and wave.last_key[i] is not None:
                goexplore.note(
                    wave.last_key[i], envs[i], depth=t,
                    globally_new=wave.last_new[i],
                )

        # Stream a live sample ~3 Hz (throttled internally; cheap when not due).
        if streamer is not None:
            streamer.maybe_write(gen, screens, wave.fitness, dead)

    for i, g in enumerate(genomes):
        g.fitness = float(wave.fitness[i])
    return n * episode_steps


def replay_champion(
    genome,
    env,
    encoder: ObsEncoder,
    archive: NoveltyArchive,
    device: torch.device,
    steps: int,
    writer: TelemetryWriter,
    max_nodes: int,
    max_conns: int,
    reset_state: str,
) -> None:
    """Replay the generation champion solo and log a few ChampionSteps."""
    pop = Population.from_genomes([genome], max_nodes=max_nodes, max_conns=max_conns)
    cp = pop.compile(device)
    screen = env.reset(reset_state)
    wram = env.raw_wram()
    for t in range(steps):
        x = encoder.encode(screen, wram)
        xt = torch.from_numpy(x[None, :]).to(device).unsqueeze(1)  # (1,1,dim)
        out = population_forward_sparse(cp, xt, steps=FORWARD_STEPS)
        action = int(out[0, 0, :].argmax().item())
        screen, wram, _done, info = env.step(action)
        is_new = archive.observe(screen, wram)
        writer.write_champion_step(
            ChampionStep(
                step=t,
                action=action,
                screen_ref=_screen_ref(screen),
                ram_tap={"map_id": int(info.get("map_id", 0))},
                reward_components={"novelty": 1.0 if is_new else 0.0},
            )
        )


# --------------------------------------------------------------------------
# adaptive speciation threshold
# --------------------------------------------------------------------------
def _greedy_species_count(
    genomes, thr: float, c1: float, c2: float, c3: float
) -> int:
    """Count species formed by fresh greedy speciation at compatibility ``thr``."""
    reps: list = []
    for g in genomes:
        if not any(
            compatibility_distance(g, r, c1, c2, c3) < thr for r in reps
        ):
            reps.append(g)
    return len(reps)


def fit_species_threshold(
    genomes,
    c1: float,
    c2: float,
    c3: float,
    target: int,
    rng: np.random.Generator,
    sample_cap: int = 96,
    iters: int = 24,
) -> float:
    """Binary-search a compatibility threshold that yields ~``target`` species.

    At gen 0 every genome shares the same innovation baseline, so the
    compatibility distance is dominated by the (tiny, tightly-clustered) mean
    weight difference and the fixed default threshold (3.0) lumps the whole
    population into one species.  Species count is monot- decreasing in the
    threshold, so a binary search reliably lands on a value that splits the
    population into ``target`` species — restoring real speciation pressure.

    Purely a loop-side tuning of the threshold fed to :class:`Speciation`; no
    change to ``evo/`` internals.
    """
    n = len(genomes)
    if n <= 1:
        return 1.0
    if n > sample_cap:
        idx = rng.choice(n, size=sample_cap, replace=False)
        sample = [genomes[int(i)] for i in idx]
    else:
        sample = list(genomes)

    # bound the search with the max pairwise distance over the sample.
    hi = 0.0
    for a in range(len(sample)):
        for b in range(a + 1, len(sample)):
            d = compatibility_distance(sample[a], sample[b], c1, c2, c3)
            if d > hi:
                hi = d
    if hi <= 0.0:
        return 1.0  # genomes identical; nothing to split
    lo = 0.0
    hi = hi + 1e-6
    target = max(2, min(target, len(sample)))
    best = hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        k = _greedy_species_count(sample, mid, c1, c2, c3)
        if k >= target:
            # too many species -> raise threshold (merge)
            best = mid
            lo = mid
        else:
            hi = mid
    return float(best)


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def train(
    gens: int,
    pop_size: int,
    players: int,
    episode_steps: int,
    obs_res: int,
    run_id: str,
    config: Config,
    device_str: str = "cuda:1",
    champion_steps: int = 48,
    live: bool = True,
    novelty_mode: str = "rarity",
    goexplore: bool = False,
    restore_prob: float = 0.5,
    goexplore_capacity: int = 2048,
    auto_species: bool = True,
    species_target: int = 6,
) -> Path:
    device = pick_device(device_str)
    rng = np.random.default_rng(config.run.seed)

    encoder = ObsEncoder(obs_res, config.vision.obs_ram_bytes)
    n_in = encoder.dim
    # Budgets: fully-connected init is (n_in+1)*N_OUT conns; leave slack to grow.
    init_conns = (n_in + 1) * N_OUT
    max_conns = init_conns + 2048
    max_nodes = n_in + 1 + N_OUT + 256

    run_dir = Path(config.run.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config.snapshot(run_dir)

    print(
        f"[train] device={device} n_in={n_in} (res={obs_res}^2+{config.vision.obs_ram_bytes} ram) "
        f"pop={pop_size} players={players} episode_steps={episode_steps} gens={gens}"
    )
    print(f"[train] run_dir={run_dir}  reset_state={config.emu.reset_state}")

    tracker = InnovationTracker(n_in=n_in, n_out=N_OUT)
    genomes = [
        make_genome(n_in, N_OUT, tracker, rng, connect="full", weight_scale=1.0)
        for _ in range(pop_size)
    ]

    rates = MutationRates(
        add_node=config.evo.mutate_add_node,
        add_conn=config.evo.mutate_add_conn,
        weight=config.evo.mutate_weight,
        toggle=config.evo.mutate_toggle,
        feedforward=not config.evo.recurrent,
    )
    spec = Speciation(threshold=config.evo.species_threshold, c1=1.0, c2=1.0, c3=0.4)
    archive = NoveltyArchive()
    go = (
        GoExplore(
            capacity=goexplore_capacity,
            rng=np.random.default_rng(config.run.seed + 1),
        )
        if goexplore
        else None
    )
    print(
        f"[train] novelty_mode={novelty_mode} goexplore={'on' if go else 'off'} "
        f"(restore_prob={restore_prob}, cap={goexplore_capacity}) "
        f"auto_species={'on' if auto_species else 'off'} (target={species_target})"
    )

    # Create the emulator pool once; reused across every wave/generation.
    rom = config.emu.rom_path
    print(f"[train] booting {players} PokeEnv instances (rom={rom}) ...")
    envs = [
        PokeEnv(
            rom,
            frame_skip=config.emu.frame_skip,
            hold_frames=config.emu.button_hold_frames,
        )
        for _ in range(players)
    ]
    reset_state = config.emu.reset_state

    # Live streaming: a dedicated showcase env that continuously plays the
    # best-genome-so-far, plus a throttled atomic writer for runs/<id>/live.json.
    streamer: LiveStreamer | None = None
    showcase_env: PokeEnv | None = None
    if live:
        showcase_env = PokeEnv(
            rom,
            frame_skip=config.emu.frame_skip,
            hold_frames=config.emu.button_hold_frames,
        )
        showcase = ChampionShowcase(
            showcase_env,
            encoder,
            device,
            reset_state,
            forward_steps=FORWARD_STEPS,
            max_nodes=max_nodes,
            max_conns=max_conns,
        )
        streamer = LiveStreamer(run_dir, showcase, run_id, hz=3.0)
        # Seed the showcase with an initial champion so the very first wave has a
        # live network to stream (replaced by the real champion each generation).
        streamer.set_champion(genomes[0], "init")
        print(f"[train] live streaming ON -> {run_dir / 'live.json'} (~3 Hz)")

    proc = psutil.Process()
    proc.cpu_percent(None)  # prime the psutil counter
    run_start = time.perf_counter()

    with TelemetryWriter(run_dir) as writer:
        for gen in range(gens):
            gen_t0 = time.perf_counter()
            archive.begin_generation()
            if go is not None:
                go.begin_generation(gen)
            psutil.cpu_percent(None)

            steps_done = 0
            for a in range(0, pop_size, players):
                wave = genomes[a : a + players]
                steps_done += evaluate_wave(
                    wave,
                    envs[: len(wave)],
                    encoder,
                    archive,
                    device,
                    episode_steps,
                    max_nodes,
                    max_conns,
                    reset_state,
                    streamer=streamer,
                    gen=gen,
                    novelty_mode=novelty_mode,
                    novelty_floor=config.reward.novelty_floor,
                    goexplore=go,
                    restore_prob=restore_prob,
                )

            fits = np.array([g.fitness for g in genomes], dtype=np.float64)
            champ_idx = int(fits.argmax())
            champion = genomes[champ_idx]

            # Update the showcase to this generation's real champion and push a
            # fresh live frame at the generation boundary.
            if streamer is not None:
                streamer.set_champion(champion, f"gen{gen}_g{champ_idx}")
                streamer.force_write(gen)

            gen_dt = time.perf_counter() - gen_t0
            cpu_pct = psutil.cpu_percent(None)
            sps = steps_done / gen_dt if gen_dt > 0 else 0.0

            # short champion replay for the live feed
            replay_champion(
                champion, envs[0], encoder, archive, device,
                min(champion_steps, episode_steps), writer,
                max_nodes, max_conns, reset_state,
            )

            # Adaptive speciation threshold: without it the gen-0 population
            # shares one innovation baseline and collapses to a single species.
            # Re-fit the threshold to the current genome spread and re-speciate
            # from scratch so the population actually splits (loop-side only).
            if auto_species:
                spec.threshold = fit_species_threshold(
                    genomes, spec.c1, spec.c2, spec.c3, species_target, rng
                )
                spec.reps.clear()
            species = spec.assign(genomes, rng)

            reward_terms = {"novelty": float(fits.max())}
            if auto_species:
                reward_terms["species_threshold"] = float(spec.threshold)
            if go is not None:
                reward_terms.update(go.stats())
            rec = GenerationRecord(
                gen=gen,
                wall_time=time.perf_counter() - run_start,
                fitness_best=float(fits.max()),
                fitness_median=float(np.median(fits)),
                fitness_worst=float(fits.min()),
                n_species=len(species),
                archive_cells=archive.size,
                archive_delta=archive.generation_delta,
                champion_id=f"gen{gen}_g{champ_idx}",
                champion_genome_ref=f"gen{gen}:idx{champ_idx}",
                reward_terms=reward_terms,
                throughput_sps=sps,
                cpu_pct=cpu_pct,
                gpu=query_gpu(),
            )
            writer.write_generation(rec)
            print(
                f"[gen {gen:2d}] best={rec.fitness_best:6.1f} med={rec.fitness_median:6.1f} "
                f"worst={rec.fitness_worst:5.1f} species={rec.n_species:3d} "
                f"cells={rec.archive_cells:6d} (+{rec.archive_delta:4d}) "
                f"{gen_dt:5.1f}s  {sps:7.1f} steps/s"
            )

            # evolve (skip on the final generation — nothing consumes it)
            if gen < gens - 1:
                genomes = reproduce(
                    genomes, species, tracker, rng, rates,
                    pop_size=pop_size,
                    tournament_size=config.evo.tournament_size,
                    elitism=config.evo.elitism,
                    crossover_rate=config.evo.crossover_rate,
                    fitness_sharing=config.evo.fitness_sharing,
                    weight_scale=1.0,
                    survival_threshold=0.4,
                    c3=0.4,
                )

    for e in envs:
        e.close()
    if showcase_env is not None:
        showcase_env.close()
    print(f"[train] done. telemetry -> {run_dir / 'telemetry.jsonl'}")
    return run_dir


def build_config(args) -> Config:
    cfg = Config()
    cfg.run.run_id = args.run_id
    cfg.run.seed = args.seed
    cfg.evo.pop_size = args.pop
    cfg.emu.n_players = args.players
    cfg.vision.obs_res = args.obs_res
    # The config default frame_skip=1/hold=0 releases the button before any tick
    # holds it, so input never registers. Use the Phase-0 contract (env.py) values
    # so buttons actually take effect and the agent can move / advance dialog.
    cfg.emu.frame_skip = args.frame_skip
    cfg.emu.button_hold_frames = args.hold_frames
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="pokeIO evolutionary training loop")
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--players", type=int, default=16)
    ap.add_argument("--episode-steps", type=int, default=300)
    ap.add_argument("--obs-res", type=int, default=24)
    ap.add_argument("--run-id", default="smoke1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frame-skip", type=int, default=24,
                    help="ticks advanced per agent step (must hold buttons to register)")
    ap.add_argument("--hold-frames", type=int, default=8,
                    help="frames a button is held within a step (< frame-skip)")
    ap.add_argument("--live", dest="live", action="store_true", default=True,
                    help="stream runs/<id>/live.json for the dashboard (default ON)")
    ap.add_argument("--no-live", dest="live", action="store_false",
                    help="disable live streaming")
    # -- Go-Explore + speciation ------------------------------------------
    ap.add_argument("--goexplore", dest="goexplore", action="store_true",
                    default=False,
                    help="restart episodes from sampled frontier cells (Go-Explore)")
    ap.add_argument("--restore-prob", type=float, default=0.5,
                    help="per-player prob of restoring from a frontier cell")
    ap.add_argument("--goexplore-capacity", type=int, default=2048,
                    help="max stored emulator states (memory bound)")
    ap.add_argument("--novelty-mode", choices=("rarity", "per_gen", "global"),
                    default="rarity",
                    help="rarity: rarity-weighted distinct-cell coverage (default); "
                         "per_gen: only cells new this generation; global: legacy")
    ap.add_argument("--auto-species", dest="auto_species", action="store_true",
                    default=True,
                    help="adaptively fit the speciation threshold (default ON)")
    ap.add_argument("--no-auto-species", dest="auto_species", action="store_false",
                    help="use the fixed config species_threshold instead")
    ap.add_argument("--species-target", type=int, default=6,
                    help="target species count for the adaptive threshold")
    args = ap.parse_args()

    cfg = build_config(args)
    train(
        gens=args.gens,
        pop_size=args.pop,
        players=args.players,
        episode_steps=args.episode_steps,
        obs_res=args.obs_res,
        run_id=args.run_id,
        config=cfg,
        device_str=args.device,
        live=args.live,
        novelty_mode=args.novelty_mode,
        goexplore=args.goexplore,
        restore_prob=args.restore_prob,
        goexplore_capacity=args.goexplore_capacity,
        auto_species=args.auto_species,
        species_target=args.species_target,
    )


if __name__ == "__main__":
    main()
