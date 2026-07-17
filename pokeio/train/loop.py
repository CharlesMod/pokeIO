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
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from pokeio.config import Config
from pokeio.emu.env import PokeEnv
from pokeio.emu.fleet import AsyncFleet, BarrierFleet, FovealEncoder, ObsEncoder
from pokeio.evo.forward import TANH, population_forward_sparse
from pokeio.evo.genome import InnovationTracker, Population, make_genome
from pokeio.evo.ops import MutationRates, Speciation, compatibility_distance
from pokeio.reward.archive import NoveltyArchive
from pokeio.reward.goexplore import GoExplore
from pokeio.reward.novelty import WaveNovelty
from pokeio.train.checkpoint import (
    LOAD_ABSENT,
    build_state as build_checkpoint_state,
    checkpoint_path,
    load_checkpoint_status,
    save_checkpoint,
)
from pokeio.train.gauntlet import run_boot_gauntlet
from pokeio.train.live import ChampionShowcase, LiveStreamer
from pokeio.telemetry.schema import (
    CHAMPION_FILENAME,
    TELEMETRY_FILENAME,
    TYPE_CHAMPION_STEP,
    TYPE_GENERATION,
    ChampionStep,
    GenerationRecord,
    TelemetryWriter,
)

_SCREEN_H = 144
_SCREEN_W = 160
N_OUT = 11  # 9 button logits (argmax) + saccade dx,dy (tanh); rebound from config.evo.n_out in train()
N_BUTTONS = 9  # up down left right A B START SELECT NOOP (the argmax-consumed head slice)
FORWARD_STEPS = 4  # propagation hops per inference (covers evolved depth)


def _split_head(outv: np.ndarray, softmax_temp: float = 0.0):
    """Split the raw (n, N_OUT) tanh output into (buttons, saccade_dx, saccade_dy).

    Active-vision spine §3.1: ``out[..., 0:9]`` is the Discrete-9 button head
    (argmax, or temperature-softmax sample when ``softmax_temp > 0``); ``out[9]``
    / ``out[10]`` are the RAW saccade commands.  The gaze integrator applies
    ``tanh`` internally (``FovealEncoder.update_gaze`` / the fleet workers), so
    the raw values are forwarded verbatim — no tanh here (double-tanh bug).
    """
    logits = outv[:, :N_BUTTONS]
    if softmax_temp and softmax_temp > 0.0:
        z = logits / float(softmax_temp)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        r = np.random.random((logits.shape[0], 1))
        buttons = (p.cumsum(axis=1) > r).argmax(axis=1).astype(np.int32)
    else:
        buttons = logits.argmax(axis=1).astype(np.int32)
    dx = np.ascontiguousarray(outv[:, N_BUTTONS], dtype=np.float32)
    dy = np.ascontiguousarray(outv[:, N_BUTTONS + 1], dtype=np.float32)
    return buttons, dx, dy


def _resp_scores(btn_hist, o_sum, o_sq, o_cnt, w_resp_var):
    """R_resp per genome (§6.3): button-histogram Shannon entropy (bits) + a
    variance term ``w_resp_var * mean_t ||o_t - o_bar||^2`` over the raw output.

    ``btn_hist`` (n, 9) counts, ``o_sum`` (n, N_OUT) running sum, ``o_sq`` (n,)
    running sum of ``||o_t||^2``, ``o_cnt`` (n,) step counts.  A constant-button
    agent scores H≈0; a policy that varies its output scores higher."""
    cnt = np.maximum(o_cnt.astype(np.float64), 1.0)
    tot = np.maximum(btn_hist.sum(axis=1, keepdims=True), 1.0)
    p = btn_hist / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0.0, np.log2(p), 0.0)
    H = -(p * logp).sum(axis=1)
    mean = o_sum / cnt[:, None]
    var = o_sq / cnt - (mean ** 2).sum(axis=1)
    var = np.clip(var, 0.0, None)
    return H + float(w_resp_var) * var


def _blind_ablation_gate(cp, probe_obs, device, *, beta, dmin, optical_hi=432):
    """E3 blind-ablation gate (§6.2): the dominant selection multiplier.

    Runs ONE batched population forward over the probe buffer ``P`` twice — real
    obs vs optical blocks ``[0:optical_hi]`` zeroed (proprio+ram preserved) — and
    scores ``Δ_i = mean_P || a_real − a_blind ||_1`` over the N_OUT output, then
    ``gate_i = sigmoid(beta*(Δ_i − dmin))``.  A screen-blind policy (Δ≈0) gates
    to ~0; a screen-dependent one to ~1.  RAM held constant isolates the OPTICAL
    dependence.  Returns ``(gate (N,), median_delta)``."""
    n = int(cp.n)
    if probe_obs is None or len(probe_obs) == 0:
        return np.ones(n, dtype=np.float64), 0.0
    P = np.ascontiguousarray(np.asarray(probe_obs, dtype=np.float32))
    Pb = P.copy()
    Pb[:, :optical_hi] = 0.0
    with torch.no_grad():
        xt = torch.from_numpy(P).to(device)          # (P, dim) -> expands to (N,P,dim)
        real = population_forward_sparse(cp, xt, steps=FORWARD_STEPS)
        xtb = torch.from_numpy(Pb).to(device)
        blind = population_forward_sparse(cp, xtb, steps=FORWARD_STEPS)
        d = (real - blind).abs().sum(dim=2).mean(dim=1)  # (N,): L1 over out, mean over P
    delta = d.detach().cpu().numpy().astype(np.float64)
    gate = 1.0 / (1.0 + np.exp(-float(beta) * (delta - float(dmin))))
    return gate, float(np.median(delta))


# --------------------------------------------------------------------------
# reproducibility (A4) + topology-budget growth (A2) + resume hygiene (A3)
# --------------------------------------------------------------------------
def _git_sha(repo_dir: str | Path | None = None) -> str:
    """Best-effort ``<sha>[-dirty]``; empty string outside a git checkout.

    Records the exact code a run used. Guarded for non-git checkouts (tarball
    deploys) — a missing ``git`` or a non-repo cwd yields ``""`` rather than
    raising.
    """
    repo = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]

    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=str(repo),
            capture_output=True, text=True, timeout=5,
        )

    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode != 0:
            return ""
        sha = head.stdout.strip()
        status = _git("status", "--porcelain")
        if status.returncode == 0 and status.stdout.strip():
            sha += "-dirty"
        return sha
    except Exception:
        return ""


def _write_resolved_run(
    run_dir: str | Path, argv, train_kwargs: dict, config
) -> Path:
    """Write ``<run_dir>/resolved-run.json`` (argv + all train() kwargs + config).

    ``config.yaml`` snapshots only the ``Config`` dataclass and omits the
    decisive CLI-only knobs (gens, episode_steps, novelty_mode, goexplore*,
    init_connect/init_k, engine, boot_gauntlet_*, recurrent_memory,
    checkpoint_every); this captures the FULL call so a run is reproducible from
    one file. Written atomically (temp + os.replace). (audit A4)
    """
    payload = {
        "argv": list(argv),
        "git_sha": config.run.git_sha,
        "seed": config.run.seed,
        "train_kwargs": train_kwargs,
        "config": config.to_dict(),
    }
    path = Path(run_dir) / "resolved-run.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _grow_topology_budget(
    max_nodes: int, max_conns: int,
    need_nodes: int, need_conns: int,
    headroom: float = 0.25,
) -> tuple[int, int]:
    """Grow padding budgets so the biggest genome fits with headroom (A2).

    NEAT's add_node/add_conn grow genomes monotonically, so a gen-0 budget is
    eventually exceeded and ``Population.from_genomes`` raises ``ValueError``,
    killing a long run with no in-flight checkpoint. When a genome comes within
    ``headroom`` of a cap, grow that cap to the next power of two above
    ``need*(1+headroom)`` (rounding up keeps regrows rare and tensors
    realloc-friendly). Never shrinks.
    """
    def _grow(cap: int, need: int) -> int:
        need = max(int(need), 1)
        if need <= int(cap * (1.0 - headroom)):
            return cap
        target = int(math.ceil(need * (1.0 + headroom)))
        grown = 1 << max(0, (target - 1)).bit_length()
        return max(cap, grown)

    return _grow(max_nodes, need_nodes), _grow(max_conns, need_conns)


def _atomic_write_lines(path: Path, lines) -> None:
    """Rewrite ``path`` with ``lines`` (bytes, newline-free) atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        for s in lines:
            fh.write(s)
            fh.write(b"\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _truncate_streams_on_resume(run_dir: str | Path, start_gen: int) -> None:
    """Drop telemetry/champion records for ``gen >= start_gen`` before appending.

    ``write_generation`` fires every gen but checkpoints are periodic, so gens
    between the last checkpoint and a crash were already written. On ``--resume``
    the loop re-runs them; without this they'd be appended a SECOND time
    (``read_generations`` does no dedup -> the dashboard sees duplicate / rewound
    generations). Both streams are rewritten atomically (temp + os.replace) so a
    crash mid-truncate can't corrupt them. (audit A3a)
    """
    import orjson  # the same serializer the telemetry writer uses

    run_dir = Path(run_dir)

    # -- telemetry.jsonl: keyed directly on each record's `gen` field ---------
    tpath = run_dir / TELEMETRY_FILENAME
    if tpath.exists():
        kept: list[bytes] = []
        dropped = 0
        with open(tpath, "rb") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    env = orjson.loads(s)
                except Exception:
                    continue  # torn trailing line from a crash mid-write
                if (
                    env.get("type") == TYPE_GENERATION
                    and int(env.get("data", {}).get("gen", -1)) >= start_gen
                ):
                    dropped += 1
                    continue
                kept.append(s)
        _atomic_write_lines(tpath, kept)
        if dropped:
            print(
                f"[resume] telemetry: dropped {dropped} record(s) for "
                f"gen>={start_gen}", flush=True,
            )

    # -- champion.jsonl: no gen field; one replay block per gen, each block
    #    begins with a step==0 record. Keep the first `start_gen` blocks. ------
    cpath = run_dir / CHAMPION_FILENAME
    if cpath.exists():
        kept = []
        blocks = 0
        dropped = 0
        dropping = False
        with open(cpath, "rb") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    env = orjson.loads(s)
                except Exception:
                    continue
                if (
                    env.get("type") == TYPE_CHAMPION_STEP
                    and int(env.get("data", {}).get("step", -1)) == 0
                ):
                    dropping = blocks >= start_gen  # block index == gen
                    blocks += 1
                if dropping:
                    dropped += 1
                    continue
                kept.append(s)
        _atomic_write_lines(cpath, kept)
        if dropped:
            print(
                f"[resume] champion: dropped {dropped} step(s) past "
                f"gen {start_gen - 1}", flush=True,
            )


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


# The observation encoder (ObsEncoder) lives in pokeio.emu.fleet so the barrier
# worker processes can import it WITHOUT pulling torch into every child; it is
# re-exported here for callers that historically imported it from the loop.


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
# live pace switching (spectate mode)
# --------------------------------------------------------------------------
class PaceController:
    """Live-toggleable wave pacing: "max" (flat-out, today's behaviour,
    bit-for-bit) or "realtime" (authentic Game Boy speed for the whole swarm).

    The dashboard writes ``runs/<id>/pace.json`` via ``/api/pace``; this
    controller stat-polls it at most every ``poll_s`` seconds (a monotonic
    clock read per round otherwise — the busy-spin hot path is untouched) and
    applies flips to a RUNNING trainer with no restart:

    * fleet: shm pace flag -> workers sleep-wait between rounds (realtime)
      or busy-spin (max).
    * streamer: emit cadence 10 Hz (realtime) / base ~3 Hz (max) + payload
      ``pace`` field.
    * rounds: in realtime, round k's release deadline is ``t0 + k * period``
      with ``period = frame_skip / 60`` s (one agent-step = frame_skip game
      frames) — an ABSOLUTE schedule, so sleep truncation never accumulates
      drift.  If a round overruns its deadline the schedule re-anchors at
      "now" instead of bursting to catch up.
    """

    def __init__(self, run_dir, frame_skip: int, fleet=None, streamer=None,
                 poll_s: float = 0.25) -> None:
        self.path = Path(run_dir) / "pace.json"
        self.period = float(frame_skip) / 60.0  # seconds of game time per round
        self.fleet = fleet
        self.streamer = streamer
        self.poll_s = float(poll_s)
        self.mode = "max"
        self.round_hz = 0.0  # rolling measured round rate (either mode)
        self._sig = None  # (mtime_ns, size) of last parsed pace.json
        self._next_poll = 0.0
        self._t0: float | None = None  # absolute-schedule anchor
        self._k = 0  # rounds since anchor
        self._last_round_ts: float | None = None
        self.poll()  # honor a pre-existing pace.json at startup

    # -- pace.json polling (cheap: clock read; stat at most every poll_s) ----
    def poll(self) -> str:
        now = time.monotonic()
        if now < self._next_poll:
            return self.mode
        self._next_poll = now + self.poll_s
        try:
            st = self.path.stat()
        except OSError:
            return self.mode  # no pace.json -> stay put
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._sig:
            return self.mode
        try:
            mode = json.loads(self.path.read_bytes()).get("mode")
        except Exception:
            return self.mode  # mid-write/corrupt: retry next poll
        self._sig = sig
        if mode in ("realtime", "max"):
            self._apply(mode)
        return self.mode

    def _apply(self, mode: str) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self._t0 = None  # (re-)anchor the schedule on entering realtime
        self._k = 0
        if self.fleet is not None:
            self.fleet.set_pace(mode == "realtime")
        if self.streamer is not None:
            self.streamer.set_pace(mode)
        print(f"[pace] mode -> {mode}", flush=True)

    # -- per-round hooks ------------------------------------------------------
    def before_round(self) -> None:
        """Poll for flips, then (realtime only) sleep to the round's deadline."""
        self.poll()
        if self.mode != "realtime":
            return
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
            self._k = 0
            return
        self._k += 1
        deadline = self._t0 + self._k * self.period
        if deadline <= now:
            # overrun: re-anchor (never burst-catch-up)
            self._t0 = now
            self._k = 0
            return
        time.sleep(deadline - now)

    def after_round(self) -> None:
        """Update the measured round rate (EMA over inter-round gaps)."""
        now = time.monotonic()
        if self._last_round_ts is not None:
            dt = now - self._last_round_ts
            if dt > 0:
                inst = 1.0 / dt
                self.round_hz = (
                    inst if self.round_hz <= 0
                    else 0.9 * self.round_hz + 0.1 * inst
                )
        self._last_round_ts = now

    def begin_wave(self) -> None:
        """Reset the inter-round timer so boundary gaps don't poison the EMA."""
        self._last_round_ts = None
        self._t0 = None  # fresh absolute anchor per wave
        self._k = 0

    def per_round_s(self) -> float:
        """Current per-round budget: exact in realtime, measured in max."""
        if self.mode == "realtime":
            return self.period
        return (1.0 / self.round_hz) if self.round_hz > 0 else 0.0


# --------------------------------------------------------------------------
# core evaluation
# --------------------------------------------------------------------------
def _slice_compiled(cp, lo: int, hi: int):
    """Row-slice a CompiledPopulation (padded rows are per-genome independent,
    so slicing is bit-identical to compiling the genome slice on its own)."""
    import dataclasses

    return dataclasses.replace(
        cp,
        node_act=cp.node_act[lo:hi],
        node_bias=cp.node_bias[lo:hi],
        conn_in_slot=cp.conn_in_slot[lo:hi],
        conn_out_slot=cp.conn_out_slot[lo:hi],
        conn_weight=cp.conn_weight[lo:hi],
        conn_valid=cp.conn_valid[lo:hi],
    )


def evaluate_wave(
    genomes,
    envs,
    encoder: FovealEncoder,
    archive: NoveltyArchive,
    device: torch.device,
    episode_steps: int,
    max_nodes: int,
    max_conns: int,
    reset_state: str,
    streamer: LiveStreamer | None = None,
    gen: int = 0,
    novelty_mode: str = "rarity",
    novelty_floor: float = 0.01,
    goexplore: GoExplore | None = None,
    restore_prob: float = 0.5,
    spawn_out: dict[int, bytes] | None = None,
    spawn_key_out: dict[int, bytes] | None = None,
    wave_offset: int = 0,
    recurrent_memory: bool = True,
    softmax_temp: float = 0.0,
    w_resp_var: float = 0.5,
    probe_sink: list | None = None,
    probe_quota: int = 0,
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
    base_depth = np.zeros(n, dtype=np.int64)  # cumulative chain depth per player
    for i in range(n):
        encoder.reset(i)  # gaze -> centre, motion -> 0.5 (per episode/restore)
        entry = None
        if (
            goexplore is not None
            and goexplore.size > 0
            and goexplore.rng.random() < restore_prob
        ):
            entry = goexplore.sample()
        if entry is not None:
            obs = goexplore.restore(envs[i], entry)
            base_depth[i] = entry.depth
            if spawn_out is not None:
                # record the champion-candidate's true spawn for the showcase
                spawn_out[wave_offset + i] = entry.state
            if spawn_key_out is not None:  # E2: cell identity for the baseline
                spawn_key_out[wave_offset + i] = entry.key
        else:
            obs = envs[i].reset(reset_state)
        screens.append(obs)
        wrams.append(envs[i].raw_wram())
    dead = [False] * n
    last_button = np.full(n, 8, dtype=np.int32)  # NOOP until the first action lands

    # R_resp accumulators (§6.3): 9-button histogram + running output moments.
    btn_hist = np.zeros((n, N_BUTTONS), dtype=np.int64)
    o_sum = np.zeros((n, N_OUT), dtype=np.float64)
    o_sq = np.zeros(n, dtype=np.float64)
    o_cnt = np.zeros(n, dtype=np.int64)
    probe_taken = 0

    pop = Population.from_genomes(genomes, max_nodes=max_nodes, max_conns=max_conns)
    cp = pop.compile(device)
    # lockstep: every env steps every round, so recurrent state advances for
    # all rows each round (no selective write-back like the async engine).
    state_t = (
        torch.zeros((n, 1, cp.M), dtype=torch.float32, device=device)
        if recurrent_memory else None
    )

    for t in range(episode_steps):
        if goexplore is not None:
            goexplore.feed_capture_budget(1.0)  # note() admits against this
        X = np.empty((n, encoder.dim), dtype=np.float32)
        for i in range(n):
            X[i] = encoder.encode(i, screens[i], wrams[i], button=int(last_button[i]))
        xt = torch.from_numpy(X).to(device).unsqueeze(1)  # (n, 1, dim)
        if recurrent_memory:
            out, state_t = population_forward_sparse(
                cp, xt, steps=FORWARD_STEPS, state=state_t, return_state=True
            )
        else:
            out = population_forward_sparse(cp, xt, steps=FORWARD_STEPS)
        outv = out[:, 0, :].detach().cpu().numpy()
        actions, gdx, gdy = _split_head(outv, softmax_temp)
        # R_resp bookkeeping (all rows step every round in lockstep).
        btn_hist[np.arange(n), actions] += 1
        o_sum += outv
        o_sq += (outv * outv).sum(axis=1)
        o_cnt += 1
        if probe_sink is not None and probe_quota > 0 and probe_taken < probe_quota:
            take = min(probe_quota - probe_taken, n)
            probe_sink.extend(X[k].copy() for k in range(take))
            probe_taken += take

        for i in range(n):
            # Efference copy: obs t's saccade steers obs t+1's fovea (§3.3).
            encoder.update_gaze(i, float(gdx[i]), float(gdy[i]))
            screen, wram, done, _info = envs[i].step(int(actions[i]))
            last_button[i] = actions[i]
            screens[i] = screen
            wrams[i] = wram
            dead[i] = dead[i] or bool(done)
            wave.observe(i, screen, wram)
            # Capture the state of every fresh frontier cell for later restarts.
            if goexplore is not None and wave.last_key[i] is not None:
                goexplore.note(
                    wave.last_key[i], envs[i], depth=int(base_depth[i]) + t,
                    globally_new=wave.last_new[i],
                )

        # Stream a live sample ~3 Hz (throttled internally; cheap when not due).
        if streamer is not None:
            streamer.maybe_write(
                gen, screens, wave.fitness, dead,
                obs=X, actions=actions, round_t=t,
            )

    resp = _resp_scores(btn_hist, o_sum, o_sq, o_cnt, w_resp_var)
    for i, g in enumerate(genomes):
        g.fitness = float(wave.fitness[i])
        g._resp = float(resp[i])
    return n * episode_steps


def evaluate_wave_parallel(
    genomes,
    fleet: BarrierFleet,
    archive: NoveltyArchive,
    device: torch.device,
    episode_steps: int,
    max_nodes: int,
    max_conns: int,
    streamer: LiveStreamer | None = None,
    gen: int = 0,
    novelty_mode: str = "rarity",
    novelty_floor: float = 0.01,
    goexplore: GoExplore | None = None,
    restore_prob: float = 0.5,
    prof: dict | None = None,
    cp_full=None,
    wave_offset: int = 0,
    pace: PaceController | None = None,
    waves_left: int = 0,
    spawn_out: dict[int, bytes] | None = None,
    spawn_key_out: dict[int, bytes] | None = None,
    taps: list[dict] | None = None,  # accepted for API parity; furnace-only v1
    recurrent_memory: bool = True,
    softmax_temp: float = 0.0,
    w_resp_var: float = 0.5,
    probe_sink: list | None = None,
    probe_quota: int = 0,
) -> int:
    """Parallel-barrier equivalent of :func:`evaluate_wave`.

    The emulators are stepped concurrently across the fleet's worker processes;
    each worker hashes the novelty cell key and encodes the observation locally,
    so the parent's per-round work is just one batched GPU forward plus cheap
    archive/Go-Explore bookkeeping.  Behaviour (novelty modes, Go-Explore
    cell-restore + capture) matches :func:`evaluate_wave`; the only intentional
    difference is that a globally-new cell's emulator state is captured one
    barrier round after discovery (while the worker still sits in that state),
    and captures on the final episode step are dropped.
    """
    n = len(genomes)  # sub-wave size (<= fleet.n_envs)
    wave = WaveNovelty(archive, n, mode=novelty_mode, floor=novelty_floor)

    def _mark(key: str, t0: float) -> float:
        t1 = time.perf_counter()
        if prof is not None:
            prof[key] = prof.get(key, 0.0) + (t1 - t0)
        return t1

    # Reset the fleet; a fraction of players restore from a sampled frontier cell.
    _t = time.perf_counter()
    if streamer is not None:
        streamer.set_phase("evolving", "restoring frontier")
    restore, base_depth = _sample_restores(
        goexplore, n, restore_prob, spawn_out, wave_offset, spawn_key_out
    )
    _t = _mark("go_sample", _t)
    # Release the reset round, then pack/compile WHILE the workers reset.
    fleet.reset_all_begin(restore if restore else None)
    if callable(cp_full):
        cp_full = cp_full()  # memoized full-population pack+compile
    if cp_full is not None:
        # The whole population was packed + compiled once for this generation;
        # slice this wave's rows out of it (bit-identical to packing the slice
        # on its own: padded rows are independent).
        cp = _slice_compiled(cp_full, wave_offset, wave_offset + n)
    else:
        pop = Population.from_genomes(
            genomes, max_nodes=max_nodes, max_conns=max_conns
        )
        cp = pop.compile(device)
    _t = _mark("pack_compile", _t)
    obs = fleet.reset_all_end()  # (n_envs, obs_dim)
    _t = _mark("reset_all", _t)
    if streamer is not None:
        streamer.set_phase("wave")

    dead = [False] * n
    n_envs = fleet.n_envs
    cap_flags = np.zeros(n_envs, dtype=np.uint8)
    pending: dict[int, tuple[bytes, int]] = {}
    actions_full = np.zeros(n_envs, dtype=np.int32)
    # R_resp accumulators (§6.3); all n envs step every barrier round.
    btn_hist = np.zeros((n, N_BUTTONS), dtype=np.int64)
    o_sum = np.zeros((n, N_OUT), dtype=np.float64)
    o_sq = np.zeros(n, dtype=np.float64)
    o_cnt = np.zeros(n, dtype=np.int64)
    probe_taken = 0

    import os as _os
    _prof = _os.environ.get("POKEIO_PROF") == "1"
    _t_fwd = _t_step = _t_book = 0.0

    if pace is not None:
        pace.begin_wave()

    # lockstep barrier: all envs step every round → state advances for all.
    state_t = (
        torch.zeros((n, 1, cp.M), dtype=torch.float32, device=device)
        if recurrent_memory else None
    )

    for t in range(episode_steps):
        _c0 = time.perf_counter() if _prof else 0.0
        X = np.ascontiguousarray(obs[:n], dtype=np.float32)
        xt = torch.from_numpy(X).to(device).unsqueeze(1)  # (n,1,dim)
        if recurrent_memory:
            out, state_t = population_forward_sparse(
                cp, xt, steps=FORWARD_STEPS, state=state_t, return_state=True
            )
        else:
            out = population_forward_sparse(cp, xt, steps=FORWARD_STEPS)
        outv = out[:, 0, :].detach().cpu().numpy()
        actions, gdx, gdy = _split_head(outv, softmax_temp)  # buttons + RAW saccade
        actions_full[:] = 0
        actions_full[:n] = actions
        # R_resp bookkeeping.
        btn_hist[np.arange(n), actions] += 1
        o_sum += outv
        o_sq += (outv * outv).sum(axis=1)
        o_cnt += 1
        if probe_sink is not None and probe_quota > 0 and probe_taken < probe_quota:
            take = min(probe_quota - probe_taken, n)
            probe_sink.extend(X[k].copy() for k in range(take))
            probe_taken += take
        if _prof:
            _c1 = time.perf_counter()
            _t_fwd += _c1 - _c0

        # Live pace: poll for flips (throttled stat) and, in realtime, sleep
        # to this round's absolute wall-clock deadline before releasing it.
        if pace is not None:
            pace.before_round()

        # Workers integrate the RAW saccade (tanh applied inside update_gaze)
        # into gaze BEFORE building next round's obs (§3.3/§3.4).
        obs, keys, dones, captured = fleet.step_all(
            actions_full,
            cap_flags if goexplore is not None else None,
            gaze_dx=gdx,
            gaze_dy=gdy,
        )
        if pace is not None:
            pace.after_round()
            if streamer is not None:
                rounds_left = (episode_steps - 1 - t) + waves_left * episode_steps
                streamer.set_eta(rounds_left * pace.per_round_s(), pace.round_hz)
        if _prof:
            _c2 = time.perf_counter()
            _t_step += _c2 - _c1

        # Fulfil captures requested last round (state is the cell as first reached).
        if goexplore is not None and pending:
            for idx, (key, depth) in pending.items():
                blob = captured.get(idx)
                if blob is not None:
                    goexplore.store_captured(key, blob, depth)
            pending = {}

        cap_flags[:] = 0
        cap_cand: list[tuple[int, bytes]] = []
        for i in range(n):
            key = keys[i].tobytes()
            globally_new = archive.add(key)
            prior = archive.visit(key)
            wave.observe_key(i, key, globally_new, prior)
            dead[i] = dead[i] or bool(dones[i])
            if goexplore is not None:
                # new cells + re-capture of still-rare evicted cells (else the
                # `seen` ratchet makes evicted cells permanently un-restorable)
                if not goexplore.revisit(key) and (globally_new or prior < 3):
                    cap_cand.append((i, key))
        if goexplore is not None:
            # Meter the 47 ms worker-side save_states to the capture budget.
            # Rotating start index so no env slot monopolises the tokens.
            goexplore.feed_capture_budget(1.0)
            off = t % len(cap_cand) if cap_cand else 0
            for j in range(len(cap_cand)):
                if not goexplore.admit_capture():
                    goexplore.n_throttled += len(cap_cand) - 1 - j
                    break
                i, key = cap_cand[(off + j) % len(cap_cand)]
                cap_flags[i] = 1
                pending[i] = (key, int(base_depth[i]) + t)

        if streamer is not None:
            # X is the obs batch this round's actions came from (a stable copy),
            # so the focus capture-forward reproduces the acted-on decision.
            streamer.maybe_write(
                gen, fleet.screens[:n], wave.fitness, dead,
                obs=X, actions=actions, round_t=t,
            )
        if _prof:
            _t_book += time.perf_counter() - _c2

    if _prof:
        tot = _t_fwd + _t_step + _t_book
        print(f"[prof] rounds={episode_steps} tot={tot:.2f}s  "
              f"fwd={_t_fwd/episode_steps*1000:.2f}ms  "
              f"step_barrier={_t_step/episode_steps*1000:.2f}ms  "
              f"book={_t_book/episode_steps*1000:.2f}ms/round")
    _mark("rounds", _t)

    resp = _resp_scores(btn_hist, o_sum, o_sq, o_cnt, w_resp_var)
    for i, g in enumerate(genomes):
        g.fitness = float(wave.fitness[i])
        g._resp = float(resp[i])
    return n * episode_steps


def _sample_restores(
    goexplore: GoExplore | None,
    n: int,
    restore_prob: float,
    spawn_out: dict[int, bytes] | None = None,
    wave_offset: int = 0,
    spawn_key_out: dict[int, bytes] | None = None,
) -> tuple[dict[int, bytes], np.ndarray]:
    """Pick each player's spawn: a Go-Explore frontier restore or newgame.

    Returns ``(restore, base_depth)`` — the slot->state-blob map for the fleet
    and each slot's CUMULATIVE chain depth (the restored entry's depth; 0 for
    newgame spawns). Cells discovered this episode store ``base_depth + t`` so
    restore chains accumulate distance-from-newgame instead of resetting per
    episode (which made Go-Explore's depth preference meaningless).

    ``spawn_key_out`` (optional) records the restored cell's KEY per global
    genome index — the identity the E2 per-cell baseline groups on (§6.1); two
    genomes restored from the same frontier cell share a key.
    """
    restore: dict[int, bytes] = {}
    base_depth = np.zeros(n, dtype=np.int64)
    if goexplore is not None and goexplore.size > 0:
        idxs = [i for i in range(n) if goexplore.rng.random() < restore_prob]
        for i, entry in zip(idxs, goexplore.sample_many(len(idxs))):
            restore[i] = entry.state
            base_depth[i] = entry.depth
            if spawn_key_out is not None:
                spawn_key_out[wave_offset + i] = entry.key
        goexplore.n_restores += len(restore)
        if spawn_out is not None:
            for i, blob in restore.items():
                spawn_out[wave_offset + i] = blob
    return restore, base_depth


def evaluate_wave_async(
    genomes,
    fleet: AsyncFleet,
    archive: NoveltyArchive,
    device: torch.device,
    episode_steps: int,
    max_nodes: int,
    max_conns: int,
    streamer: LiveStreamer | None = None,
    gen: int = 0,
    novelty_mode: str = "rarity",
    novelty_floor: float = 0.01,
    goexplore: GoExplore | None = None,
    restore_prob: float = 0.5,
    prof: dict | None = None,
    cp_full=None,
    wave_offset: int = 0,
    pace: PaceController | None = None,
    waves_left: int = 0,
    spawn_out: dict[int, bytes] | None = None,
    spawn_key_out: dict[int, bytes] | None = None,
    taps: list[dict] | None = None,
    recurrent_memory: bool = True,
    softmax_temp: float = 0.0,
    w_resp_var: float = 0.5,
    probe_sink: list | None = None,
    probe_quota: int = 0,
) -> int:
    """Free-running ("furnace") equivalent of :func:`evaluate_wave_parallel`.

    ``taps`` (optional): mined progress counters riding in the obs RAM tail —
    each ``{"slots": (j,) | (j, j+1), "dir": +1|-1}`` with slot indices into
    the tail. Direction-aligned positive deltas accumulate into ``g.progress``
    (reverts, e.g. blackout money-halving, count at half weight against it).

    No per-step barrier: each env advances the moment its next action arrives,
    and the parent continuously batches whichever envs are ready into one GPU
    forward. Per-env trajectories are bit-identical to the barrier engine (an
    env's action k is a function of its own obs k only); what differs is the
    ORDER envs hit the novelty archive within the wave, so rarity credit and
    Go-Explore capture ownership are timing-dependent rather than reproducible
    round-by-round. Measured 2.9x the barrier engine at the live config
    (2,797 -> 8,129 steps/s, quiet box, 2026-07 audit3).

    Realtime pace: the parent releases env i's k-th action on the absolute
    schedule ``t0 + k * period`` (period = frame_skip/60), so every emulator
    individually runs at authentic Game Boy speed; overruns re-anchor per env
    instead of burst-catching-up.
    """
    n = len(genomes)  # sub-wave size (<= fleet.n_envs)
    R = int(episode_steps)
    wave = WaveNovelty(archive, n, mode=novelty_mode, floor=novelty_floor)

    def _mark(key: str, t0: float) -> float:
        t1 = time.perf_counter()
        if prof is not None:
            prof[key] = prof.get(key, 0.0) + (t1 - t0)
        return t1

    # Reset the fleet; a fraction of players restore from a sampled frontier
    # cell (identical sampling path to the barrier engine).
    _t = time.perf_counter()
    if streamer is not None:
        streamer.set_phase("evolving", "restoring frontier")
    restore, base_depth = _sample_restores(
        goexplore, n, restore_prob, spawn_out, wave_offset, spawn_key_out
    )
    _t = _mark("go_sample", _t)
    fleet.reset_all_begin(restore if restore else None)
    if callable(cp_full):
        cp_full = cp_full()  # memoized full-population pack+compile
    if cp_full is not None:
        cp = _slice_compiled(cp_full, wave_offset, wave_offset + n)
    else:
        pop = Population.from_genomes(
            genomes, max_nodes=max_nodes, max_conns=max_conns
        )
        cp = pop.compile(device)
    _t = _mark("pack_compile", _t)
    fleet.reset_all_end()
    _t = _mark("reset_all", _t)
    if streamer is not None:
        streamer.set_phase("wave")

    # Optional CUDA-graph forward: collapses the ~60 kernel launches of the
    # 4-hop sparse forward + argmax into one replay (~3ms -> <1ms). Opt-in
    # until it has a quiet-box A/B (POKEIO_CUDAGRAPH=1).
    graph_fwd = None
    if os.environ.get("POKEIO_CUDAGRAPH") == "1" and device.type == "cuda":
        # The streamer's pump thread runs the champion-showcase forward on
        # this SAME device; a kernel launched mid-capture aborts the capture
        # ("operation failed due to a previous error during capture", races
        # per launch). Hold the streamer lock for the ~50ms capture so the
        # showcase sits out, and fall back to eager on any capture failure —
        # a lost 2ms/round optimization must never kill a training run.
        _cap_lock = streamer._lock if streamer is not None else None
        try:
            if _cap_lock is not None:
                _cap_lock.acquire()
            torch.cuda.set_device(device)
            x_static = torch.zeros((n, 1, fleet.obs_dim), dtype=torch.float32,
                                   device=device)
            warm = torch.cuda.Stream(device)
            warm.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warm):
                for _ in range(3):
                    _o = population_forward_sparse(cp, x_static, steps=FORWARD_STEPS)
                    _ = _o[:, 0, :N_BUTTONS].argmax(dim=1)
            torch.cuda.current_stream(device).wait_stream(warm)
            torch.cuda.synchronize(device)
            # recurrent memory needs the node-state tensor as a static graph
            # in/out so it can be fed back each replay (see the loop below).
            st_static = (
                torch.zeros((n, 1, cp.M), dtype=torch.float32, device=device)
                if recurrent_memory else None
            )
            if recurrent_memory:
                with torch.cuda.stream(warm):
                    for _ in range(3):
                        _o, _s = population_forward_sparse(
                            cp, x_static, steps=FORWARD_STEPS,
                            state=st_static, return_state=True,
                        )
                        _ = _o[:, 0, :N_BUTTONS].argmax(dim=1)
                torch.cuda.current_stream(device).wait_stream(warm)
                torch.cuda.synchronize(device)
            _graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(_graph):
                if recurrent_memory:
                    _o, _st_out = population_forward_sparse(
                        cp, x_static, steps=FORWARD_STEPS,
                        state=st_static, return_state=True,
                    )
                else:
                    _o = population_forward_sparse(cp, x_static, steps=FORWARD_STEPS)
                # Capture the RAW output rows (not the fused argmax): the parent
                # splits buttons + saccade (dx,dy) and accumulates R_resp from
                # the same tensor, so the graph and eager paths share one seam.
                _out_static = _o[:, 0, :].contiguous()
            torch.cuda.synchronize(device)

            def graph_fwd(X_np, cur_state):
                x_static.copy_(torch.from_numpy(X_np).unsqueeze(1))
                if recurrent_memory:
                    st_static.copy_(cur_state)  # feed last step's state back in
                _graph.replay()
                # _st_out lives in the graph pool; the caller copies the fresh
                # rows it needs out (state_t[idx]=) before the next replay.
                return (
                    _out_static.cpu().numpy(),
                    _st_out if recurrent_memory else None,
                )
        except RuntimeError as e:
            print(f"[train] CUDA-graph capture failed ({e}); eager fallback",
                  flush=True)
            graph_fwd = None
            torch.cuda.synchronize(device)
        finally:
            if _cap_lock is not None:
                _cap_lock.release()

    obs_seq = fleet.arr["obs_seq"]
    obs_shm = fleet.arr["obs"]
    keys_shm = fleet.arr["keys"]
    dones_shm = fleet.arr["dones"]
    actions_shm = fleet.arr["actions"]
    act_seq = fleet.arr["act_seq"]
    cap_flag = fleet.arr["cap_flag"]
    cap_done = fleet.arr["cap_done"]
    cap_state = fleet.arr["cap_state"]
    cap_len = fleet.arr["cap_len"]

    dead = [False] * n
    acted = np.full(n, -1, dtype=np.int64)   # newest obs index acted on
    booked = np.zeros(n, dtype=np.int64)     # obs index bookkept through
    pending: dict[int, tuple[bytes, int]] = {}
    last_X = None
    last_actions = np.zeros(n, dtype=np.int32)
    # R_resp accumulators (§6.3): only rows that actually stepped (ready) tick.
    btn_hist = np.zeros((n, N_BUTTONS), dtype=np.int64)
    o_sum = np.zeros((n, N_OUT), dtype=np.float64)
    o_sq = np.zeros(n, dtype=np.float64)
    o_cnt = np.zeros(n, dtype=np.int64)
    probe_taken = 0
    # Per-env recurrent node-state (N,1,M), zeroed at wave start (= episode
    # start). Advances ONLY for envs that actually consume an action this
    # cycle — the batched forward computes new state for all n envs, but a
    # non-ready env didn't step, so its memory must not tick (audit EVO#4).
    state_t = (
        torch.zeros((n, 1, cp.M), dtype=torch.float32, device=device)
        if recurrent_memory else None
    )

    # realtime pacing state (absolute per-wave schedule, per-env re-anchor)
    if pace is not None:
        pace.begin_wave()
    period = pace.period if pace is not None else 0.0
    next_due = np.zeros(n, dtype=np.float64)  # env i's next action due time
    t_anchor: float | None = None

    # aggregate step-rate EMA for the dashboard ETA
    rate = 0.0
    rate_prev_total = 0
    rate_prev_ts = time.perf_counter()
    alive_check = 0

    # mined progress-counter scoring state (vectorized once per parent cycle)
    progress = np.zeros(n, dtype=np.float64)
    if taps:
        tail0 = fleet.obs_dim - fleet.obs_ram
        tap_cols = np.array(
            [tail0 + j for tp in taps for j in tp["slots"]], dtype=np.intp
        )
        # (S slots -> T taps) little-endian byte combiner
        _S, _T = len(tap_cols), len(taps)
        tap_comb = np.zeros((_S, _T), dtype=np.float64)
        s = 0
        for ti, tp in enumerate(taps):
            for b in range(len(tp["slots"])):
                tap_comb[s, ti] = 256.0 ** b
                s += 1
        tap_dirs = np.array([float(tp["dir"]) for tp in taps])
        tap_scale = np.array(
            [1.0 / (256.0 ** len(tp["slots"]) - 1.0) for tp in taps]
        )
        tap_prev = np.full((n, _T), np.nan)

    fleet.begin_wave(n, R)

    while True:
        snap = obs_seq[:n].copy()

        # ---- mined-counter progress: one vectorized pass over the envs that
        # published new obs this cycle (dir-aligned gains count full, reverts
        # — blackouts, spent money — count half against).
        changed = np.nonzero(snap > booked)[0]
        if taps and changed.size:
            tail = np.rint(obs_shm[np.ix_(changed, tap_cols)] * 255.0)
            vals = tail @ tap_comb  # (m, T) counter values
            prev = tap_prev[changed]
            d = (vals - prev) * tap_dirs * tap_scale
            d[np.isnan(d)] = 0.0
            progress[changed] += (
                np.clip(d, 0.0, None) + 0.5 * np.clip(d, None, 0.0)
            ).sum(axis=1)
            tap_prev[changed] = vals

        # ---- bookkeeping: every obs published since the last cycle.
        # (An env advances at most one step per parent cycle, so nothing skips.)
        for i in changed:
            ii = int(i)
            for k in range(int(booked[ii]) + 1, int(snap[ii]) + 1):
                key = keys_shm[ii].tobytes()
                globally_new = archive.add(key)
                prior = archive.visit(key)
                wave.observe_key(ii, key, globally_new, prior)
                dead[ii] = dead[ii] or bool(dones_shm[ii])
                if goexplore is not None:
                    # n booked obs ~= one swarm round of capture budget.
                    goexplore.feed_capture_budget(1.0 / n)
                if (
                    goexplore is not None and k < R
                    and not cap_flag[ii] and not cap_done[ii]
                    and not goexplore.revisit(key)
                    # capture genuinely-new cells AND re-capture still-rare
                    # cells whose stored state was evicted (or that predate
                    # go-explore) — without this, `seen` ratchets every
                    # evicted cell permanently un-restorable.
                    and (globally_new or prior < 3)
                    and goexplore.admit_capture()
                ):
                    # Worker captures obs k's state before applying its action.
                    # (While a capture is outstanding, further discoveries by
                    # the same env are not re-flagged — a rare, harmless drop.)
                    cap_flag[ii] = 1
                    pending[ii] = (key, int(base_depth[ii]) + k)
            booked[ii] = snap[ii]
        if goexplore is not None and pending:
            # snapshot: workers flip cap_done concurrently, and np.nonzero on
            # a mutating shm array raises "number of non-zero array elements
            # changed during function execution" (took down a live run).
            done_snap = cap_done[:n].copy()
            for i in np.nonzero(done_snap)[0]:
                ii = int(i)
                got = pending.pop(ii, None)
                if got is not None:
                    key, depth = got
                    goexplore.store_captured(
                        key, bytes(cap_state[ii, : int(cap_len[ii])]), depth
                    )
                cap_done[ii] = 0

        # ---- issue actions for every env whose newest obs is unanswered
        ready = (snap > acted) & (snap < R)
        realtime = pace is not None and pace.mode == "realtime"
        now = time.perf_counter()
        if realtime:
            if t_anchor is None:
                t_anchor = now
                next_due[:] = now
            ready &= next_due <= now
        else:
            t_anchor = None
        if ready.any():
            X = np.ascontiguousarray(obs_shm[:n], dtype=np.float32)
            if graph_fwd is not None:
                outv, new_state = graph_fwd(X, state_t)  # outv: (n, N_OUT)
            elif recurrent_memory:
                xt = torch.from_numpy(X).to(device).unsqueeze(1)
                out, new_state = population_forward_sparse(
                    cp, xt, steps=FORWARD_STEPS, state=state_t, return_state=True
                )
                outv = out[:, 0, :].detach().cpu().numpy()
            else:
                xt = torch.from_numpy(X).to(device).unsqueeze(1)
                out = population_forward_sparse(cp, xt, steps=FORWARD_STEPS)
                outv = out[:, 0, :].detach().cpu().numpy()
                new_state = None
            # Split the tanh head: buttons (argmax) + RAW saccade (dx,dy). The
            # worker applies tanh inside update_gaze, so pass the raw floats.
            acts, gdx, gdy = _split_head(outv, softmax_temp)
            idx = np.nonzero(ready)[0]
            # R_resp: accumulate only the rows that actually consume an action.
            btn_hist[idx, acts[idx]] += 1
            o_sum[idx] += outv[idx]
            o_sq[idx] += (outv[idx] * outv[idx]).sum(axis=1)
            o_cnt[idx] += 1
            if (
                probe_sink is not None and probe_quota > 0
                and probe_taken < probe_quota
            ):
                take = min(probe_quota - probe_taken, len(idx))
                probe_sink.extend(X[int(k)].copy() for k in idx[:take])
                probe_taken += take
            # submit_actions publishes gaze + buttons FIRST, act_seq LAST (§3.4).
            fleet.submit_actions(idx, acts[idx], gdx[idx], gdy[idx], snap[idx])
            acted[idx] = snap[idx]
            # advance memory ONLY for envs that stepped (ready rows)
            if new_state is not None:
                state_t[torch.from_numpy(idx).to(device)] = new_state[
                    torch.from_numpy(idx).to(device)
                ]
            last_X, last_actions = X, acts
            if realtime:
                nd = next_due[idx] + period
                overrun = nd < now  # stalled past a full period: re-anchor
                nd[overrun] = now + period
                next_due[idx] = nd
        elif bool((snap >= R).all()) and not pending:
            break
        else:
            # idle: nothing ready (all envs mid-step, or paced). The 20us nap
            # keeps this core polite; the GPU sync above busy-waits anyway.
            time.sleep(1e-3 if realtime else 2e-5)

        # ---- periodic upkeep (pace flips, liveness, live feed, ETA)
        alive_check += 1
        if alive_check % 512 == 0:
            fleet.check_alive()
        if pace is not None:
            pace.poll()
        now = time.perf_counter()
        if now - rate_prev_ts >= 0.25:
            total = int(snap.sum())
            inst = (total - rate_prev_total) / (now - rate_prev_ts)
            rate = inst if rate <= 0 else 0.9 * rate + 0.1 * inst
            rate_prev_total, rate_prev_ts = total, now
            if pace is not None and n > 0:
                # per-env step cadence == the barrier engine's "round rate"
                pace.round_hz = rate / n
        if streamer is not None and last_X is not None:
            steps_left = int((R - snap).clip(min=0).sum()) + waves_left * R * n
            if rate > 0:
                streamer.set_eta(steps_left / rate, rate / max(1, n))
            streamer.maybe_write(
                gen, fleet.screens[:n], wave.fitness, dead,
                obs=last_X, actions=last_actions, round_t=int(snap.min()),
            )

    fleet.end_wave()
    _mark("rounds", _t)

    resp = _resp_scores(btn_hist, o_sum, o_sq, o_cnt, w_resp_var)
    for i, g in enumerate(genomes):
        g.fitness = float(wave.fitness[i])
        g.progress = float(progress[i])  # mined-counter advancement (0 w/o taps)
        g._resp = float(resp[i])
    return n * R


def _quantile_ranks(vals: np.ndarray) -> np.ndarray:
    """(rank+0.5)/n quantiles with exact ties sharing their average rank."""
    order = vals.argsort(kind="stable")
    ranks = np.empty(len(vals), dtype=np.float64)
    ranks[order] = np.arange(len(vals), dtype=np.float64)
    _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.shape[0], dtype=np.float64)
    np.add.at(sums, inv, ranks)
    return (sums[inv] / cnt[inv] + 0.5) / len(vals)


def _cell_baseline_advantage(
    raw_f: np.ndarray,
    restored: set[int],
    spawn_keys: dict[int, bytes] | None,
) -> np.ndarray:
    """E2 per-cell leave-one-out advantage (§6.1) — the difference-reward that
    cancels cross-cell spawn-value luck.

    For each restore cell ``c`` with member set ``M_c``:
      ``|M_c| >= 2``: ``A_i = f_i − (Σ_{j∈M_c} f_j − f_i)/(|M_c|−1)`` (leave-one-out);
      ``|M_c| == 1``: ``A_i = 0`` (no counterfactual);
      newgame (shared fixed spawn): ``A_i = f_i`` (no cross-spawn luck to cancel).

    Two genomes restored to the SAME cell with identical policy get equal ``A_i``,
    so a screen-blind constant at a rich spawn can no longer out-rank a seeing
    policy at a poor one.  Grouping is by the Go-Explore cell KEY (``spawn_keys``);
    a restored genome with no recorded key is treated as its own singleton.
    """
    raw_f = np.asarray(raw_f, dtype=np.float64)
    A = raw_f.copy()  # newgame default: A_i = f_i
    groups: dict = {}
    for i in restored:
        key = (spawn_keys or {}).get(i)
        if key is None:
            key = ("__singleton__", int(i))  # unknown cell -> own singleton
        groups.setdefault(key, []).append(int(i))
    for members in groups.values():
        m = len(members)
        if m >= 2:
            s = float(sum(raw_f[j] for j in members))
            for i in members:
                A[i] = raw_f[i] - (s - raw_f[i]) / (m - 1)
        else:
            A[members[0]] = 0.0
    return A


def cohort_rank_normalize(
    genomes,
    restored: set[int],
    progress_weight: float = 0.0,
    *,
    spawn_keys: dict[int, bytes] | None = None,
    gate: np.ndarray | None = None,
    resp: np.ndarray | None = None,
    emp: np.ndarray | None = None,
    w_resp: float = 0.0,
    w_emp: float = 0.0,
    restore_baseline: bool = False,
    blind_gate: bool = False,
) -> None:
    """Replace ``g.fitness`` with the coupled within-cohort selection fitness (§6.5).

    Ledger 2 (policy fitness). Within each spawn cohort (restored / newgame), map
    each component to ``(rank+0.5)/n`` quantiles via :func:`_quantile_ranks` and
    combine::

        policy_i = gate_i · q(A_i) + w_resp · q(R_resp_i) + w_emp · q(Emp_i)
        sel_i    = (1−w_prog)·q(policy_i) + w_prog·q(progress_i)   if progress live
                 = q(policy_i)                                     otherwise

    where ``A_i`` is the E2 per-cell leave-one-out advantage (``restore_baseline``;
    replaces raw novelty ``f_i`` as the task-credit input), ``gate_i`` the E3
    blind-ablation multiplier (``blind_gate``; ~0 for screen-blind policies),
    ``R_resp_i`` responsiveness (``g._resp`` from the wave), and ``Emp_i``
    empowerment (Phase 1+, ``emp``/``w_emp``; 0-effect in Phase 0).

    Backward compatible: with all keyword extras at their defaults this reduces to
    the previous plain within-cohort rank-normalize (``q(A_i)=q(f_i)`` idempotent
    under re-ranking), so existing callers and the progress blend are unchanged.
    """
    n = len(genomes)
    raw_f = np.array([g.fitness for g in genomes], dtype=np.float64)
    # E2: A_i replaces raw f_i as the novelty/task-credit input.
    A = (
        _cell_baseline_advantage(raw_f, restored, spawn_keys)
        if restore_baseline else raw_f
    )
    if resp is None:
        resp = np.array(
            [float(getattr(g, "_resp", 0.0)) for g in genomes], dtype=np.float64
        )
    else:
        resp = np.asarray(resp, dtype=np.float64)
    gate_v = None if gate is None else np.asarray(gate, dtype=np.float64)
    emp_v = None if emp is None else np.asarray(emp, dtype=np.float64)

    cohorts = (
        [i for i in range(n) if i in restored],
        [i for i in range(n) if i not in restored],
    )
    w = float(progress_weight)
    blend = w > 0.0 and any(
        float(getattr(g, "progress", 0.0)) != 0.0 for g in genomes
    )
    for idxs in cohorts:
        if not idxs:
            continue
        ia = np.asarray(idxs, dtype=np.intp)
        qA = _quantile_ranks(A[ia])
        # gate is the DOMINANT multiplier on the task-credit quantile.
        policy = (gate_v[ia] * qA) if (blind_gate and gate_v is not None) else qA
        if w_resp:
            policy = policy + float(w_resp) * _quantile_ranks(resp[ia])
        if w_emp and emp_v is not None:
            policy = policy + float(w_emp) * _quantile_ranks(emp_v[ia])
        qpolicy = _quantile_ranks(policy)
        if blend:
            p = np.array(
                [float(getattr(genomes[i], "progress", 0.0)) for i in idxs],
                dtype=np.float64,
            )
            sel = (1.0 - w) * qpolicy + w * _quantile_ranks(p)
        else:
            sel = qpolicy
        for j, i in enumerate(idxs):
            genomes[i].fitness = float(sel[j])


def idle_autonomous_bytes(
    rom_path: str,
    reset_state: str,
    frame_skip: int = 24,
    steps: int = 300,
) -> set[int]:
    """Full-resolution WRAM addresses that change with ZERO input.

    Music pointers, timers, and other free-running engine state advance
    monotonically and consistently across rollouts — to the progress-counter
    miner they look exactly like progress (the smoke run promoted three audio
    channel pointers to fitness taps). Anything that moves while the agent
    does nothing cannot be a progress counter; drop it from candidacy.
    Game-agnostic and deterministic.
    """
    env = PokeEnv(rom_path=rom_path, frame_skip=frame_skip)
    try:
        env.reset(reset_state)
        prev = env.raw_wram().copy()
        auto = np.zeros(prev.size, dtype=bool)
        for _ in range(steps):
            env.release_all()
            env.tick_frames(frame_skip)
            cur = env.raw_wram()
            auto |= cur != prev
            prev = cur.copy()
        addrs = {0xC000 + int(i) for i in np.nonzero(auto)[0]}
        print(f"[miner] idle-autonomous bytes excluded: {len(addrs)}")
        return addrs
    finally:
        env.close()


def calibrate_wram_mask(
    rom_path: str,
    reset_state: str,
    frame_skip: int = 24,
    wram_stride: int = 64,
    wram_levels: int = 16,
    steps: int = 400,
    runs: int = 2,
    threshold: float = 0.02,
    seed: int = 0,
) -> tuple[int, ...]:
    """Learn which strided WRAM bytes to keep in the novelty cell key.

    Runs a seeded random policy from the reset state and measures each strided
    byte's QUANTIZED change rate (matching the digest's quantization).  Bytes
    that flip more than ``threshold`` of steps are dropped: on Gen-1 those are
    the tile-map buffer (a copy of the on-screen tiles — redundant with the
    screen digest and churning at every camera scroll) and audio scratch.
    Game-agnostic: nothing here names an address; any game's per-frame buffers
    get masked the same way.  Deterministic given (rom, state, seed).
    """
    env = PokeEnv(rom_path=rom_path, frame_skip=frame_skip)
    try:
        rng = np.random.default_rng(seed)
        step_q = max(1, 256 // wram_levels)
        changes: np.ndarray | None = None
        total = 0
        for _ in range(runs):
            env.reset(reset_state)
            prev = None
            for _t in range(steps):
                env.step(int(rng.integers(0, N_BUTTONS)))  # buttons only (0..8)
                q = env.raw_wram()[::wram_stride] // step_q
                if prev is not None:
                    if changes is None:
                        changes = np.zeros(q.shape[0], dtype=np.int64)
                    changes += q != prev
                    total += 1
                prev = q
        assert changes is not None
        keep = np.nonzero(changes <= total * threshold)[0]
        dropped = changes.shape[0] - keep.shape[0]
        print(
            f"[train] wram churn-mask: keeping {keep.shape[0]}/{changes.shape[0]} "
            f"strided bytes ({dropped} churners masked, probe {total} steps)"
        )
        return tuple(int(i) for i in keep)
    finally:
        env.close()


def replay_champion(
    genome,
    env,
    encoder: FovealEncoder,
    archive: NoveltyArchive,
    device: torch.device,
    steps: int,
    writer: TelemetryWriter,
    max_nodes: int,
    max_conns: int,
    reset_state: str,
    spawn_state: bytes | None = None,
    record_wram: bool = False,
    recurrent_memory: bool = True,
    softmax_temp: float = 0.0,
) -> np.ndarray | None:
    """Replay the generation champion solo and log a few ChampionSteps.

    ``spawn_state``: replay from the champion's actual eval spawn (frontier
    restore) instead of newgame — the trajectory then exercises the game
    context the champion actually earned its fitness in, which is exactly the
    data the progress-counter miner needs (battles, dialogue, shops).
    ``record_wram``: return the (steps, 8192) WRAM trace for the miner.

    The champion drives the movable fovea: each step's saccade steers the next
    step's fovea crop through ``encoder.update_gaze`` (parent path, env slot 0).
    """
    pop = Population.from_genomes([genome], max_nodes=max_nodes, max_conns=max_conns)
    cp = pop.compile(device)
    encoder.reset(0)  # gaze -> centre, motion -> 0.5 (solo replay uses env slot 0)
    if spawn_state is not None:
        env.load_state(spawn_state)
        screen = env.reset(None)  # clear held input + settle a frame
    else:
        screen = env.reset(reset_state)
    wram = env.raw_wram()
    trace = np.empty((steps, wram.size), dtype=np.uint8) if record_wram else None
    state = None  # recurrent node-state carried across the replay episode
    last_button = 8  # NOOP until the first action lands
    for t in range(steps):
        x = encoder.encode(0, screen, wram, button=int(last_button))
        xt = torch.from_numpy(x[None, :]).to(device).unsqueeze(1)  # (1,1,dim)
        if recurrent_memory:
            out, state = population_forward_sparse(
                cp, xt, steps=FORWARD_STEPS, state=state, return_state=True
            )
        else:
            out = population_forward_sparse(cp, xt, steps=FORWARD_STEPS)
        outv = out[0, 0, :].detach().cpu().numpy()[None, :]  # (1, N_OUT)
        acts, gdx, gdy = _split_head(outv, softmax_temp)
        action = int(acts[0])
        encoder.update_gaze(0, float(gdx[0]), float(gdy[0]))  # steer next fovea
        last_button = action
        screen, wram, _done, info = env.step(action)
        time.sleep(0)  # cooperative GIL handoff for the live pump thread
        # READ-ONLY: the replay is telemetry, not evaluation — writing here
        # would burn fresh keys into the archive with no state capture,
        # making those cells permanently un-restorable (audit REWARD#5).
        is_new = not archive.contains(screen, wram)
        writer.write_champion_step(
            ChampionStep(
                step=t,
                action=action,
                screen_ref=_screen_ref(screen),
                ram_tap={"map_id": int(info.get("map_id", 0))},
                reward_components={"novelty": 1.0 if is_new else 0.0},
            )
        )
        if trace is not None:
            trace[t] = wram
    return trace


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
    prof: dict | None = None,
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
    _t0 = time.perf_counter()
    _n_dist = 0
    hi = 0.0
    for a in range(len(sample)):
        for b in range(a + 1, len(sample)):
            d = compatibility_distance(sample[a], sample[b], c1, c2, c3)
            _n_dist += 1
            if d > hi:
                hi = d
    _t1 = time.perf_counter()
    if prof is not None:
        prof["spec_fit_bound"] = prof.get("spec_fit_bound", 0.0) + (_t1 - _t0)
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
    if prof is not None:
        prof["spec_fit_search"] = (
            prof.get("spec_fit_search", 0.0) + (time.perf_counter() - _t1)
        )
        prof["spec_fit_bound_calls"] = _n_dist
    return float(best)


# --------------------------------------------------------------------------
# vectorized speciation (pairwise compatibility matrix)
# --------------------------------------------------------------------------
def compat_matrix(
    genomes,
    c1: float,
    c2: float,
    c3: float,
    device: torch.device,
    normalize_min: int = 20,
) -> np.ndarray | None:
    """Full pairwise NEAT compatibility-distance matrix, vectorized.

    Semantics match :func:`pokeio.evo.ops.compatibility_distance` pair-by-pair:
    the matching / disjoint / excess counts are integer-exact; the mean matching
    weight difference (W̄) is summed in innovation-sorted order on the GPU, which
    can differ from the scalar ``np.mean(set-ordered list)`` by float rounding in
    the last ulps.  Returns ``None`` if any genome has no connections (callers
    fall back to the scalar path).
    """
    n = len(genomes)
    if n == 0:
        return None
    counts = [len(g.conns) for g in genomes]
    if min(counts) == 0:
        return None
    innovs = [
        np.fromiter(g.conns.keys(), dtype=np.int64, count=counts[i])
        for i, g in enumerate(genomes)
    ]
    all_innov = np.unique(np.concatenate(innovs))
    U = int(all_innov.size)
    P = np.zeros((n, U), dtype=bool)
    W = np.zeros((n, U), dtype=np.float64)
    last = np.zeros(n, dtype=np.int64)  # column of each genome's max innov
    for i, g in enumerate(genomes):
        cols = np.searchsorted(all_innov, innovs[i])
        P[i, cols] = True
        W[i, cols] = np.fromiter(
            (c.weight for c in g.conns.values()), dtype=np.float64, count=counts[i]
        )
        last[i] = int(cols.max())

    cnt = P.sum(axis=1).astype(np.int64)  # (n,)
    # matching-gene counts via 0/1 matmul (exact in fp32 for counts < 2^24)
    Pt = torch.from_numpy(P).to(device)
    Pf = Pt.to(torch.float32)
    match = (Pf @ Pf.T).round().to(torch.int64).cpu().numpy()  # (n, n)
    only = cnt[:, None] + cnt[None, :] - 2 * match  # |k1 ^ k2|

    # excess: genes with innov strictly greater than min(max1, max2).
    csum = np.cumsum(P, axis=1, dtype=np.int64)  # inclusive counts <= column
    cut = np.minimum(last[:, None], last[None, :])  # (n, n) cutoff column
    ii = np.arange(n)
    le_a = csum[ii[:, None], cut]  # a's genes with innov <= cutoff
    le_b = csum[ii[None, :], cut]  # b's genes with innov <= cutoff
    excess = (cnt[:, None] - le_a) + (cnt[None, :] - le_b)
    disjoint = only - excess

    # W̄: mean |w1 - w2| over matching genes (row-blocked on the GPU).
    Wt = torch.from_numpy(W).to(device)
    wsum = torch.empty((n, n), dtype=torch.float64, device=device)
    for a in range(n):
        d = (Wt[a].unsqueeze(0) - Wt).abs()
        d = d * (Pt[a].unsqueeze(0) & Pt).to(torch.float64)
        wsum[a] = d.sum(dim=1)
    wsum_np = wsum.cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        wbar = np.where(match > 0, wsum_np / np.maximum(match, 1), 0.0)

    N = np.maximum(cnt[:, None], cnt[None, :]).astype(np.float64)
    N[N < normalize_min] = 1.0
    D = c1 * excess / N + c2 * disjoint / N + c3 * wbar
    np.fill_diagonal(D, 0.0)
    return D


def _greedy_species_count_mat(Dl: list[list[float]], thr: float) -> int:
    """Matrix-backed equivalent of :func:`_greedy_species_count`."""
    reps: list[int] = []
    for i in range(len(Dl)):
        row = Dl[i]
        placed = False
        for r in reps:
            if row[r] < thr:
                placed = True
                break
        if not placed:
            reps.append(i)
    return len(reps)


def fit_species_threshold_mat(
    D: np.ndarray,
    n: int,
    target: int,
    rng: np.random.Generator,
    sample_cap: int = 96,
    iters: int = 24,
) -> float:
    """Matrix-backed :func:`fit_species_threshold` (same rng consumption)."""
    if n <= 1:
        return 1.0
    if n > sample_cap:
        idx = rng.choice(n, size=sample_cap, replace=False)
        sidx = np.array([int(i) for i in idx], dtype=np.int64)
    else:
        sidx = np.arange(n, dtype=np.int64)
    Ds = D[np.ix_(sidx, sidx)]
    hi = float(Ds.max())
    if hi <= 0.0:
        return 1.0
    lo = 0.0
    hi = hi + 1e-6
    target = max(2, min(target, len(sidx)))
    best = hi
    Dl = Ds.tolist()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        k = _greedy_species_count_mat(Dl, mid)
        if k >= target:
            best = mid
            lo = mid
        else:
            hi = mid
    return float(best)


def assign_species_mat(spec, genomes, D: np.ndarray, rng: np.random.Generator):
    """Matrix-backed :meth:`Speciation.assign` for the cleared-reps case.

    Requires ``spec.reps`` to be empty on entry (the auto-species path clears it
    each generation), so every representative is a member of ``genomes`` and its
    distances are rows of ``D``.  Consumes rng identically to the original
    (only the final rep-refresh draws).
    """
    assert not spec.reps, "assign_species_mat requires cleared reps"
    Dl = D.tolist()
    thr = spec.threshold
    species: dict[int, list] = {}
    rep_idx: dict[int, int] = {}  # sid -> genome index of the live rep
    for i, g in enumerate(genomes):
        row = Dl[i]
        placed = False
        for sid, ri in rep_idx.items():
            if row[ri] < thr:
                species.setdefault(sid, []).append(g)
                g.species_id = sid
                placed = True
                break
        if not placed:
            sid = spec._next
            spec._next += 1
            rep_idx[sid] = i
            species.setdefault(sid, []).append(g)
            g.species_id = sid
    # refresh reps to a random current member; drop empty species (as original).
    spec.reps = {
        sid: members[int(rng.integers(len(members)))].copy()
        for sid, members in species.items()
    }
    return species


def _relabel_species(species, prev_reps, spec):
    """Give this generation's fresh clusters STABLE cross-generation ids.

    ``assign_species_mat`` re-clusters from scratch every generation and mints
    fresh sids, so species had no identity over time — which made stagnation
    tracking structurally impossible (audit EVO#2). Match each new cluster to
    the closest previous representative within the compat threshold (greedy,
    biggest clusters claim first) and reuse that sid; unmatched clusters keep
    their fresh sid (a genuinely new species). ~n_species^2 scalar distance
    calls per generation — trivial next to the vectorized NxN matrix.
    """
    if not prev_reps:
        return species
    out: dict[int, list] = {}
    used: set[int] = set()
    for sid, members in sorted(species.items(), key=lambda kv: -len(kv[1])):
        rep = members[0]
        best_sid, best_d = None, None
        for psid, prep in prev_reps.items():
            if psid in used:
                continue
            d = compatibility_distance(rep, prep, spec.c1, spec.c2, spec.c3)
            if best_d is None or d < best_d:
                best_sid, best_d = psid, d
        if best_sid is not None and best_d < spec.threshold:
            used.add(best_sid)
            nsid = best_sid
        else:
            nsid = sid
        out[nsid] = members
        for g in members:
            g.species_id = nsid
    return out


# --------------------------------------------------------------------------
# fast reproduction (rng-stream-identical to pokeio.evo.ops.reproduce)
# --------------------------------------------------------------------------
# These reimplement ops.crossover / Genome.copy / ops.mutate_genome with the
# EXACT same rng call sequence and the exact same resulting genomes, but build
# gene objects with direct constructors instead of dataclasses.replace and bind
# rng methods locally (the originals spend most of their time in attribute
# lookups + `replace`).  Structural mutations (add_conn / add_node / toggle) are
# rare and delegate to the ops originals unchanged.
def _fast_copy(g):
    from pokeio.evo.genome import ConnGene as _CG
    from pokeio.evo.genome import Genome as _G
    from pokeio.evo.genome import NodeGene as _NG

    return _G(
        n_in=g.n_in,
        n_out=g.n_out,
        nodes={k: _NG(v.id, v.type, v.act, v.bias) for k, v in g.nodes.items()},
        conns={
            k: _CG(c.in_id, c.out_id, c.weight, c.enabled, c.innov)
            for k, c in g.conns.items()
        },
        fitness=g.fitness,
        age=g.age,
        species_id=g.species_id,
    )


def _fast_crossover(p1, p2, rng, disabled_inherit_prob: float = 0.75):
    from pokeio.evo.genome import ConnGene as _CG
    from pokeio.evo.genome import Genome as _G
    from pokeio.evo.genome import NodeGene as _NG

    if p2.fitness > p1.fitness:
        p1, p2 = p2, p1
    elif p1.fitness == p2.fitness and len(p2.conns) < len(p1.conns):
        p1, p2 = p2, p1

    child = _G(n_in=p1.n_in, n_out=p1.n_out)
    cc = child.conns
    p2c = p2.conns
    rr = rng.random
    for innov, c1 in p1.conns.items():
        c2 = p2c.get(innov)
        if c2 is not None:
            src = c1 if rr() < 0.5 else c2
            en = True
            if (not c1.enabled or not c2.enabled) and rr() < disabled_inherit_prob:
                en = False
            cc[innov] = _CG(src.in_id, src.out_id, src.weight, en, innov)
        else:
            cc[innov] = _CG(c1.in_id, c1.out_id, c1.weight, c1.enabled, innov)

    needed: set[int] = set()
    for c in cc.values():
        needed.add(c.in_id)
        needed.add(c.out_id)
    for nid in list(p1.input_ids()) + [p1.bias_id] + list(p1.output_ids()):
        needed.add(nid)
    p1n = p1.nodes
    p2n = p2.nodes
    cn = child.nodes
    for nid in needed:
        ng = p1n.get(nid) or p2n.get(nid)
        if ng is not None:
            cn[nid] = _NG(ng.id, ng.type, ng.act, ng.bias)
    return child


def _fast_perturb(g, rng, rates) -> None:
    # rng.normal(0.0, s) computes `0.0 + s * standard_normal()` in C; for any
    # finite z, `0.0 + s*z == s*z` bit-for-bit, and both consume the same raw
    # stream — so `s * standard_normal()` is a drop-in with ~30% less call
    # overhead (verified bit-equal over 200k draws).
    rr = rng.random
    sn = rng.standard_normal
    p_reset = rates.weight_reset_prob
    s_reset = rates.weight_reset_scale
    s_pert = rates.weight_perturb_sigma
    clamp = rates.weight_clamp
    for c in g.conns.values():
        if rr() < p_reset:
            c.weight = float(s_reset * sn())
        else:
            c.weight += float(s_pert * sn())
        if clamp > 0.0:
            # clamp consumes no rng draws — stream stays identical to ops.py
            if c.weight > clamp:
                c.weight = clamp
            elif c.weight < -clamp:
                c.weight = -clamp


def _fast_mutate(g, tracker, rng, rates, weight_scale: float = 1.0):
    from pokeio.evo.ops import (
        _HIDDEN_ACTS,
        mutate_add_connection,
        mutate_add_node,
        mutate_toggle,
    )
    from pokeio.evo.genome import HIDDEN as _HID
    from pokeio.evo.genome import OUTPUT as _OUT

    if rng.random() < rates.weight:
        _fast_perturb(g, rng, rates)
    if rng.random() < rates.add_conn:
        mutate_add_connection(g, tracker, rng, rates, weight_scale)
    if rng.random() < rates.add_node:
        mutate_add_node(g, tracker, rng)
    if rng.random() < rates.toggle:
        mutate_toggle(g, rng)
    if rates.mutate_act and rng.random() < rates.mutate_act:
        hids = [nid for nid, ng in g.nodes.items() if ng.type in (_HID, _OUT)]
        if hids:
            nid = int(rng.choice(hids))
            g.nodes[nid].act = _HIDDEN_ACTS[int(rng.integers(len(_HIDDEN_ACTS)))]
    return g


def fast_reproduce(
    genomes,
    species,
    tracker,
    rng,
    rates,
    pop_size: int,
    tournament_size: int = 3,
    elitism: int = 1,
    crossover_rate: float = 0.75,
    fitness_sharing: bool = True,
    weight_scale: float = 1.0,
    survival_threshold: float = 0.4,
    c3: float = 0.4,
):
    """Drop-in for :func:`pokeio.evo.ops.reproduce` (same rng stream + results)."""
    from pokeio.evo.ops import tournament_select

    min_fit = min(g.fitness for g in genomes)
    shift = -min_fit if min_fit < 0 else 0.0

    species_adj: dict[int, float] = {}
    for sid, members in species.items():
        size = len(members)
        adj = sum(
            (g.fitness + shift) / (size if fitness_sharing else 1) for g in members
        )
        species_adj[sid] = adj

    total_adj = sum(species_adj.values())
    if total_adj <= 0:
        total_adj = 1.0
        for sid in species_adj:
            species_adj[sid] = len(species[sid])

    raw_alloc = {sid: species_adj[sid] / total_adj * pop_size for sid in species}
    alloc = {sid: int(np.floor(v)) for sid, v in raw_alloc.items()}
    remainder = pop_size - sum(alloc.values())
    frac_order = sorted(species, key=lambda s: raw_alloc[s] - alloc[s], reverse=True)
    for i in range(remainder):
        alloc[frac_order[i % len(frac_order)]] += 1

    # Diversity guards (audit EVO#2/#6): cap any one species at ~40% of the
    # population (a lucky champion's species could otherwise take ~90% in one
    # step and the adaptive threshold would cosmetically re-split its clones),
    # and guarantee every surviving species >= 1 slot so its elite is never
    # extinguished by allocation rounding alone.
    if len(alloc) > 1:
        cap = max(1, int(np.ceil(0.4 * pop_size)))
        over = [sid for sid in alloc if alloc[sid] > cap]
        if over:
            excess = sum(alloc[sid] - cap for sid in over)
            for sid in over:
                alloc[sid] = cap
            order = sorted(
                (s for s in alloc if s not in set(over)),
                key=lambda s: species_adj.get(s, 0.0), reverse=True,
            )
            i = 0
            while excess > 0 and order:
                alloc[order[i % len(order)]] += 1
                excess -= 1
                i += 1
        for sid in alloc:
            if alloc[sid] <= 0:
                donor = max(alloc, key=lambda s: alloc[s])
                if alloc[donor] > 1:
                    alloc[donor] -= 1
                    alloc[sid] = 1

    new: list = []
    for sid, members in species.items():
        n_off = alloc[sid]
        if n_off <= 0:
            continue
        ranked = sorted(members, key=lambda g: g.fitness, reverse=True)

        n_keep = min(elitism, n_off)
        for e in range(n_keep):
            if e < len(ranked):
                champ = _fast_copy(ranked[e])
                champ.age += 1
                new.append(champ)
        n_off -= n_keep

        cut = max(1, int(np.ceil(len(ranked) * survival_threshold)))
        pool = ranked[:cut]

        for _ in range(n_off):
            if len(pool) > 1 and rng.random() < crossover_rate:
                p1 = tournament_select(pool, tournament_size, rng)
                p2 = tournament_select(pool, tournament_size, rng)
                child = _fast_crossover(p1, p2, rng)
            else:
                child = _fast_copy(tournament_select(pool, tournament_size, rng))
            _fast_mutate(child, tracker, rng, rates, weight_scale)
            child.fitness = 0.0
            child.age = 0
            new.append(child)
            # Cooperative GIL handoff (sched_yield, zero idle time) so the live
            # pump thread keeps emitting through this pure-Python stretch.
            time.sleep(0)

    while len(new) < pop_size:
        base = _fast_copy(tournament_select(genomes, tournament_size, rng))
        _fast_mutate(base, tracker, rng, rates, weight_scale)
        base.fitness = 0.0
        base.age = 0
        new.append(base)
        time.sleep(0)
    return new[:pop_size]


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
    goexplore_caps_per_round: float = 4.0,
    auto_species: bool = True,
    species_target: int = 6,
    parallel: bool = True,
    envs_per_worker: int = 1,
    init_connect: str = "full",  # §8: fan-in-scaled full seed (was "sparse")
    init_k: int = 32,  # §8: sparse-fallback fan-in (was 12)
    engine: str = "furnace",
    boot_gauntlet_every: int = 10,
    boot_gauntlet_steps: int = 0,  # 0 = auto (4 * episode_steps)
    resume: bool = False,  # append to existing telemetry (checkpoint-resume)
    recurrent_memory: bool = True,  # persist node state across agent-steps
    checkpoint_every: int = -1,  # -1 = config default; 0 = off
) -> Path:
    device = pick_device(device_str)
    rng = np.random.default_rng(config.run.seed)
    # Seed torch's global RNG too (A4): weight-init / any torch sampling was
    # previously unseeded, so nominally "seed=0" runs weren't reproducible.
    torch.manual_seed(int(config.run.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.run.seed))

    # Output layout is config-driven (§3.1): 9 button logits + 2 saccade = 11.
    global N_OUT
    N_OUT = int(config.evo.n_out)

    # Active-vision spine (§2/§3): stateful foveal obs (periphery + gaze-driven
    # fovea + motion + proprio + connect-protected RAM taps). The parent encoder
    # is used for the solo champion/miner replay (env slot 0) and the serial
    # (non-parallel) wave path; fleet workers auto-select their own FovealEncoder
    # when obs_dim == 3*periph_grid^2 + 14 + n_ram. Sized to `players` so the
    # serial path can encode a whole sub-wave.
    encoder = FovealEncoder(
        max(1, players),
        periph_grid=config.vision.periph_grid,
        fovea_native_px=config.vision.fovea_native_px,
        n_ram=config.vision.obs_ram_bytes,
        saccade_gain=config.vision.saccade_gain,
        saccade_every_k=config.vision.saccade_every_k,
        episode_steps=episode_steps,
    )
    n_in = encoder.dim  # 454 for the committed defaults
    softmax_temp = float(getattr(config.evo, "softmax_temp", 0.0))
    w_resp = float(getattr(config.reward, "w_resp", 0.0))
    w_emp = float(getattr(config.reward, "w_emp", 0.0))
    w_resp_var = float(getattr(config.reward, "w_resp_var", 0.5))
    restore_baseline = bool(getattr(config.reward, "restore_baseline", True))
    blind_gate_on = bool(getattr(config.reward, "blind_gate", True))
    blind_gate_beta = float(getattr(config.reward, "blind_gate_beta", 8.0))
    blind_gate_dmin = float(getattr(config.reward, "blind_gate_dmin", 0.05))
    # Budgets: sized to the seed density + slack to grow. Sparse seeds are
    # ~(init_k+1)*N_OUT conns, so the pack/compile tensors shrink ~10x vs full.
    if init_connect == "full":
        init_conns = (n_in + 1) * N_OUT
    else:
        init_conns = (init_k + 1) * N_OUT
    max_conns = init_conns + 2048
    max_nodes = n_in + 1 + N_OUT + 256

    run_dir = Path(config.run.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Reproducibility (A4): stamp the exact code + the FULL resolved call.
    if not config.run.git_sha:
        config.run.git_sha = _git_sha()
    config.snapshot(run_dir)
    _write_resolved_run(
        run_dir,
        argv=list(sys.argv),
        train_kwargs=dict(
            gens=gens, pop_size=pop_size, players=players,
            episode_steps=episode_steps, obs_res=obs_res, run_id=run_id,
            device_str=device_str, champion_steps=champion_steps, live=live,
            novelty_mode=novelty_mode, goexplore=goexplore,
            restore_prob=restore_prob, goexplore_capacity=goexplore_capacity,
            goexplore_caps_per_round=goexplore_caps_per_round,
            auto_species=auto_species, species_target=species_target,
            parallel=parallel, envs_per_worker=envs_per_worker,
            init_connect=init_connect, init_k=init_k, engine=engine,
            boot_gauntlet_every=boot_gauntlet_every,
            boot_gauntlet_steps=boot_gauntlet_steps, resume=resume,
            recurrent_memory=recurrent_memory, checkpoint_every=checkpoint_every,
        ),
        config=config,
    )

    _G = config.vision.periph_grid
    print(
        f"[train] device={device} n_in={n_in} N_OUT={N_OUT} "
        f"(foveal: 3x{_G}^2 periph/fovea/motion + 14 proprio + "
        f"{config.vision.obs_ram_bytes} ram) "
        f"pop={pop_size} players={players} episode_steps={episode_steps} gens={gens}"
    )
    print(f"[train] run_dir={run_dir}  reset_state={config.emu.reset_state}")

    tracker = InnovationTracker(n_in=n_in, n_out=N_OUT)
    genomes = [
        make_genome(
            n_in, N_OUT, tracker, rng,
            connect=init_connect, weight_scale=1.0, sparse_k=init_k,
            # Decisive TANH head (§8) + connect-protect the trailing sensory
            # block: proprio(14) then RAM taps(obs_ram_bytes) are wired to every
            # output at init and exempted from toggle/split, so they are never
            # topologically dead (the measured Δoutput=0 pathology).
            output_act=TANH,
            n_ram=int(config.vision.obs_ram_bytes),
            n_proprio=14,
            protect_ram_taps=bool(config.evo.protect_ram_taps),
            protect_proprio=bool(config.evo.protect_proprio),
        )
        for _ in range(pop_size)
    ]
    n_c0 = sum(len(g.conns) for g in genomes) // max(1, len(genomes))
    print(f"[train] seed genomes: connect={init_connect} (~{n_c0} conns/genome)")

    # Weight-mutation scales anchored to the seed's fan-in-scaled init std (§8):
    # full -> 1/sqrt(n_in+1), sparse -> 1/sqrt(init_k+1); perturb sigma
    # 0.1*init_std, reset scale = init_std, clamp 4*init_std. This pins the
    # perturb/reset random walk's stationary weight std at ~init_std; the legacy
    # defaults (0.5/1.0, unclamped) drift to std ~1.8 and re-saturate the outputs
    # within ~10 gens (constant-action collapse). The vision fix is STRUCTURAL —
    # keep sigma at 0.1*init_std; a larger sigma just re-saturates the head.
    init_std = (
        1.0 / math.sqrt(n_in + 1) if init_connect == "full"
        else 1.0 / math.sqrt(init_k + 1)
    )
    rates = MutationRates(
        add_node=config.evo.mutate_add_node,
        add_conn=config.evo.mutate_add_conn,
        weight=config.evo.mutate_weight,
        toggle=config.evo.mutate_toggle,
        feedforward=not config.evo.recurrent,
        weight_perturb_sigma=0.1 * init_std,
        weight_reset_scale=init_std,
        weight_clamp=4.0 * init_std,
    )
    spec = Speciation(threshold=config.evo.species_threshold, c1=1.0, c2=1.0, c3=0.4)
    # Calibrate the WRAM churn-mask before the archive exists: strided WRAM
    # bytes that flip constantly under a random policy are the tile-map buffer
    # (redundant with the screen digest, churns at every camera scroll) and
    # audio scratch — pure key noise. Measured on this box: drops ~14/128
    # bytes and ~15% of minted cells with zero loss of state separability.
    wram_mask = calibrate_wram_mask(
        config.emu.rom_path,
        config.emu.reset_state,
        frame_skip=config.emu.frame_skip,
        seed=config.run.seed,
    )
    archive = NoveltyArchive(wram_mask=wram_mask)
    go = (
        GoExplore(
            capacity=goexplore_capacity,
            caps_per_round=goexplore_caps_per_round,
            rng=np.random.default_rng(config.run.seed + 1),
        )
        if goexplore
        else None
    )
    print(
        f"[train] novelty_mode={novelty_mode} goexplore={'on' if go else 'off'} "
        f"(restore_prob={restore_prob}, cap={goexplore_capacity}, "
        f"caps/round={goexplore_caps_per_round}) "
        f"auto_species={'on' if auto_species else 'off'} (target={species_target})"
    )

    # Create the emulator pool once; reused across every wave/generation.
    rom = config.emu.rom_path
    reset_state = config.emu.reset_state
    archive_kwargs = dict(
        screen_cells=archive.screen_cells,
        screen_levels=archive.screen_levels,
        wram_stride=archive.wram_stride,
        wram_levels=archive.wram_levels,
        wram_mask=archive.wram_mask,  # workers must hash identical keys
    )

    fleet: BarrierFleet | AsyncFleet | None = None
    envs: list[PokeEnv] = []
    replay_env: PokeEnv | None = None
    if parallel:
        fleet_cls = AsyncFleet if engine == "furnace" else BarrierFleet
        print(
            f"[train] PARALLEL fleet [{engine}]: {players} PokeEnv workers "
            f"(envs_per_worker={envs_per_worker}, rom={rom}) ..."
        )
        fleet = fleet_cls(
            n_envs=players,
            obs_dim=encoder.dim,
            obs_res=obs_res,
            obs_ram=config.vision.obs_ram_bytes,
            rom_path=rom,
            frame_skip=config.emu.frame_skip,
            hold_frames=config.emu.button_hold_frames,
            reset_state=reset_state,
            archive_kwargs=archive_kwargs,
            wram_stride=archive.wram_stride,
            goexplore=bool(go),
            envs_per_worker=envs_per_worker,
            # Active-vision knobs: with obs_dim=454 (=3*12^2+14+8) the workers
            # auto-select FovealEncoder and must match the parent's geometry.
            periph_grid=config.vision.periph_grid,
            fovea_native_px=config.vision.fovea_native_px,
            saccade_gain=config.vision.saccade_gain,
            saccade_every_k=config.vision.saccade_every_k,
            episode_steps=episode_steps,
        )
        print(f"[train] fleet up: {fleet.n_workers} worker procs")
        # A single parent-side env for the per-generation champion replay.
        replay_env = PokeEnv(
            rom,
            frame_skip=config.emu.frame_skip,
            hold_frames=config.emu.button_hold_frames,
        )
    else:
        print(f"[train] SERIAL: booting {players} PokeEnv instances (rom={rom}) ...")
        envs = [
            PokeEnv(
                rom,
                frame_skip=config.emu.frame_skip,
                hold_frames=config.emu.button_hold_frames,
            )
            for _ in range(players)
        ]
        replay_env = envs[0]

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
        streamer = LiveStreamer(
            run_dir, showcase, run_id, hz=3.0, archive=archive, goexplore=go
        )
        # Seed the showcase with an initial champion so the very first wave has a
        # live network to stream (replaced by the real champion each generation).
        streamer.set_champion(genomes[0], "init", gen=0, fitness=0.0)
        print(f"[train] live streaming ON -> {run_dir / 'live.json'} (~3 Hz)")

    # Live spectate toggle: /api/pace writes runs/<id>/pace.json; the controller
    # applies flips to the running fleet + streamer with no restart.  frame_skip
    # is derived from the env config (one agent-step = frame_skip game frames).
    pace: PaceController | None = None
    if parallel and fleet is not None:
        pace = PaceController(
            run_dir, config.emu.frame_skip, fleet=fleet, streamer=streamer
        )
        print(
            f"[train] pace control ON -> {run_dir / 'pace.json'} "
            f"(realtime budget {pace.period * 1000:.0f} ms/round, "
            f"mode={pace.mode})"
        )

    proc = psutil.Process()
    proc.cpu_percent(None)  # prime the psutil counter
    run_start = time.perf_counter()

    # Cross-generation species lineage state (see _relabel_species): last
    # generation's best-member representative per stable sid, and each
    # lineage's (best raw fitness, gen it was set) for stagnation kills.
    prev_reps: dict[int, object] = {}
    species_best: dict[int, tuple[float, int]] = {}
    species_stagnation = int(getattr(config.evo, "species_stagnation", 15))

    # Progress-counter mining state (audit MATH#1 — the endgame gradient):
    # champion-replay WRAM traces feed the game-agnostic miner every
    # `miner_every` gens; the top counters become obs RAM taps (the agent SEES
    # them) and a progress fitness term (dir-aligned advancement, reverts
    # penalized) blended with novelty at reproduction time. Novelty explores;
    # progress rewards what novelty cannot: winning, XP, story flags.
    from collections import deque as _deque

    from pokeio.reward.miner import MinerConfig, mine

    miner_rollouts: _deque = _deque(maxlen=8)
    miner_cfg = MinerConfig.from_reward_config(config.reward)
    miner_every = int(getattr(config.reward, "miner_every", 10))
    progress_weight = float(getattr(config.reward, "progress_weight", 0.5))
    taps: list[dict] = []  # {"slots": (j,)|(j,j+1), "dir": +/-1, "addr": int}
    # free-running engine bytes (music pointers, timers) are counter mimics
    miner_exclude = idle_autonomous_bytes(
        config.emu.rom_path, config.emu.reset_state,
        frame_skip=config.emu.frame_skip,
    )

    # -- resume from a checkpoint (survive power loss) ----------------------
    if checkpoint_every < 0:  # -1 sentinel → config default
        checkpoint_every = int(getattr(config.run, "checkpoint_every_gens", 25))
    start_gen = 0
    total_agent_steps = 0  # cumulative across all agents (drives Play Years)
    if resume:
        ckpt, ck_status = load_checkpoint_status(run_dir)
        if ckpt is None and ck_status != LOAD_ABSENT:
            # A checkpoint is PRESENT but unusable (corrupt / truncated / wrong
            # schema). Silently restarting at gen 0 would discard the run and
            # append gen-0 telemetry onto the old history. Fail loud + non-zero
            # so the operator fixes it or drops --resume deliberately. (A3b)
            print(
                f"[checkpoint] --resume requested but the checkpoint is "
                f"unusable ({ck_status}) at {checkpoint_path(run_dir)}; "
                f"refusing to restart at gen 0. Fix/move it or drop --resume.",
                file=sys.stderr, flush=True,
            )
            raise SystemExit(2)
        if ckpt is not None:
            # Resume guard (§2.2): changing n_in or N_OUT shifts every node-id /
            # innovation number, so an old-shape checkpoint cannot load into the
            # new obs/head. Refuse loudly rather than silently corrupt the pack.
            ck_tracker = ckpt["tracker"]
            ck_pair = (int(ck_tracker.n_in), int(ck_tracker.n_out))
            if ck_pair != (int(n_in), int(N_OUT)):
                print(
                    "[checkpoint] --resume refused: checkpoint shape "
                    f"(n_in={ck_pair[0]}, n_out={ck_pair[1]}) != freshly built "
                    f"(n_in={n_in}, n_out={N_OUT}). The active-vision restart "
                    "changed the obs/head; old 584/9 checkpoints are unloadable "
                    "(the Go-Explore archive survives on its own). Start a fresh "
                    "run id, or drop --resume.",
                    file=sys.stderr, flush=True,
                )
                raise SystemExit(2)
            genomes = ckpt["genomes"]
            archive = ckpt["archive"]
            go = ckpt["goexplore"]
            tracker = ckpt["tracker"]
            spec = ckpt["spec"]
            prev_reps = ckpt["prev_reps"]
            species_best = ckpt["species_best"]
            rng.bit_generator.state = ckpt["rng_state"]
            taps = ckpt["taps"] or []
            miner_rollouts = _deque(ckpt["miner_rollouts"], maxlen=8)
            miner_exclude = ckpt["miner_exclude"]
            start_gen = int(ckpt["gen"])
            # pre-feature checkpoints lack the cumulative step count: estimate
            # it from the resume gen (each gen ran pop_size*episode_steps steps)
            # so Play Years stays roughly continuous across the upgrade.
            total_agent_steps = int(
                ckpt.get("total_agent_steps", start_gen * pop_size * episode_steps)
            )
            # The streamer captured the ORIGINAL (empty) archive/goexplore refs
            # at construction; re-point them at the checkpoint's objects or the
            # dashboard's archive panel reads the empty ones (shows 0 cells).
            if streamer is not None:
                streamer.archive = archive
                streamer.goexplore = go
            # re-arm the obs taps the checkpoint was training with
            if taps:
                addrs = [0] * int(config.vision.obs_ram_bytes)
                for tp in taps:
                    for b, j in enumerate(tp["slots"]):
                        addrs[j] = tp["addr"] + b
                encoder.set_taps(addrs)
                if fleet is not None and hasattr(fleet, "set_tap_addrs"):
                    fleet.set_tap_addrs(addrs)
            print(
                f"[checkpoint] resumed at gen {start_gen}: "
                f"{len(genomes)} genomes, {archive.size} cells, "
                f"{go.size if go else 0} go-states, {len(taps)} taps",
                flush=True,
            )
            # Drop any telemetry/champion records for gens the checkpoint didn't
            # cover (written between the last checkpoint and the crash); we're
            # about to re-run and re-append them. (A3a)
            _truncate_streams_on_resume(run_dir, start_gen)
        else:
            print("[checkpoint] --resume set but no checkpoint yet; fresh start",
                  flush=True)

    with TelemetryWriter(run_dir, resume=resume) as writer:
        gen_wall_prev = time.perf_counter()
        for gen in range(start_gen, gens):
            gen_t0 = time.perf_counter()
            prof: dict = {}

            def _phase(key: str, t0: float) -> float:
                t1 = time.perf_counter()
                prof[key] = prof.get(key, 0.0) + (t1 - t0)
                return t1

            archive.begin_generation()
            if go is not None:
                go.begin_generation(gen)
            psutil.cpu_percent(None)

            steps_done = 0
            # Grow the topology padding budget before packing (A2). NEAT's
            # add_node/add_conn grow genomes monotonically, so the gen-0 budget
            # is eventually exceeded and Population.from_genomes raises
            # ValueError — killing the run with no in-flight checkpoint. Sizing
            # to the live population every gen also right-sizes on --resume (the
            # budget otherwise resets to the gen-0 value while resumed genomes
            # are already larger).
            need_nodes = max(len(g.nodes) for g in genomes)
            need_conns = max(len(g.conns) for g in genomes)
            new_nodes, new_conns = _grow_topology_budget(
                max_nodes, max_conns, need_nodes, need_conns
            )
            if new_nodes != max_nodes or new_conns != max_conns:
                print(
                    f"[gen {gen}] topology budget grown: "
                    f"nodes {max_nodes}->{new_nodes} conns {max_conns}->{new_conns} "
                    f"(need {need_nodes}/{need_conns})",
                    flush=True,
                )
                max_nodes, max_conns = new_nodes, new_conns
                # The live showcase froze the gen-0 budget at construction; keep
                # it in step or its champion re-pack (and focus-mode packs of
                # this gen's genomes) would hit the same ValueError off-thread.
                if streamer is not None and getattr(streamer, "showcase", None):
                    streamer.showcase.max_nodes = max_nodes
                    streamer.showcase.max_conns = max_conns

            # Pack + compile the WHOLE population once per generation (lazily,
            # so it overlaps the first wave's fleet reset); waves slice rows.
            _cp_cache: list = []
            _gen_genomes = genomes

            def _cp_provider(_cache=_cp_cache, _gs=_gen_genomes):
                if not _cache:
                    pop_full = Population.from_genomes(
                        _gs, max_nodes=max_nodes, max_conns=max_conns
                    )
                    _cache.append(pop_full.compile(device))
                return _cache[0]

            n_waves = (pop_size + players - 1) // players
            # genome idx -> the emulator state its episode spawned from (only
            # restored players appear; the rest spawned from newgame). Blobs
            # are refs into the Go-Explore archive, not copies.
            spawn_states: dict[int, bytes] = {}
            # genome idx -> restored cell KEY (E2 per-cell baseline grouping).
            spawn_key_states: dict[int, bytes] = {}
            # E3 probe buffer: real obs sampled across the whole population this
            # gen (fixed within the gen), fed to the blind-ablation gate below.
            probe_sink: list = []
            probe_quota = max(1, 256 // max(1, n_waves))  # ~256 total, spread
            for wi, a in enumerate(range(0, pop_size, players)):
                wave = genomes[a : a + players]
                # Give the streamer this wave's slot -> genome mapping so the
                # focus protocol can capture-forward any selected player slot.
                if streamer is not None:
                    streamer.begin_wave(wave, gen, a, wi, n_waves=n_waves)
                if parallel:
                    eval_fn = (
                        evaluate_wave_async if engine == "furnace"
                        else evaluate_wave_parallel
                    )
                    steps_done += eval_fn(
                        wave,
                        fleet,
                        archive,
                        device,
                        episode_steps,
                        max_nodes,
                        max_conns,
                        streamer=streamer,
                        gen=gen,
                        novelty_mode=novelty_mode,
                        novelty_floor=config.reward.novelty_floor,
                        goexplore=go,
                        restore_prob=restore_prob,
                        prof=prof,
                        cp_full=_cp_provider,
                        wave_offset=a,
                        pace=pace,
                        waves_left=n_waves - 1 - wi,
                        spawn_out=spawn_states,
                        spawn_key_out=spawn_key_states,
                        taps=taps if (taps and engine == "furnace") else None,
                        recurrent_memory=recurrent_memory,
                        softmax_temp=softmax_temp,
                        w_resp_var=w_resp_var,
                        probe_sink=probe_sink,
                        probe_quota=probe_quota,
                    )
                else:
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
                        spawn_out=spawn_states,
                        spawn_key_out=spawn_key_states,
                        wave_offset=a,
                        recurrent_memory=recurrent_memory,
                        softmax_temp=softmax_temp,
                        w_resp_var=w_resp_var,
                        probe_sink=probe_sink,
                        probe_quota=probe_quota,
                    )

            _bt = time.perf_counter()
            fits = np.array([g.fitness for g in genomes], dtype=np.float64)
            champ_idx = int(fits.argmax())  # champion by RAW wave fitness (unchanged)
            champion = genomes[champ_idx]

            # E3 blind-ablation gate (§6.2): the dominant selection multiplier
            # AND the smoke's input-sensitivity metric. Two batched forwards over
            # the shared probe buffer on the memoized full-population cp; median Δ
            # must clear Δ_min and rise across gens if the terms bite.
            gate = None
            gate_median_delta = 0.0
            if blind_gate_on and probe_sink:
                _cp_gate = _cp_provider()
                gate, gate_median_delta = _blind_ablation_gate(
                    _cp_gate, probe_sink, device,
                    beta=blind_gate_beta, dmin=blind_gate_dmin,
                    optical_hi=encoder._o_proprio,  # zero periphery+fovea+motion
                )
            _bt = _phase("blind_gate", _bt)

            # Update the showcase to this generation's real champion and push a
            # fresh live frame at the generation boundary.
            if streamer is not None:
                streamer.set_phase("evolving", "champion replay")
                streamer.set_champion(
                    champion, f"gen{gen}_g{champ_idx}",
                    gen=gen, fitness=float(fits.max()),
                    spawn_state=spawn_states.get(champ_idx),
                )
                streamer.force_write(gen)
            _bt = _phase("live_champ", _bt)

            gen_dt = time.perf_counter() - gen_t0
            cpu_pct = psutil.cpu_percent(None)
            sps = steps_done / gen_dt if gen_dt > 0 else 0.0
            # Cumulative game-time played across ALL agents: one agent-step is
            # frame_skip/60 game-seconds. This is the wall's "Play Years" tile.
            total_agent_steps += steps_done
            if streamer is not None:
                streamer.set_play_seconds(
                    total_agent_steps * config.emu.frame_skip / 60.0
                )

            # Champion replay: live-feed telemetry AND the miner's data source.
            # Replaying from the champion's true eval spawn (not newgame)
            # exercises the contexts it earned fitness in — battles, dialogue —
            # which is exactly where progress counters move. Full episode
            # length so slow counters (XP, flags) get room to tick.
            trace = replay_champion(
                champion, replay_env, encoder, archive, device,
                episode_steps, writer,
                max_nodes, max_conns, reset_state,
                spawn_state=spawn_states.get(champ_idx),
                record_wram=True,
                recurrent_memory=recurrent_memory,
                softmax_temp=softmax_temp,
            )
            if trace is not None and trace.shape[0] >= 2:
                # Dedup at the source (A9): the champion replay is a single
                # DETERMINISTIC rollout, so an unchanged champion+spawn yields a
                # byte-identical trace every gen. Feeding duplicates to the
                # miner makes a one-time ramp score consistency=1.0 (a rewarded
                # tap). Only distinct episodes enter the deque, so the
                # `len(miner_rollouts) >= 4` gate below means 4 DISTINCT traces.
                dup = any(
                    t.shape == trace.shape and np.array_equal(t, trace)
                    for t in miner_rollouts
                )
                if not dup:
                    miner_rollouts.append(trace)
            _bt = _phase("replay", _bt)

            # Re-mine progress counters on schedule; install as obs taps.
            if (
                gen % miner_every == miner_every - 1
                and len(miner_rollouts) >= 4
            ):
                cands = mine(list(miner_rollouts), cfg=miner_cfg, top_k=16)
                new_taps: list[dict] = []
                used_addrs: set[int] = set()
                slot = 0
                n_slots = int(config.vision.obs_ram_bytes)
                for c in cands:
                    span = range(c.address, c.address + c.width)
                    if slot + c.width > n_slots or any(
                        a in used_addrs or a in miner_exclude for a in span
                    ):
                        continue
                    new_taps.append({
                        "slots": tuple(range(slot, slot + c.width)),
                        "dir": 1 if c.direction == "increasing" else -1,
                        "addr": c.address,
                    })
                    used_addrs.update(span)
                    slot += c.width
                    if slot >= n_slots:
                        break
                same = [
                    (tp["addr"], tp["dir"], len(tp["slots"])) for tp in new_taps
                ] == [
                    (tp["addr"], tp["dir"], len(tp["slots"])) for tp in taps
                ]
                if new_taps and not same:  # skip no-op reinstalls: each install
                    # shifts obs semantics under evolved nets (one-time cost)
                    taps = new_taps
                    addrs = [0] * n_slots
                    for tp in taps:
                        for b, j in enumerate(tp["slots"]):
                            addrs[j] = tp["addr"] + b
                    encoder.set_taps(addrs)  # parent showcase/replay obs
                    if fleet is not None and hasattr(fleet, "set_tap_addrs"):
                        fleet.set_tap_addrs(addrs)  # workers pick up at reset
                    print(
                        "[miner] taps installed: "
                        + ", ".join(
                            f"0x{tp['addr']:04X}"
                            f"{'x2' if len(tp['slots']) == 2 else ''}"
                            f"({'+' if tp['dir'] > 0 else '-'})"
                            for tp in taps
                        )
                    )

            # Boot gauntlet (docs/specs/boot-gauntlet.md): every N gens,
            # measure how deep into the archived frontier the champion
            # re-reaches from power-on, solo, STRICTLY read-only. gen 0 is
            # skipped (archive still seeding; the -1 sentinel marks no-run).
            boot_metrics: dict[str, float] | None = None
            if (
                go is not None
                and boot_gauntlet_every > 0
                and gen > 0
                and gen % boot_gauntlet_every == 0
            ):
                if streamer is not None:
                    streamer.set_phase("evolving", "boot gauntlet")
                _bsteps = boot_gauntlet_steps or 4 * episode_steps
                boot_metrics = run_boot_gauntlet(
                    champion, replay_env, encoder, archive, go, device,
                    _bsteps, max_nodes, max_conns, reset_state,
                    forward_steps=FORWARD_STEPS,
                    recurrent_memory=recurrent_memory,
                )
                _fr = int(max((e.depth for e in go.cells.values()), default=0))
                print(
                    f"[gen {gen}] boot-gauntlet: depth "
                    f"{boot_metrics['boot_depth']:.0f}/{_fr} "
                    f"({100 * boot_metrics['boot_depth_frac']:.1f}%) cells "
                    f"{boot_metrics['boot_cells']:.0f} (known "
                    f"{boot_metrics['boot_cells_known']:.0f}) in "
                    f"{boot_metrics['boot_steps']:.0f} steps"
                )
            _bt = _phase("gauntlet", _bt)
            if pace is not None:
                pace.poll()  # pick up mid-boundary flips before the next wave

            # Adaptive speciation threshold: without it the gen-0 population
            # shares one innovation baseline and collapses to a single species.
            # Re-fit the threshold to the current genome spread and re-speciate
            # from scratch so the population actually splits (loop-side only).
            # The pairwise distances are computed ONCE as a vectorized matrix
            # (GPU) and shared by the threshold search and the assignment —
            # this was ~30s of scalar Python per generation at pop 224.
            if streamer is not None:
                streamer.set_phase("evolving", "speciation")
            D = None
            if auto_species:
                D = compat_matrix(genomes, spec.c1, spec.c2, spec.c3, device)
                _bt = _phase("spec_mat", _bt)
                if D is not None:
                    spec.threshold = fit_species_threshold_mat(
                        D, len(genomes), species_target, rng
                    )
                else:  # scalar fallback (a genome had zero connections)
                    spec.threshold = fit_species_threshold(
                        genomes, spec.c1, spec.c2, spec.c3, species_target, rng,
                        prof=prof,
                    )
                spec.reps.clear()
            _bt = _phase("spec_fit", _bt)
            if auto_species and D is not None:
                species = assign_species_mat(spec, genomes, D, rng)
            else:
                species = spec.assign(genomes, rng)
            del D
            # Stable species identity + stagnation (audit EVO#2): relabel this
            # generation's clusters to last generation's ids by rep distance,
            # track each lineage's best RAW fitness, and remove species that
            # haven't improved in `species_stagnation` gens (the champion's
            # species and a 2-species floor are protected).
            if auto_species:
                # (spec.assign keeps stable ids natively; only the from-scratch
                # matrix clustering needs relabeling)
                species = _relabel_species(species, prev_reps, spec)
            champ_sid = genomes[champ_idx].species_id
            for sid, members in species.items():
                b = max(g.fitness for g in members)
                rec = species_best.get(sid)
                if rec is None or b > rec[0] + 1e-9:
                    species_best[sid] = (b, gen)
            stagnant = {
                sid for sid in species
                if sid != champ_sid
                and gen - species_best[sid][1] >= species_stagnation
            }
            while stagnant and len(species) - len(stagnant) < 2:
                stagnant.pop()  # keep at least 2 species alive
            n_stagnant_killed = len(stagnant)
            for sid in stagnant:
                del species[sid]
            # bounded bookkeeping: forget lineages that went extinct
            species_best = {s: v for s, v in species_best.items() if s in species}
            prev_reps = {
                sid: max(members, key=lambda g: g.fitness).copy()
                for sid, members in species.items()
            }
            _bt = _phase("spec_assign", _bt)

            reward_terms = {"novelty": float(fits.max())}
            if auto_species:
                reward_terms["species_threshold"] = float(spec.threshold)
                reward_terms["species_stagnant_killed"] = float(n_stagnant_killed)
            if taps:
                reward_terms["progress_best"] = float(
                    max(getattr(g, "progress", 0.0) for g in genomes)
                )
                reward_terms["progress_taps"] = float(len(taps))
            if boot_metrics is not None:
                # mirrored so the wall side panel shows it with zero UI work
                reward_terms.update(boot_metrics)
            if go is not None:
                reward_terms.update(go.stats())

            # Push the real side-panel payloads to the live stream (species +
            # per-term reward breakdown; archive stats are read live).
            if streamer is not None:
                species_list = sorted(
                    (
                        [int(sid), len(members),
                         float(max(m.fitness for m in members))]
                        for sid, members in species.items()
                    ),
                    key=lambda s: -s[1],
                )
                streamer.set_side_stats(reward_terms, species_list)
                # Refresh the stream mid-boundary (replay + speciation +
                # reproduce can take many seconds with no wave emits) so the
                # dashboard's side panels and champion don't look frozen.
                streamer.force_write(gen)
            _bt = _phase("live_side", _bt)
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
                **(boot_metrics or {}),
            )
            writer.write_generation(rec)
            _bt = _phase("telemetry", _bt)
            print(
                f"[gen {gen:2d}] best={rec.fitness_best:6.1f} med={rec.fitness_median:6.1f} "
                f"worst={rec.fitness_worst:5.1f} species={rec.n_species:3d} "
                f"cells={rec.archive_cells:6d} (+{rec.archive_delta:4d}) "
                f"blindΔ={gate_median_delta:.4f} "
                f"{gen_dt:5.1f}s  {sps:7.1f} steps/s"
            )

            # evolve (skip on the final generation — nothing consumes it)
            if gen < gens - 1:
                if streamer is not None:
                    streamer.set_phase("evolving", "reproduction")
                # SELECTION-ONLY fitness transform (§6.5): couple selection to
                # POLICY, not spawn luck. Within each spawn cohort combine the E2
                # per-cell leave-one-out advantage q(A_i), gated by the E3 blind-
                # ablation multiplier (screen-blind -> gate~0), shaped by R_resp
                # responsiveness, then blended with mined progress exactly as
                # before. Raw fitness already drove the champion pick + telemetry.
                cohort_rank_normalize(
                    genomes, set(spawn_states),
                    progress_weight=progress_weight,
                    spawn_keys=spawn_key_states,
                    gate=gate,
                    w_resp=w_resp,
                    w_emp=w_emp,
                    restore_baseline=restore_baseline,
                    blind_gate=blind_gate_on,
                )
                genomes = fast_reproduce(
                    genomes, species, tracker, rng, rates,
                    pop_size=pop_size,
                    tournament_size=config.evo.tournament_size,
                    elitism=config.evo.elitism,
                    crossover_rate=config.evo.crossover_rate,
                    fitness_sharing=config.evo.fitness_sharing,
                    weight_scale=init_std,  # add_conn weights at init scale
                    survival_threshold=0.4,
                    c3=0.4,
                )
            _bt = _phase("reproduce", _bt)

            # ---- checkpoint (survive power loss) --------------------------
            # After reproduction: `genomes` is the NEXT population and the
            # archives are quiescent (no wave mutating them), so a synchronous
            # snapshot here is race-free. gen+1 = the generation to resume at.
            if (
                checkpoint_every > 0
                and gen < gens - 1
                and (gen + 1) % checkpoint_every == 0
            ):
                _ck_dt = save_checkpoint(
                    run_dir,
                    build_checkpoint_state(
                        gen=gen + 1, genomes=genomes, archive=archive,
                        goexplore=go, tracker=tracker, spec=spec,
                        prev_reps=prev_reps, species_best=species_best,
                        rng=rng, taps=taps, miner_rollouts=miner_rollouts,
                        miner_exclude=miner_exclude,
                        total_agent_steps=total_agent_steps,
                    ),
                )
                print(f"[checkpoint] gen {gen + 1} saved in {_ck_dt:.1f}s",
                      flush=True)
            _bt = _phase("checkpoint", _bt)

            # ---- per-generation boundary profile -------------------------
            gen_wall = time.perf_counter()
            wall_dt = gen_wall - gen_wall_prev
            gen_wall_prev = gen_wall
            rounds_t = prof.get("rounds", 0.0)
            boundary_t = wall_dt - rounds_t
            eff_sps = steps_done / wall_dt if wall_dt > 0 else 0.0
            keys = (
                "go_sample", "reset_all", "pack_compile", "live_champ",
                "replay", "spec_mat", "spec_fit", "spec_assign", "live_side",
                "telemetry", "reproduce",
            )
            parts = "  ".join(f"{k}={prof.get(k, 0.0):.2f}" for k in keys)
            accounted = sum(prof.get(k, 0.0) for k in keys)
            print(
                f"[boundary gen {gen}] wall={wall_dt:.1f}s rounds={rounds_t:.1f}s "
                f"boundary={boundary_t:.1f}s (accounted={accounted:.1f}s "
                f"other={boundary_t - accounted:.1f}s) eff_sps={eff_sps:.0f}\n"
                f"    {parts}  "
                f"spec_fit_bound={prof.get('spec_fit_bound', 0.0):.2f} "
                f"spec_fit_search={prof.get('spec_fit_search', 0.0):.2f} "
                f"(bound_calls={prof.get('spec_fit_bound_calls', 0)})",
                flush=True,
            )

    if streamer is not None:
        streamer.close()
    if fleet is not None:
        fleet.close()
    for e in envs:
        e.close()
    if parallel and replay_env is not None:
        replay_env.close()
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
    ap.add_argument("--boot-gauntlet-every", type=int, default=10,
                    help="run the from-boot champion gauntlet every N gens "
                         "(0 = off; see docs/specs/boot-gauntlet.md)")
    ap.add_argument("--boot-gauntlet-steps", type=int, default=0,
                    help="gauntlet episode length (0 = auto: 4*episode-steps)")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="append to existing telemetry instead of rotating it "
                         "aside (for checkpoint-resume; default starts fresh)")
    ap.add_argument("--no-recurrent-memory", dest="recurrent_memory",
                    action="store_false", default=True,
                    help="disable cross-agent-step recurrent state (memoryless "
                         "policies; default persists node state within an episode)")
    ap.add_argument("--checkpoint-every", type=int, default=-1,
                    help="save a resumable checkpoint every N gens (0=off; "
                         "-1=use config.run.checkpoint_every_gens, default 25)")
    ap.add_argument("--goexplore-capacity", type=int, default=16384,
                    help="max stored emulator states (memory bound)")
    ap.add_argument("--goexplore-caps-per-round", type=float, default=4.0,
                    help="max state captures per swarm round (0 = unlimited). "
                         "Each capture is a ~47ms worker-side save_state; the "
                         "archive only holds --goexplore-capacity states, so "
                         "capturing every new cell (~20/round once agents "
                         "explore) is pure churn — measured ~4 cores of "
                         "save_state burn and the gen-over-gen steps/s decay")
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
    # -- parallelism ------------------------------------------------------
    ap.add_argument("--parallel", dest="parallel", action="store_true", default=True,
                    help="evaluate waves across a shared-memory worker fleet (default ON)")
    ap.add_argument("--no-parallel", dest="parallel", action="store_false",
                    help="single-process serial evaluation (legacy fallback)")
    ap.add_argument("--envs-per-worker", type=int, default=1,
                    help="emulators owned by each worker process (1 = max parallelism)")
    ap.add_argument("--engine", choices=("furnace", "barrier"), default="furnace",
                    help="wave engine: furnace = free-running async fleet "
                         "(no per-step barrier, ~3x measured); barrier = "
                         "lockstep spin-barrier fleet (legacy)")
    # -- gen-0 seeding ------------------------------------------------------
    ap.add_argument("--init-connect", choices=("sparse", "full", "none"),
                    default="full",
                    help="gen-0 wiring: full (default; §8) fan-in-scales every "
                         "input->output edge so the decisive TANH head stays "
                         "responsive and every latent/proprio/RAM-tap input is "
                         "wired; sparse gives each output --init-k random inputs")
    ap.add_argument("--init-k", type=int, default=32,
                    help="inputs per output for --init-connect sparse (§8)")
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
        goexplore_caps_per_round=args.goexplore_caps_per_round,
        auto_species=args.auto_species,
        species_target=args.species_target,
        parallel=args.parallel,
        envs_per_worker=args.envs_per_worker,
        init_connect=args.init_connect,
        init_k=args.init_k,
        engine=args.engine,
        boot_gauntlet_every=args.boot_gauntlet_every,
        boot_gauntlet_steps=args.boot_gauntlet_steps,
        resume=args.resume,
        recurrent_memory=args.recurrent_memory,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
