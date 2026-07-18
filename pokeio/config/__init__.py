"""pokeIO configuration — the single source of truth for every knob.

A nested dataclass ``Config`` with sections that mirror the pipeline:
``emu, vision, evo, reward, llm, run``. Defaults are derived from the roadmap
in ``TODO.md`` (frame-skip 24, Discrete(8) action space, N-channel foveated
vision, GLM-4.7-Flash oversight, etc.).

Round-trips through YAML (pyyaml). Every run snapshots its resolved config to
``<run_dir>/config.yaml`` for reproducibility.

Dependency-light: stdlib + pyyaml only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

CONFIG_FILENAME = "config.yaml"


@dataclass
class EmuConfig:
    """Emulator / environment knobs (see TODO Phase 1)."""

    rom_path: str = "roms/pokemon_yellow.gb"
    reset_state: str = "roms/yellow_newgame.state"
    dmg_mode: bool = True  # monochrome DMG mode
    headless: bool = True  # window="null"
    sound_emulated: bool = False
    frame_skip: int = 1  # ticks advanced per agent step (Phase 1.5: reflex fs=1)
    button_hold_frames: int = 0  # frames a button is held within a step (fs>1 only)
    # Concurrent emulator instances ("players"). Measured operating point on this
    # box (bench_40k sweep): 128 players -> ~23.2k agent-steps/s (~3x realtime per
    # player, ~390x aggregate). Pure-throughput peak is ~28 players @ 26.6k, but
    # 128 buys 4.5x the in-flight behavioral diversity for ~13% less throughput.
    n_players: int = 128
    action_space: int = 9  # Discrete(9): up down left right A B START SELECT NOOP
    action_names: list[str] = field(
        default_factory=lambda: [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "A",
            "B",
            "START",
            "SELECT",
            "NOOP",
        ]
    )


@dataclass
class VisionConfig:
    """N-channel configurable vision pipeline (see TODO Phase 1)."""

    screen_height: int = 144
    screen_width: int = 160
    shades: int = 4  # monochrome normalized to {0, 1/3, 2/3, 1}
    coarse_size: int = 32  # whole screen downscaled to coarse_size x coarse_size
    # -- foveal crop (Phase 1 obs spec) --------------------------------------
    # A center crop of the NATIVE screen at native resolution (no resample):
    # game-agnostic since the player sprite is screen-centered in GB overworld
    # games, and it stays an informative center crop in menus/battles. Padded
    # if the crop exceeds the screen bounds.
    fovea_size: int = 64  # side length of the native-resolution center crop
    # Small square grayscale obs fed to the direct-encoded NEAT loop (train/loop.py).
    # The full 6176-dim obs is too wide for direct NEAT; obs_res x obs_res is tractable.
    obs_res: int = 24  # (legacy) side length of the old flat grayscale obs
    obs_ram_bytes: int = 8  # count of strided normalized WRAM bytes appended to obs
    # -- Active-vision spine (docs/specs/active-vision-spine.md) --------------
    # obs mode selector / phased-fallback rung:
    #   "foveal"       Phase-0 pixel obs: periphery + gaze-driven fovea + motion + proprio + ram
    #   "fovea_static" sub-fallback: fovea fixed screen-centered, N_OUT back to 9
    #   "retina"       Phase-1: learned SPR/FSQ latent replaces the pixel blocks
    mode: str = "foveal"
    periph_grid: int = 12  # G: periphery/fovea/motion area-resample side (G*G each)
    fovea_native_px: int = 48  # F: native fovea crop side (resampled F->G)
    saccade_gain: float = 32.0  # px/step velocity applied to the fovea center
    saccade_every_k: int = 1  # gaze update cadence in agent-steps
    proprio: bool = True  # 14-d efference-copy block (gaze + last saccade + last buttons + step)
    # -- legacy foveal knobs (superseded by fovea_size; kept for back-compat) -
    foveal_size: int = 32  # (unused by ObsBuilder) old downscaled foveal res
    foveal_crop: int = 48  # (unused by ObsBuilder) old source-pixel crop side
    motion_channel: bool = True  # coarse_t - coarse_{t-1} difference sheet
    ram_aux_dim: int = 32  # length-K RAM aux vector (placeholder until Phase 3)
    # Resulting sheets: coarse global + foveal + optional motion (+ ram_aux vec).
    n_channels: int = 3
    normalize: bool = True


@dataclass
class EvoConfig:
    """Evolution core knobs (see TODO Phase 2)."""

    # Genomes per generation, evaluated in waves of emu.n_players (1024/128 = 8 waves).
    pop_size: int = 1024
    tournament_size: int = 4
    elitism: int = 2
    species_threshold: float = 3.0  # compatibility distance for speciation
    mutate_add_node: float = 0.03
    # Raised 0.05 -> 0.4 for the efficacy redesign: at 0.05 topology grew ~5 conns
    # / 100 gens, freezing the net at the random init (diagnosis). See B3.
    mutate_add_conn: float = 0.4
    mutate_weight: float = 0.8
    mutate_toggle: float = 0.01
    crossover_rate: float = 0.75
    fitness_sharing: bool = True
    # kill species whose best raw fitness hasn't improved in this many gens
    # (champion's species + a 2-species floor are always protected)
    species_stagnation: int = 15
    recurrent: bool = True  # allow evolved recurrent connections
    max_nodes: int = 512  # padding bound for tensorized genome
    max_conns: int = 4096
    # -- Active-vision spine (docs/specs/active-vision-spine.md) --------------
    n_out: int = 11  # 9 button logits + 2 saccade (dx, dy); was 9
    output_act: str = "tanh"  # decisive head: argmax over spread tanh drive (was sigmoid)
    init_connect: str = "full"  # "full" (fan-in-scaled) | "sparse" | "none"
    init_k: int = 32  # sparse-fallback fan-in (raised from 12)
    prefer_unconnected_src: bool = True  # bias add-conn toward unwired input sources
    prefer_unconnected_weight: float = 4.0
    protect_ram_taps: bool = True  # connect tap->output at init, exempt from toggle/split
    protect_proprio: bool = True  # same for proprio->output edges
    softmax_temp: float = 0.0  # 0 = plain argmax over out[:9]; >0 = temperature-softmax
    # -- Neural-native timing: evolvable per-neuron time constant (spec [TC]) ---
    # CTRNN leaky integration a_t = (1-alpha)*a_{t-1} + alpha*f(net), one gene per
    # node (stored on NodeGene.alpha). alpha=1 = fast reflex (legacy overwrite);
    # small alpha = slow integrator (~1/alpha-step native dwell-timer). Evolution
    # discovers the timescale mix per game/state.
    time_constants: bool = True  # False => force alpha=1 everywhere (exact legacy)
    tau_min: float = 0.05  # slowest neuron: alpha floor (~20-step integration window)
    tau_init_fast_frac: float = 0.5  # fraction of nodes seeded alpha≈1 (reflex) at init
    tau_init_slow_lo: float = 0.1  # slow-band alpha lower bound (log-uniform seed)
    tau_init_slow_hi: float = 0.5  # slow-band alpha upper bound
    mutate_tau: float = 0.2  # per-genome prob of a log-scale alpha mutation
    tau_perturb_sigma: float = 0.15  # sigma of the log10(alpha) Gaussian step


@dataclass
class RewardConfig:
    """Reward stack / Go-Explore / manifest knobs (see TODO Phase 3)."""

    # Per-distinct-cell backbone credit. Keep SMALL relative to the fresh-cell
    # unit (1.0): with ~90k observations/gen every trodden cell's rarity term
    # decays to the floor within a generation or two, so fitness degenerates
    # to `floor * distinct_cells` — a churn counter that rewards re-walking
    # known ground over opening frontier. At 0.01 a fresh cell outbids ~100
    # trodden ones (was 11 at the old 0.1).
    novelty_floor: float = 0.01
    archive_cell_downscale: int = 8  # screen downscale before hashing a cell
    use_ram_hash: bool = True  # prefer RAM-hash over pixel-hash (noisy-TV guard)
    rarity_weighted: bool = True
    manifest_path: str = ""  # runs/<id>/manifest.json (LLM-generated)
    reward_pop_size: int = 32  # co-evolving reward candidate population
    miner_entropy_mask: float = 0.95  # mask addresses above this entropy fraction
    # -- mined progress counters (loop wiring) -------------------------------
    miner_every: int = 10  # re-mine counters from champion traces every N gens
    # selection-fitness blend: (1-w)*novelty_rank + w*progress_rank, applied
    # only once any player registers counter advancement in a generation
    progress_weight: float = 0.5
    # -- Efficacy redesign: couple selection to policy (docs/specs/active-vision-spine.md §6)
    # E2: per-cell leave-one-out baseline-subtracted advantage for restored players
    restore_baseline: bool = True
    baseline_ema: float = 0.9  # cross-gen smoothing of the per-cell baseline
    # E3: blind-ablation gate — crush fitness of screen-invariant policies
    blind_gate: bool = True
    blind_gate_beta: float = 8.0
    blind_gate_dmin: float = 0.05  # min |Δoutput| (real vs optical-zeroed) to pass
    # R_resp: responsiveness (button-histogram entropy + output variance)
    w_resp: float = 0.1
    w_resp_var: float = 0.5
    # Emp: empowerment via the SPR inverse head (Phase 1+; 0 effect in Phase 0)
    w_emp: float = 0.1
    # -- E1: play-from-boot selection eval (the composition-gap cure) ----------
    # Each generation, run a SHORT no-restore rollout of the population from the
    # canonical newgame state (restore_prob=0, Go-Explore capture OFF) and score
    # every genome by exploration+progress achieved FROM BOOT (distinct cells
    # opened; mined-progress advancement when taps are live). The score is
    # rank-blended into selection fitness (``w_boot``) and picks the showcased/
    # mined champion, so the population is pressured to PLAY from newgame instead
    # of specialising on deep Go-Explore restore spawns. 0 = OFF (skips the extra
    # env-steps entirely). This is the E1 the active-vision spine spec deferred,
    # now real and ON by default for the Phase-0 foveal spine (no effect in
    # retina mode, which has no controller-latent transform hook yet).
    policy_eval_steps: int = 300
    # Genomes to boot-eval each gen: 0 = the WHOLE population (cleanest signal;
    # ~doubles per-gen eval cost when policy_eval_steps == episode_steps). A
    # positive value evaluates only that many genomes — always including the
    # current top-raw-fitness candidates — to bound the tax on long runs; the
    # rest get a neutral boot quantile.
    policy_eval_sample: int = 0
    # Blend weight of from-boot competence into selection fitness:
    #   sel_i <- (1 - w_boot)*sel_i + w_boot*q(boot_competence_i).
    w_boot: float = 0.4
    # -- Backward-shift restore curriculum (Explore-Go / boot-gauntlet D2) ------
    # Bias Go-Explore restores toward SHALLOW (early-game) cells early, annealing
    # the eligible bottom depth-quantile deeper as generations progress, so the
    # population must re-earn the opening before deep frontier spawns dominate
    # (the evolutionary analog of Go-Explore's backward algorithm).
    restore_backward: bool = True
    restore_backward_q0: float = 0.3  # gen 0: only bottom-30% depth cells eligible
    restore_backward_anneal_gens: int = 60  # widen to the full frontier by this gen


@dataclass
class LLMConfig:
    """GLM oversight client knobs (see TODO Phase 0 / Phase 3)."""

    # Served model name is lowercase (llama.cpp reports "glm-4.7-flash"); the
    # old "GLM-4.7-Flash" mismatched and 404'd. base_url must NOT carry the /v1
    # suffix: the client appends "/v1/chat/completions" (and "/health") itself,
    # so a trailing /v1 here produced a double "/v1/v1/..." 404. (audit A11)
    model: str = "glm-4.7-flash"
    base_url: str = "http://127.0.0.1:8080"
    # UD-Q3_K_XL (13.78GB): largest GLM-4.7-Flash quant that fits ENTIRELY on one
    # 16GB P100 with a 4k ctx (measured 13.75GB on card 0, card 1 free).
    quant: str = "UD-Q3_K_XL"
    context: int = 4096  # tight ctx so the model fits a single card
    device: int = 0  # card 0 = oversight; card 1 stays free for training
    timeout_s: float = 60.0
    max_retries: int = 3
    temperature: float = 0.2
    cache_dir: str = "runs/llm_cache"
    json_mode: bool = True


@dataclass
class ACConfig:
    """[AC] Adaptive cadence — learned dwell via the commit-gate output.

    See ``docs/specs/adaptive-cadence.md``. Dwell is the leaky-integrated
    activation of ONE appended OUTPUT neuron (the commit gate); that node's
    evolvable time-constant ``alpha`` (spec [TC]) IS the dwell clock. A parent-
    side ``MotorClock`` re-emits the held button until the gate opens (or an
    interrupt fires). ``enable`` drives the effective ``N_OUT`` (11 off, 12 on);
    every other knob is inert when ``enable`` is False (bit-identical legacy).
    """

    enable: bool = False            # single OFF switch; drives N_OUT (11 off, 12 on)
    commit_thresh: float = 0.0      # gate_raw >= thresh -> re-decide (open)
    min_dwell: int = 1              # reflex floor reachable; >1 forces a minimum hold
    max_dwell: int = 64             # hard liveness cap (0 = uncapped) — no-stall guarantee
    salience_interrupt: bool = True  # master switch for the self-calibrating reflex
    # [AC §11b] self-calibrating salience reflex (no fixed game-specific magnitude):
    # fire when this frame's motion is ``salience_z`` std ABOVE THIS env's OWN recent
    # motion — a per-env EMA-z of the motion signal itself (dimensionless z + EMA
    # decay + warmup), mirroring the WRAM-churn / novelty-rarity EMA machinery.
    salience_z: float = 1.5         # z-score (std above the per-env motion baseline)
    salience_ema_decay: float = 0.99  # per-env EMA decay for the motion mean/var baseline
    salience_warmup: int = 16       # min per-env steps before a salience break can fire
    reflex_margin: float = 0.0      # logit[argmax]-logit[held] > margin -> re-decide (§11b: default OFF; the gate learns "a better button appeared")
    saccade_interrupt: bool = False  # optional gaze->motor coupling (eye jump -> re-decide hand)
    saccade_deadband: float = 0.05  # |saccade| beyond this counts as "moved"
    seed_gate_slow: bool = True     # rng-free: seed gen-0 gate alpha into the slow band so the LEARNED α (not interrupts) is the primary dwell driver (§11b)
    gate_seed_alpha: float = 0.3


@dataclass
class RetinaConfig:
    """Learned decoder-free retina (Phase 1; docs/specs/active-vision-spine.md §4).

    A frozen sensory organ, never part of the genotype. SPR latent self-prediction
    + inverse-dynamics, FSQ latent (cannot codebook-collapse). Trained on the swarm's
    own frames on ``train_card``; population consumes a frozen ``snapshot()`` on
    ``infer_card``, swapped every ``swap_gens`` generations.
    """

    enable: bool = False  # Phase-1 gate; Phase-0 ships with pixel obs
    z_periph: int = 48
    z_fovea: int = 32
    spr_k: int = 5  # SPR prediction horizon
    ema_tau: float = 0.0  # 0.0 = hard target-encoder copy
    loss: str = "cosine"  # SPR loss (never L2)
    inverse_dynamics: bool = True  # inverse head q_psi (reused for empowerment)
    fsq_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 5, 5])  # 10,240 cells
    warmup_frames: int = 200000  # train before any genome consumes the latent
    swap_gens: int = 10  # freeze-and-swap cadence
    train_card: int = 0
    infer_card: int = 1
    aug_shift_px: int = 4
    aug_jitter: float = 0.05


@dataclass
class PrefConfig:
    """[MF] Frozen LLM preference-potential Φ (docs/specs/manifest-reward.md §3/§5/§10).

    A Motif-style Bradley-Terry potential over the pixel-latent slice, distilled
    offline from LLM pairwise progress-preferences and entering selection as
    potential-based shaping. ``enable`` is the single OFF switch: with it False the
    loop is byte-identical to legacy (no pref term, no annotation worker). Every
    other field is a CADENCE / BUDGET knob — none is a reward magnitude. The one
    weight, ``w_pref_ref``, inherits the loop's existing add-on regime (defaults to
    ``reward.w_resp``); the EFFECTIVE weight is ``c_acc * w_pref_ref`` where
    ``c_acc`` is the model's own dimensionless held-out reliability (§5), so a
    chance-level Φ self-zeroes and no new operating magnitude is introduced.
    """

    enable: bool = False        # single OFF switch; False => byte-identical legacy
    pref_swap_gens: int = 10     # freeze-and-swap cadence (mirrors retina.swap_gens)
    pref_train_steps: int = 300  # BT distillation grad steps per swap
    pref_pairs_per_round: int = 64   # LLM pair budget enqueued per generation (cap)
    pref_uniform_frac: float = 0.25  # reserve fraction drawn uniformly (active-learning guard)
    pref_gamma: float = 0.99     # potential-shaping discount (derives from episode horizon)
    pref_buf_cap: int = 4096     # max stored clips/labels before oldest are evicted
    pref_val_frac: float = 0.2   # held-out split fraction for the accuracy gate
    pref_subsample_k: int = 8    # frames per clip (K≈8, ~1s of progress transition)
    pref_infer_card: int = 1     # card hosting the frozen PrefScorer (card 0 = LLM)
    # w_pref_ref is NOT a new hand-tuned magnitude: it inherits the loop's existing
    # policy add-on regime (defaults to reward.w_resp = 0.1). At full reliability the
    # pref term carries the weight a trusted responsiveness term does; muted when
    # unreliable via c_acc. Keep this equal to reward.w_resp unless w_resp changes.
    w_pref_ref: float = 0.1


@dataclass
class GoExploreConfig:
    """Go-Explore cell source (docs/specs/active-vision-spine.md §5)."""

    # "pixel" = existing screen+wram hash (obs-independent, permanent safety net);
    # "fsq" = Phase-2 latent cell key from retina.fsq_code(periphery).
    cell_source: str = "pixel"


@dataclass
class RunConfig:
    """Run-level bookkeeping (see TODO Cross-Cutting)."""

    run_id: str = "dev"
    seed: int = 0
    runs_dir: str = "runs"
    device_map: dict[str, int] = field(
        default_factory=lambda: {"llm": 0, "evo": 1, "autoencoder": 1}
    )
    checkpoint_every_gens: int = 25
    telemetry_enabled: bool = True
    git_sha: str = ""  # filled at run start


@dataclass
class Config:
    """Top-level resolved configuration."""

    emu: EmuConfig = field(default_factory=EmuConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    evo: EvoConfig = field(default_factory=EvoConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ac: ACConfig = field(default_factory=ACConfig)
    retina: RetinaConfig = field(default_factory=RetinaConfig)
    pref: PrefConfig = field(default_factory=PrefConfig)
    goexplore: GoExploreConfig = field(default_factory=GoExploreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return _build_dataclass(cls, d or {})

    def save(self, path: str | Path) -> Path:
        """Write this config to a YAML file, returning the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load a config from a YAML file (missing keys fall back to defaults)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def snapshot(self, run_dir: str | Path) -> Path:
        """Write the resolved config to ``<run_dir>/config.yaml``."""
        return self.save(Path(run_dir) / CONFIG_FILENAME)


def _build_dataclass(cls: type, d: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a dict.

    Unknown keys are ignored; nested dataclass fields recurse into their type.
    """
    # Resolve string annotations (PEP 563) to real types.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _build_dataclass(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


__all__ = [
    "Config",
    "EmuConfig",
    "VisionConfig",
    "EvoConfig",
    "RewardConfig",
    "ACConfig",
    "RetinaConfig",
    "PrefConfig",
    "GoExploreConfig",
    "LLMConfig",
    "RunConfig",
    "CONFIG_FILENAME",
]
