# pokeIO — Master Roadmap

**Vision:** A game-agnostic evolutionary neural-network system that learns to play — and eventually
complete — console games from **optical input (primary) + RAM (auxiliary)**, with a visibly evolving
network at its core, no per-game hand-coding, and no downloaded priors. First target: **Pokémon Yellow
(Game Boy, monochrome)**. Long arc: up the console ladder (GB → NES → SNES → N64).

**Status:** Phase 0 COMPLETE (2026-07-16). Repo scaffolded, deps in, torch verified on P100, emulator +
throughput + determinism proven, data contracts tested, GLM confined to one card. Next: Phase 1 (in progress).

### Phase 0 — RESULTS (measured on this box)
- **torch 2.7.1+cu126 runs on both P100s** (sm_60 kernels present, matmul OK on cuda:0 & cuda:1). Pin this; torch ≥2.8 drops Pascal.
- **Single-core raw emulation: ~22,300 frames/s** (matches ~20k precedent).
- **Single-core agent-steps/s: ~354** (frame_skip=24).
- **Aggregate at 56 procs: ~10,183 agent-steps/s — meets the ≥10⁴ target.** Near-linear to 28 physical cores (25.5×); 28→56 hyperthreads add only ~9%.
- **RSS per env: ~45 MB** (vs 200–400 MB estimated) → thousands of envs fit in 128 GB; **CPU, not RAM, is the ceiling.**
- **Determinism: PASS** (save-state + seeded 1000-action replay → byte-identical screen + WRAM).
- **Canonical reset state** `roms/yellow_newgame.state`: controllable overworld (Pallet Town house, map 18, coords (3,1)), timing-independent scripted intro.
- **GLM-4.7-Flash on ONE card:** `UD-Q3_K_XL` (13.78 GB) fits card 0 alone (~13.8 GB, card 1 FREE for training); 350 tok/s prefill / 39.5 tok/s decode @ 4k ctx. Standalone server :8080 (needs systemd for persistence).
- **STRATEGIC IMPLICATION:** we are firmly **CPU-bound at ~10k agent-steps/s**; the P100s can infer far more tiny-nets/s than that. So the levers are **sample efficiency, episode-length/frame-skip budgeting, and (later) more machines** — NOT GPU optimization. Rough budget: pop 256 × ~4k steps ≈ 1M steps ≈ 100s/generation ≈ ~36 gen/hr.

### Audit3 — FURNACE engine (async free-running fleet, 2026-07-16)
- **Barrier round decomposition** (112 players / 28 workers, goexplore, quiet box; `scripts/bench_audit2_fleet.py`): round=40ms → **2,797 steps/s real-workload**. Barrier wait = 34ms of which mean worker work is only ~19ms — the rest is straggler tax (emulation cost is game-state dependent: 3.7ms typical, 9–12ms in scroll/dialog states) plus a ~5ms GPU forward all workers idle through. Go-Explore `save_state` = **47ms** and stalls the whole fleet when it lands on the barrier.
- **FURNACE (`AsyncFleet` + `evaluate_wave_async`, `--engine furnace`, now default): 8,129–8,470 steps/s at the same config — 2.9x.** No per-step barrier: per-env `obs_seq`/`act_seq` counters in shm; workers step the moment their action lands; the parent batches whatever is ready into each forward. Straggler tax → 0 (a slow env only slows itself), forward overlaps stepping, captures stall one worker slice not the fleet. Per-env trajectories are bit-identical to barrier (verified: identical gen-0 champion); within-gen rarity-credit/capture ORDER is timing-dependent, so runs are not round-reproducible.
- **Traps discovered:** (1) pure busy-spin in the async workers measured ~2x WORSE than 200-spin+50µs naps — the barrier's busy-spin lore does not transfer to multi-ms waits; (2) in-fleet step cost is ~2x the solo micro-bench (memory-system contention of 28 concurrent PyBoys, not frequency — verified ~3.1GHz under load); (3) a CUDA-graph forward (0.86ms vs 2.9ms) plus min-batch/fixed-cadence pacing knobs exist in `scripts/bench_audit3_async.py` (`POKEIO_CUDAGRAPH=1` opt-in is wired in `evaluate_wave_async`).
- **Production A/B (2026-07-16, ~6 cores lost to a realtime-paced live1 in the background; 2 gens each, pop 224, 400 steps, goexplore, live on):**
  | config | gen0 / gen1 steps/s |
  |---|---|
  | barrier, 112 players | 3,339 / 3,995 |
  | furnace, 112 players | 4,873 / 6,020 |
  | furnace + CUDA graph, 112 players | 5,919 / 6,147 |
  | furnace, 224 players (single wave, epw 8) | 7,126 / 5,248 |
  | **furnace + CUDA graph, 224 players single-wave** | **7,500 / 6,839 (eff 6,530–6,832 incl. boundaries)** |
  Best config ≈ **1.8x barrier contemporaneous**; boundary time collapses to ~1-2s in single-wave mode. CUDA-graph capture worked cleanly in production at both 112 and 224 (validated identical eager-vs-replay argmax at capture time). Expect ~8-9k+ on a truly idle box.
- **Anomaly to investigate:** the restarted live1 burns ~6.2 cores at REALTIME pace (old build idled at load <1). Something in the current realtime path isn't sleeping.
- **Realtime spectate** still works under furnace: per-env absolute schedule (env i's action k at t0+k·period) instead of the barrier's per-round deadline.

---

## Guiding Invariants (never violate without an explicit decision)

- [ ] **Game-agnostic core.** No file outside a per-game *generated* manifest may contain game-specific
      constants (addresses, "badges", map names). If you're typing `0xD356` into engine code, stop.
- [ ] **Optical is primary, RAM is auxiliary.** The agent must be able to function on pixels alone; RAM
      features augment, they don't replace vision. (Proves generalization up the console ladder.)
- [ ] **Evolution is visible.** A real neural network's topology must be watchable as it evolves
      (the CPPN genotype). This is a product requirement, not a nice-to-have.
- [ ] **Pure learning.** No downloaded checkpoints, no longplay imitation, no pretrained backbones on
      external data. Self-supervised learning on the game's *own* frames is allowed.
- [ ] **The manifest is the only contract.** Reward stack and dashboard consume the same LLM-generated
      Progress Manifest. Neither knows the word "Pokémon."
- [ ] **Determinism where it counts.** Given a save-state + action sequence, emulation must replay
      identically. Seeds are logged. Runs are resumable.
- [ ] **Pascal reality.** All GPU code must run on sm_60 (no tensor cores). Pin `torch==2.7.*+cu126`.
      Card 0 = GLM-4.7-Flash oversight; Card 1 = evolutionary training.

---

## Hardware Budget (the machine we optimize for)

| Resource | Spec | Allocation |
|---|---|---|
| CPU | 2× Xeon E5-2690 v4 — 28 cores / 56 threads, 2 NUMA nodes, AVX2 | Emulator fleet (NUMA-pinned) |
| RAM | 128 GB | ~200–400 MB/env → thousands of envs feasible |
| GPU 0 | Tesla P100 16 GB (sm_60, HBM2 ~720 GB/s) | GLM-4.7-Flash IQ3_XS (~13 GB) + KV |
| GPU 1 | Tesla P100 16 GB | Population inference + autoencoder lane |
| Disk | 1.6 TB free | Save-state archive, checkpoints, telemetry logs |

**Throughput target (to be measured Phase 0):** community precedent ≈ 10k agent-steps/s on one box;
raw headless PyBoy ≈ 20k frames/s/core → aim for ≥ 10⁴ agent-steps/s aggregate.

---

## Phase 0 — Foundation & Smoke Test
*Goal: prove every primitive works on THIS box and lock the numbers/contracts everything else keys off.*

### Repo & environment
- [ ] `git init`; commit structure below; `.gitignore` (venv, ROMs, checkpoints, `*.state`, telemetry).
- [ ] Proposed layout:
      `pokeio/{emu,vision,evo,reward,manifest,llm,telemetry,dash,config,utils}`, `scripts/`, `tests/`,
      `roms/` (gitignored), `runs/` (gitignored), `TODO.md`, `pyproject.toml`.
- [ ] Python 3.12 venv; pin deps: `pyboy>=2.7`, `torch==2.7.*+cu126`, `numpy`, `pyyaml`, `psutil`,
      `msgpack`/`orjson`, `pytest`, `numactl` (system).
- [ ] Verify CUDA on Pascal: `torch.cuda.is_available()`, run a real matmul on card 1, confirm sm_60
      kernels execute (no "no kernel image" error). Document torch/cuda versions in `runs/env.txt`.

### Emulator primitives
- [ ] Confirm ROM present (`roms/pokemon_yellow.gb`); verify checksum; **DMG (monochrome) mode** in PyBoy.
- [ ] Boot headless (`window="null"`, `sound_emulated=False`); tick past intro to controllable state;
      save that as `roms/yellow_newgame.state` (the canonical reset point).
- [ ] Read `pyboy.screen.ndarray` (144×160×4) → grayscale; read arbitrary WRAM via `pyboy.memory[...]`.
- [ ] `save_state`/`load_state` to `io.BytesIO`; measure `.state` size and per-instance RSS.
- [ ] **Determinism test:** load state, replay fixed 1000-action script twice → assert identical final
      frame + WRAM digest. (Flush queued inputs before saving — known PyBoy gotcha.)

### GLM-4.7-Flash oversight
- [ ] Obtain GLM-4.7-Flash **IQ3_XS** GGUF; serve via llama.cpp on **card 0 only** (reconfigure/relocate
      the existing `llama-server` so card 1 is free — verify with `nvidia-smi`).
- [ ] Smoke a structured completion (JSON out); measure decode tok/s + prefill latency on P100.
- [ ] Wrap in a thin `llm/client.py` (timeout, retry, JSON-schema-validated responses, on-disk cache).

### Contracts & baselines
- [ ] **Define telemetry schema v0** (`telemetry/schema.py`): per-generation record (gen, wall, fitness
      best/median, species, archive delta, champion genome ref, reward-term weights) + per-step champion
      stream (frame ref, action, RAM tap). Version it.
- [ ] **Throughput benchmark** (`scripts/bench_throughput.py`): 1 core fps; scaling 1→56 procs; report
      aggregate steps/s, RSS/env, optimal env count. Save to `runs/bench0.json`.
- [ ] Random-agent harness end-to-end (env → random actions → telemetry emitted) as the measurement floor.

**✅ Definition of done:** one command boots Yellow headless, we know exact box throughput & RSS,
determinism holds, GLM answers on card 0 with card 1 free, telemetry schema frozen.

---

## Phase 1 — Emulator Fleet & Vision Wiring
*Goal: a fast, NUMA-pinned fleet feeding the exact observation tensor the evolving net will consume.*

### Environment wrapper
- [ ] `emu/env.py`: gym-like `reset(from_state)` / `step(action)`; frame-skip ≈ 24 ticks, button held
      ~8 frames, `render=False` during skip.
- [ ] Action space `Discrete(8)`: ↑ ↓ ← → A B START SELECT (SELECT/START pruneable later).
- [ ] **Action abstraction for the console ladder** (build the seam now, even if unused at fs=1 GB): a
      pluggable action head so the output layer can grow discrete-8 → more buttons → **continuous/analog
      (stick X/Y) + camera** without a rewrite. Action space is the biggest genuine change going up the
      ladder — drive env action-application and the evo output-node count from an action-spec, not a
      hardcoded 8. (See console-ladder §.)
- [ ] RAM feature extractor (generic byte reads → normalized vector; addresses come from manifest later,
      raw-dump mode for the miner now).

### Vision pipeline (invest heavily here)
- [ ] `vision/preprocess.py`, **N-channel configurable from day one**:
  - [ ] Monochrome capture → normalize 4 shades → {0, ⅓, ⅔, 1}.
  - [ ] **Multi-scale foveation:** coarse global (whole screen → ~32×32) + fine foveal crop around the
        player/attention center. Concatenate as sheets.
  - [ ] **Motion:** one difference channel (frame_t − frame_{t−1}).
  - [ ] Assemble the observation tensor; document its exact shape (drives the substrate layout).
- [ ] Unit tests: shape, range, determinism of preprocessing.

### Vectorized fleet
- [ ] `emu/fleet.py`: shared-memory observation transfer (PufferLib-style), worker pool, crash recovery,
      graceful shutdown.
- [ ] **NUMA pinning:** bind workers to cores/sockets (psutil affinity); keep obs buffers node-local.
- [ ] Save-state manager: archive of states keyed by cell id (for Go-Explore returns in Phase 3).
- [ ] Re-run throughput bench with the *real* obs pipeline (not just raw emulation); tune env count.

### Config
- [ ] `config/` dataclass+YAML system; single source of truth for every knob (frame-skip, obs shape,
      pop size, device map, etc.); every run snapshots its resolved config into `runs/<id>/config.yaml`.

**✅ Definition of done:** N-thousand parallel envs sustain the target step rate; observation tensor is
finalized and documented; states seed/restore correctly.

---

## Phase 1.5 — Path to 40k+ agent-steps/sec (GPU-offloaded throughput)
*Goal: honor **frame_skip=1** (per-frame reflex control for Mario-class games) AND hit ≥40k agent-steps/sec by moving the per-frame pipeline onto the GPU and decoupling it from CPU emulation.*

**Reframing:** frame_skip only sets how often the agent ACTS — the emulator always simulates every frame; nothing is dropped. At fs=1, raw emulation is NOT the bound (40k frames/s ≈ <2 cores of raw emulation). The bound becomes render+obs+inference on EVERY frame (4× the pipeline invocations vs fs=24). So the whole game is: make the per-frame pipeline cheap and parallel → push it to the P100s.

**Architecture — decoupled async actor pipeline:**
- [ ] **CPU emulation workers** (28 cores): emulate render-on, grab a compact grayscale framebuffer (~23 KB), push to a shared-memory ring, apply the action returned for the previous frame. Never block on the GPU (**1-frame pipeline latency**, ~17 ms — imperceptible).
- [ ] **GPU inference/vision server** (card 1): pull batches of framebuffers → **batched vision preprocessing on-GPU** (coarse/fovea/motion as tensor ops) → **whole-population batched forward pass** → actions back. Obs never touches CPU numpy.
- [ ] **CPU step-path optimization** (approved #1): batch the 24→1 tick into one C call, zero-copy framebuffer access, minimal Python, render-on. Drive per-core toward raw render-on emulation.
- [ ] **Transfer:** compact framebuffer + pinned host memory + batched async H2D (40k × 23 KB ≈ 0.9 GB/s, far under PCIe).
- [ ] **Async double-buffering:** while GPU processes batch N, CPU emulates batch N+1 — hides CPU↔GPU serialization.

**GPU vision kernels:**
- [ ] Start with torch GPU ops (`F.interpolate` downsample, slice crop, subtract motion — already CUDA kernels).
- [ ] If per-batch kernel-launch overhead dominates, hand-write ONE fused CUDA kernel (raw framebuffers → obs tensor) via `torch.utils.cpp_extension`, compiled for **sm_60** (nvcc 12.8 on-box). NOTE: Triton dropped pre-Volta — use **raw CUDA C on Pascal, not Triton**.

**Validation:**
- [ ] **Profile-first:** decompose per-frame cost (render-on fps/core, framebuffer grab, transfer, GPU obs, batched inference) — measure before optimizing.
- [ ] Target ≥40k agent-steps/sec sustained end-to-end at fs=1; ceiling likely higher (render-on emulation supports >100k/s; GPU handles obs+inference trivially).
- [ ] Config default `frame_skip=1`; motion channel = true consecutive-frame diff; no action-hold.
- [ ] Keep the CPU obs/fleet path as a correctness reference & fallback.

**Episode budgeting note:** at fs=1 an episode has ~24× more agent-steps per unit game-time — more inferences per episode, but the P100 has the headroom and reflex precision is the point.

---

## Phase 2 — Evolution Core (the visible star)
*Goal: a GPU-batched, tensorized, topology-evolving network — starting simple, ending at ES-HyperNEAT.*

### EVO CORE RESULTS (2026-07-16) — foundation proven
- **Tensorized NEAT works on Pascal (cuda:1).** XOR solved **10/10 seeds, median 12 gen**, evolving hidden nodes from a zero-hidden start → the visibly-evolving-topology requirement is met.
- **125,257 net-evals/sec** (pop 256, dim 6144, sparse path) — **~13× the ~10k CPU emu ceiling**, confirming the GPU is not the bottleneck (the whole architecture premise holds).
- **Use the SPARSE forward path for wide optical nets.** Dense `(N,M,M)` adjacency is pathological at 6144 inputs (691 evals/s, ~99% zeros, bandwidth-bound); sparse edge-list is O(edges) → 125k. Both verified bit-identical.
- **CPPN/HyperNEAT plug-in point ready:** `propagate()` takes a pre-built weight tensor, so a painted substrate feeds straight in — no connection list needed.
- Recurrence + variable depth handled by **iterated propagation** (cyclic adjacency = unrolled RNN = evolved memory). 50/50 tests pass.
- Follow-up: add a public `InnovationTracker.new_node()` (demo_xor reaches into a private field).

### Tensorized substrate (port TensorNEAT idea to PyTorch)
- [ ] `evo/genome.py`: padded-tensor genome (nodes, connections, innovation numbers, enable flags).
- [ ] `evo/forward.py`: batched forward pass over the whole population on **card 1**; supports
      **recurrent** connections (evolved memory); handles variable topo via padding + masks.
- [ ] `evo/ops.py`: mutation (add-node, add-connection, weight-perturb, toggle), crossover, and
      **speciation** (compatibility distance); tournament selection + **aging/regularized-evolution** +
      **fitness sharing**.

### Lane A — fixed-topology GA baseline (prove the loop first)
- [ ] Simple MLP/conv over foveated obs; **seed-list genome encoding** (Deep-GA style, cheap parallelism).
- [ ] Evolve on a trivial objective (maximize screen novelty / "get somewhere") end-to-end with telemetry.

### Lane B — ES-HyperNEAT (the real vision encoder)
- [ ] `evo/cppn.py`: CPPN genome (the visible genotype), evolved by NEAT.
- [ ] **Canonical CPPN inputs — NO egocentric priors.** Query `CPPN(x1,y1, x2,y2, bias)` on the SOURCE
      and TARGET substrate coordinates of each connection. Do NOT feed hand-picked `center dist` / radial
      `r` / `phase` — those are 2D-egocentric priors that won't transfer up the ladder (see console-ladder
      §). Let evolution DISCOVER radial/symmetry structure via gauss/sin nodes only where a game rewards it.
      (Replaces the demo CPPN's toy single-point inputs; update the dashboard labels when this lands.)
- [ ] `evo/substrate.py`: substrate layout — input sheet = screen grid (matches obs geometry), hidden
      sheet(s), output = 8 buttons; **RAM aux inputs as a distinct coordinate region**.
- [ ] Query CPPN → phenotype weights (batched); convolution-like receptive fields should emerge.
- [ ] **ES-HyperNEAT:** evolve hidden-node placement (density adapts to information).
- [ ] Verify a CPPN can express spatial features (sanity task: detect sprite / respond to motion).

### Lane C — learned-encoder retina (non-NEAT competitor + THE 3D / console-ladder path)
*Makes zero geometric assumptions about the world → the forward-compatible lane. Going up the ladder we
shift weight from Lane B (2D-geometric) toward this one; we don't rebuild the architecture.*
- [ ] `evo/retina.py`: self-supervised autoencoder trained by gradient on the game's own frames
      (no external data) → compact features → evolved controller on top (ERL-Re² pattern).
- [ ] **3D-from-2D readiness:** feed the encoder temporal input (motion channel + short frame history)
      and give the controller evolved recurrence, so depth/perspective is inferred from motion — the
      mechanism that carries us to Mode-7 GBA and N64 with NO substrate change.
- [ ] Shares the exploration archive with Lanes A/B; the lanes **race** on the same telemetry.
- [ ] **Milestone:** validate Lane C on a pseudo-3D / perspective game (a Mode-7 GBA title) BEFORE N64,
      to prove the "2D substrate + temporal inference" thesis on a real 3D-ish world.

### Novelty backbone (fitness that works day one)
- [ ] Behavior characterization = set of visited screen-hash / RAM-state cells.
- [ ] Population-global novelty (singleton-env regime) + episodic novelty tiebreak.

**✅ Definition of done:** a population visibly evolves topology on card 1, three lanes runnable and
comparable, CPPN paints spatially-structured vision, novelty backbone drives real exploration.

---

## Phase 3 — Reward Stack & Progress Manifest
*Goal: game-agnostic fitness — LLM proposes, RAM mining grounds, evolution validates.*

### Go-Explore archive
- [ ] `reward/archive.py`: cell = hash(downscaled screen ⊕ RAM digest); global population archive;
      **rarity-weighted** novelty; frontier tracking; **return-via-save-state** seeding.

### RAM progress-counter auto-miner (the publishable gap)
- [ ] `reward/miner.py`: from random+novelty rollouts, per-address stats — change frequency, entropy
      under a no-op policy, within-episode monotonicity, **cross-population correlation with archive
      growth**.
- [ ] Classify progress-counter candidates; **mask high-entropy noise addresses** (anti-noisy-TV).
- [ ] Output: ranked candidate progress dimensions with types (counter/flag/bitfield/level).

### Progress Manifest
- [ ] `manifest/schema.py`: the JSON contract (progress_dimensions[], spatial, milestone_label, labels,
      icons, sources). Versioned + validated.
- [ ] `manifest/generate.py`: GLM prompt = ROM name + mined candidates (+ optional disassembly symbols)
      → manifest JSON; validate; store `runs/<id>/manifest.json`.
- [ ] `reward/compile.py`: manifest → weighted fitness terms.

### LLM roles (bursty, per-generation — never per-step)
- [ ] **Captioner:** RAM-diff → text events ("entered Route 3; party 12→14; flag set").
- [ ] **Judge (Motif):** pairwise rank of champion episode transcripts → feeds tournament directly
      (preferences suffice) and/or distills a light reward model. Cache + rate-control.
- [ ] **Reward engineer (EUREKA):** generate candidate fitness functions + reflection loop using
      archive-growth feedback.

### Reward co-evolution (anti-Goodhart)
- [ ] `reward/coevo.py`: second population of reward candidates; **meta-fitness = predicts frontier
      growth + agrees with held-out judge**; ensemble/median; novelty-backbone floor always > 0;
      stochastic held-out judge. A hacked reward stops predicting exploration → it dies.
- [ ] Noisy-TV guards: prefer RAM-hash over pixel-hash; optional temporal-reachability separation.

**✅ Definition of done:** point the system at Yellow with zero game-specific code and it generates a
sane manifest, mines real progress bytes, and produces a co-evolved fitness that beats novelty-only.

---

## Phase 3.5 — Boot Gauntlet (from-boot competence guard)
*Goal: measure — then, only if needed, fix — the Go-Explore chain gap: no single genome has ever
demonstrated newgame→frontier play. Full spec: `docs/specs/boot-gauntlet.md` (Opus-ready).*

- [ ] **D1 — boot gauntlet eval** (`pokeio/train/gauntlet.py`): every N gens, run the champion solo
      from `yellow_newgame.state` for `~4×episode_steps` at full speed, STRICTLY read-only vs both
      archives (audit REWARD#5 rule); score = max `CellEntry.depth` among visited cells present in
      the Go-Explore archive (+ `boot_depth_frac` = /`goexplore_max_depth`, same-snapshot).
      Cell keys via existing `archive.cell_key` — verified byte-identical to the worker compact path.
- [ ] **D1 — telemetry**: defaulted `boot_*` fields on `GenerationRecord` (−1 = "didn't run");
      mirror into `reward_terms` so the wall shows it with zero dashboard work; CLI
      `--boot-gauntlet-every` (default 10) / `--boot-gauntlet-steps` (0 = auto).
- [ ] **D1 — tests**: pure-metric unit test (synthetic archive), read-only guarantee (archive
      counters unchanged), `cell_key`/`cell_key_compact` equality regression guard.
- [ ] **D1 — stretch**: dedicated wall chart, boot_depth_frac vs frontier depth by generation.
- [ ] **D2 — backward-shift robustification** (GATED — build only if D1 shows boot_depth_frac
      stalling ≥5 gauntlets while frontier depth grows ≥20%): bias `_sample_restores` toward
      shallow cells (bottom-q depth quantile, annealed) — the evolutionary analog of Go-Explore's
      backward algorithm. Design sketch in the spec; no commitment beyond it.

**✅ Definition of done:** every run plots how deep the champion gets from power-on vs how deep the
archive's frontier is; the two curves diverging is now a measured signal, not a fear.

---

## Phase 4 — The Training Wall (real, from the mock)
*Goal: the dashboard becomes a live instrument — spectator view and debugging view are one view.*

- [ ] `telemetry/emit.py`: integrate emitter into the loop (per-gen records + per-step champion stream);
      ensure it never bottlenecks training (async, backpressure).
- [ ] Transport: JSONL tail or lightweight websocket server.
- [ ] Build real dashboard from the mock (artifact:
      https://claude.ai/code/artifact/c5ceb6e8-2f87-4ea5-8e9f-5470da82277d), componentized & data-bound:
  - [ ] **Champion Cage** — live optical feed + **substrate-heatmap overlay** (CPPN receptive fields
        over the screen) + RAM tap (mined bytes highlighted).
  - [ ] **The Swarm** — sampled N live agent screens, elite/dead states.
  - [ ] **Champion Topology** — **live evolving CPPN genotype**.
  - [ ] **Fitness by generation** — best/median/novelty (dataviz rules: one axis, direct labels).
  - [ ] **Reward Genome** — co-evolved terms, mutation flashes.
  - [ ] **Exploration Archive** — Go-Explore coverage grid.
  - [ ] **Milestone Feed** — **manifest-labeled** events (no hardcoded "badges").
  - [ ] **Compute Fabric** — real `nvidia-smi` + `psutil` gauges.
- [ ] All panel labels/units bind to the manifest (prove by swapping game → labels change, no code edit).
- [ ] Stretch: "director" auto-spotlight on whichever agent just hit a milestone.

**✅ Definition of done:** a single screen shows a live run legibly; swapping the ROM re-labels the UI
with zero code changes.

---

## Phase 5 — Scale, The Run, & Prove Generality
*Goal: beat real Yellow milestones, then prove the thesis on a game we never touched.*

- [ ] `evo/loop.py`: **async steady-state** evaluation (no generational stragglers on variable-length
      episodes); keeps cores + card 1 saturated.
- [ ] Full-run checkpoint/restore: population, archive, manifest, reward population, RNG, config.
- [ ] Long Yellow run; track the milestone ladder (below); hyperparameter sweeps.
- [ ] **Generalization acid test:** point the untouched system at **Super Mario Land** (and Kirby,
      Tetris) — confirm manifest/reward/UI regenerate with **zero Pokémon-specific code**.
- [ ] **D3 — multi-game fitness** (blocked on the acid test working at all): evaluate each genome on
      ≥2 ROMs, aggregate per-(game × spawn-cohort) ranks — the path to a literal single cross-game
      genome, vs the default "the *harness* drops in, the population re-evolves per game".
      Pointer in `docs/specs/boot-gauntlet.md` §D3.
- [ ] Ablations & write-up: CPPN vs autoencoder lane; novelty-only vs +RAM-mining vs +LLM-judge;
      document what actually drove progress.

**✅ Definition of done:** meaningful Yellow progress under a fully game-agnostic pipeline, and a second
game learned with no code changes.

---

## Cross-Cutting (do continuously, not a phase)

- [ ] **Testing:** unit (genome ops, hashing, preprocessing, miner stats, manifest validation) +
      integration smoke (env→evo→telemetry) run in CI.
- [ ] **Reproducibility:** global seed control; every run logs seeds, config, git SHA, env versions.
- [ ] **Experiment tracking:** per-gen stats to Parquet/SQLite in `runs/<id>/`; easy diffing across runs.
- [ ] **Performance:** profile the hot loop each phase; keep a `runs/perf.md` of throughput regressions.
- [ ] **Code quality:** ruff + type hints; small, tested modules; no game constants in engine code.
- [ ] **Docs:** keep `README.md` current; an `ARCHITECTURE.md` once Phase 2 stabilizes.

---

## Research Bets (higher-risk, higher-reward — schedule opportunistically)

- [ ] **Automated RAM progress-counter mining** — no published end-to-end system exists; this is
      potentially publishable on its own. Treat the miner as a first-class experiment, not just plumbing.
- [ ] **Indirect vs learned vision** — rigorous CPPN(HyperNEAT) vs autoencoder-retina head-to-head on
      identical envs/budget.
- [ ] **LLM-as-reward-engineer generality** — how many distinct ROMs get a working manifest zero-shot?
- [ ] **Open-endedness / auto-curriculum** — Voyager-style GLM milestone proposal driving the archive.
- [ ] **CPPN substrate for motion** — can the encoding natively represent temporal/velocity features?

---

## Milestone Ladder — Pokémon Yellow (measurable progress checkpoints)

- [ ] **M0** — Random agent runs; telemetry + wall live.
- [ ] **M1** — Agent reliably leaves the player's house / navigates Pallet Town.
- [ ] **M2** — Reaches Viridian City.
- [ ] **M3** — Catches a Pokémon / clears Viridian Forest.
- [ ] **M4** — Defeats Brock — **Badge 1** (first real "it's learning the game" moment).
- [ ] **M5** — Badge 2 (Misty).
- [ ] **M6** — Reaches Vermilion / Badge 3.
- [ ] **M7+** — Onward toward the Elite Four (stretch; long-horizon).
- [ ] **MG** — **Generalization:** Super Mario Land world 1 cleared with zero code changes.

---

## Console-Ladder Roadmap (the long arc — informs architecture now)

Each rung stresses the design; keep the vision front-end and manifest flexible so these are config, not
rewrites:

- [ ] **NES** — libretro core / nes-py; color required, larger obs, bigger RAM map.
- [ ] **SNES** — snes9x/stable-retro; higher res, more complex control, slower emulation.
- [ ] **N64** — mupen; 3D optical input (huge jump — vision front-end will need real depth), much slower
      emulation, GPU-side rendering considerations.

*Implication for today:* N-channel/color-ready vision, arbitrary-size RAM handling, action-space
abstraction, and manifest generality are the forward-compatibility bets we're already making.

### 2D-vs-3D — the optical substrate never changes (design principle)
The **observation space is always a 2D framebuffer** (a screen is 2D), even for 3D worlds (Mario 64,
Mode-7 GBA). The agent, like a human, infers 3D from the 2D projection. So **3D is a temporal +
representation problem, NOT a substrate-geometry problem** — recovered via the motion channel + evolved
recurrence + learned features, never a "3D substrate." Consequences:
- [ ] **Do NOT hand the CPPN egocentric priors** (`center dist`, radial `r`, and the demo-only `phase`).
      Use canonical HyperNEAT inputs — source+target substrate coords `CPPN(x1,y1,x2,y2,bias)` — and let
      evolution DISCOVER radial/symmetry structure (via gauss/sin nodes) only where a game rewards it.
      (The dashboard's "screen x / center dist / phase / bias" labels are the DEMO CPPN's toy inputs, not
      the real query — replace when the real evo core drives the panels.)
- [ ] **The learned-encoder lane (autoencoder retina / conv + evolved controller) is the 3D path.** It
      makes zero geometric assumptions; going up the ladder we shift weight from the geometric HyperNEAT
      lane toward it, rather than rebuilding. HyperNEAT-2D stays great for 2D games.
- [ ] What actually changes up the ladder: **action space** (analog stick + camera → continuous/more
      outputs — the biggest real delta), **color** (front-end already channel-agnostic), **emulation speed**
      (N64 GPU-rendered + slow). None require abandoning the 2D optical substrate.

---

## Open Decisions (revisit as we learn)

- [ ] GLM IQ3_XS quality sufficient for manifest/judge? (Fallback: Q4 split across both cards.)
- [ ] SELECT/START in the action space, or prune to reduce branching?
- [ ] Foveal-crop centering: player-locked (needs a coarse RAM position) vs learned attention vs fixed.
- [ ] When to introduce Lane C (autoencoder) — parallel from the start, or only after Lanes A/B baseline?
- [ ] Checkpoint cadence vs disk (save-state archive can grow large).
