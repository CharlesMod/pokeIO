# pokeIO — Fix Ledger + Active-Vision/Efficacy Redesign (2026-07-17)

Plan of record + **bug tracker** for: (1) the diagnosis-driven *network-efficacy* redesign,
(2) all 39 verified audit findings, and (3) the aggressive active-vision spine — then wipe the
population and restart at square 0.

**Sources:** `memory/pokeio-audit-2026-07-17.md` (39 verified findings) + the live-run diagnosis of
`runs/live1` gen-100 champions (this session). **User decisions (2026-07-17):** land on **master +
push**; **aggressive** live spine (learned SPR retina + saccadic foveation wired live); **delete**
`runs/live1`, restart under a new run-id.

**Diagnosis headline:** the gen-100 policies are *near-constant, screen-blind linear perceptrons*,
indistinguishable from the random init — because Go-Explore's restore+novelty harvested the fitness
the policy was supposed to earn (0 selection pressure to see/act), the 8 RAM taps are never connected
in topology (Δoutput = 0.000000 across all 224 genomes), only 19% of pixels are wired, and the
sigmoid head bunches all 9 outputs in [0.3,0.7]. **Priority order is therefore: efficacy (Wave B)
first, on a correct/reproducible base (Wave A), with perception (Wave C) as the vehicle.**

Legend: ☐ todo · ◐ in progress · ☑ done · ⏭ deferred (post-restart).

---

## Wave A — Correctness, reproducibility & hygiene *(land first: makes everything measurable)*

- ☐ **A1 · Un-RED the test suite + CI green-gate.** `tests/test_schemas.py:130` asserts
  `action_space==8` (real 9, NOOP added). Fix to real value; assert `"noop" in ACTIONS` &
  `len(ACTIONS)==action_space`; add a gate that fails on any red test. *(finding #1)*
- ☐ **A2 · Gen-0 topology budget crash.** `loop.py:1697` freezes `max_conns/max_nodes` at gen 0;
  NEAT grows monotonically → uncaught `ValueError` in `Population.from_genomes` kills long runs with
  no in-flight checkpoint. Grow-and-recompile the pack at the wave boundary (+headroom). *(finding #10)*
- ☐ **A3 · Resume corrupts telemetry.** (a) `loop.py:1943` re-appends every gen between last ckpt and
  crash (no dedup). (b) `loop.py:1939` silently restarts at gen 0 on absent/corrupt ckpt while
  appending to old telemetry. Dedup/truncate on `gen>=start_gen` atomically; distinguish absent vs
  corrupt; exit non-zero when `--resume` set but nothing usable. *(findings #9, #24)*
- ☐ **A4 · Reproducibility snapshot.** `config/__init__.py:162` `git_sha` never set; `loop.py:1702`
  `snapshot()` omits gens/episode_steps/novelty_mode/goexplore*/init_k/engine/boot_gauntlet*/
  recurrent_memory/checkpoint_every; torch RNG never seeded. Add `git rev-parse HEAD`(+`-dirty`),
  `resolved-run.json` (argv + all `train()` kwargs), `torch.manual_seed`. *(findings #2, #4)*
- ☐ **A5 · save_state truncation.** `fleet.py:609` caps blobs at `_MAX_STATE=262144` (Yellow 200592,
  23% headroom) → silent corruption for larger states. Probe real size at fleet init; skip+log,
  never store a truncated blob. *(finding #18)*
- ☐ **A6 · Go-Explore detachment.** `goexplore.py:260` recency term (`0.5^(age/4)`) evicts proven deep
  hubs; `seen` append-only → never re-captured. Additive depth floor / exempt top-N deepest. *(#23)*
- ☐ **A7 · Rarity credit order-dependence.** `novelty.py:110` shared monotonic `visit()` gives
  same-cycle co-visitors position-dependent credit (low player-index bias). Score against a gen-start
  per-cell visit snapshot. *(finding #33)*
- ☐ **A8 · Hung-worker watchdog.** `fleet.py:765` parent hot-wait detects only crashed workers, not
  hung-but-alive ones (one wedged env busy-spins a core forever). Add a per-round wall-clock deadline
  that raises with env-index/pid. *(finding #37)*
- ☐ **A9 · Miner consistency loophole.** `loop.py:2076` miner is fed one deterministic champion replay
  → byte-identical traces score `consistency=1.0`; a one-time ramp becomes a rewarded tap. Dedup
  identical traces / feed distinct genomes+spawns. *(finding #22)*
- ☐ **A10 · Tests for load-bearing paths.** furnace/AsyncFleet transport (`fleet.py:922`), checkpoint
  round-trip (`checkpoint.py`), `fast_reproduce` rng-identity, miner + novelty math, obs contents
  (`tests/test_vision.py`). *(findings #6, #12, #13, #26, #29, #31, #39)*

## Wave A′ — LLM / manifest / honesty *(unblocks the game-agnostic thesis; low runtime risk)*

- ☐ **A11 · LLM client entry-point + hardening.** `config/__init__.py:137` `LLMConfig.base_url`
  double-`/v1` + `GLM-4.7-Flash` vs `glm-4.7-flash` case; add `LlamaClient.from_config` (normalizes);
  cache invalidation (`client.py:342`: version + served-model fingerprint); offline tests
  (`client.py:191`). *(findings #21, #30, #31)*
- ☐ **A12 · Manifest generate + consume.** `manifest/schema.py:5` docstring claims consumption that
  doesn't happen. Add `manifest/generate.py` (mined candidates → GLM → validated manifest); load a
  manifest at run start; route UI/analytics RAM taps (`live.py:51`, `analytics/yellow.py`) through it,
  quarantined behind `manifest.game`. *(findings #3, #7, #19, #20)*
- ☐ **A13 · Honesty pass.** Label Lanes B/C + fs=1 pipeline **experimental** in docstrings/TODO; fix
  the dashboard's false "GLM judge" / anti-Goodhart labels; rename the duplicate `AsyncFleet`
  (`pipeline/async_fleet.py:188` vs `emu/fleet.py`); GPUVision LUT parity-or-quarantine
  (`gpu_vision.py:54`). *(findings #14, #15, #17, #28, #36)*

---

## Wave B — Network Efficacy Redesign *(the diagnosis; THE reason to restart)*

- ☐ **B1 · Couple selection to the policy (root cause).** Fitness currently credits *where the archive
  teleports you*, not what the policy did → 0 pressure to see/act (`fitness_best` flat since gen 30;
  entropy flat 135 gens). Fixes: (a) **marginal, baseline-subtracted credit** for restored players —
  subtract cells a no-op/reference reaches from the *same* spawn (restore tracks `base_depth`,
  `loop.py:562`), score only the excess; (b) **input-conditioned behavior term** — reward obs→action
  dependence (MI) / penalize near-constant or screen-invariant policies; (c) **separate exploration
  credit from policy fitness** — drive selection from a short **no-restore** eval, keep restores for
  archive growth. *(subsumes findings #8, #25, #32, #38)*
- ☐ **B2 · Connect & protect the RAM taps.** The 8 mined progress-tap inputs (ids 576–583) are wired
  as inputs but **never connected in topology** (Δoutput=0.0 across all 224). Hard-wire the tap inputs
  → all 9 outputs at init and protect them from pruning (`make_genome` sparse branch,
  `genome.py:192`). *(diagnosis #2)*
- ☐ **B3 · Grow the topology so it can see.** `mutate_add_conn=0.05` grows ~5 conns/100 gens → frozen
  at random init (champion sees 19% of pixels, near-linear, no hidden features). Raise
  `evo.mutate_add_conn`→~0.3–0.5; bias `mutate_add_connection` (`ops.py:93`) to prefer
  **currently-unconnected input sources**; raise `init_k` (`loop.py:1678`, 12→~48–96) or seed a hidden
  layer. *(diagnosis #2, findings #16 degenerate-compat, #35 weight-sigma)*
- ☐ **B4 · Decisive output head.** Sigmoid-then-argmax bunches all 9 outputs in [0.3,0.7]; argmax set
  by bias, not screen. Use **identity/tanh pre-activation + argmax** (or argmax over `out−out.mean()`,
  or temperature-softmax) over relative drive (`genome.py:162` output_act, `loop.py:343/470` argmax).
  Only bites once B1–B3 make drive screen-dependent. *(diagnosis #3, finding #35)*
- ☐ **B5 · Fix speciation (diversity for the search).** `loop.py:1320` fits the compat threshold on a
  96-genome subsample applied to all N → species count oscillates at pop 224. `sample_cap>=len(genomes)`;
  address degenerate compat distance at wide sparse init (`ops.py:230`, Jaccard-on-edges or delay
  adaptive speciation). *(findings #5, #16)*
- ☐ **B6 · Cohort-normalize the champion.** `loop.py:2036` `fits.argmax()` = luckiest restore; seeds
  miner + gauntlet + showcase. Pick from cohort-normalized rank (`loop.py:950`); raw fitness for
  labels only. *(findings #25, #32 — also serves B1)*
- ☐ **B7 · Noise/blank-frame ablation as a standing control.** Periodically eval top genomes on
  scrambled/constant frames; flag any whose fitness barely drops (catches screen-blind policies before
  they dominate again). *(research: Hausknecht noise-screen; ties to B1)*

---

## Wave C — Perception Spine (aggressive; wired into the live restart)

Detailed design → `docs/specs/active-vision-spine.md` (produced by the design workflow this session).
Replaces the 24×24 thumbnail obs (`loop.py:1689`) and pure-optical policy.

- ☐ **C1 · Saccadic foveation (biomimetic centerpiece).** Obs = **coarse peripheral** (whole screen,
  low-res) ⊕ **movable high-res fovea** crop ⊕ **motion diff** ⊕ **fovea-center proprioception**
  (efference copy). Controller emits, alongside 9 buttons, a **saccade (dx,dy)** that repositions the
  fovea next step; recurrence integrates glimpses. *(subsumes findings #11, #17, #27 motion, #34 reset)*
- ☐ **C2 · Learned decoder-free retina (SPR / VQ latent).** Small conv encoder, gradient-trained
  self-supervised (latent self-prediction + augmentation, NOT pixel reconstruction), VQ discrete
  latent, on the idle GPU. Controller consumes the latent (+ proprioception + motion). *(findings #14,
  #15, #29 lanes-parked)*
- ☐ **C3 · VQ latent → Go-Explore cells.** Discrete codes replace the pixel+WRAM cell hash →
  game-agnostic cells. *(ties B1/C2; retires hand-tuned hash)*
- ☐ **C4 · World model (imagination) — staged.** RSSM/MDN-RNN (categorical latents, NOT transformer —
  sm_60); evolve the controller in imagination on the spare P100. Tested module first; wire live only
  if it smoke-trains cleanly. *(research #14)*
- ☐ **C5 · Action/obs plumbing.** Action-spec drives env application + evo output-node count
  (buttons+saccade); obs dims drive `n_in`. Files: env.py, preprocess.py, fleet.py, genome.py/forward.py,
  goexplore.py/archive.py, loop.py, config.
- ☐ **C6 · Motion reset on restore.** `env_wrap.py:71` `VisionEnv.load_state` must reset ObsBuilder
  motion memory (no phantom spike at every Go-Explore restart). *(finding #34)*

---

## Deferred (post-restart, tracked so they're not lost)
- ⏭ Credibility harness: geometric-mean-of-manifest-milestones metric (IQM+CIs), random-weight-search
  standing baseline, pre-registered 2nd-game frozen-core protocol. *(research #16)*
- ⏭ Full anti-Goodhart co-evolved reward population + LLM distill-then-freeze judge (ONI/Motif).
  *(findings #8, #38; B1/B7 buy most of it cheaply first.)*

---

## Verified-findings ledger (all 39 → item)
A1←#1 · A2←#10 · A3←#9,#24 · A4←#2,#4 · A5←#18 · A6←#23 · A7←#33 · A8←#37 · A9←#22 ·
A10←#6,#12,#13,#26,#29,#31,#39 · A11←#21,#30,#31 · A12←#3,#7,#19,#20 · A13←#14,#15,#17,#28,#36 ·
B1←#8,#25,#32,#38 · B3←#16,#35 · B4←#35 · B5←#5,#16 · B6←#25,#32 · C1←#11,#17,#27,#34 · C2←#14,#15,#29 ·
C6←#34. *(Every finding #1–#39 is covered by ≥1 item above.)*

---

## Process — restart at square 0
- ☐ Wave A/A′ → `pytest tests/` green → **commit** (master).
- ☐ Wave B + Wave C → tests green → **smoke-run gate** (input-sensitivity must measurably rise;
  policy conditions on screen; fovea saccades; no crash) → **commit** (master).
- ☐ `git push origin master`.
- ☐ `rm -rf runs/live1` (user chose delete) — **only after the diagnosis is banked** (done).
- ☐ Launch fresh run under a new run-id from square 0 on the new spine (record final config here).
