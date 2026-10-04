# Execution Plan — "best sim/AI" build (2026-07-17)

North star: **near-term Pokemon progress** (leave the house → milestones), with the **game-agnostic
thesis track in parallel**. Everything routed through the live dashboard (Tailscale
http://100.74.178.26:8600) so testing is watchable. Each build item = agent → `--live` smoke on the
wall → commit. The real multi-day run launches once the controller improvements are in.

**BINDING CONSTRAINT — cross-game species bootstrap (user, 2026-07-17).** A goal is to seed the NEXT
game's population with THIS game's evolved species. A genome only transfers if its input weights read a
**transfer-stable, game-agnostic observation**. So: keep the genome obs transfer-stable — the **foveal
encoder qualifies** (hand-defined, fixed n_in=454 / N_OUT=12, game-invariant grid semantics); the
**per-game learned retina latent does NOT** (transfer-breaking unless made a universal/shared retina).
[MF]/Φ is orthogonal (genomes transfer via the obs; Φ re-warms per game). The only game-specific piece
of the foveal obs is the 8 RAM taps → reset/re-discover them on transfer. Bootstrap TOOLING (export
species → seed pop, reset taps, re-base InnovationTracker) deferred until a run earns it. See
`memory: pokeio-cross-game-transfer`. Do not commit to the per-game retina latent in a way that breaks this.

Legend: ☐ todo · ◐ in progress · ☑ done. Handles in [brackets].

## Critical path — SEQUENTIAL (all touch loop.py / genome / forward; one at a time, commit between)

- ☑ **[PB] Play-from-boot — the top blocker.** Both spines' champions are Go-Explore frontier
  specialists that don't play from newgame (composition gap). Make selection reward from-boot
  competence: (a) enable/refine the from-newgame eval blended into selection fitness
  (`reward.policy_eval_steps`, scored by progress+exploration from boot); (b) backward-shift the
  restore distribution toward SHALLOW cells early, annealing deeper (Explore-Go / boot-gauntlet D2);
  (c) pick the showcased/mined champion by BOOT competence (from-newgame depth), not raw frontier
  fitness. Files: train/loop.py, reward/*, config. **Accept (UI):** a `--live` smoke where the
  champion, from newgame, reaches materially more distinct screens / leaves the start area vs the
  frontier-specialist baseline; boot-gauntlet depth rises.
- ☑ **[TC] Time-constants — neural-native timing.** Evolvable per-neuron time constant (CTRNN leaky
  integration `h=(1-α)h+α·f(Wx+b)`), one gene/node, log-scale mutation, bounded, seeded with a
  timescale spread. Files: evo/genome.py, evo/forward.py, evo/ops.py, config. **Accept (UI):** XOR/
  timing unit tests pass; a `--live` smoke shows the learned α distribution differentiates (slow +
  fast neurons emerge); no instability. **DONE `d3c6dcf`** — 259 tests green (incl. tau-active
  fast_reproduce determinism); `tc_live` on the wall showed a stable bimodal split (α p10≈0.15 slow /
  p90=1.0 fast, slow_frac≈0.5) across gens with no instability.
- ☑ **[AC] Adaptive cadence.** Learned (button, dwell-duration) with a reflex floor (k=1), decoupled
  gaze vs motor clocks, salience-interruptible dwell (uses the magno signal when available; a
  surprise term for now). **Design of record: `docs/specs/adaptive-cadence.md` (LatchGate — dwell = a
  commit-gate output neuron whose TC α IS the dwell clock; reuses [TC], no evo-core edit; feature
  lives in config + loop.py).** Files: train/loop.py, config (env/genome/forward/ops/fleet unchanged).
  **Accept (UI):** a `--live` smoke where dwell is state-dependent (long in menus/dialogue, ~1 in
  motion); reflex floor + break-cause histogram reachable; gate-α slow; no stall; boot-gauntlet holds.
  **DONE — skeleton `30b0a8a` + self-tuning `dc03413`.** 288 tests green; determinism intact
  (fast_reproduce==ops at N_OUT 11 & 12, AC-off byte-identical, engine-parity). Self-tuning mandate
  §11b honored: per-env EMA-z salience (no fixed magnitude), reflex-margin off, R_resp→salience↔commit
  correlation. `ac_selftune_live` wall showed the bimodal split emerge in the champion (gen-2:
  dwell_lo_sal 65 ≫ hi_sal 33.5, slow gate-α). Stability of dwell-dominance is a real-run question.
- ☐ **[RUN] Launch the real run.** Wipe `runs/live1` (authorized), launch a fresh square-0 `--live`
  Phase-0 run with PB+TC+AC folded in, furnace engine, boot gauntlet on, watchable on the wall.
  Runs for days — this item = launch + verify healthy, not finish.

## Parallel tracks — ISOLATED files (run alongside the critical path)

- ☑ **[CR] Credibility harness.** New `pokeio/eval/`: noise/blank-frame ablation control,
  random-weight-search baseline, geometric-mean-of-milestones metric (IQM + CIs). Isolated new
  module + tests. **Accept:** runnable on a checkpoint; metric computed from telemetry. **DONE `561a5c2`.**
- ☐ **[MF] Manifest/LLM loop (thesis keystone).** Wire manifest generate→consume and a
  distill-then-FREEZE reward (ONI/Motif: async caption annotation → small frozen reward model the CPU
  loop optimizes; never a live inner-loop judge). Non-loop parts (llm/, manifest/, reward/) build in
  parallel; the loop.py wiring sequences AFTER [AC]. **Accept:** a manifest is generated + consumed;
  a frozen reward term influences selection; game-agnostic (no Yellow constants leak).

## Shelved research branch (revisit after PB+TC+AC show progress on a real run)
Framing (generalization thought-experiment, 2026-07-17 — Tetris/Mario/BoxBoy): the perception+control
substrate (optical obs, saccade, TC, AC) is game-agnostic by construction; generalization is
bottlenecked by the game-agnostic PROGRESS signal (→ [MF] is the keystone), and each shelved branch
closes a *specific game-class* gap, not a random research bet:
- Retinal channels (ON/OFF + magno/parvo + motion-saccade) — layer onto retina once it earns the spine.
  **Closes the scroller gap:** in side-scrollers (Mario) ego-motion scrolls the whole screen, so the
  motion-sheet salience fires on self-motion not objects; an optic-flow/magno channel disentangles it.
- Retina-as-spine commitment + FSQ cells — bench-race vs Phase 0 only after the base actually plays.
  **Closes the Go-Explore cell-abstraction gap:** learned latent cells generalize where pixel-hash cells
  explode (Tetris boards) or blur progress.
- World model / imagination — far-later bet. **Closes the planning gap:** puzzles (Box Boy) need
  multi-step lookahead a reactive evolved policy + Go-Explore memorization can't supply.

## Process
- Dashboard up on :8600 (newest-run view) for the whole session → http://100.74.178.26:8600.
- Each critical-path item: build (agent) → `--live` smoke on the wall → tests green → commit.
- [CR] and [MF]-non-loop run in parallel from the start.
- After [AC]: fold in [MF] loop-wiring → then [RUN].
- "Execute to the end" = every buildable item built/tested/committed + the real run launched & healthy.
