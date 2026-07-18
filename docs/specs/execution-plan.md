# Execution Plan — "best sim/AI" build (2026-07-17)

North star: **near-term Pokemon progress** (leave the house → milestones), with the **game-agnostic
thesis track in parallel**. Everything routed through the live dashboard (Tailscale
http://100.74.178.26:8600) so testing is watchable. Each build item = agent → `--live` smoke on the
wall → commit. The real multi-day run launches once the controller improvements are in.

Legend: ☐ todo · ◐ in progress · ☑ done. Handles in [brackets].

## Critical path — SEQUENTIAL (all touch loop.py / genome / forward; one at a time, commit between)

- ☐ **[PB] Play-from-boot — the top blocker.** Both spines' champions are Go-Explore frontier
  specialists that don't play from newgame (composition gap). Make selection reward from-boot
  competence: (a) enable/refine the from-newgame eval blended into selection fitness
  (`reward.policy_eval_steps`, scored by progress+exploration from boot); (b) backward-shift the
  restore distribution toward SHALLOW cells early, annealing deeper (Explore-Go / boot-gauntlet D2);
  (c) pick the showcased/mined champion by BOOT competence (from-newgame depth), not raw frontier
  fitness. Files: train/loop.py, reward/*, config. **Accept (UI):** a `--live` smoke where the
  champion, from newgame, reaches materially more distinct screens / leaves the start area vs the
  frontier-specialist baseline; boot-gauntlet depth rises.
- ☐ **[TC] Time-constants — neural-native timing.** Evolvable per-neuron time constant (CTRNN leaky
  integration `h=(1-α)h+α·f(Wx+b)`), one gene/node, log-scale mutation, bounded, seeded with a
  timescale spread. Files: evo/genome.py, evo/forward.py, evo/ops.py, config. **Accept (UI):** XOR/
  timing unit tests pass; a `--live` smoke shows the learned α distribution differentiates (slow +
  fast neurons emerge); no instability.
- ☐ **[AC] Adaptive cadence.** Learned (button, dwell-duration) with a reflex floor (k=1), decoupled
  gaze vs motor clocks, salience-interruptible dwell (uses the magno signal when available; a
  surprise term for now). Files: emu/env.py, evo/genome.py+forward.py, train/loop.py, emu/fleet.py,
  config. **Accept (UI):** a `--live` smoke where the learned dwell distribution adapts (long dwells
  in menus/dialogue, short in motion); reflex floor reachable; no stall.
- ☐ **[RUN] Launch the real run.** Wipe `runs/live1` (authorized), launch a fresh square-0 `--live`
  Phase-0 run with PB+TC+AC folded in, furnace engine, boot gauntlet on, watchable on the wall.
  Runs for days — this item = launch + verify healthy, not finish.

## Parallel tracks — ISOLATED files (run alongside the critical path)

- ☐ **[CR] Credibility harness.** New `pokeio/eval/`: noise/blank-frame ablation control,
  random-weight-search baseline, geometric-mean-of-milestones metric (IQM + CIs). Isolated new
  module + tests. **Accept:** runnable on a checkpoint; metric computed from telemetry.
- ☐ **[MF] Manifest/LLM loop (thesis keystone).** Wire manifest generate→consume and a
  distill-then-FREEZE reward (ONI/Motif: async caption annotation → small frozen reward model the CPU
  loop optimizes; never a live inner-loop judge). Non-loop parts (llm/, manifest/, reward/) build in
  parallel; the loop.py wiring sequences AFTER [AC]. **Accept:** a manifest is generated + consumed;
  a frozen reward term influences selection; game-agnostic (no Yellow constants leak).

## Shelved research branch (revisit after PB+TC+AC show progress on a real run)
- Retinal channels (ON/OFF + magno/parvo + motion-saccade) — layer onto retina once it earns the spine.
- Retina-as-spine commitment + FSQ cells — bench-race vs Phase 0 only after the base actually plays.
- World model / imagination — far-later bet.

## Process
- Dashboard up on :8600 (newest-run view) for the whole session → http://100.74.178.26:8600.
- Each critical-path item: build (agent) → `--live` smoke on the wall → tests green → commit.
- [CR] and [MF]-non-loop run in parallel from the start.
- After [AC]: fold in [MF] loop-wiring → then [RUN].
- "Execute to the end" = every buildable item built/tested/committed + the real run launched & healthy.
