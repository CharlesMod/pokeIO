# pokeIO roadmap (v3)

**Goal:** a game-agnostic agent that learns console games from pixels plus spec-declared RAM,
with no demos and no pretrained weights. Pokémon Yellow is the current challenge, not the product.
The v1/v2 roadmap is archived in docs/archive/.

## Phase 0: bring-up on the training box (nothing here has been executed yet)
- [ ] `pip install -e .[dev,tb]` with torch ≤ 2.14 (cu126). Check `torch.cuda.get_arch_list()` includes sm_60.
- [ ] `pytest`: all tests are ROM-free (gridworld fake console). Fix whatever the first run turns up.
- [ ] `python -m pokeio train --spec games/gridworld/spec.yaml --config configs/smoke.yaml`; open the wall on :8600.
- [ ] Yellow ROM in `roms/`. Run `python -m pokeio make-state --spec games/pokemon_yellow/spec.yaml`, then check `roms/yellow_newgame.png`.
- [ ] `python -m pokeio probe --spec games/pokemon_yellow/spec.yaml`. Confirm the `[D]` addresses: HP/max HP plausible after getting Pikachu, options byte = text speed, Pokédex popcount 0 then 1. Calibrate `player_cell_px` from `probe/start_obs.png`.
- [ ] Build `roms/pokeyellow.sym` (pret/pokeyellow `make`) to enable `wait_while` text skipping.
- [ ] `python -m pokeio bench --workers 52 --envs 6 --batch-workers 26`. Target ≥ 12k env-steps/s. Also try `cgb: true` vs `false` and 4, 6 or 12 envs per worker.

## Phase 1: reproduce the known-good baseline
- [ ] Red cross-check: train `games/pokemon_red` from puffer's Bulbasaur state. Brock within ~30–60 min at ~10k sps is the bar puffer reports.
- [ ] Yellow from the new-game state: got_starter → Viridian → Brock. Run `pokeio eval` every ~100M steps; keep `--blind` within budget.
- [ ] Two seeds per config (one per GPU). Report milestone rates with CIs, never single curves.

## Phase 2: throughput
- [ ] Profile env-only vs learner time (`time_collect` / `time_update` in metrics). Is GPU idle time hidden by async batches?
- [ ] fp16 autocast A/B on P100. A CUDA-graph inference step (the v2 repo had one working).
- [ ] PyBoy hooks instead of polling for `wait_while`, if profiling shows memory reads matter.
- [ ] Rust/C++ emulator core (~1.4–1.5× per Karten et al. 2026). Only after the above.

## Phase 3: claw back genericity (each one is an A/B against the Phase 1 baseline)
- [ ] Learned/pixel position: replace RAM (room, x, y) with tilemap-hash or scroll-register positions; measure the loss.
- [ ] Auto-discovered progress counters: a RAM miner proposing `bitcount`/`value` fields (v1 had a prototype).
- [ ] Population-novelty exploration boosts (generic stand-in for puffer's map-ID boosts): weight cells by how rarely the *population* visits their room.
- [ ] Stronger interaction novelty (dialog-box detection from pixels).
- [ ] An LLM drafts specs from a RAM map or disassembly. The spec is the manifest.

## Phase 4: research bets (after Phase 1 works)
- [ ] A self-supervised auxiliary loss (SPR / inverse dynamics) on the encoder.
- [ ] Achievement-distillation-style contrastive loss on milestones.
- [ ] Foveal retina from v2 as an *extra* input stream (never the only one).
- [ ] LLM subgoals → RL refinement (the PokéAgent 2025 winner's pattern), if the no-priors rule is relaxed.

## Phase 5: the console ladder
- [ ] Super Mario Land with `generic_gb`, then with a real spec (position and progress).
- [ ] An NES platform (one `Platform` module) and a first NES spec.
