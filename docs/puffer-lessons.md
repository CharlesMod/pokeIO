# What pokemonred_puffer taught us, and how pokeIO v3 applies it

Sources:
- pokemonred_puffer code (809 commits, Jan 2024 – Dec 2025).
- The PokeRL write-up, https://drubinstein.github.io/pokerl/
- The HN thread, https://news.ycombinator.com/item?id=43269330
- Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden 2025, "Pokémon Red via Reinforcement Learning", arXiv 2502.19920. This is the only source with ablations and multiple seeds.

Tags: [C] seen in their code or commits · [W] stated in the write-up · [P] the Pleines paper.

## 1. What actually made the difference

| Lesson | Evidence | pokeIO v3 |
|---|---|---|
| **Use a plain small network.** A Nature-CNN with an LSTM, about 5M parameters, trained with PPO. They tried ResNet-34 and the Procgen ResNet first and reverted both. | [C] 712a580 → 337f79e, 7390b0b | `policy.py`: 32k8s2 / 64k4s2 / 64k3s2 conv layers, a 512 encoder and an LSTMCell(512). |
| **Show the whole screen**, grayscale and 2× downsampled by striding. A min-pool variant was tried and reverted. | [C] 97a77bc, da859b6 → d0d3f52 | `screen.py`: stride downsample, 4 palette levels. |
| **Give the policy a visited-cell mask.** "Without it, the agent revisited the same areas repeatedly." | [W] | `exploration.VisitedMask`: a second pixel channel aligned to the screen. |
| **Reward each new (map, x, y) cell.** Removing that reward gives 0% progress. Making it 10× larger also gives 0%, because the agent stops fighting. | [C] 345a218 "explore reward is needed for reals"; [P] ablations | `CellMemory` plus `rewards.exploration.cell_weight`. |
| **Use memory, not frame stacking.** Stacking was never really used. Pleines' GRU roughly doubled success on multi-step quests. | [C] 8ae25d1, [P] | LSTM with BPTT 16. No frame stack and no clock input. |
| **Run mini-episodes.** Never reset the emulator; wipe the exploration memory every ~20k steps. | [C] 9b7ea93 | `episode.memory_reset_steps` (19816 ± 2000 of jitter). |
| **Swarm.** When any env reaches a new milestone, every env loads its save state. This was the single biggest stabilizer; erratic data "plagued us for months" before it. | [W], [C] 839cec2 | `swarm.py`, triggered by a spec-defined score. |
| **Hold the button 8 frames out of 24**, and auto-press A while the game ignores input (`wJoyIgnore`). | [C] 972c3c5, 94ed20d | `controls.press_frames` / `wait_while`, both declared in the spec. |
| **Keep the level reward, soft-capped.** "Any attempt to remove it led to a failed experiment." In Pleines' setup, removing it actually helped slightly. | [W] vs [P] | `sum_softcap` term, weight 0.5, knee 22. It's spec-tunable because the evidence conflicts. |
| **Pack observations.** Moving observations around was "the biggest bottleneck". They used 2-bit pixels and bit-packed events, unpacked on the GPU. | [W], [C] 23c845d | `screen.pack`, the `bits` obs key, `Policy.unpack_pixels`. |
| **Use an async env pool** with several envs per worker (288 envs over 24 workers, env batch 36). | [C] b22f061 | `ProcVec` with `batch_workers` set to half of `num_workers`. |
| **Tune rewards, not PPO.** CARBS sweeps barely changed the PPO settings (lr 2e-4, γ 0.998, λ 0.95, clip 0.1, entropy 0.01); they mostly moved reward weights. | [C] config history | `configs/default.yaml` uses their PPO numbers; the reward weights live in the spec. |

## 2. What they tried that didn't work, so we skip it

- **A global map observation.** Tried four times between January and June 2024, then turned off ([C] 467c7d9).
- **Rewards for pressing A anywhere, and for menus.** Agents spammed A instead of navigating ([W]). We use *effective* interaction instead: A only pays if it visibly changes the screen at a new (cell, facing). This is our hypothesis, not a proven result.
- **A battle reward.** "Trash and didn't work" (HN, Addis). Agents learn to spam the menu and pick damaging attacks on their own.
- **Blackout penalties**, five variants ([C] Feb–Mar 2024).
- **Variable-length "hook ticking"** steps, reverted after one day ([C] 0f22379 → 15097a7).
- **NOOP action** ([C] 089bdd5 → 9fde27d). **SELECT** was never in the action set.
- **Swarm v1** (move the bottom X% every N updates): "lots of gotchas and really unstable". Copying LSTM state between agents was also dropped.
- **LR re-annealing tricks, `max_steps_scaling`, and `required_tolerance`.** All were removed or never enabled.
- **Decaying exploration memory (DecayWrapper).** It was not in the pipeline that beat the game, which wiped binary memory instead. We ship it as an option (`half_life_steps > 0`) with binary memory as the default.

## 3. Puffer bugs we designed around

These are from reading their code; they don't come from their write-up.

1. **The reward is a delta of a running sum, and the memory wipe at a mini-episode boundary isn't rebased.** That gives a large negative reward spike every ~20k steps ([C] since 970f5f4). In v3, exploration pays per-visit gains, and every swap of game state calls `Progress.rebase`.
2. **LSTM rollout state is never zeroed**, not on done and not on swarm migration. v3 marks swarm loads with done=2, zeroes the state, and masks the action that was never executed out of the loss.
3. **Training replays each minibatch from a zero LSTM state, then carries that state across unrelated envs.** v3 stores (h, c) at every BPTT chunk start and replays from there.
4. **GAE doesn't bootstrap at segment bounds** (their own TODO). v3 bootstraps every row end from the held next observation.
5. **There is no held-out evaluation.** Progress is read off training curves, where the swarm teleports agents forward. v3's `pokeio eval` runs from the start state with the swarm off and reports milestone rates with bootstrap CIs.
6. **No reward or value normalization.** Rewards span 0.005 per tile up to 10+ per badge. v3 normalizes the value targets.

## 4. What's game-specific in puffer, and the generic replacement in v3

| Puffer (Pokémon Red only) | pokeIO v3 (any game, set in the spec) |
|---|---|
| ~35 hand-written reward terms, many keyed to named events | 5 generic term kinds (`bitcount`, `value`, `sum_softcap`, `ratio_gain`, `distinct`) over spec fields |
| A "required events/items" list (~80, curated) as the swarm key | `swarm.score`: the weighted sum of state-function terms, or the number of milestones reached |
| Map-ID exploration boosts gated on story events (×9.5) | **Not ported.** It's pure game knowledge; the authors call it their "least favorite addition". See the roadmap: population-novelty boosts. |
| Sign, NPC, hidden-object, warp and menu hooks via pokered symbols | `InteractionNovelty`: a visible screen change from an interaction button at a new (cell, facing) |
| Scripted Cut, Surf, Strength, Flash, item tossing, infinite money | **None.** Our scripting surface is limited to `memory_patches` at reset (fast text, animations off). |
| `wJoyIgnore` auto-advance | `controls.wait_while`: any spec condition, any button |
| A global Kanto map for the visited mask | Per-room grids cropped around the player, with `camera: follow | fixed` |

## 5. Numbers to calibrate against

- Peak throughput: ~10k agent steps/s on an i9-14900K and RTX 4090 ([W]). The PufferLib paper reports ~7k on a desktop.
- Beating Brock takes under 30 minutes; with wild battles disabled, badge 1 in 10 minutes ([W]). Their `early_stop` cutoffs are Brock 30 min, Misty 300, HM01 600, Surge 1200 ([C]).
- A full run with scripts takes 7 hours to a week ([W]).
- Pleines (no scripts, 4 reward terms): Brock 99%, Cerulean 85% in ~25k steps of game time, Misty 27%, Cut 0%. Each run is ~36 h on CPU.
- Random actions never leave Pallet Town ([HN]).
