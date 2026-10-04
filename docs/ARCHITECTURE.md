# pokeIO v3 architecture

```
games/<game>/spec.yaml ──► GameSpec (spec.py)            the ONLY game-specific contract
                              │
          ┌───────────────────┼───────────────────────────────────────────┐
          ▼                   ▼                                           ▼
   Platform (platforms/)   Memory (memory.py)                      display: names, palette
   PyBoy | gridworld       typed, cached field reads               (dashboard only)
          │                   │
          └──────► GameEnv (env.py) ◄── CellMemory / VisitedMask / InteractionNovelty /
                      │                 ScreenNovelty (exploration.py), Progress (progress.py)
                      │  obs: pixels (2-bit packed screen [+visited mask]), bits, scalars, cats, aux
                      ▼
               ProcVec (vec.py)   W workers × K envs, shared memory, async batches
                      │
                      ▼
               Trainer (train.py) ── Policy (policy.py): CNN + bits + scalars + embeddings → LSTM
                 │    │    └── SwarmCoordinator (swarm.py): frontier states → population
                 │    └── LiveHub (dash/hub.py) ──► HTTP server (dash/server.py) ──► wall.html
                 ▼
          runs/<run>/  metrics.jsonl · ckpt/ · frontier/*.state · dash_state.json · tb/
```

## The invariant

Nothing under `pokeio/` may contain a game constant: no addresses, map numbers or
event names. Everything game-specific lives in a spec. The test is simple: grep
`pokeio/` for a hex address or a game name. The gridworld platform is the only
exception, because it *is* a fake game.

Supporting a new Game Boy game means writing a spec. Supporting a new console
means adding one `Platform` (around 100 lines: buttons, tick, screen, read/write,
save/load) and specs for its games. See games/README.md.

## One agent step (env.py)

1. Press the button for `press_frames`, release it, then tick to `frames_per_action` (24) without rendering.
2. While `wait_while` holds (the game is ignoring input: text, cutscenes), tap `wait_button` and keep ticking. The agent never spends a decision there.
3. Render one frame and read memory once (cached per step).
4. Exploration reward: a refresh gain for the current cell, plus effective-interaction and screen-novelty bonuses.
5. Progress reward: `Σ weight · Δphi` over the spec's terms (monotone running max by default).
6. Info: milestones, a frontier report (score plus save state) when the score rises by `min_delta`, and episode stats at memory wipes.

`done` codes: 0 = continuing; 1 = the env jumped to a start state (hard or stall reset); 2 = the swarm swapped the state, so the action sent was not executed.

## Observation

| key | dtype | content | policy path |
|---|---|---|---|
| `pixels` | uint8 [C, H, W·bpp/8] | screen levels (+ visited mask), bit-packed | unpacked on device → CNN |
| `bits` | uint8 [B] | packed binary features (badges, event flags) | unpack → Linear(→128) |
| `scalars` | float32 [S] | scaled values, ratios, previous action one-hot | concatenated |
| `cats` | int32 [K] | categorical codes (map id, species) | embeddings, shared per feature |
| `aux` | int32 [5] | room, x, y, paused, score×100 | **dashboard only**, never shown to the policy |

The policy has no clock input. With a deterministic start, a step counter invites open-loop scripts, which is exactly how v2 collapsed.

## Learner (train.py)

- Each env fills its own row of `rollout_len` steps in whatever order the async pool returns workers. Workers with full rows are held, and their latest observation is both the GAE bootstrap and the first step of the next rollout.
- LSTM (h, c) is stored at every BPTT chunk start. Training replays each chunk from its stored state and zeroes state at episode starts.
- PPO-clip with value clipping, normalized value targets, and advantages normalized per minibatch over valid steps. Transitions with done=2 are masked out.
- Optional fp16 autocast (the P100 has 2×-rate fp16 but no bf16 and no tensor cores). `torch.compile` is opt-in; Inductor needs sm_70+.

## Swarm (swarm.py)

Envs report `{score, state}` when their frontier score rises by `min_delta`. If a report beats the global best (after `min_interval_steps`), `fraction` of the population loads that state and adopts it as their restart point. Every frontier state is archived to `runs/<run>/frontier/` and restored on resume.

## Evaluation (evaluate.py)

Episodes start from the spec's start states with the swarm and hard resets off. The report gives each milestone's reach rate with a bootstrap 95% CI and the median steps to reach it. `--blind` zeroes the screen channel: if rates don't drop, the policy isn't using vision. That was v1's failure mode.

## The training wall (dash/)

Served live by the trainer on `0.0.0.0:8600`, or offline from `runs/<run>/dash_state.json` with `python -m pokeio dash --run runs/<run>`. Every panel is real data:

- **Focus agent.** What one agent sees (the screen in a Game Boy palette) and what it remembers (its visited-cell channel, as an overlay or side by side). It also shows the live action probabilities, value estimate and reward sparklines, and a periodic saliency map of where the chosen action depends on the screen (|∂ logit / ∂ pixel|). It follows the frontier agent by default, and you can replay at 1×, 4× or 16× Game Boy speed.
- **The wall.** 16 agents playing simultaneously, with live labels for room and score. A tile flashes gold when it reaches a milestone; click a tile to focus it.
- **Milestones.** The ladder from the spec, with the first-ever reach (steps and wall time) and hit counts, plus a toast for every first.
- **World.** Population footprint heatmaps for every room discovered, in discovery order, with how many agents are there now.
- **Swarm frontier.** Score over time, and a gallery of the exact frames the population was teleported to.
- **Events.** A feed of first milestones, new areas, swarm migrations and checkpoints. Agent numbers are clickable.
- **Learning vitals, reward anatomy, buttons.** Return, tiles per mini-episode, SPS, entropy, KL, explained variance and value loss. Which reward terms are paying, and which buttons the population presses.
- **Machine.** Per-core CPU and per-GPU utilization, memory and temperature (psutil / nvidia-smi).
- **Top bar.** Agent steps, *game time played* (steps × 24 frames ÷ 59.7 fps, so "years of Pokémon"), SPS, tiles explored, milestones, and a STOP button (checkpoints, then exits cleanly).

Cost to training: per step, the trainer passes the batch it already has; the hub copies frames for about 16 agents and does a vectorized heatmap update. Saliency costs one backward pass every 2 s for a single agent.
