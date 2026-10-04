# pokeIO: SOTA review and reset plan (2026-10-02)

Premise: assume our implementation choices were naive. What does the 2023–2026 literature say works for
Pokémon-class RL from pixels, how far is pokeIO v2.0 from that, and what should we change to train
faster, perform better, and actually make progress through the game?

Inputs: a code audit of this repo (file:line cites below) plus three literature/engineering surveys.
Numbers marked *(est.)* are our own extrapolations, not published figures.

---

## 1. TL;DR

1. **The problem is solved-ish, and the solution is boring.** `pokemonred_puffer` (Rubinstein, Whidden et
   al., Feb 2025) beat Pokémon Red with **plain PPO + a small CNN→LSTM (~5M params)**, a 72×80 grayscale
   screen + a **visited-tiles mask** channel, ~35 shaped reward terms, and **"swarming"** (when any env hits
   a new required milestone, all envs load its save state). ~10k steps/s on one i9 + RTX 4090 box;
   a full-game run takes 7 h to ~1 week. The cleaner academic version (Pleines et al., arXiv 2502.19920)
   reaches Cerulean in 85% of runs with a 4-term reward and no scripts.
2. **pokeIO v2 diverges from that recipe on almost every axis that the literature says matters**:
   no exploration reward, no recurrence, a lossy foveal canvas instead of the full screen, a clock input
   that invites memorized scripts, an entropy *penalty*, 896-step episodes, ~10–23M-step runs (vs 10⁸–10⁹),
   and a synchronous Python-heavy fleet. Result: it reliably gets Pikachu (with a scripted demo +
   backward curriculum), then collapses into an open-loop script. Nothing records reaching Viridian.
3. **World models, fancy encoders and foveation are not the lever.** Our bottleneck is exploration and
   credit assignment over 10⁴-step horizons, not pixel sample-efficiency. The emulator is cheap; on-policy
   PPO with many envs is the regime that has won here.
4. **Throughput: ~1.5–2.5× is available on this box** *(est. 12–18k env-steps/s)* via the fixes the puffer
   team found, plus **skipping non-decision frames** (text/animations), which is worth a further ~1.3–2×
   in game-progress-per-hour *(est.)*. No GPU Game Boy emulator exists; a Rust/C++ core buys ~1.4×.
5. **The real decision is philosophical, not technical:** every successful Pokémon RL agent leans on
   game-specific RAM (coordinates, event flags, map IDs). Our invariants say "game-agnostic, optical
   primary." Recommended path: **first reproduce the proven recipe (accepting RAM features via the
   manifest), then replace game-specific pieces one at a time with generic ones, measuring each swap
   against the baseline.** Right now we have neither a working baseline nor a generic agent.

---

## 2. Where pokeIO v2.0 actually is (code audit)

| Area | pokeIO v2.0 (as of 2026-07-20) | Cite |
|---|---|---|
| Learner | PPO-clip + GAE, 1.27M-param conv(2→16→32→32)+MLP, no RNN, no frame stack | `brain/actor_critic.py:80-144`, `brain/rl_loop.py:383-435` |
| Policy input | 2×96×96 "canvas" (mostly 12×12 periphery upsampled 8×8 + 32×29 foveal stamps) + 24 extras | `emu/fleet.py:1100-1151, 689-692` |
| Clock input | `p[13] = nstep / episode_steps` → with deterministic boot, learnable open-loop scripts | `emu/fleet.py:1186` |
| "RAM aux" | 8 bytes at `wram[::1024]` (0xC000, 0xC400, …) — arbitrary, not progress features | `emu/fleet.py:1197-1205` |
| Reward | Potential over running maxes: maps +1, party +5, levels +1, badges +50, event flags +2. **No exploration bonus.** | `brain/reward.py:83-162`, `brain/from_boot.py:100-125` |
| Entropy | `0.02·(1−ema) − 0.01·ema` → **goes negative** as success rises | `brain/rl_loop.py:47-48, 401` |
| Normalization | Advantages only. No value/return/obs normalization; γ=0.999 with reward jumps of 1–50 | `brain/rl_loop.py:391` |
| Episodes | 896-step cap, reset every iteration; Go-Explore archive not checkpointed | `rl_loop.py:169-191`, `train/brain_loop.py:120-124` |
| Scale | 28–64 envs, 400 iters → ~10M steps/arm (brain6), ~23M (brain4) | `scripts/train_pipeline.sh` |
| Fleet | `BarrierFleet` (sync); 1 env/process; float64 encode ≈ emulation cost; full 8 KB WRAM copy + cell hash every step; per-env Python reward in parent; GPU `.cpu()` sync each step; blocking eval (~17%) | `train/brain_loop.py:81`, `emu/fleet.py:1721-1722`, commit c9b1f94 |
| Actions | Discrete(9) incl. START, SELECT, NOOP; d-pad **held all 24 frames** | `emu/env.py:169-209` |
| Gaze | Reflex soft-argmax; learned gaze head scored **below random** on every stream | commit c6b9e38 |
| Results | Pikachu from boot ~100% (stochastic) via demo + backward curriculum; brain4/5 "mode collapse at progress 32"; brain6 outcomes unrecorded | commits 915fe54, 832e835; `train_pipeline.sh:9-11` |

Also: `brain_eval.py` likely can't load canvas checkpoints (`brain_eval.py:39-44`); `eval/spine.py`
(IQM/bootstrap CIs) isn't wired into v2; `train/loop.py` (5,052 lines of NEAT) is dead weight for v2.

---

## 3. The reference recipes

| | **Whidden v2** (2024) | **Pleines et al.** (2025) | **pokemonred_puffer** (2025) | **pokeIO v2** |
|---|---|---|---|---|
| Screen | 72×80 gray, 3-frame stack | 72×80 gray, 3-frame stack | 72×80 gray, 2-bit packed | 96×96 foveal canvas |
| Visited mask | 48×48 local (RAM coords) | 48×48 crop (RAM coords) | 72×80 channel (RAM coords) | none |
| Other obs | HP, levels, badges, 728 event bits, last 3 actions | party HP+levels, events | map ID, items, party 6×11, events, facing, battle type | clock, last button, gaze, 8 junk bytes |
| Memory | frame stack | frame stack / GRU (GRU doubled Bill's-quest success) | **LSTM**, BPTT 16 | none |
| Params | small CNN | ~2M (4M GRU) | ~5M | 1.27M |
| Actions | 7, press 8 / release 16 of 24 | 7, press 8 / release 16 | 7, press 8 + **auto-tick through text** (`wJoyIgnore`) | 9, held 24 |
| Exploration | **+per new (map,x,y)** | **+0.005 per new coord** | **+0.029/coord**, signs, warps, per-map boosts, decayed maps | none |
| Progress | events, heal, badges | +2/event, heal, level (soft-capped) | ~35 terms; required events ≫ generic | milestones only |
| Entropy | 0.01 | 0 (tried, didn't help) | 0.01 | **negative when winning** |
| γ / λ | 0.997 | 0.997 / 0.95 | 0.998 / 0.95 | 0.999 / 0.95 |
| Batch | 64 envs × 2560 | 32 × 2048 = 65k | 288 envs, 65k batch, mb 2048, 3 epochs | 28–64 envs × ≤896 |
| Episode | 163,840 steps | 10,240 + 2,048 per event | never reset game; wipe explore-memory every ~20k | **896** |
| State sharing | — | — | **swarm** on new required milestone | Go-Explore spawns (pixel cells, not persisted) |
| Scale | ~10¹⁰+ steps budget | ~36 h / run | ~10k sps; 7 h–1 wk full game | ~10–23M steps |
| Result | Cerulean | Brock 99%, Cerulean 85%, Misty 27% | **beat the game** (with some scripts: Surf/Strength/Flash/money) | Pikachu, then script collapse |

What those authors say was **critical**: the visited mask, coordinate exploration reward ("random play
never leaves Pallet Town after billions of steps"), treating *any interaction* as exploration (pure coord
reward wanders forever and never talks to Brock), swarming ("plagued us for months" without it), the
soft-capped level reward, recurrence, long/dynamic episodes. What **didn't matter much**: battle
rewards (A-spam emerges), fancy encoders.

Documented **reward hacks** to design against: staring at animated water/grass under pixel novelty
(Whidden v1), Leech Seed heal-farming (Pleines), 10× nav reward → never fights Brock, Gastly grinding in
Lavender Tower, PC deposits dropping level-sum → avoiding Pokémon Centers, discount bias in starter choice.

Other 2025–26 data points:
- **PokéAgent Challenge (NeurIPS 2025, Emerald):** winner = LLM-written scripted policies → distilled by
  imitation → refined by RL. 2nd = pure recurrent PPO with milestone rewards (~2× slower). Raw VLMs ≈ 0%.
- **Karten et al. (arXiv 2603.12145):** Rust port of PyBoy, 128 envs/process: PPO 14.5k vs 9.9k sps on
  32 cores (1.4–1.5×). No code release stated.
- **PokeRL (arXiv 2604.10812):** small-scale; anti-loop tile penalties + anti-button-spam cut loop episodes
  41% → 4.7%. Early game only.
- LLM agents (Gemini, Claude) have beaten Gen-1/3 games with harnesses, at ~10⁵ actions but weeks of
  wall-clock. Not comparable to our constraints; relevant only as a possible subgoal prior later.

---

## 4. Ranked diagnosis: why v2 stalls

1. **No exploration signal inside a map.** Only map-entry (+1) and rare flags pay. The literature is
   unanimous that per-tile novelty + a visible visited mask is the single biggest ingredient.
2. **Memoryless policy + clock input + deterministic boot = open-loop script.** The observed collapse is
   exactly what this setup predicts. Dialogue/menus/battles are non-Markov; the canvas is not memory.
3. **Entropy penalty when succeeding** (with constant LR, no KL stop, no value norm) actively drives the
   collapse that was then diagnosed.
4. **Episodes too short to matter.** 896 steps ≈ 6 min of game time; the Pikachu demo alone uses ~506.
   Pleines' agents need ~25k steps to reach Cerulean.
5. **Budget 10–100× too small.** ~10M steps/run vs 10⁸–10⁹ for Pokémon agents.
6. **The percept discards the screen.** Most of the canvas is 8×8-block upsampled 12×12 periphery; the
   learned gaze is worse than random. SUGARL (NeurIPS 2023) shows foveated agents only match/beat full
   observation *with* a full periphery — and only by small margins on Atari. On a 160×144 4-shade screen
   a full-frame CNN is essentially free, so foveation adds partial observability + a second credit
   assignment problem for no compute benefit.
7. **Action timing.** D-pad held all 24 frames (≈1.5 tiles, misaligned) vs the standard press-8/release;
   SELECT in the action space; no skipping of text/animation frames.
8. **Throughput left on the table** (§6).
9. **Complexity outran validation.** EMA-z invalidation, PTZ cadence derivation, priority-map at weight 0,
   neuromod framing, dashboards, cron — while the learned components (NEAT, gaze head) both failed and
   bugs slipped through (off-by-one RAM taps in v1, `brain_eval.py` arch mismatch).

---

## 5. Recommended plan

### Phase A — Reproduce the boring baseline (target: Brock, then Cerulean)

Goal: a known-good agent on our box, so every later idea is measured against something real.

- **Env** (new, small, separate from the canvas fleet):
  - PyBoy `window="null"`, `sound_emulated=False`; benchmark `cgb=False` vs CGB for Yellow.
  - Actions: 7 (↑↓←→ A B START), **press 8 frames, release, tick to 24**, render only the last frame.
  - Starting save state with **text speed FAST, battle animations OFF**.
  - Obs: 72×80 grayscale (`[::2, ::2]` of one channel, stable 4-shade mapping — drop per-frame shade-rank
    normalization), **+ 72×80 visited-mask channel**, + a small vector (party HP/levels, badge bits, event
    bits) read via the manifest. **No clock.**
- **Reward** (start from Pleines' 4 terms, they're cleaner than puffer's 35): +2 per event flag,
  +0.005–0.03 per new (map, x, y), heal term, soft-capped level term (slope ¼ above ~22). Add
  "interactions are exploration" (signs/warps/NPC dialogue) only if it stalls before Brock.
- **Policy:** CNN 32k8s2 → 64k4s2 → 64k3s2 (tile-sized kernels) + MLP for the vector → Linear 512 →
  **LSTM/GRU** (BPTT 16) → policy/value heads. Orthogonal init, LayerNorm.
- **PPO:** γ≈0.997–0.998, λ 0.95, clip 0.1–0.2, **ent 0.01 (never negative)**, lr 2–3e-4, batch ~65k,
  minibatch ~2k, 3 epochs, value/return normalization (or HL-Gauss two-hot critic), grad-clip 0.5.
- **Episodes:** dynamic (10,240 + 2,048 per event) or puffer-style "never reset the game; wipe
  exploration memory every ~20k steps". Jitter episode ends so envs don't reset in lockstep.
- **Swarm:** on a new required milestone, persist the state (in memory + on disk) and load it into all
  envs. This replaces the current backward curriculum and the scripted Pikachu demo.
- **Eval:** wire `eval/spine.py`; track % of runs reaching each milestone (Pikachu, Viridian, Brock,
  Mt. Moon, Cerulean) and steps-to-milestone, à la Pleines. Keep the screen-ablation probe
  (`coupling.py` blind_delta) — it's a genuinely good check.
- **Honesty check:** ~1B steps should be within reach in ~1 day *(est.)*; Pleines got Brock in ~5.6k
  agent-steps of game time and puffer got Brock in <30 min wall-clock on comparable throughput.

Pragmatic option: start from `pokemonred_puffer` itself (Red-targeted, MIT) and port to Yellow
(addresses via pret `pokeyellow.sym` + `pyboy.symbol_lookup`), rather than re-deriving everything.
Caveat: their `torch.compile` path uses Inductor, which **does not run on P100** (Triton needs sm_70+);
disable compile or use manual CUDA graphs (which we already have working).

### Phase B — Throughput (in parallel with A)

- **Measure first:** env-only sps (random policy, no learner) vs learner samples/s. Our own numbers
  (~22k raw frames/s/core → ~900 decisions/s/core) imply an env ceiling of *est.* 18–24k sps on 28
  physical cores; we sit around 7–8k (v1 furnace) and v2 is likely lower.
- **Env side:** multiple envs per worker (puffer: 288 envs / 24 workers); one worker per *physical*
  core; async EnvPool-style batching (run ~2× the envs of one inference batch so stepping overlaps GPU);
  uint8 obs, 2-bit packing, unpack on GPU (~64× less host→device traffic); compute reward in the worker
  from sparse RAM reads/PyBoy hooks, not full 8 KB WRAM copies; no per-step cell hashing.
- **Decision skipping:** auto-advance while the game ignores input (puffer uses `wJoyIgnore`; for us this
  goes in the manifest). Expect a large cut in wasted decisions.
- **Learner side:** keep torch on the cu126 line (Pascal wheels end at **2.14**; pin ≤2.14). No Inductor.
  CUDA graphs for inference + update; pinned memory + non-blocking copies; fp16 autocast is worth a test
  (P100 has native 2× fp16, no tensor cores, no bf16). Use GPU0 for inference/env-serving and GPU1 for
  learning, or run two seeds — not two architecturally different A/B arms with N=1 each.
- **Later / optional:** Rust/C++ emulator core (~1.4×, weeks of work). Beyond ~25k sps needs more cores.

### Phase C — Win back the invariants, one swap at a time

Each swap is an A/B against the Phase-A baseline at a fixed step budget, ≥3 seeds, reporting
milestone rates with CIs.

| Game-specific crutch in baseline | Generic replacement to try | Risk |
|---|---|---|
| (map, x, y) from RAM for visited mask + novelty | Learned or hashed position: hash of BG tilemap + hardware scroll regs; or **Latent Go-Explore** cells; or episodic count on 4-shade frame hashes with animated tiles masked | Pixel novelty is hacked by animated water/grass (Whidden v1) — must be tested, not assumed |
| Event flags / badges from RAM | v1's RAM progress-counter **miner** (`reward/miner.py`) feeding the manifest; LLM manifest proposes which counters matter | Off-by-one taps bit us before — validate with the from-boot gate |
| Hand-set reward weights | LLM-as-judge preference reward (`reward/preference.py`, Motif/ONI-style), used offline | Breaks "no pretrained priors" if we're strict; expect reward hacking |
| Swarm on "required" events | Swarm on any new mined milestone / archive frontier cell | Slower; swarm on noise |

The manifest stays the single contract: game-specific addresses live there, never in engine code, which
keeps "game-agnostic core" honest even in Phase A.

### Phase D — Research bets (only after A works)

- **Foveal retina as an *extra* stream** alongside the full-frame CNN, never the only pathway. Keep the
  existing work; just stop making it load-bearing.
- **One self-supervised auxiliary loss** (SPR-style latent prediction or inverse dynamics) to guarantee
  the encoder tracks dynamics. Modest gains expected.
- **Achievement-distillation-style** contrastive loss over milestones (Crafter: PPO 21.8% vs DreamerV3
  14.5% at 1M steps).
- **Plasticity:** shrink-and-perturb on long runs as the state distribution shifts across the game.
- **PQN** as a cheaper off-policy alternative; **STORM/DreamerV3-S** only as a side branch.
- **LLM subgoals → RL refinement** (the PokéAgent winner's pattern) if we relax the no-priors rule —
  the most plausible way past text-gated quests (Bill, Cut, Silph Scope).

### What to stop / retire

- The clock input, the entropy penalty, the 896-step cap, the scripted Pikachu demo + backward
  curriculum, the 8 strided RAM bytes, the canvas-only percept, N=1 architectural A/Bs.
- Move `train/loop.py` (NEAT) and the v1 dashboard plumbing out of the critical path.
- Don't build more retina machinery until a full-frame baseline exists to compare it against.

---

## 6. Hardware notes (56 threads, 2× P100 16 GB)

- We will be **CPU/emulator-bound**. A 2–5M-param CNN-LSTM at batch 65k fits one P100 easily *(est. 15–20k
  learner samples/s)*; the second GPU is best spent on a second seed or an aux/RND network.
- Pascal: no tensor cores, no bf16, no Triton/Inductor. CUDA graphs + fp16 autocast are the tools.
  PyTorch 2.14 (cu126) is the last Pascal wheel; CUDA 13 drops Pascal.
- Expected throughput: tuned PyBoy *(est.)* 12–18k sps; with a native core *(est.)* 18–25k; plus
  decision-skipping multiplies effective progress-per-hour.
- At ~12k sps *(est.)*: 1B steps ≈ 23 h. That's the right order of magnitude for Cerulean-and-beyond.

---

## 7. Yellow-specific cautions

- Many WRAM addresses differ from Red — use pret `pokeyellow` symbols via `pyboy.symbol_lookup` (our v1
  off-by-one taps came from using R/B addresses).
- Starter is a fixed Pikachu, which can't learn Cut — the route needs a Cut-capable catch (puffer chose
  Bulbasaur/Charmander for exactly this reason).
- Following-Pikachu changes the visuals and adds an interactable sprite near the player.
- Yellow runs in CGB mode under PyBoy by default; we currently read only the red channel. Test `cgb=False`.

---

## 8. Open decisions for the owner

1. **Do we accept RAM-derived features (coords, events) in Phase A** — via the manifest — to get a
   working baseline, then claw back genericity in Phase C? *(Recommended: yes.)*
2. **Build on `pokemonred_puffer` or rewrite the env/learner inside pokeIO?** *(Recommended: port
   puffer's env ideas into a new lean pokeIO env + learner; reuse our manifest, eval spine, dashboard.)*
3. **Save-state swarming:** acceptable under "pure learning"? It's not imitation, but it is a reset
   oracle. *(Recommended: yes; it's Go-Explore, which the roadmap already embraces.)*
4. **Fate of the foveal retina:** keep as a research stream (Phase D) or shelve.
5. **Any scripts allowed** (puffer still scripts Surf/Strength/Flash)? Decide before mid-game.
6. **brain6 results:** pull `runs/brain6r` and `runs/brain6g` from the training box before deciding
   anything — they're the only v2 data not in this repo.

---

## Sources

- Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden — *Pokémon Red via Reinforcement Learning*,
  arXiv 2502.19920 — https://arxiv.org/abs/2502.19920
- Rubinstein et al. — PokeRL write-up — https://drubinstein.github.io/pokerl/ ; code
  https://github.com/drubinstein/pokemonred_puffer (config.yaml, environment.py)
- Whidden — PokemonRedExperiments (v1, v2) — https://github.com/PWhiddy/PokemonRedExperiments
- PufferLib — https://arxiv.org/html/2406.12905 ; https://puffer.ai/blog/ppo/ ;
  https://puffer.ai/blog/engineering-4.0/
- PokéAgent Challenge (NeurIPS 2025) — https://arxiv.org/html/2603.15563
- Karten, Appapogu, Jin — Rust PyBoy port — https://arxiv.org/html/2603.12145
- PokeRL (small-scale) — arXiv 2604.10812
- PyBoy — https://github.com/Baekalfen/PyBoy
- DreamerV3 — https://arxiv.org/abs/2301.04104 ; Dreamer 4 — https://arxiv.org/abs/2509.24527
- EfficientZero V2 — https://arxiv.org/abs/2403.00564 ; STORM — arXiv 2310.09615 ; DIAMOND —
  https://arxiv.org/abs/2405.12399 ; TWISTER — https://arxiv.org/abs/2503.04416
- MR.Q — https://arxiv.org/abs/2501.16142 ; PQN — https://arxiv.org/abs/2407.04811
- Stop Regressing (HL-Gauss) — https://arxiv.org/abs/2403.03950 ; PPO plasticity (Juliani & Ash) —
  https://arxiv.org/abs/2405.19153
- Craftax — https://arxiv.org/abs/2402.16801 ; Achievement Distillation — https://arxiv.org/abs/2307.03486
- RLeXplore — https://arxiv.org/abs/2405.19548 ; Latent Go-Explore — https://arxiv.org/abs/2208.14928 ;
  Intelligent Go-Explore — https://arxiv.org/abs/2405.15143
- Motif — https://arxiv.org/abs/2310.00166 ; ONI — https://arxiv.org/abs/2410.23022
- SUGARL (active vision RL) — https://arxiv.org/abs/2306.00975
- Triton/Inductor sm_70 requirement — https://discuss.pytorch.org/t/torch-compile-triton-cuda-capability/182068
- PyTorch Pascal wheel deprecation notice —
  https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432

Unverified / flagged by the research: total step count for puffer's full-game run (*est.* ~250M);
puffer LSTM size (docs 128 vs config 512); PufferLib 3.0/4.0 trainer internals and Pascal compatibility;
the PyTorch 2.14 Pascal claim rests on one dev-discuss post.
