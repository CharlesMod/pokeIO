# The Game-Playing Brain — Biomimetic Architecture of Record

**Goal:** the best game-playing brain we can build — beat SOTA — by mapping the system onto real
neuroanatomy, so each module has a defined job and the *missing* regions are obvious. Three cooperating
systems (System-1 reflex, System-2 deliberation, and the evolutionary process that shaped them) realized
as brain regions, on a high-throughput core (20k+ actions/s → millions with scale).

Legend: ✅ built · ◐ partial · ✗ missing.

---

## The regions

### Perception — Occipital lobe + eye-movement control  ✅ BUILT (the part we have)
- **Retina / LGN** → screen capture + periphery/fovea split.
- **V1/V2/V4 (feature hierarchy)** → foveal encoder (edges/motion/features).
- **Dorsal "where" stream (V5/MT)** → motion sheet.
- **Superior colliculus** (reflexive orienting) → reflex gaze (motion × staleness).
- **Frontal eye fields** (voluntary saccades) → evolved top-down gaze modulation.
- **Parietal trans-saccadic memory** → the persistent foveal buffer (change-blindness, corollary discharge).
- *Gap inside vision:* **Ventral "what" stream / inferotemporal (IT)** — explicit object/entity recognition
  ("this is an NPC / a menu / a Pokémon"). Currently implicit; may be covered by System-2's vision. ◐

### Reinforcement learning — Basal ganglia + dopamine  ✗ MISSING (the core we're adding)
- **Striatum (actor) + dopamine (critic / reward-prediction-error)** → the gradient-RL actor-critic. Dopamine
  RPE *is* temporal-difference learning; this is the biological RL engine. We ran neuroevolution here
  instead — the wrong tissue for this job. **This is System-1's learning core.**
- **Reward signal ("dopamine")** → the real progress signal (RAM game-state milestones + evolved reward
  program), NOT the exploration proxy that failed.

### Motor — Motor cortex + premotor/SMA + cerebellum  ◐ PARTIAL
- **M1 (execution)** → the fast policy's button output.
- **Premotor / SMA (action sequencing, motor chunks)** → adaptive cadence (commit-gate, dwell) — action
  chunking/timing. ✅ (built, from the AC work)
- **Cerebellum (forward model, fine timing, error-correction)** → a *cheap* motor-consequence predictor
  (predict the sensory result of an action) — smooth execution + a lightweight forward model that gives us
  a slice of planning without a full MuZero world model. ✗ (candidate addition)

### Memory — Hippocampus + entorhinal cortex  ◐ PARTIAL (map yes, replay no)
- **Place/grid cells → cognitive map** → the Go-Explore cell archive (the map of the game's states). ✅
- **Episodic memory** → stored trajectories in the archive.
- **Replay / consolidation** (offline replay of experience to train cortex) → **backward robustification**:
  replay the archive's deep trajectories from progressively earlier starts to grow from-boot competence.
  This is the missing half of Go-Explore and the hippocampus's signature trick. ✗→◐ (the key new work)

### Executive / deliberation — Prefrontal cortex  ✗ MISSING (System-2)
- **Dorsolateral PFC (planning, working memory, goal maintenance, reasoning, "mental calculations")** →
  the **async vision-LLM director**. Sets goals/strategy over the long horizon; the net executes the latest
  directive between updates. This supplies the long-horizon credit assignment neither RL nor evolution can.
- **Working memory** → the LLM's context + a persistent goal/plan state the policy is conditioned on.

### Cognitive control — Anterior cingulate + orbitofrontal  ✗ MISSING (the coordinator)
- **Anterior cingulate (ACC): conflict/error/effort monitoring → recruit PFC** → **the trigger that decides
  WHEN System-2 re-plans** (net is stuck / surprised / a novel screen / low value). This is the biomimetic
  answer to the "when does the async LLM wake up?" problem — a real architecture gap with a real brain
  region as the solution. ✗ (high value, currently undefined)
- **Orbitofrontal (OFC): value / expected-reward** → the critic / value model. ◐

### Neuromodulation — the global self-tuning layer  ✗ MOSTLY MISSING (elegant no-knobs win)
The mandate "no hand-tuned behavior knobs" is *literally* what neuromodulators do — they self-regulate
global parameters from the system's own signals:
- **Dopamine (VTA/SNc)** → reward / learning signal (see basal ganglia).
- **Norepinephrine (locus coeruleus)** → explore↔exploit gain + arousal, driven by surprise/uncertainty →
  our exploration temperature / novelty weighting, self-set from prediction error. ✗
- **Acetylcholine (basal forebrain)** → attention + learning-rate gain under expected uncertainty →
  self-modulated learning rate / attention gating. ✗
- **Serotonin** → patience / temporal discounting → self-tuned γ / dwell persistence. ✗
Building these as a small **neuromodulatory controller** (state signals → global gains) is how explore/exploit,
learning rate, discount, and attention get set *without hand-tuned constants* — the mandate, realized.

### Communication — Thalamus + Global Workspace  ✗ MISSING (the bus)
- **Thalamus (central relay + attention gate)** + **Global Workspace (Baars/Dehaene: the shared broadcast
  where regions compete for and share a common representation)** → **the inter-region communication protocol**:
  how System-2's goal reaches the policy, how the ACC's "stuck" signal reaches the PFC, how memory is read/written.
  In a multi-module brain this plumbing is not optional — it's what makes the regions one brain rather than
  three programs. ✗ (essential; currently undefined)

### Amygdala — salience / value tagging  ◐ minor
- Tags states with valence/urgency (danger, big reward) to prioritize memory + attention. Partially present
  as motion-salience; value-tagging of *memorable/valuable* states is a small add. ◐

### Phylogeny + development — EVOLUTION (not a region — the process that grows the brain)  ◐ REFRAME
Brains are *evolved*. Evolution's honest home here is the outer, throughput-hungry loop that shaped the brain
and its innate drives — and it's where evolution genuinely escapes local minima that gradient descent can't:
- **Evolved innate reward / drives ("instincts")** → the evolved reward program (selected against true
  from-boot progress), the game-agnostic differentiator.
- **Quality-Diversity (MAP-Elites / Go-Explore / novelty)** → diverse exploration that won't collapse into one
  basin — SOTA hard-exploration, and throughput-scaled.
- **Population-based (ERL / PBT)** → a population of policies escaping the local optima a single gradient
  learner falls into.
These are throughput-bound → the 20k+/s (→ millions) core is the fuel that turns evolution from garnish to engine.

---

## Missing / under-developed, prioritized

1. **Basal ganglia (RL core)** — the dopaminergic actor-critic. The learning engine; being added now. ✗→◐
2. **Prefrontal cortex (async LLM planner, System-2)** — long-horizon reasoning + goals. ✗
3. **Anterior cingulate (re-plan trigger / conflict monitor)** — the System-1↔2 coordination signal. ✗
4. **Neuromodulatory layer (DA/NE/ACh/5-HT)** — self-tuned explore/exploit, learning rate, discount, attention
   = the no-hand-tuned-knobs mandate, realized. ✗
5. **Thalamus / Global Workspace** — the inter-region communication bus. ✗
6. **Hippocampal replay → consolidation into the policy** (backward robustification). ✗→◐
7. **Cerebellum (cheap forward model / motor timing)** — a slice of planning without a full world model. ✗ (optional)
8. **Ventral-stream / IT object recognition** — "what am I looking at" (maybe covered by System-2 vision). ◐
9. **Amygdala value-tagging** — prioritize valuable/memorable states. ◐ (minor)

## How the three systems map on
- **System-1 (reflex, realtime):** occipital + basal ganglia + motor/cerebellum — see, value, act, fast.
- **System-2 (deliberation, async):** prefrontal + ACC — plan, reason, decide when to intervene.
- **Evolution (phylogeny, outer loop):** evolved drives + QD + population — escape local minima at scale.
- **Shared organs:** hippocampus (memory/map), neuromodulators (global tuning), thalamus/workspace (the bus).

## ⚠ RED-TEAM REVISION (Fable, 2026-07-18) — this section governs the build

The map above names the *destination*; it is NOT a parallel build schedule. This project's prior death was
an abundance of **unwired** components (gen-90 policies ≈ random init; Go-Explore restore harvested the
fitness the policy should have earned). Nine regions at once repeats that with a nervous-system diagram on
top. **v1 is scoped to the System-1 core ONLY, gated on one question; nothing else is built until it passes.**

### The gate (the whole project rides on this)
**Does from-boot competence on Yellow climb off the plateau (boot gauntlet rising from ~4.4%) with System-2
ABSENT?** Yes → the reflex learns and the rest earns its place. No → no brain region fixes a learning signal
uncoupled from perception + action; the diagnosis is the reward or the percept, not missing regions.

### The ONE thing to get right (what killed 90 gens; RL alone does NOT fix it)
The reward/credit signal must be **COUPLED to what the policy perceives and does, and not harvestable without
competent play.** Requirements: **dense**, **decoupled from save-state teleport**, **never fed to the agent as
an input** (the miner-Goodhart trap), **validated by a from-boot metric**. Instrument the coupling PERMANENTLY
(obs→action MI, RAM-tap ablation Δ, no-restore selection eval) — the failure is silent.

### Two absent pieces that come FIRST (co-equal with the reward)
- **A working from-boot progress metric** — boot gauntlet ≈4.4% ≈ noise; backward-robustification, evolved
  reward, and QD selection all depend on a from-boot signal that barely exists. Build it before rewarding.
- **The evaluation spine** — rliable IQM+CIs, a pre-registered 2nd game, a frozen-core git-diff protocol,
  noise-frame + random-weight controls. Without it we cannot tell competence from Goodhart.

### Region ruling (v1 vs deferred/demoted)
- **Basal ganglia (gradient actor-critic)** — the ONE load-bearing mapping. **v1 core.**
- **Hippocampus** — archive as **TRAJECTORIES not save-state blobs** + backward-robustification. **v1, with the RL core.**
- **Occipital** — reuse (built), but **verify the percept carries motion/fovea into the gradient**. **v1 (validate).**
- **Motor/premotor** — actor output + reuse AC cadence. **v1.**
- **Thalamus/global-workspace** — DEMOTED: it's the **IPC we already have**. No bus in a System-1-only v1. **Deferred.**
- **Neuromodulation** — NOT a subsystem (a controller-with-weights is itself a knob-bag). **Inline self-
  calibrating scalars** (NE→explore-temp EMA; ACh→LR; 5-HT→γ), one line each. **Woven in, not a region.**
- **ACC (re-plan trigger)** — sound, wrong time. **Deferred with System-2.**
- **PFC = live async LLM director** — premature + contradicts the project's own conclusion (LLM = distill-then-
  FREEZE offline, never a live judge; GLM already eats card 0). v1 LLM job = **offline frozen reward/manifest**. **Async-director deferred.**
- **Cerebellum forward model** — model-based creep, sm_60-hostile. **Deferred (explicit bet).**
- **Amygdala** — fold into **archive eviction priority** (a bug-fix, not a region).

### Pascal honesty
2× P100, **no tensor cores**, card 0 = the LLM → **~1 P100 of training compute**, not 2. Every GPU-hungry region
contends for that one card. State it.

### Throughput reality (revises the 20k+/millions line)
Training already ≈ **7k sps at max pace** (the "0.7k" was the realtime *spectator* budget). ~**9–11k** = PyBoy
engineering ceiling (GPU-side encode + pipelined forward are **nearly free — GPUs are idle — but do NOT move the
ceiling**, which is **CPU frame-stepping**); ~16–20k = hard PyBoy bound; **50k+ needs a native C core
(gambatte/binjgb)** — multi-week FFI; "millions" = many boxes. **Target ~10k for v1; native core ONLY after
competence moves AND we're proven throughput-bound.**

## Build order (Fable-revised — v1 = steps 1–4; DO NOT start 5 until the gate passes)
1. **From-boot progress metric + evaluation spine** (prerequisites — else we optimize noise).
2. **Reward signal** — dense, teleport-decoupled, never-an-input, coupled to perception+action + permanent coupling instrumentation.
3. **System-1 core** — gradient actor-critic on the existing percept + Go-Explore-as-trajectories + backward-robustification, on the current ~7–10k furnace engine.
4. **THE GATE** — from-boot competence climbs off ~4.4% with System-2 absent. Everything below is blocked on this.
5. *(after gate)* throughput demand-driven (native core iff throughput-bound) → offline frozen reward distillation → System-2 (PFC + ACC) → neuromodulation-inline + evolutionary outer loop (QD/ERL/PBT) at scale.
