# [MF] Manifest / LLM Frozen Reward — Design of Record (2026-07-17)

**Status:** buildable spec. Thesis keystone: the game-agnostic PROGRESS signal.
Supersedes the three candidate designs + judge panels that fed this synthesis.

**Chosen base design:** **MF-Preference** — a Motif-style *frozen preference
potential* Φ over the pixel-derived controller latent, distilled from offline
LLM pairwise progress-preferences, entering selection as a **potential-based
shaping** term. Unanimous judge winner (34 / 33 / 32). Grafts folded in from
CaptionPotential (active-learning sampler, manifest weak-label cross-check,
latent-delta input, reserve-uniform guard) and MF-MM (anonymized
milestone-ordering as the *default text-only* preference substrate, shuffled-order
negative control, kNN-anchor cold-start fallback, engine-agnostic screen-digest
descriptor, never-crash discipline).

---

## 0. Why this design (one paragraph)

Today fitness is driven by pixel-cell novelty (Go-Explore), a game-blind RAM
miner, boot-competence, empowerment, and responsiveness — none robustly means
"playing WELL" for score/skill/puzzle games. MF supplies that missing signal by
learning a scalar **progress potential Φ(s)** from an offline LLM's *relative*
judgments ("which of these two short clips shows more progress toward completing
this game"). Relative preferences are the right primitive here for three
repo-grounded reasons: (1) they are **calibration-free**, so the miner's measured
init-saturation pathology (a counter that starts high reads as "done") has no
analog — a pairwise judgment carries no scale to saturate; (2) they distill into
an **ordinal** potential that slots directly into the existing scale-free
quantile selection algebra (`cohort_rank_normalize`, loop.py ≈2166) with no new
magnitude to hand-balance; (3) a forced-choice is the **lowest-variance** LLM
query, which matters because the served model (GLM-4.7-Flash, text-only today) is
noisy. Turned into reward via Ng-1999 potential-based shaping
`F_t = γ·Φ(s_{t+1}) − Φ(s_t)`, the term is provably policy-invariant: it cannot be
farmed by cycling and cannot move the exploration optimum.

---

## 1. Non-negotiable constraints — how each is met

| # | Constraint | Mechanism |
|---|---|---|
| 1 | **Frozen, never a live judge** | LLM runs async on card0 in a background `AnnotationWorker` that only appends labels to disk. The rollout queries ONLY a **frozen `PrefScorer` snapshot** on card1. Distill + snapshot-swap happen between generations every `pref_swap_gens` (default 10), mirroring the retina freeze-and-swap (loop.py ≈4160). |
| 2 | **Game-agnostic — no Yellow leak** | Hard **input wall**: Φ consumes only the pixel-latent slice `[z_periph(48) | z_fovea(32)]` (+ optional generic proprio + one-step latent delta). The game-specific `ram(8)` tail at controller-latent `[94:102]` is **sliced off** before the model's forward. RAM/manifest/game-name are used ONLY offline to *select/order/validate* pairs, then discarded (ONI/Motif: privileged info supervises, pixels infer). A unit test asserts `build_pair_prompt` contains no game nouns and that no RAM/manifest value enters `PrefModel.forward`. |
| 3 | **Self-tuning, no reward magic (§11b)** | The term's effective weight is `w_pref_eff = c_acc · w_pref_ref`, where `c_acc = clamp((EMA(val_acc) − 0.5)/0.5, 0, 1)` is the model's own dimensionless held-out reliability (chance → 0), and `w_pref_ref` **inherits the loop's existing add-on regime** (defaults to `reward.w_resp`, not a new literal). No fixed reward magnitude is introduced. γ derives from the episode horizon; label smoothing derives from LLM confidence; pair selection uses percentile/uncertainty and the existing per-env EMA-z surprise threshold — all data-calibrated. |
| 4 | **Cheap in the CPU loop** | Φ is a ~13k-param MLP evaluated on the controller latent `RetinaObsPipe.transform` ALREADY computes each step (zero marginal input cost). One tiny batched GEMM over ready envs on card1, sub-millisecond, plus a per-`cell_key` Φ cache. All annotation + distillation run off the hot path on card0. |
| 5 | **Isolated build first** | `reward/preference.py`, `reward/pairs.py`, `llm/annotate.py` build + unit-test in parallel with no loop import. loop.py wiring is a small, clearly-specified follow-on AFTER [AC]. A `pref_enable=False` default keeps the loop byte-identical; a dedicated `SeedSequence` child keeps `fast_reproduce` bit-identical. |

---

## 2. What the LLM annotates

**Pairwise forced-choice progress-preference over short K-frame clips** (K≈8
frames spanning ~1s; progress is a *transition*, not a state).

For each sampled pair `(clipA, clipB)` the LLM returns, via a jsonschema handed to
`LlamaClient.complete` (enum-enforced, prose rejected):

```json
{"winner": "A" | "B" | "tie", "confidence": 0.0-1.0}
```

The prompt uses a **fixed, generic rubric** — "Which clip shows more progress
toward completing whatever game this is, or tie?" — that names no game and no
mechanic. Each pair is annotated once at temperature>0; the on-disk client cache
(`runs/llm_cache/`, keyed by a stable clip-pair hash) makes re-runs free.

### 2a. The clip/pair substrate — text-only is the DEFAULT path

The served client is **text-only today** (client.py `complete(prompt, schema,
system)`, no image arg) and GLM-4.7-Flash is text-only. So the working v1 ships a
**text substrate**, and a multimodal image path is an *optional enhancement*, not
a dependency:

- **Default (text) substrate** — the prompt describes each clip with
  game-agnostic structured features: the **anonymized milestone-ordering digest**
  (grafted from MF-MM: discovered region/counter nodes presented as anonymized
  ids like `region_A`, `counter_3→L2`, ROM stem stripped), the **manifest
  progress-dimension deltas** across the clip (offline only), and **generic scene
  stats** (motion energy, novelty/cell-depth, ‖z_t − z_{t−1}‖ surprise). This is a
  strictly stronger fallback than a bare "manifest-delta" line and works TODAY.
- **Optional (vision) substrate** — when a multimodal card0 server is present,
  `client.complete(..., images=[...])` sends the two rendered clips as base64
  thumbnails. Gated on server capability; text path unchanged when absent
  (graceful degradation).

In BOTH substrates every game-specific fact lives only in *label generation* and
is **discarded** — the frozen Φ is trained to predict the preference from PIXELS
(the latent) alone.

### 2b. Pair selection — active learning + reserved uniform

Clips are drawn from data the loop already produces every generation: the
between-gen `_retina_collect` segments (per-env periph+fovea rollouts), the
champion replay trace (env slot 0), and Go-Explore archive cells (diverse anchors
with depth quantiles). `pairs.py` selects pairs by five game-agnostic strategies,
capped at `pref_pairs_per_round` (mirrors the goexplore caps-per-round throttle):

1. **temporal** — two clips one trajectory apart by `dt` (weak "later ≥ earlier"
   prior the LLM confirms/overturns);
2. **depth-straddle** — a shallow vs a deep archive cell (archive depth-quantile);
3. **manifest-straddle** — clips straddling a change in a manifest
   progress-dimension (highest-information; offline only);
4. **milestone-order** *(graft, MF-MM)* — pairs straddling a discovered milestone
   boundary, pre-ordered by the anonymized DAG; a structured text-only
   supervision channel that works without a VLM;
5. **uncertainty / active-learning** *(graft, CaptionPotential)* — pairs where the
   CURRENT frozen Φ is nearest 0.5 AND where an *eventful frame* fired (a mined
   counter moved via `ram_addresses_from_manifest`, or per-env EMA-z latent
   surprise exceeded its self-calibrated threshold), spending the small LLM budget
   on near-decision-boundary transitions.

**Reserve guard (graft):** a fixed fraction `pref_uniform_frac` of each round's
budget is drawn uniformly at random so active learning never starves Φ of
idle/negative examples.

**Manifest weak-label cross-check (graft):** when the manifest independently
confirms a counter moved across a pair in the direction the LLM chose, that pair's
training confidence is raised; on disagreement it is lowered. A cheap offline
de-noiser that tightens the accuracy gate and works text-only. This never becomes
a model *input* — it only reweights the label.

---

## 3. The frozen reward model

### 3a. `PrefModel` (Bradley-Terry potential)

- **Input:** the pixel-latent slice `[z_periph(48) | z_fovea(32)]` = 80-d, plus
  the **one-step latent delta `(z_t − z_{t−1})`** (graft: progress is a
  transition) and optionally generic proprio(14). The `ram(8)` tail is
  **explicitly excluded** (input wall). In foveal mode (no retina latent) the
  input is the **engine-agnostic screen-digest descriptor** (16×14 quantized
  screen ⊕ cell-depth scalar; graft from MF-MM) so MF is not retina-locked.
- **Architecture:** MLP `in → 128 → 64 → 1`, SiLU + LayerNorm, ~13k params.
  Output a single scalar Φ(s). A clip's score is the mean Φ over its frames.
- **Objective:** Bradley-Terry `P(A≻B) = σ(Φ(A) − Φ(B))`, trained with
  confidence-weighted soft-label BCE (LLM confidence sets label smoothing; ties →
  0.5 target), AdamW, on a held-out split of accumulated pairs. Because BT
  constrains only *differences* of Φ, its absolute scale is unidentified — which
  is exactly why the downstream blend must (and does) use rank/quantile, never the
  raw value.
- **Cold-start fallback (graft):** a **kNN-over-anchors** table (Φ =
  DAG-order-fraction of the nearest satisfied milestone anchor) warm-starts Φ and
  bounds pathological MLP extrapolation off-distribution when few labels exist.
  Anchors are supervision/warm-start ONLY — never a model input, preserving the
  input wall.
- **Buffer alignment:** the buffer stores RAW clip frames (game-agnostic pixels);
  Φ is (re-)fit on latents produced by the CURRENT frozen retina snapshot. On a
  retina snapshot swap, the buffer is re-encoded before the next distill so Φ
  stays aligned to the encoder the population is actually fed.

### 3b. `PrefScorer` (frozen inference)

Exposes `.snapshot()` / `score_batch(latent) -> np.ndarray` mirroring
`retina.snapshot()` / `encode_np`, plus a per-player **potential-return
accumulator**. CPU-portable `state_dict` for the checkpoint.

---

## 4. Fitness integration

Φ enters **Ledger-2 selection fitness ONLY** — never Ledger-1
exploration/archive credit, so Go-Explore frontier growth is untouched.

**Per-player credit = TRUE telescoping potential-shaping return** (NOT
max-over-trajectory — the decisive robustness call; a peak formulation lets a
single transient high-Φ frame score):

```
pref_i = Σ_t ( γ · Φ(s_{t+dt}) − Φ(s_t) )        over sampled steps (subsample-safe)
```

It is passed into `cohort_rank_normalize` and added exactly like the existing
`w_resp` / `w_emp` terms, INSIDE the policy channel (NOT the miner-gated progress
channel — that would make MF silent precisely when the miner finds nothing, the
case MF exists to cover):

```
policy_i = gate_i · q(A_i) + w_resp · q(R_resp_i) + w_emp · q(Emp_i) + w_pref_eff · q(pref_i)
sel_i    = (1 − w_prog) · q(policy_i) + w_prog · q(progress_i)      [progress blend untouched]
```

Because it enters as a within-cohort quantile rank (never a raw scalar) it is
scale-free and cannot drown exploration — it re-orders within a cohort, it does
not add magnitude. The dominant multiplicative backbone `gate·q(A)` still governs
task credit; pref is only an additive, re-quantiled nudge.

**Optional A/B (graft, decided by the acceptance harness, not by hand):** a
spawn-relative **max-advance** readout `max_t Φ(z_t) − Φ(z_spawn)` may credit a
genome that reached deep then died better than endpoint-only telescoping under
max-fitness evolution. Both enter as scale-free quantiles; keep whichever wins the
transfer-AUC metric. Default is telescoping.

---

## 5. Self-tuning (§11b) — no reward magic

```
c_acc      = clamp( (EMA(val_acc) − 0.5) / 0.5 , 0, 1 )      # dimensionless reliability, chance→0
w_pref_eff = c_acc · w_pref_ref                               # w_pref_ref defaults to reward.w_resp
```

- `c_acc` is the model's own **held-out pairwise accuracy**, EMA-smoothed across
  swaps (reuses the §11b EMA/percentile machinery). A chance-level model
  contributes exactly 0; a reliable one ramps up.
- `w_pref_ref` is **not a new hand-tuned magnitude** — it inherits the loop's
  existing policy add-on regime (the already-configured `w_resp` scale). At full
  reliability the pref term carries the same weight a trusted responsiveness term
  does; when unreliable it is muted. No new reward-magnitude literal survives.
- γ derives from the episode horizon; label smoothing from LLM confidence; the
  active-learning surprise threshold reuses the existing per-env EMA-z machinery.
  The only new config scalars (`pref_swap_gens`, `pref_pairs_per_round`,
  `pref_uniform_frac`, buffer caps) are cadence/budget knobs, not reward
  magnitudes.

**Verify before trust:** a unit test drives a synthetic *gamed/noisy* label stream
and asserts `w_pref_eff → ~0` at chance before the term is trusted in-loop.

---

## 6. Manifest generate → consume loop

- **GENERATE** (untouched): `pokeio/manifest/generate.py::generate_manifest`
  labels mined WRAM candidates into a validated `Manifest`
  (`fallback_manifest` guarantees validity offline). No schema change.
- **CONSUME** (the first real consume beyond `ram_addresses_from_manifest`):
  `pairs.py` reads `progress_dimensions` to (a) **stratify** informative pairs
  (manifest-straddle + eventful-frame active learning), (b) supply the
  **anonymized milestone-ordering digest** and manifest-delta text substrate for
  the prompt, and (c) provide the held-out validation axis for the acceptance
  metric and dashboard. A tiny helper `eventful_frames(manifest, wram_trace)` is
  added to `generate.py` reusing `ram_addresses_from_manifest`, keeping the
  manifest the single game-specific contract.
- The manifest is consumed **only offline**, in label generation and pair
  selection. Its game-specific labels/addresses **never** enter Φ's input or
  weights.

---

## 7. Freeze + refresh cadence

Mirrors the validated retina freeze-and-swap discipline exactly (loop.py
≈4160–4200):

1. A background `AnnotationWorker` on card0 drains a pair queue → LlamaClient
   verdicts → appends to `runs/<id>/pref/labels.jsonl`. The CPU/card1 loop NEVER
   blocks on it.
2. At each generation boundary the loop enqueues freshly sampled pairs from that
   gen's collection segments / champion / archive (capped).
3. Every `pref_swap_gens` (default 10): read all labels ready, run
   `pref_train_steps` of BT distillation on card0, compute held-out pairwise
   accuracy → `c_acc`, then **swap the frozen `PrefScorer` snapshot at the gen
   boundary** — with the SAME **skip-the-final-gen guard** the retina uses
   (`gen < gens - 1`), so the population selected this gen always matches the Φ it
   evolved under (the off-by-one). Between swaps Φ is a stationary evolution target
   (anti-drift §4.5).
4. On a retina snapshot swap, re-encode the buffer with the new snapshot before
   the next distill.
5. The learner + optimizer + frozen snapshot (+ buffer manifest) are persisted in
   the pickle checkpoint via `_pref_build_ckpt` mirroring `_retina_build_ckpt`, so
   resume continues the SAME Φ, not a re-warmed one.

---

## 8. Anti-reward-hacking — fixed at the source, not with a weight

1. **Potential-based shaping (Ng-Harada-Russell 1999):** the credit telescopes to
   a bounded function of endpoint potentials, so any returning cycle nets exactly
   0 — the canonical "farm the boundary" exploit is impossible by construction,
   and the term provably cannot shift the exploration optimum.
2. **Relative, not absolute:** pairwise preferences carry no scale, so the miner's
   measured init-saturation failure mode has no analog — nothing to saturate.
3. **Accuracy gate:** `w_pref_eff` scales with held-out accuracy; a model that
   learned to be gamed, or is unreliable on a new game, collapses toward chance and
   self-zeroes.
4. **Stationary target:** Φ is frozen between swaps and consumes the frozen latent,
   so the population cannot chase a moving/self-referential reward within a window.
5. **Rank-only, additive under the dominant gate·q(A) backbone:** the term is
   re-quantiled and cannot dominate by magnitude.
6. **Input wall:** Φ reads the SSL retina latent, whose encoder the policy does not
   control (frozen), so genomes cannot adversarially reshape z's meaning.

---

## 9. Game-agnosticism guarantee

**Agnostic-by-pipeline, not agnostic-by-frozen-transfer** (honest framing per the
judge guidance). Game-agnosticism here means **no game-specific CODE and no
authored Yellow semantics; identical pipeline on any ROM** — NOT one frozen net
for all games. The retina SSL encoder is itself trained on the current game's
pixels, so Φ over its latent does not transfer zero-shot; the real bar is "same
pipeline, retina retrained per game."

Three hard walls make it hold:

1. **INPUT wall** — Φ consumes only the pixel-latent slice (+ delta + optional
   proprio); the `ram(8)` tail is sliced off. No RAM byte, address, manifest label,
   milestone string, or game name is a feature of the frozen weights.
2. **LABEL wall** — the annotation rubric names no game and no mechanic; the
   manifest/RAM side-channel is offline and discarded (privileged info supervises,
   pixels infer). Even a "Pikachu fainted" observation reduces to the generic
   preference the model learns from pixels.
3. **MANIFEST wall** — the manifest's game-specific labels touch only clip
   selection / pair stratification / label validation, never model input or target.

Swap the ROM → the miner finds that game's counters, the cell graph finds that
game's regions/anchors, the LLM orders/prefers them, the SAME code re-fits that
game's Φ. A Yellow constant cannot leak because none is referenced in the
substrate input, the prompt, or the scorer.

---

## 10. Isolated-files-first implementation plan

### Isolated modules (build + unit-test in PARALLEL, no loop import)

| File | Change | Contents |
|---|---|---|
| `pokeio/reward/preference.py` | **NEW** | `PrefModel` (BT MLP on the `[z_periph\|z_fovea]` slice + latent-delta, ram sliced off); `train_bt(labels, encode_fn)` (confidence-weighted BCE + held-out accuracy); kNN-anchor warm-start; `PrefScorer` (frozen `score_batch` + telescoping potential-return accumulator); `snapshot()`/freeze; `_pref_build_ckpt`/`_pref_load_ckpt` mirroring `_retina_build_ckpt`. |
| `pokeio/reward/pairs.py` | **NEW** | Clip extraction from retina segments + champion trace + archive cells; the five pair strategies (temporal, depth-straddle, manifest-straddle, milestone-order, uncertainty/active-learning) + reserve-uniform guard + manifest weak-label cross-check; dedup + per-round cap; stable pair-hash for cache keys; engine-agnostic screen-digest fallback descriptor. |
| `pokeio/llm/annotate.py` | **NEW** | `PAIR_SCHEMA` (`winner ∈ {A,B,tie}` + confidence); `build_pair_prompt` (fixed generic rubric, NO game constants); `annotate_pair(client, clipA, clipB)`; background `AnnotationWorker` draining a pair queue on card0 → `labels.jsonl` (async, never blocks); verdicts cached via the client's on-disk cache. |
| `pokeio/manifest/generate.py` | **EDIT (tiny)** | Add `eventful_frames(manifest, wram_trace)` + anonymized milestone-ordering digest builder, reusing `ram_addresses_from_manifest`. No schema change. |
| `pokeio/config/__init__.py` | **EDIT (small)** | `RewardConfig` / new `PrefConfig` fields: `pref_enable` (default False → byte-identical legacy), `pref_swap_gens` (mirrors `retina.swap_gens`), `pref_train_steps`, `pref_pairs_per_round`, `pref_uniform_frac`, `pref_gamma`, `pref_buf_cap`, `pref_val_frac`, `pref_subsample_k`, `pref_infer_card`, `w_pref_ref` (defaults to `w_resp`; NOT a new operating magnitude). |
| `pokeio/llm/client.py` | **EDIT (small, optional)** | Optional `images` arg on `complete`/`_request` to send base64 image parts for a multimodal server; text-only path unchanged when omitted. |
| `tests/test_preference.py`, `tests/test_pairs.py`, `tests/test_annotate.py` | **NEW** | BT convergence on synthetic monotone labels; telescoping/potential-invariance of the shaping return; accuracy-gate → weight 0 at chance/gamed labels; pair-strategy + dedup + cache-key stability; assert `build_pair_prompt` contains NO game constants and no RAM value enters `PrefModel.forward`. |

### Follow-on loop.py wiring (small, sequenced AFTER [AC])

`pokeio/train/loop.py` — mirror the retina collect/train/swap block:

1. Build `PrefScorer` + `AnnotationWorker` at `train()` start (retina precedent).
2. During the eval wave, accumulate the telescoping potential-return per genome
   by piggybacking the per-cycle latent scatter (the same `X_lat` the retina obs
   already produces — no new forward pass); store as `g._pref`.
3. Pass `pref` + `w_pref_eff` into `cohort_rank_normalize` (≈2166) as an additive
   `q(pref)` term next to `w_resp`/`w_emp` — in the policy channel, not the
   progress channel.
4. At the swap boundary (between-gen block ≈4160): enqueue sampled pairs, distill
   on card0, recompute `c_acc`, swap the frozen snapshot with the skip-final-gen
   guard (`gen < gens - 1`); re-encode buffer on retina swap.
5. Telemetry: `reward_terms['pref_acc' / 'pref_best' / 'w_pref_eff']`.
6. Persist the pref snapshot in the checkpoint alongside retina.

`pokeio/train/checkpoint.py` — **EDIT**: add a `pref=None` param stored under a
`"pref"` key (mirror the existing `retina` handling), round-tripping
learner+optimizer+frozen snapshot (+ buffer manifest) so resume/`fast_reproduce`
continue the same Φ. Missing key → `None` → safe re-warm.

---

## 11. fast_reproduce determinism note

The frozen Φ term is just another `reward_terms`/genome-attribute entry, exactly
like `g.progress` and `g._resp`. `g._pref` is computed from the FROZEN snapshot
(stationary between swaps) over obs the wave already produced, **draws NO rng**,
and is consumed by `cohort_rank_normalize`, which runs BEFORE `fast_reproduce`.
`fast_reproduce` is the only rng consumer at the generation boundary and is
unchanged, so the reproduce rng stream is bit-identical with MF on or off — the
same reason the retina snapshot needs no `fast_reproduce` edit.

All stochastic parts (pair sampling, LLM annotation, BT distillation) use a
**dedicated `SeedSequence` child derived from `config.run.seed`**, entirely
separate from the mutation/crossover reproduce stream. The swap boundary is
deterministic (`every pref_swap_gens, gen < gens-1`). With `pref_enable=False` the
loop is byte-identical to today.

---

## 12. Acceptance metric — prove Φ is a REAL, game-agnostic progress signal

All on a FROZEN Φ; report IQM + CIs via the `pokeio/eval/` credibility harness
([CR]).

1. **Held-out concordance (input-wall proof).** On held-out champion replays of
   the training game, Spearman ρ between Φ's terminal potential and a **manifest
   counter it never read as input** (badges/level) exceeds a run-measured floor —
   Φ predicts human-labeled progress from PIXELS alone.
2. **Shuffled-order negative control** *(graft, MF-MM).* The real model must beat a
   **label-shuffled** control on held-out pairwise accuracy — proving the
   *ordering* carries the signal, not merely the descriptor.
3. **Cross-game agnosticism** *(honest bar).* Run the UNCHANGED pipeline (retina
   **retrained**) on a second GB ROM (Tetris/Mario): with zero code change it must
   produce a Φ whose held-out pairwise accuracy clears chance+margin. As a leak
   guard, the training-game's **un-refreshed** scorer applied to game-B
   trajectories must score **≈ chance** — the signal is discovered per-game, not a
   leaked game-A prior.
4. **Accuracy-gate safety.** On a synthetic gamed/noisy label stream,
   `w_pref_eff → ~0` — failure is safe (self-zeroes), never harmful.
5. **In-loop lift.** Turning MF on lifts `boot_eval_cells` / manifest-counter
   advancement of the champion vs an MF-off control at equal wall-clock, WITHOUT
   collapsing `archive_delta` (exploration preserved).

---

## 13. Risks + mitigations

1. **Text-only under-informs.** The default text substrate routes supervision
   partly through counter-deltas/scene-stats; mitigated by the milestone-ordering
   digest (structured, game-agnostic) and the accuracy gate (bad labels →
   weight ~0). Upgrades cleanly when a VLM lands.
2. **Cold-start.** Early gens have few labels and a weak Φ; the accuracy gate keeps
   it near-zero (no early harm, no early help), and the kNN-anchor warm-start
   bounds extrapolation.
3. **Manifest quality.** Manifest-straddle degrades to temporal/depth/milestone
   strategies when the miner finds no good counters — MF still works, just with
   less targeted pairs.
4. **LLM progress-bias on puzzles** (visual "more action" ≠ progress). Potential
   shaping bounds the damage (policy-invariant) and the gate demotes an inaccurate
   model.
5. **Buffer re-encode on retina swap** adds a bounded between-gen cost, capped by
   `pref_buf_cap`.
6. **Annotation outrunning the loop** on a very fast run leaves stale (still-valid,
   older) snapshots; the swap-every-N + on-disk cache absorb this.
