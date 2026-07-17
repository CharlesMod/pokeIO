# pokeIO — Network-Efficacy + Active-Vision Redesign (Waves B & C): Implementable Spec

**Status:** final build spec. Deterministic. Where a number appears it is a committed default, not a suggestion. File/line anchors are verified against the current tree (`/home/cmod/pokeIO/pokeio/`).

**Owning decision (from the judge):** build **Phase 0 this session** — it inverts every measured symptom at near-zero compute over mostly-existing code. Then execute the **learned retina + biomimetic saccade** as the committed **Phase 1/2** target under per-feature smoke gates. **Ship E2+E3 before E1. Do not build a world model.**

---

## 1. Rationale, tied to the live diagnosis

The gen-100 `runs/live1` pathology has one shape: the controller was handed **584 undifferentiated low-resolution inputs and no reason to wire them**, and **selection was decoupled from policy** so a screen-blind constant at a lucky Go-Explore spawn out-scored a seeing policy at a bad spawn. Concretely measured: champion wires 111/576 pixels, output-std across 400 screens ≈ 0.005, all 9 sigmoid outputs bunched in [0.3,0.7], zeroing the 8 RAM taps → Δoutput = 0.000000 for all 224 genomes, `fitness_best` flat since gen 30.

Four levers, each mapped to a cause:

| Diagnosis cause | Fix | Ships |
|---|---|---|
| Screen-blind flat perceptron (111/576 wired) | Compact **foveated** obs (real acuity where it looks) + wiring pressure | Phase 0 |
| RAM taps topologically dead | **connect+protect** taps at init + decisive TANH head | Phase 0 |
| Fitness decoupled from policy (spawn luck) | **Split ledgers + per-cell baseline advantage (E2) + blind-ablation gate (E3)** | Phase 0 |
| User's vision ask (biomimetic saccade/foveation) | Evolved `(dx,dy)` saccade over the fovea, integrated through existing recurrence | Phase 0 (pixels) → Phase 1 (latent) |
| Sample efficiency (idle P100 → control-relevant features) | Decoder-free **SPR + inverse-dynamics** learned retina, FSQ latent | Phase 1/2 |

**Invariant preserved:** the retina and the FSQ codebook are **frozen sensory organs, never part of the genotype**. The NEAT graph remains the sole evolved, watchable artifact. `population_forward_sparse` re-clamps input slots and bias every hop (`forward.py` docstring line 11), so injecting a latent as the input vector needs **zero forward-pass change**.

**Why SSL is mandatory, not optional (state this to pre-empt the objection):** the strongest "just augment an end-to-end CNN" result relies on backprop from the loss into pixels. NEAT is gradient-free — there is no backprop path from fitness to an encoder — so self-supervision is the *only* way an encoder becomes control-relevant.

---

## 2. Observation tensor — coarse periphery + movable fovea + motion + proprioception

### 2.1 Phase 0 (`vision.mode = "foveal"`, primary ship): pixel obs

Replace `ObsEncoder` (`emu/fleet.py:388`, `dim = res*res + n_ram`) with a **stateful `FovealEncoder`** in the same file (workers already import it). It produces a **fixed 454-dim float32 vector** with a contiguous, named block layout. All streams are grayscale in [0,1] via the existing `_area_matrix` resample (`fleet.py`, mirrored in `vision/preprocess.py:36`); shade normalization reuses `ObsBuilder.normalize_shades` (`preprocess.py:101`).

Let `G = vision.periph_grid = 12`, screen `H=144, W=160`, fovea native side `F = vision.fovea_native_px = 48`.

| Block | Slots | Dims | Source |
|---|---|---|---|
| `periphery` | `[0:144]` | G²=144 | full 144×160 → area-resample to G×G |
| `fovea` | `[144:288]` | G²=144 | **native F×F crop centered at gaze `(gy,gx)`**, then area-resample F→G (zero-padded if the crop overhangs an edge) |
| `motion` | `[288:432]` | G²=144 | `(periphery_t − periphery_{t-1} + 1)/2`; 0.5 when no prev (stateful) |
| `proprio` | `[432:446]` | 14 | efference copy (below) |
| `ram` | `[446:454]` | 8 | mined tap bytes (`vision.obs_ram_bytes`), **connect-protected trailing block** |
| **n_in** | | **454** | `encoder.dim` |

**Proprio (14), all in [−1,1]:** `[gx/W·2−1, gy/H·2−1, dx_prev, dy_prev, held_up, held_down, held_left, held_right, last_A, last_B, last_start, last_select, last_noop, step_frac]` — 2 gaze + 2 last-saccade (efference copy) + 9 last-button one-hot + 1 episode-step fraction. This is what makes glimpse integration over the evolved recurrence work: the controller always knows where it is looking and what it just did.

**Fovea acuity is the point:** an `F=48` native crop resampled to `12×12` carries ~4× the effective resolution over the fixation region versus the `12×12` periphery. The flat 24×24 threw this away.

**State & the three build sites (must emit byte-identical vectors or novelty/tap keys desync):**
1. Parent encoder — `loop.py:1874` (`encoder = ObsEncoder(...)`).
2. Barrier worker `_emit` — `fleet.py:637`.
3. Async worker `_emit` — `fleet.py:1134`.

`FovealEncoder` holds per-env `_prev_periphery` and current `(gy,gx)`; `.reset()` on every episode/restore boundary (mirror `VisionEnv.reset`, `vision/env_wrap.py:47`; reset gaze to screen center `(72,80)`). **Fix the existing tap inconsistency while here:** the barrier `_emit` (`fleet.py:637`) does *not* overlay mined taps; the async `_emit` (`fleet.py:1140-1143`) does. Both must overlay taps onto the `ram` block identically, or connect-protected taps get garbage in the barrier path.

### 2.2 Phase 1 (`vision.mode = "retina"`): latent obs, same block contract

The `periphery`+`fovea`+`motion` pixel blocks are replaced by the learned latent `[z_periph(48) | z_fovea(32)]`; `proprio(14)` and `ram(8)` are unchanged. **n_in = 102.** Flat order `[z_periph 0:48 | z_fovea 48:80 | proprio 80:94 | ram 94:102]`. Fresh-population restart (see §12).

**Resume-compat (both phases):** changing `n_in` (or `N_OUT`) shifts the global node-id convention (`genome.py:15-24`) and every innovation number → **old checkpoints are unloadable; population starts fresh.** The Go-Explore/novelty archive is keyed on raw screen+wram (`archive.py:123`) and **survives** — the explored frontier carries across the restart. Add an explicit resume guard: at `loop.py` train entry (`:1837`, near the `obs_dim=encoder.dim` snapshot at `:1995`) assert the checkpoint's `(n_in, N_OUT)` equals the freshly built pair; refuse with a clear error on mismatch.

---

## 3. Saccade action encoding, output-node layout, env application

### 3.1 Output layout

**`N_OUT: 9 → 11`** at `loop.py:71`. `Genome.output_ids()` (`genome.py:94`) and `InnovationTracker.n_io` (`genome.py:122`) resize automatically off `n_out`.

- `out[..., 0:9]` — **button head**, Discrete-9 (`up,down,left,right,A,B,START,SELECT,NOOP`), consumed by `argmax`.
- `out[..., 9]` — saccade `dx` (tanh).
- `out[..., 10]` — saccade `dy` (tanh).

**Head activation:** `output_act = TANH` (§8). Buttons argmax over unbounded tanh drive (decisive, not bunched sigmoids); the two saccade outputs are read as continuous tanh directly.

### 3.2 The 8 argmax sites

Replace `out[:, 0, :].argmax(dim=1)` at **all 8 action sites** — `loop.py:523, 650, 862, 878, 890, 1058, 1062, 1298` — with a split: `button = out[..., :9].argmax(...)`; `dx, dy = out[..., 9], out[..., 10]`. (`loop.py:2286` `fits.argmax()` is champion *selection*, leave it. `loop.py:1236` random-action baseline samples buttons only: keep `rng.integers(0, 9)`.)

### 3.3 Env application — dynamics and clamp

Gaze is a **velocity on the fovea center**, integrated through the controller's existing recurrence (state already threads across agent-steps; `recurrent_memory=True`). `GAIN = vision.saccade_gain = 32` px/step, applied every `vision.saccade_every_k = 1` steps in Phase 0:

```
gx ← clip(gx + GAIN * tanh(out_dx),  F/2, W - F/2)   # = clip(.., 24, 136)   (column)
gy ← clip(gy + GAIN * tanh(out_dy),  F/2, H - F/2)   # = clip(.., 24, 120)   (row)
```

`(0,0)` gaze at center; clamp keeps the F×F window fully on-screen. **No REINFORCE** — the saccade is evolved end-to-end with the buttons under one fitness (the documented hard-attention instability is sidestepped because we are already an ES). One-tick efference-copy delay is intentional and biological: the saccade emitted from obs *t* crops the fovea of obs *t+1*.

### 3.4 The one non-trivial seam: action plumbing

Today only a scalar `actions` int32 per env crosses shared memory (`fleet.py:619/1107` allocate `("actions",(n,),int32)`; workers call `env.step_fast(int(actions[gi]), ...)` at `:699/:1221`; parent writes at `step_all` `:971`). The worker builds the fovea crop, so it needs the gaze. **Add two sibling shared-memory arrays** `gaze_dx (n,) float32`, `gaze_dy (n,) float32` (exact, no quantization). Parent writes `tanh(out_dx/dy)` after the argmax split; the worker, at the top of its step, updates its per-env `(gy,gx)` per §3.3 **before** `_emit` builds the obs, and writes the new gaze + last `(dx,dy)` into the proprio block. Register the arrays in both `BarrierFleet` and `AsyncFleet` alloc blocks and both worker readers.

---

## 4. Learned retina (Phase 1/2) — decoder-free SSL, FSQ latent, off-genotype

Rewrite `evo/retina.py`. It is currently an **MSE reconstruction autoencoder** (`train_step` → `F.mse_loss(recon, batch)` at `:199`; decoder `from_z`/`decode` at `:167/:185`) — the reconstruction trap for mostly-static Pokémon tiles/text. **Remove the decoder.** Keep the scaffolding: `RetinaStacker` (`:114`), `downscale` (`:84`), `build_stack` (`:92`), `encode_np` (`:223`), `fit` (`:205`).

### 4.1 Class interface

```python
class Retina(nn.Module):
    def __init__(self, z_periph=48, z_fovea=32, spr_k=5, ema_tau=0.0,
                 inverse_dynamics=True, fsq_levels=(8,8,8,5,5), n_act=9): ...
    # inference (card1, no_grad):
    def encode_np(self, periph_stack, fovea_stack) -> np.ndarray   # -> (z_periph+z_fovea,) fp32
    def fsq_code(self, periph_stack) -> np.ndarray                 # -> (5,) int, the cell code
    # training (card0):
    def train_step(self, batch, opt) -> dict                       # SPR + inv-dyn losses
    def snapshot(self) -> "Retina"                                 # frozen copy for the population
```

### 4.2 Architecture (Nature-CNN, ~1M params)

- **Input:** periphery 4-frame stack → 84×84 → `[B,4,84,84]`; fovea 48×48 native → resized 84×84 → `[B,4,84,84]`. **Single shared-weight encoder** over both (do not split magno/parvo until a smoke shows the split pays).
- **Convs:** channels `[32,64,64]`, kernels `8/4/3`, strides `4/2/1`, ReLU → `[B,64,7,7]`.
- **Heads:** global-pool + linear → `z_periph=48` (from periphery), `z_fovea=32` (from fovea). Policy latent = `concat = 80`.

### 4.3 SSL loss (SPR + inverse dynamics; decoder-free)

- **SPR latent self-prediction**, horizon `K = spr_k = 5`. Transition model: 2× 64-ch 3×3 conv on the 7×7 map, BN after the first conv only; the **11-way action one-hot broadcast to every spatial cell**. Projection head 256-d (online) + **asymmetric linear predictor** (do not remove — collapse guard). Target encoder = **EMA of online**, `ema_tau = 0.0` (hard copy) with augmentation. Loss = normalized cosine `−Σ_k cos(pred_{t+k}, sg(tgt_{t+k}))`, weight `λ = 2`. **Cosine, never L2.**
- **Inverse-dynamics head:** MLP predicts the button (9-way cross-entropy) from `(z_t, z_{t+1})`. Forces features onto agent-controllable content, away from HUD/animation. **This network is `q_ψ`, reused directly for the empowerment term (§6).**
- **Augmentation:** random shift ±4 px + intensity jitter (scale 0.05). **No horizontal flips** (they invert left/right button semantics). No decoder, no VAE.

### 4.4 FSQ latent (not VQ)

Project `z_periph` → 5 dims, quantize each to levels `fsq_levels = [8,8,8,5,5]` → **10,240 codes**. FSQ cannot codebook-collapse by construction and needs none of the VQ machinery (commitment loss, EMA codebook, dead-code reseed). The integer 5-tuple **is** the Go-Explore cell code (§5), computed on periphery so cells are invariant to where the fovea points.

### 4.5 Training loop, GPU cards, cadence (anti-drift discipline)

A drifting encoder is the original fitness-decoupling bug in a new guise — genomes chasing a moving feature space. Guard exactly:

1. **Warm-up:** background learner trains on a replay buffer of the swarm's own frames for `retina.warmup_frames = 200000` frames **before** any genome consumes the latent. Genomes never see a random-init encoder's output.
2. **Freeze-and-swap:** the population is fed a **frozen `.snapshot()`**, swapped only every `retina.swap_gens = 10` generations. Between swaps the feature space is stationary relative to evolution. On each swap: recompute FSQ cell codes for the archive (§5), reusing the streamer/goexplore re-pointing discipline already in the tree (commit 21754b0).
3. **Cards:** learner on **card0** consuming replay; population inference on the frozen snapshot on **card1** (matches the existing split). Batch all 224 agents' periphery+fovea stacks into one forward per step. P100 (GP100) has 2:1 fp16, but at ~1M params the encoder is never the bottleneck (the ~10k env-step/s CPU wall is) — fp32 is fine.

### 4.6 How the controller consumes it while staying the visible genotype

The frozen retina runs under `torch.no_grad()` on card1 → `[z_periph|z_fovea]`; the parent concatenates `proprio + ram` → the 102-d input vector → `population_forward_sparse` unchanged. The genome topology is the only evolved object and is *more* legible than before (102 labeled input axes vs 584 pixels). The retina is separately watchable (feature-map / FSQ-code viewer) — a sensory organ, not a gene.

---

## 5. FSQ latent → Go-Explore cell

- **Phase 0 and default everywhere (`goexplore.cell_source = "pixel"`):** keep `NoveltyArchive.cell_key` unchanged (`archive.py:123`, screen digest 16×14 @ 4 levels + strided/masked wram digest). Obs-independent, proven, survives the obs/topology restart. **Permanent safety net.**
- **Phase 2 option (`cell_source = "fsq"`):** the cell key is `retina.fsq_code(periphery).tobytes()`. Because worker-side cell keys can't see the encoder cheaply, when `fsq="fsq"` the **parent** recomputes cell keys from the periphery latent of the `screens` it already receives back from workers (parent already runs card1 inference), overriding the worker-shipped pixel keys before `archive.add`. On every snapshot swap, re-key the archive (recompute codes for stored cells) at the swap boundary only, never mid-generation. If FSQ churns the archive in smoke (§11), flip one flag back to `"pixel"`.

---

## 6. Fitness / selection redesign — couple credit to policy

Two ledgers that **never share a scalar** (Go-Explore's frontier≠robust-policy lesson).

**Ledger 1 — exploration credit (archive only).** The rarity-novelty of newly discovered cells (`novelty.py:140/:188`, `floor + 1/√(1+prior_visits)`) stays exactly as-is and continues to grow the frontier and pick restore cells. It is **not** the genome's selection fitness.

**Ledger 2 — policy fitness (drives NEAT selection).** Computed per genome from the wave, per the formulas below.

### 6.1 E2 — per-cell baseline-subtracted advantage (default ON, zero added env-steps)

`base_depth` is already tracked (`loop.py:480/:491/:534`, `_sample_restores` `:747/:752`, async `:819`) but never subtracted. For each restore cell `c` with the set `M_c` of genomes restored into it this generation:

```
|M_c| ≥ 2 :  A_i = f_i − (Σ_{j∈M_c} f_j − f_i) / (|M_c| − 1)      # leave-one-out
|M_c| = 1 :  A_i = 0                                              # singleton → no counterfactual
newgame   :  A_i = f_i                                            # shared fixed spawn: no cross-spawn luck
```

where `f_i` is the wave rarity fitness. Optionally smooth `b(c)` across generations with `reward.baseline_ema = 0.9` for recurring cells. This is the COMA/difference-reward counterfactual; it exactly cancels the measured cross-cell spawn-value gradient ("a constant policy at a good spawn out-scores a seeing policy at a bad spawn").

### 6.2 E3 — blind-ablation gate (default ON; dominant multiplier; doubles as smoke gate)

Maintain a rolling probe buffer `P` of ~256 recently-observed real obs vectors (sampled across the population each gen; fixed within a gen). Run **one batched** `population_forward_sparse` over `P` twice for all genomes at once (reuses the compiled `cp` the loop holds): (i) real obs, (ii) **optical blocks `[0:432]` zeroed, `proprio`+`ram` preserved**.

```
Δ_i    = mean over P of || a_i^real − a_i^blind ||₁        # over the 11-d output
gate_i = sigmoid(β · (Δ_i − Δ_min)),   β = reward.blind_gate_beta = 8,  Δ_min = reward.blind_gate_dmin = 0.05
```

`Δ_i ≈ 0` ⟹ screen-blind ⟹ `gate_i ≈ 0` ⟹ fitness crushed. Causal (RAM held constant → isolates optical dependence). Two extra batched forwards per gen; no env-steps.

### 6.3 R_resp — responsiveness (near-free, reuses outputs already computed)

Over each genome's rollout accumulate the 9-button argmax histogram and the running mean `ō_i` / variance of the raw output vector:

```
R_resp_i = H(button_hist_i)  +  λ_resp · mean_t || o_t − ō_i ||²,   λ_resp = reward.w_resp_var = 0.5
```

`H` = Shannon entropy in bits (max `log2(9)=3.17`). A constant-button agent → `H≈0` → killed.

### 6.4 Emp — empowerment (Phase 1+, one shared net/gen on card0)

Reuse the SPR inverse head `q_ψ(a | z_t, z_{t+1})` (§4.3):

```
Emp_i = mean_t [ log q_ψ(a_t | z_t, z_{t+1}) − log π_i(a_t | z_t) ]
```

Rewards buttons that predictably change the next frame; constant agents → 0. Inverse form only (far less noisy-TV-prone; GB is near-deterministic). `w_emp = 0` in Phase 0.

### 6.5 The combined selection transform (extends `cohort_rank_normalize`, loop.py:1130)

Within each spawn cohort (restored / newgame), map each component to `(rank+0.5)/n` quantiles via the existing `_quantile_ranks` (`loop.py:1119`), then:

```
policy_i = gate_i · q(A_i)  +  w_resp · q(R_resp_i)  +  w_emp · q(Emp_i)
           # w_resp = reward.w_resp = 0.1,  w_emp = reward.w_emp = 0.1 (0 in Phase 0)

# blend with mined progress exactly as today (progress_weight, config.reward:129),
# but only when some player registered progress this gen (existing `blend` guard):
sel_i = (1 − w_prog) · q(policy_i) + w_prog · q(progress_i)   if progress live
      = q(policy_i)                                            otherwise

genomes[i].fitness = sel_i
```

`gate` is the dominant multiplier; `w_resp/w_emp` shape rather than replace task credit. Feed `A_i` (not raw `f_i`) as the "novelty" input to the extended function. Tune by watching the `Δ_i`/`Emp_i` distributions shift upward across gens — they must, if the terms bite.

### 6.6 E1 — separate no-restore eval (default OFF; judge override)

`reward.policy_eval_steps = 0` disables it. E2+E3 alone kill the measured pathology at zero added env-steps. Enable a short (`policy_eval_steps` e.g. 400) `evaluate_wave(restore_prob=0.0, goexplore=None)` from a shared rotating start-bank **only** if E2+E3 leave residual *within-cell* spawn luck in the smoke. ~90k extra env-steps/gen is real cost on the binding constraint — hence off by default.

**Model-error firewall (standing invariant, even though the world model is cut):** selection ground truth is the real wave; the archive frontier grows from **real frames only**. No imagined sample ever grows the frontier or overrides `sel_i`.

---

## 7. RAM-tap connect+protect at init

The mined taps reach obs (data plane OK: `fleet.py:1140-1143` async, and the barrier fix in §2.1) but are topologically dead. Fix in the genome factory.

- **Signature:** `make_genome(..., n_ram: int = 0, n_proprio: int = 0)` (`genome.py:153`). Pass `n_ram = config.vision.obs_ram_bytes`, `n_proprio = 14` at the loop call (`loop.py:1920-1921`).
- **`connect="full"` (Phase-0 primary, §8):** every input incl. all taps and proprio is wired at init — but the current full branch (`genome.py:185-191`) uses **unscaled** `weight_scale=1.0` (the documented saturation trap in the docstring). **Fan-in-scale the full branch:** per-edge `std = weight_scale / √(n_in + 1)`. That makes full unsaturated with a TANH head and *guarantees* every latent dim, every proprio input, and all 8 taps are wired.
- **`connect="sparse"` (fallback):** after the `k` random picks (`genome.py:196-205`), unconditionally add an edge from every id in `range(n_in − n_ram − n_proprio, n_in)` to every output (fan-in-scaled), so taps+proprio are live from gen 0.
- **Protect from disable in both modes:** add `protected: bool = False` on `ConnGene` (`genome.py:68`); set `True` on tap→output (and proprio→output) edges. Exclude protected edges from `mutate_toggle` (`ops.py:150`) and from `mutate_add_node`'s split-candidate list (`ops.py:133`, so a protected shortcut is never severed). Evolution may re-weight taps but not cut the channel.

---

## 8. Mutation-rate / init_k / decisive-head changes (concrete values)

- **Decisive head:** `output_act = TANH` at the `make_genome` call (`loop.py:1920-1921`; default is `SIGMOID` at `genome.py:161`). Buttons argmax over spread tanh drive; saccade reads tanh directly. `evo.softmax_temp = 0.0` (0 = plain argmax; >0 enables temperature-softmax over `out[:9]` at the 8 sites for early exploration — off by default).
- **Connect mode:** `evo.init_connect = "full"` (primary, with the fan-in fix in §7). Sparse fallback keeps `evo.init_k = 32` (raised from 12; `loop.py` init-conns budget at `:1881` and `init_std` at `:1933` follow).
- **Weight regime (`loop.py:1933-1936`):** `init_std = 1/√(n_in+1)` for full, `1/√(init_k+1)` for sparse; perturb `sigma = 0.1·init_std`, `reset_scale = init_std`, `clamp = 4·init_std` (matches the `MutationRates` guidance comment at `ops.py:45-53` — keeps the stationary weight distribution at init scale, preventing the silent re-saturation that undid the last run).
- **Wiring pressure:** `evo.mutate_add_conn = 0.4` (from 0.05; `config.evo` `:95`, applied `loop.py:1936`). **Prefer-unconnected source sampler** in `mutate_add_connection` (`ops.py:112`, currently uniform `rng.choice(src_pool)`): build a weight over `src_pool` where input nodes with zero out-edges get weight `evo.prefer_unconnected_weight = 4.0` and all others `1.0`; sample proportionally. `evo.mutate_add_node = 0.03` unchanged.

---

## 9. Config knobs (names + defaults)

Add to `config/__init__.py` (`VisionConfig:58`, `EvoConfig:86`, `RewardConfig:109`, new `RetinaConfig`, new `GoExploreConfig`).

```yaml
vision:
  mode:              foveal        # {foveal, fovea_static, retina}   ← phase/fallback selector
  periph_grid:       12            # G: periphery/fovea/motion resample side
  fovea_native_px:   48            # F: native fovea crop side (resampled F->G)
  saccade_gain:      32            # px/step velocity on the fovea center
  saccade_every_k:   1             # gaze update cadence (ticks)
  proprio:           true          # 14-d efference-copy block
  obs_ram_bytes:     8             # unchanged; trailing connect-protected taps
  motion_channel:    true          # unchanged
evo:
  n_out:             11            # 9 buttons + 2 saccade
  output_act:        tanh          # decisive head (was sigmoid)
  init_connect:      full          # fan-in-scaled full (primary); sparse fallback
  init_k:            32            # sparse fallback fan-in
  mutate_add_conn:   0.4           # was 0.05
  mutate_add_node:   0.03
  prefer_unconnected_src:    true
  prefer_unconnected_weight: 4.0
  protect_ram_taps:  true
  protect_proprio:   true
  softmax_temp:      0.0           # 0 = plain argmax over tanh
reward:
  restore_baseline:  true          # E2 per-cell leave-one-out advantage
  baseline_ema:      0.9
  blind_gate:        true          # E3
  blind_gate_beta:   8.0
  blind_gate_dmin:   0.05
  w_resp:            0.1
  w_resp_var:        0.5           # λ on output-variance inside R_resp
  w_emp:             0.1           # Phase 1+; 0 effect in Phase 0
  policy_eval_steps: 0             # E1 off by default (judge override)
  progress_weight:   0.5           # unchanged blend with mined progress
retina:
  enable:            false         # Phase 1+
  z_periph:          48
  z_fovea:           32
  spr_k:             5
  ema_tau:           0.0
  loss:              cosine
  inverse_dynamics:  true
  fsq_levels:        [8,8,8,5,5]   # 10,240 cells
  warmup_frames:     200000
  swap_gens:         10
  train_card:        0
  infer_card:        1
  aug_shift_px:      4
  aug_jitter:        0.05
goexplore:
  cell_source:       pixel         # {pixel, fsq}; pixel is the permanent safety net
```

---

## 10. Module / file change list (mapped to real seams)

| File | Change |
|---|---|
| `emu/fleet.py` | `ObsEncoder`→`FovealEncoder` (stateful periphery/fovea/motion/proprio, gaze-driven crop, tap overlay) `:388`; add `gaze_dx/gaze_dy` float32 shared arrays in both fleet alloc blocks (`:619/:1107` region) and both worker readers (`:699/:1221`); apply gaze update before `_emit` (`:637` barrier — **also add the missing tap overlay here**, `:1134` async); parent writes gaze in `step_all` `:971`. |
| `evo/genome.py` | `make_genome(..., n_ram, n_proprio)` `:153`; fan-in-scale the `full` branch `:185-191`; connect-protect taps/proprio in `full`+`sparse`; `protected` field on `ConnGene` `:68`. |
| `evo/ops.py` | prefer-unconnected src sampler in `mutate_add_connection` `:112`; skip `protected` edges in `mutate_toggle` `:150` and `mutate_add_node` split-candidates `:133`. |
| `evo/forward.py` | **no change** (inputs/bias re-clamped every hop; latent injection is transparent). |
| `train/loop.py` | `N_OUT=11` `:71`; split button/saccade at the 8 argmax sites (523,650,862,878,890,1058,1062,1298); `make_genome` call `:1920` (`output_act=tanh`, `connect=full`, `n_ram`, `n_proprio`); weight regime `:1933-1936`; extend `cohort_rank_normalize` `:1130` with E2 `A_i` + `gate` + `R_resp` + `Emp` (§6.5); compute `Δ_i` probe (§6.2); resume guard on `(n_in,N_OUT)` near `:1995`; optional E1 no-restore eval; (Phase 1) retina warm-up/snapshot/swap phase + card0 learner (no hook exists today). |
| `reward/novelty.py` | attribute rarity strictly to the cell/archive ledger; expose per-genome raw `f_i` for the E2 baseline (no credit-semantics change). |
| `reward/archive.py` | (Phase 2) optional FSQ cell segment behind `cell_source` `:123`; parent-side re-key on snapshot swap. |
| `evo/retina.py` | rewrite MSE autoencoder → decoder-free SPR + inverse-dynamics + FSQ (`:144/:194/:199/:167/:185`); add `fsq_code`, `snapshot`; keep `RetinaStacker`/`downscale`/`build_stack`/`encode_np`. |
| `config/__init__.py` | all §9 knobs. |
| `vision/preprocess.py`, `pipeline/gpu_vision.py`, `evo/substrate.py`, `evo/cppn.py` | reuse `_area_matrix`/`normalize_shades`/motion from `preprocess.py`; substrate/CPPN/HyperNEAT **out of scope** (direct-encoded NEAT stays — keeps evolution maximally visible). |

---

## 11. Unit-test plan

1. **`test_foveal_encoder`** — `FovealEncoder.encode` returns 454 dims with the exact block boundaries; parent `encode` and both worker paths produce **byte-identical** vectors (checksum); motion resets to 0.5 after `.reset()`; gaze clamps at `[24,136]×[24,120]`; moving `(gy,gx)` changes only the `fovea` block `[144:288]`.
2. **`test_saccade_plumbing`** — a scripted `(dx,dy)` sweep walks the fovea crop across the screen; `gaze_dx/gaze_dy` float32 round-trip through shared memory exactly; proprio records the applied `(dx,dy)` and new `(gx,gy)`.
3. **`test_genome_taps`** — every genome from `make_genome(n_ram=8, n_proprio=14, connect="full")` (and `"sparse"`) has all 8 tap→output and 14 proprio→output edges live; **zeroing the tap inputs yields Δoutput > 0 for 100% of genomes** (the exact inverse of the diagnosis's Δ=0); `mutate_toggle`/`mutate_add_node` never sever a `protected` edge over 10k mutations.
4. **`test_head_decisive`** — with TANH + fan-in-scaled full init, on 400 real screens the top-2 button gap exceeds 0.05 for ≫ half the population (vs 101/224 coin-flips) and no output saturates to exact ±1 at init.
5. **`test_blind_gate`** — a hand-built constant-button genome scores `Δ≈0`, `gate≈0`; a genome wired to the fovea scores `Δ≫Δ_min`, `gate≈1`.
6. **`test_selection_credit`** — two genomes restored to the same cell with identical policy but different spawn get equal `A_i` (within rank noise); a screen-blind genome at a lucky restore no longer out-ranks a seeing genome at a bad restore.
7. **`test_forward_latent_injection`** — an arbitrary 102-d latent vector drives `population_forward_sparse` with no code change and inputs re-clamped each hop.
8. **(Phase 1) `test_retina_spr`** — SPR cosine loss decreases; latent does **not** collapse (per-dim std > ε on a held-out batch); inverse-dynamics accuracy > 1/9; FSQ code entropy stays high (no dead codes) over warm-up.

---

## 12. Smoke acceptance gate (go/no-go before any long run)

Run a **20-generation** Phase-0 restart. All of the following must hold, or drop a rung (§13):

- **Input-sensitivity (required, vs the gen-100 baseline ≈ 0.005):**
  - `output_std_vs_screen` = mean over the 9 button outputs of their std across a fixed 400-real-screen probe set, for the champion, must exceed **0.05** (≥10× the 0.005 baseline).
  - **I(obs;action) proxy** rises above baseline: fit a tiny logistic `q(a|obs)` on the champion's rollout transitions and report `Î = H(a) − H(a|obs)` in bits; require champion **Î ≥ 0.10 bits** (baseline ≈ 0). Equivalently, the population-median blind-ablation `Δ` must clear **Δ_min = 0.05** and shift upward across the 20 gens.
  - **RAM taps live:** zeroing the 8 tap inputs changes the output for **≥ 95%** of genomes (inverts the diagnosis's Δ=0 for all 224).
- **Behavior off its stuck values:** champion action-entropy rises above **0.23**; `fitness_best` is no longer flat.
- **Fovea-saccade movement:** over a 500-step champion rollout, the fovea center visits **≥ 8** distinct 8-px-quantized positions and gaze-path length > 0 (the saccade actually moves and is not pinned to a clamp corner).
- **No crash:** 20 gens complete; a checkpoint saves and reloads; the `(n_in,N_OUT)` resume guard passes.

---

## 13. Phased fallback ladder

Every rung is a shippable stopping point. Only the learned retina and the biomimetic saccade depth are at risk; the two non-negotiables — **compact foveated inputs** and **selection coupled to seeing** — land at Phase 0 with almost no compute.

**Phase 0 — SHIP THIS SESSION (zero gradients, zero drift, whole cure to the measured pathology).**
Foveal pixel obs (454-d) + evolved `(dx,dy)` saccade + TANH decisive head + fan-in-scaled `full` connect + connect/protect RAM taps + `mutate_add_conn=0.4` + prefer-unconnected sampler + **split ledgers + E2 baseline advantage + E3 blind gate + R_resp** + existing pixel Go-Explore cells. Files: `evo/genome.py`, `evo/ops.py`, `train/loop.py`, `emu/fleet.py`, `reward/novelty.py`, `config`.
- *Sub-fallback if the evolved saccade misbehaves (smoke movement/entropy regresses vs a static run):* `vision.mode="fovea_static"`, `N_OUT→9`, fovea static-centered + periphery (peripheral+static-fovea already beats full-frame). Removes recurrence-integration risk, keeps acuity.
- *Floor of the floor:* periphery-only 12×12 + proprio + taps — still spatially structured, still E2+E3-gated. Guaranteed better than the diagnosed baseline because E2+E3 alone decouple fitness from spawn and kill screen-blindness.

**Phase 1 — ADD THE LEARNED RETINA + BIOMIMETIC SACCADE (the user's aggressive path).**
Rewrite `evo/retina.py` to decoder-free SPR + inverse-dynamics + FSQ; add the card0 warm-up/snapshot/swap phase. Replace fovea/periphery *pixels* with the latent (n_in=102); add the **empowerment** term (reuses `q_ψ`); optional FSQ cells behind `cell_source`, pixel cells stay default. Ships only after `test_retina_spr` passes and a 20-gen smoke beats Phase-0 champion metrics.
- Biomimetic layer, each behind its own smoke gate (adopt the "earns its keep + has a degradation rung" discipline): **saccadic suppression** (zero `z_fov` + set a suppression proprio bit on saccade ticks — justified only by SPR-target stability); **log-polar fovea** (option; Cartesian crop fallback — its near-affine-under-saccade payoff only exists once the SPR encoder exists); **k-tick fixation dwell** (`saccade_every_k≈4`, ~3–4 saccades/s). Keep single shared-weight encoder until a smoke shows a magno/parvo split pays.
- *Retina sub-fallbacks:* SPR unstable → BYOL/SimSiam single-step (no transition model) → **frozen random-init CNN** (a real, immediately-stable baseline) with pixel cells; FSQ unstable → classic downsample-and-quantize pixel cells; log-polar unstable → Cartesian crop.

**Phase 2 — QD PROTECTION + FSQ EXPLORATION (only after Phase 1 beats Phase 0).**
MAP-Elites niche grid over BC `[Δ, Emp, action-entropy, mean-motion-magnitude]`, elite-per-cell (a lucky blind agent can never out-average a seeing one in its niche; the filling grid is a watchable evolution-visible artifact). Turn on `cell_source="fsq"` so Go-Explore explores in latent space, guarded by an archive-collapse smoke, with the one-flag revert to pixel cells always available.

**Cut / deferred:** the VQ-RSSM world model + imagination evals — not built. Revisit only as a speculative Phase 3+ behind the model-error firewall, and only if the retina+selection stack is healthy and the ~10k-step/s CPU wall is still binding.