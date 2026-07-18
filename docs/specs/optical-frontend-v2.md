# Optical Front-End v2 — Sharp Fovea + Reflex Gaze + Evolved Modulation

**Status:** buildable spec (design of record). Motivated by a measured defect + a design fix the user
approved (2026-07-17). Successor to the coarse-vision foveal spine `live1` runs on; `live1` stays up as
the baseline to beat. Preserves the **cross-game transfer constraint** (`pokeio-cross-game-transfer`):
the observation stays hand-defined and game-agnostic, just sharper.

---

## 0. The defect this fixes

The foveal observation cannot resolve sprites/text — even a human can't read the "what the AI sees" panel:

| stream | content | resolution |
|---|---|---|
| periphery `[0:144]` | whole 160×144 screen → 12×12 | ~13 px/cell (blurry thumbnail) |
| **fovea `[144:288]`** | 48×48 gaze crop **area-resampled 48→12** | **4 px/cell** |
| motion `[288:432]` | whole-screen frame-diff → 12×12 | ~13 px/cell |

A Game Boy tile/character is 8×8 px → a **2×2 smudge** in the fovea, sub-pixel in the periphery. The
"high-res fovea" downsamples its crop 4×, defeating its purpose. (The retina spine, which "actually
played" in the head-to-head, upsamples the *same* crop to 84×84 — 7× sharper per axis; much of
"retina plays, foveal wall-hugs" was **acuity**, not learned features.) Coarse vision is a ceiling on
everything else (TC/AC/MF can only exploit what the controller can see).

**Two coupled fixes:** (A) make the fovea actually sharp; (B) because a small sharp fovea is useless if
aimed badly, give gaze a competent **reflex** front-end and let the controller **evolve top-down
modulation** on top. Decoupling "where to look" (self-supervisable, game-general) from "what to do"
(evolved, task-driven) also fixes the current terrible credit assignment (gaze gets credit only through
the same sparse fitness as buttons).

---

## 1. The three-part decomposition + trans-saccadic memory

| module | question | training |
|---|---|---|
| encoder | *what a glimpse looks like* | hand-defined foveal (sharp fovea) |
| **memory** | ***what the whole scene looks like*** | **trans-saccadic stamp buffer (§1a) — the keystone** |
| **gaze** | ***where to glimpse*** | reflex (motion × staleness) + evolved top-down modulation |
| controller | *what to do* | NEAT-evolved (unchanged) |

Stage 1 (this spec) = reflex gaze (no new trainer). Stage 2 (§6, escalation) = a learned self-supervised
gaze net if Stage 1 proves gaze is still the ceiling.

### 1a. Trans-saccadic foveal memory (the keystone — user, 2026-07-18)

**The problem a bare sharp fovea has:** a small sharp fovea sees detail only where it looks *right now*;
its field of view is tiny. **The fix biology uses:** stamp each foveal glimpse into a persistent spatial
buffer and *maintain* it, running on the stored percept until the low-res periphery flags a region
changed, then re-saccade to refresh (trans-saccadic memory; the "grand illusion" of stable rich vision;
adaptive **change-blindness** as the efficient flip-side). This turns the fovea from "see where I look"
into "**build and maintain a high-res model of the whole scene**" — and it's what makes the small sharp
fovea viable (the memory is the wide FOV; the fovea is the sharp sampler that fills it in).

**The closed loop (unifies fovea + reflex + memory into one active-vision system):**
```
fovea stamps hi-res patch  →  periphery watches that region for change
   →  change/brightness delta invalidates the stamp (region goes stale)
   →  (motion × staleness) draws the reflex gaze there  →  re-saccade  →  re-stamp
```

**Mechanism (per-env, parent-side, ZERO rng, transfer-stable):**
- **Buffer:** a persistent per-env scene buffer at a **controller-readable medium resolution**
  `mem_grid` (default ~32×32 — far sharper than the 12×12 periphery, far cheaper than native 160×144
  which is 23k inputs the genome can't use). Reset per episode.
- **Stamp:** each step, paste the sharp fovea patch (§2) into the buffer at the current gaze location
  (screen-coord warp; the fovea's native crop → buffer cells).
- **Invalidate:** compare the always-current low-res periphery against the stamped content per region;
  when they diverge past a **self-calibrated** threshold (EMA/percentile-z of the change signal — a
  dimensionless surprise, *never* a fixed magnitude, per the §11b no-tuned-knobs mandate, reusing the
  [AC] per-env EMA machinery), mark that region **stale** and decay it toward the live low-res value.
- **Staleness/freshness map:** a per-region scalar (steps since last stamp, gated by peripheral change)
  the controller reads as a confidence channel.

**The controller's observation becomes** (foveal mode): instantaneous periphery `[12×12]` (change/motion
detector, biological low-acuity periphery) **+ the persistent foveal-memory buffer `[mem_grid²]`
(accumulated sharp scene)** + a **freshness/staleness map `[mem_grid²]`** + the current sharp fovea patch
+ proprio + ram. The buffer + freshness are the new, powerful inputs. `n_in` grows accordingly; the
sparse-init genome (`init_k` fan-in) attends to a learned subset, so a larger obs stays tractable.

**Reflex-gaze salience (§3) upgrades to `motion × staleness`** — orient to what changed OR what hasn't
been refreshed lately (the biological orienting drive).

**Determinism/transfer:** pure per-env arithmetic (stamp/compare/decay), zero rng → `fast_reproduce`
untouched; hand-defined over the game-agnostic screen → transfer-stable (species-bootstrap constraint
preserved). Off-switch `vision.foveal_memory=False` ⇒ the v2 obs without the buffer.

---

## 2. Sharp fovea — decouple `fovea_grid` from `periph_grid`

Add `vision.fovea_grid: int` (default `= periph_grid`, so **omitted ⇒ byte-identical legacy**). In
`FovealEncoder`, the fovea resample target becomes `FG = fovea_grid` instead of `G = periph_grid`:
- `_frow = _area_matrix(F, FG)`, `_fcol = _area_matrix(F, FG).T`; `n_fovea = FG*FG`.
- recompute the block offsets + `dim`; the fovea block is `[o_fovea : o_fovea + FG*FG]`.

**Keep the periphery coarse** (biomimetically correct — low-acuity periphery). Recommended default for
the successor run: `fovea_native_px = 32`, `fovea_grid = 32` → **1 px/cell native fovea** over a 32×32
window (a GB tile = 8×8 cells, fully legible), or `F=48, FG=24` → 2 px/cell over a wider window.
`n_in` grows from 454 to `144 + FG² + 144 + 14 + 8` (e.g. 32² → 1334; 24² → 886). **`n_in` change ⇒
fresh run** (or warm-started pop — the transfer machinery). Transfer-stability preserved: still a fixed,
hand-defined, game-invariant observation.

**Motion sheet stays at `G` (12×12)** — it is a peripheral change-detector (biologically magno/low-res),
and the reflex gaze (§3) reads it; no need to sharpen it.

---

## 3. Reflex gaze — motion soft-argmax (bottom-up, zero new trainer)

The motion sheet `obs[o_motion : o_proprio]` (G×G, whole-screen frame-difference, already computed for
[AC] salience) IS the bottom-up "where is something happening" signal. Compute a reflex gaze target
parent-side, in the loop's gaze-update path (mirrors [AC]'s salience read):

```
M = motion_ij  (G×G, |diff| after removing the 0.5 no-motion baseline)
soft-argmax (center of mass over a softmax of M) -> (r*, c*) in screen coords
reflex_target = (r*, c*)  ->  reflex velocity toward it (scaled like a saccade delta)
```

- **Zero rng, pure arithmetic** over resident floats → no `fast_reproduce` impact, engine-parity-safe
  (per-env, deterministic; identical serial vs furnace).
- When motion is flat (no salient change) the softmax is ~uniform → reflex ≈ center/no-pull (harmless).
- Runs on the **gaze clock** (`saccade_every_k`), already decoupled from the motor clock by [AC].
- Foveal-only (needs the motion sheet); retina mode keeps the current learned saccade (its reflex is the
  deferred magno channel).

---

## 4. Evolved top-down modulation (the "one or both")

The controller's **existing 2 saccade outputs `out[9], out[10]`** become the **top-down** term — an
additive correction on the reflex (superior-colliculus reflex + cortical override):

```
gaze_delta = reflex_delta + learned_delta        # learned_delta = the controller's (dx,dy)
```

- The controller can **follow** the reflex (learned≈0), **nudge** it, or **override** it (large learned).
- **No N_OUT change** (default) — reuses the saccade outputs, so `fast_reproduce` determinism is
  untouched and evolution simply re-purposes those outputs as a top-down correction.
- **Expose the reflex target to proprio** (2 efference dims: where the reflex is pulling) so the
  controller can condition its correction on the reflex — the top-down/bottom-up handshake.
- *Optional refinement (only if additive underperforms):* an evolved blend gate `g∈[0,1]`
  (`gaze = (1−g)·reflex + g·learned`) as one appended output (N_OUT+1), same determinism story as the
  [AC] commit gate. Not the default; a documented knob.

**Off-switch:** `vision.reflex_gaze=False` (default) + `fovea_grid=periph_grid` ⇒ byte-identical legacy.
The successor run turns both on.

---

## 5. Determinism + config

- Reflex + blend are pure inference-time arithmetic (zero rng) → `fast_reproduce` / `_fast_mutate` /
  `_fast_crossover` **untouched**; v2-off is byte-identical legacy; engine-parity holds (per-env).
- Config: `vision.fovea_grid` (default = periph_grid), `vision.reflex_gaze: bool=False`,
  `vision.reflex_gain: float` (self-calibrate to the motion distribution per §11b mandate — an
  EMA/percentile-normalized pull, NOT a fixed magnitude — reusing [AC]'s per-env EMA machinery).
  CLI: `--fovea-grid`, `--fovea-native-px` (exists), `--reflex-gaze`.

---

## 6. Stage 2 — learned info-gain gaze (documented escalation, NOT built now)

If Stage-1 reflex gaze proves gaze is still the ceiling: a gradient-trained self-supervised gaze net
with an **info-gain objective** (place the fovea to best reduce next-frame / next-latent prediction
error — RAM/DRAW/active-perception line, or the retina SSL's inverse-dynamics loss), trained on **card0
with freeze-and-swap** (the retina-learner discipline), the controller still evolving top-down
modulation. Run **parallel** to evolution (slow updates + periodic snapshot swap) so it is a stationary
target within a generation but keeps improving from the population's experience. A game-general gaze net
is also a **transferable module** for the species-bootstrap goal. Build only on Stage-1 evidence.

---

## 7. Acceptance (on the wall + measured)

1. **Legibility:** the "what the AI sees" fovea panel is now human-readable (sprites/text resolvable) —
   the direct fix for the reported defect. Side-by-side vs coarse `live1`.
2. **Motion-capture rate:** the fovea center tracks the on-screen motion/change peak materially above a
   random-gaze null (proves the reflex aims).
3. **Top-down evidence:** the learned correction is non-trivial in some contexts (the controller isn't
   just deferring to the reflex everywhere), and pure-reflex is reachable.
4. **Play:** boot-gauntlet depth / milestone progress beats the coarse-vision `live1` baseline at equal
   wall-clock — the whole point. If not, the acuity/gaze hypothesis is falsified; keep `live1`.

---

## 8. Implementation plan

1. **Sharp fovea** (#5): `fovea_grid` config + `FovealEncoder` resample-to-FG + offsets/dim + n_in
   derivation in `train()`; resolution unit test. Retina assert / obs-dim checks updated.
2. **Trans-saccadic foveal memory buffer** (#13, §1a — the keystone): per-env `mem_grid²` persistent
   scene buffer + fovea stamp/warp + episode reset; zero-rng, per-env. The buffer + a placeholder
   freshness channel enter the obs; n_in + offsets updated. Unit test: a stamp persists across steps;
   buffer reset on episode boundary.
3. **Peripheral-change invalidation + staleness** (#14, §1a): per-region compare (live periphery vs
   stamped) with a self-calibrated EMA/percentile-z threshold (§11b — no fixed magnitude, reuse [AC]
   EMA); mark stale → decay toward live low-res; maintain the freshness map. Unit test: an unchanged
   region stays fresh (change-blindness); a changed region invalidates; threshold self-calibrates
   (constant-high change stops firing; a relative spike fires — the [AC] proof shape).
4. **Reflex gaze on `motion × staleness`** (#6): parent-side soft-argmax of `motion × staleness` →
   reflex delta in the gaze-update path; self-calibrated `reflex_gain`; zero-rng; unit test (aims at a
   planted change/stale region; flat→no pull).
5. **Top-down modulation** (#7): `gaze_delta = reflex + learned`; reflex target → proprio;
   determinism (fast_reproduce untouched) + engine-parity tests.
6. **Tests + determinism** (#8): resolution, stamp/persist, invalidation self-calibration, soft-argmax,
   blend, v2-off==legacy, engine-parity, full suite.
7. **Rendering** (#12): stream the focused agent's gaze + fovea box + **the foveal-memory buffer +
   staleness** into `live.json`; `wall.html` draws blurry periphery + sharp fovea box + persisting
   hi-res stamps that fade as they go stale (a direct window on §1a). Click-to-inspect already exists
   (swarm + `/api/select`).
8. **Live smoke** (#9): legible fovea + persistent percept building up + motion-capture + top-down on
   the wall; watch change-blindness / re-saccade happen.
9. **Successor run / A-B** (#10): launch v2 (sharp fovea + foveal memory + reflex gaze) vs coarse
   `live1`; measure §7.4.

---

## 9. Portability & generalization roadmap (design the GB build as a portable active-vision core)

**Design principle (user, 2026-07-18):** build the active-vision system as a **sensor- and
actuator-agnostic core**, so it drops onto more advanced game systems (SNES/N64/modern, color, higher
res) and eventually a **real robot with a webcam + PTZ camera** with only the edges swapped. A PTZ
camera *is* a physical fovea+saccade rig: **pan/tilt = gaze direction, zoom = foveal magnification.**
GB is just the first, cheapest substrate. This is not future scope creep — it is how the GB code is
structured *now* (clean interfaces), so the generalizations are drop-ins, not rewrites.

### 9a. Two abstraction seams (build into the GB core — tasks #15/#16)
- **Sensor interface** — provides frames `(H, W, C)`: resolution-agnostic (the periph/fovea/mem grids
  abstract over any input size) and channel-agnostic (`C=1` grayscale now, `C=3` RGB later). GB emu,
  console renderer, and webcam all implement the same `next_frame()`.
- **Gaze-actuator interface** — consumes a gaze command `(Δpan, Δtilt, [Δzoom])` and returns the
  realized gaze. **Software crop now** (can teleport the fovea); **PTZ later** (velocity-limited,
  latency, physical). The saccade is *already* a velocity command (`gain·tanh(dx)`), so it maps to a
  PTZ slew rate directly; zoom is a natural third DOF (foveal scale = accommodation/vergence).

### 9b. Advanced game systems (tasks #17/#18)
- **Color** — `C=3` across periphery/fovea/memory + the stamp buffer; encoder handles N channels.
- **Higher resolution** — grids already parameterize this; a proper area/Gaussian pyramid for the
  periphery. **Zoom DOF** — the controller controls fovea scale, trading FOV for acuity (a 3rd gaze
  output), useful when detail sizes vary (far vs near, small UI vs big sprites).

### 9c. Real robot / PTZ (tasks #19–#22 — documented, far-later)
The trans-saccadic memory is where a *moving camera* differs from a *fixed screen*:
- **Spatiotopic (world-anchored) memory + gaze-compensated remapping (#19).** On a fixed GB screen,
  retinotopic == spatiotopic — stamps land at screen coords, no remap. On a PTZ camera the field
  *moves*, so the buffer must be **world-anchored** and stamps **remapped by the pan/tilt delta** (the
  brain's *predictive remapping* in LIP/FEF around saccades). We already carry an **efference copy** of
  the saccade in proprio — that is exactly the **corollary-discharge** signal remapping needs. Build the
  buffer with a remap hook now (identity on GB); it becomes gaze-compensated on PTZ.
- **Saccadic suppression + actuation latency (#20).** A physical pan/tilt takes time and smears the
  frame mid-move; suppress stamping during large gaze moves / until the fovea settles (biological
  saccadic suppression). Model PTZ slew-rate + settle latency; the fovea arrives *after* a delay.
- **PTZ actuator driver + sim harness (#21).** Map gaze velocity → pan/tilt/zoom; a **simulated
  moving-camera sensor** (crop-from-a-large-scene with slew + latency) validates the whole core on a
  moving field *before* touching hardware.
- **Real bring-up (#22, far future):** webcam capture + a real PTZ mount (e.g. ONVIF/serial control).

### 9d. Cross-substrate validation (#23)
The portability proof: the **same active-vision core** drives GB → an advanced console → the sim-PTZ
moving-camera with only sensor/actuator swaps — no core edits. That is the deliverable that says "this
generalizes." (Ties to [[pokeio-cross-game-transfer]]: a transfer-stable, sensor-agnostic obs is also
what lets evolved species bootstrap across substrates.)
