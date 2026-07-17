# Phase 1 — Retina-in-Loop Integration Spec

Wire the decoder-free SPR/FSQ retina (`evo/retina.py`, already unit-smoke-validated) into the live
training loop as the perceptual spine, replacing Phase-0 pixel obs with a learned 102-d latent obs.
Gate: must **beat the Phase-0 champion metrics** in a smoke before it launches the multi-day run;
otherwise we launch on Phase 0 (committed `9ea606c`). Builds on the Phase-0 saccade/foveation/coupling.

## Obs pipeline (the big change)
- **Worker (retina mode, `vision.mode="retina"`):** same gaze/saccade logic as Phase 0, but ships the
  retina's PIXEL INPUT, not a 12×12 vector: `periph84` (84×84, full screen area-downscaled) ⊕
  `fovea84` (84×84, 48px native crop at gaze, upsampled) ⊕ `proprio`(14) ⊕ `ram`(8) = **14134 f32**.
  Use `retina.downscale`/`retina.crop_fovea` so the 84×84 tensors are exactly what the encoder expects.
- **Parent (card1):** maintains per-env 4-frame rings for periph84 + fovea84 (reset on episode/restore),
  builds `[4,84,84]` stacks, batch-encodes all ready envs through the **frozen** `retina.snapshot()` →
  `[z_periph(48)|z_fovea(32)]`, concatenates `proprio(14)+ram(8)` → **102-d** obs → `population_forward_sparse`.
- **Obs block order (n_in=102):** `[z_periph 0:48 | z_fovea 48:80 | proprio 80:94 | ram 94:102]`.
  Genome: `make_genome(n_in=102, n_out=11, connect="full", n_ram=8, n_proprio=14, output_act=TANH)`.
  Blind-ablation gate zeroes the **latent** `[0:80]` (optical) and keeps proprio+ram → `optical_hi=80`.

## Learner (card0) — warm-up + freeze-and-swap (synchronous, no threading)
- **Warm-up:** at run start collect `(periph_stack, fovea_stack, button, saccade)` transitions into a
  replay buffer while the population runs, and `train_step` the retina on card0 until
  `retina.warmup_frames` (smoke: pass a smaller value, e.g. 40–60k). Genomes NEVER consume a
  random-init encoder — evolution only starts after the first `snapshot()`.
- **During evolution:** stash a sample of the parent's encode-inputs (+button+saccade) each wave into
  the replay buffer; run a batch of `train_step`s on card0 between gens; every `retina.swap_gens`
  re-`snapshot()` the inference encoder. Population inference always uses the frozen snapshot
  (stationary between swaps). Windows must not cross episode/restore boundaries.
- **Cards:** learner (grad) on `retina.train_card=0`; population inference on frozen snapshot on
  `retina.infer_card=1`.

## Fitness / cells
- Keep the Phase-0 coupled fitness (E2 advantage + E3 blind gate w/ `optical_hi=80` + R_resp).
- **Empowerment (§6.4):** `Emp_i` via the frozen `q_psi`; start `w_emp=0` for the first smoke, enable
  0.1 once stable.
- Go-Explore cells stay **pixel** (`goexplore.cell_source="pixel"`) — the safety net; FSQ cells are Phase 2.

## Resume / config
- `vision.mode="retina"`, `retina.enable=True`, `n_in=102`. Resume guard asserts `(n_in,n_out)`.
- Model-error firewall preserved: selection ground truth is the real wave; no imagined sample.

## Smoke gate (must pass before live)
Warm-up + ~15 gens; then the SAME champion gate as Phase 0 (`docs/specs/active-vision-spine.md §12`),
PLUS: retina SPR cosine loss decreased during warm-up; latent does not collapse; the latent obs is
stationary between swaps; and the champion's gate metrics **≥ Phase-0 champion** (output-std-vs-screen,
I(obs;action), action-entropy, saccades). If it does not clearly beat Phase 0 → launch on Phase 0.
