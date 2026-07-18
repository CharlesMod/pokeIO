# Experiment: Biological Retinal Channels (ON/OFF + Magno/Parvo) + Motion-Driven Saccades

**Status:** planned — the first upgrade to test *once the learned retina is the spine* (see
`docs/specs/retina-in-loop.md`, `pokeio-spine-decision`). Deferred from the original "biomimetic-max"
design; this is the disciplined A/B that decides whether it earns its keep.

## Hypothesis
The head-to-head showed the saccade's value **scales with the retina behind it**: with the crude foveal
front-end the champion wall-hugged and barely used its gaze; with the learned retina it played all nine
buttons and saccaded 2× more. Biology suggests why, and where the next gain is: in a real retina the
saccade is *driven* by a dedicated fast pathway. **Give the saccade its natural input — a transient,
ON/OFF, magnocellular motion signal from the periphery — and gaze should become purposeful (motion- and
change-seeking) rather than something the controller has to reconstruct from a single blended latent.**

## The biology → our architecture
A retina never transmits brightness; it transmits **contrast on parallel channels**, and those channels
plus the saccade form one loop:
- **ON/OFF center-surround** — bipolar/ganglion cells rectify local spatial contrast into "got brighter
  here" (ON) and "got darker here" (OFF), via lateral inhibition. → a fixed **Difference-of-Gaussians**
  preprocess, split `ON=relu(DoG)`, `OFF=relu(-DoG)`.
- **Magnocellular (transient)** — fast, motion-sensitive, low spatial detail, dominates the **periphery**;
  the substrate for reflexive orienting. → a **temporal-contrast** (frame-diff), ON/OFF, low-spatial
  peripheral stream.
- **Parvocellular (sustained)** — slow, high spatial detail, dominates the **fovea**. → a high-spatial,
  ON/OFF, low-temporal foveal stream.
- **The loop:** peripheral magno motion → **saccade** to fixate → parvo fovea inspects in detail.

Our current retina is a *single grayscale conv* over a periphery+fovea split — the right *geometry* but
the *wrong front-end*: no ON/OFF rectification, no magno/parvo specialization, and crucially **no direct
motion→gaze pathway** (the controller reads the whole 80-d latent and must infer where to look).

## The three channels (each an ablation arm)
- **A1 · ON/OFF center-surround.** Fixed DoG conv on periphery + fovea before the learned encoder; feed
  the rectified ON and OFF maps as 2 channels each. Input transform only — no learned params, cheap.
- **A2 · Magno/Parvo two-stream (A1 +).** Split the single encoder into two stems: **magno** = ON/OFF
  *temporal contrast* of the coarse periphery (motion, low-res, downsampled harder) → `z_magno`; **parvo**
  = ON/OFF current *fovea* (detail, high-res, minimal temporal) → `z_parvo`. Concat → latent (replaces the
  current `z_periph|z_fovea`). The SPR + inverse-dynamics objective is unchanged; only the input structure
  and stem split change.
- **A3 · Motion-driven saccade (A2 +, the key arm).** From the magno stem produce a 2-D **saliency map**
  (channel-norm of the magno feature map, or raw peripheral motion magnitude). Soft-argmax (center of
  mass) → a candidate reflexive gaze target in screen coords → convert to a `(dx,dy)` toward it. **Blend
  with the controller's learned `(dx,dy)` through an evolved gate** `g∈[0,1]` (a genome-controlled or
  extra-output weight): `saccade = (1-g)·learned + g·reflex`. This is the superior-colliculus reflex with
  cortical override — the controller can *lean on* motion salience for gaze while keeping button control,
  or override it. Also expose the saliency-peak location to the controller as extra proprioception.

## Experiment design (disciplined; each arm must beat the retina baseline)
- **Baseline** = the current learned-retina spine (its head-to-head champion metrics).
- **Arms:** A1 → A2 → A3, cumulative. Run each at the matched furnace config used for the spine head-to-head
  (same pop/seed/gens), with the retina warm-up + freeze-and-swap unchanged.
- **Standard gate (must not regress):** output-std-vs-screen, I(obs;action), action-entropy — the
  comparable champion metrics.
- **New gaze-purpose metrics (the actual test of the hypothesis):**
  - **motion-capture rate** — fraction of saccades that move the fovea *toward* the region of highest
    recent on-screen change (vs a random-gaze null). If A3 works, this jumps.
  - **fixation-on-salience** — correlation between fovea center and the peak of the change/motion map.
  - **saccade→reward coupling** — does gaze fixate on the sprites/objects whose interaction drives
    progress (event-delta / mined-counter changes), more than chance?
- **Exploration/throughput:** cells/agent-step and cells/wall-second (the DoG + two-stream add a little
  parent-side compute — confirm it doesn't erase the gain).
- **Decision rule:** an arm ships only if it beats the retina baseline on the gaze-purpose metrics
  **without** regressing the standard gate or per-wall-clock exploration. Keep the single-stream retina as
  the fallback (one config flag).

## Implementation mapping
| Piece | Where | Notes |
|---|---|---|
| DoG ON/OFF preprocess | `evo/retina.py` input adapter (or `FovealEncoder`) | fixed conv, `sigma1<sigma2`, `relu(±DoG)`; applies to periph84 + fovea84 |
| Magno/parvo two-stream | `evo/retina.py` `NatureCNN` → two stems + concat heads | magno = temporal-diff coarse periph; parvo = fovea; `z_magno`/`z_parvo` replace `z_periph`/`z_fovea` |
| Saliency map + soft-argmax | `evo/retina.py` (expose `magno_saliency(periph_stack)->(2,)`) | channel-norm → 2-D softmax center-of-mass |
| Reflex/learned saccade blend | `train/loop.py` saccade split + an evolved gate output (N_OUT 11→12) or a genome gene | `saccade=(1-g)·learned+g·reflex`; feed saliency peak into the proprio block |
| Config knobs | `config.retina` | `channels={grayscale,onoff,magnoparvo}`, `saccade_reflex: bool`, `reflex_gate_evolved: bool` |
| Gaze-purpose metrics | new `pokeio/eval/gaze_metrics.py` + a measure script | motion-capture rate, fixation-on-salience, saccade→reward coupling |

Lineage / precedent: ON/OFF center-surround = classical DoG/retinal ganglion models; magno/parvo split =
computational-vision standard and the event-camera analogy (transient vision); reflexive
saliency-driven fixation = superior-colliculus models + the RAM/Attention-Agent hard-attention line;
temporal self-supervision stays SPR (`retina-in-loop.md`).

## Honest risk
"More biological" is not automatically better — the magno/parvo split was *cut* from the first design
precisely to avoid complexity without payoff, and that caution stands. The difference now is a concrete,
falsifiable prediction (motion-capture rate rises) and a cheap first arm (A1 is a fixed conv). Run the
ladder, keep what beats the baseline, drop what doesn't. Do **not** build A2/A3 before A1 shows the
ON/OFF rectification at least doesn't hurt.
