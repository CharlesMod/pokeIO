# [AC] Adaptive Cadence — Design of Record

**Status:** buildable spec. Base design **LatchGate — the TC-timed commit gate** (judge tally 2–1),
absorbing the completeness/rigor grafts from **Commit-Gate Latch** and the telemetry/future-proofing
grafts from **DwellHead**. Foveal-first (Phase-0 spine); retina deferred behind a documented hook.

Depends on: **[TC] time-constants** (committed `d3c6dcf`; `NodeGene.alpha`, `mutate_tau`,
`_seed_alpha`, per-node leaky integration in `propagate_sparse`). AC is the explicit motor-cadence
**readout** built on top of the TC **substrate**.

---

## 1. The one-sentence answer to the deep question

**Dwell is not a predicted number and not new timing machinery — it is the leaky-integrated activation
of a single new output neuron (the "commit gate"), and that neuron's evolvable TC `alpha` *is* the
dwell clock.**

- `alpha ≈ 1` (fast reflex) → the gate recomputes open/closed fresh every agent-step → **twitchy
  short dwells** = the legacy re-decide-every-step motor as the limiting case (`k=1`).
- `alpha` small (slow integrator) → the gate leak-integrates its drive over ~`1/alpha` steps → once
  pushed closed it stays closed (hysteresis) → **long, stable holds** robust to per-frame optical
  noise; dwell length ≈ the gate's integration window `1/alpha`.

So AC and TC are **one mechanism seen from two angles**: TC supplies the clock (that node's `alpha`,
already perturbed by `mutate_tau`, already seeded with the fast/slow spread by `make_genome._seed_alpha`),
AC supplies the readout (a commit gate + a light parent-side latch). Evolution tunes cadence two ways
at once — the gate's **fan-in weights** (state-dependent *when* to commit) and its **`alpha`** (the
*hysteresis / hold length*). We deliberately **reject a dedicated duration head** (DwellHead): a
`round()`-quantized integer countdown gives a stepped, non-smooth fitness landscape, is open-loop
inside a hold (a stale `D` governs future steps), splits credit across two co-adapting heads, and
costs a `genome.py` edit — where the emergent-`alpha` gate reuses `mutate_tau`'s already-blessed
search dimension and needs no core-evolution edit at all.

> **TC dependency, stated plainly.** The headline native dwell-timer requires `config.evo.time_constants=True`
> (the current default). With TC **off**, the gate is memoryless (`alpha=1`) and dwell degrades to
> pure input-driven sign-hysteresis on the gate's net input — still functional, but weaker. **Run AC
> with TC on.**

---

## 2. The four required properties

| # | Property | Mechanism |
|---|----------|-----------|
| (a) | **LEARNED (button, dwell-duration)** | *Which* button = `argmax(out[0:9])`. *How long* = how long the commit gate stays closed, governed by the gate node's evolved fan-in weights **and** its evolved `alpha`. Both halves are per-genome learned. |
| (b) | **DECOUPLED gaze vs motor clocks** | GAZE runs on `FovealEncoder._nstep % vision.saccade_every_k` (untouched). MOTOR runs on the parent-side `MotorClock.dwell_len` counter, advanced only by the gate/interrupts. The two counters never read each other. Optional `saccade_interrupt` couples them *only at interrupt time* (eye jumps → re-decide the hand). |
| (c) | **REFLEX FLOOR (k=1)** | The batched forward runs **every** agent-step for every ready env (required anyway for gaze + gate + salience), so the gate is re-evaluated at the finest grain. A closed gate is a *preference, never a lock*: two learned override paths force a re-decide mid-dwell — (i) **reflex margin** `logit[argmax] − logit[held] > ac.reflex_margin`, (ii) **salience** force-open. `ac.min_dwell=1` keeps single-step re-decision reachable. |
| (d) | **SALIENCE-INTERRUPTIBLE dwell** | A parent-side surprise scalar force-opens the gate when this frame's motion is `ac.salience_z` std above the env's own recent motion (per-env EMA-z; §3.4/§11b — self-calibrated, no fixed magnitude). Foveal: reuse the encoder's existing gaze-invariant **motion sheet** (zero extra work). Retina: controller-latent L1 (deferred). Swaps to the real magno channel when the retinal-channels branch lands, with no other change. |

---

## 3. Neural mechanism (exact)

### 3.1 The genome delta
`N_OUT: 11 → 12` when `ac.enable`. Append **one** OUTPUT node — the commit gate — at the highest
output id (`n_in + 1 + 11`). Head layout becomes a pure append, so every existing slice is unchanged:

```
out[0:9]   button logits   (argmax / softmax head)     — unchanged
out[9]     saccade dx      (raw, tanh inside encoder)   — unchanged
out[10]    saccade dy                                    — unchanged
out[11]    COMMIT GATE     (tanh, thresholded)           — NEW    GATE_IDX = N_BUTTONS + 2 = 11
```

The gate is a **bog-standard OUTPUT node**: `output_act = tanh` (raw in `(-1,1)`, thresholded at
`ac.commit_thresh` default `0.0` ≈ 50% commit at gen-0 so dwell has something to differentiate from);
bias `0.0`; fan-in-scaled incoming weights; `alpha` seeded by the **same** `make_genome._seed_alpha()`
as every other output (so under TC it inherits the fast/slow timescale spread for free). **No new
`NodeGene`/`ConnGene` field, no new gene type, no new `MutationRates` field, no new mutation operator,
no `genome.py`/`forward.py`/`ops.py` edit.** `make_genome(n_out=12)` builds it; `propagate_sparse`
already emits all `N_OUT` columns and already applies the per-node `alpha` that gives the gate its
hysteresis.

### 3.2 The parent-side latch (`MotorClock`)
Per-env bookkeeping, **zero rng, pure inference-time threshold** (so it perturbs neither the reproduce
stream nor rollout determinism):

```
state per env i:  held_btn[i] : int32   (init NOOP = 8)
                  dwell_len[i]: int32   (consecutive closed-gate steps)
                  prev_lat[i] : float32[102]  (retina mode only; L1 surprise)

decide(argmax_btn, gate_raw, logits, salience, saccade_moved, ready_idx) -> emitted_btn:
    gap  = logits[argmax_btn] - logits[held_btn]                      # reflex margin
    open = (gate_raw   >= ac.commit_thresh)                           # learned commit
         | (ac.salience_interrupt  & (s − μ_i)     > ac.salience_z·√var_i)  # §11b self-calibrated (warmup+var gated)
         | (ac.reflex_margin > 0   & (gap          > ac.reflex_margin))     # §11b default OFF
         | (ac.saccade_interrupt   & saccade_moved)                   # optional gaze->motor
         | (ac.max_dwell > 0       & (dwell_len    >= ac.max_dwell))  # liveness cap
    # min_dwell only forces a hold when > 1 (default 1 = no forcing, reflex floor intact)
    if ac.min_dwell > 1:  open &= (dwell_len >= ac.min_dwell)
    on open  : held_btn <- argmax_btn ;  dwell_len <- 0 ;  record break-cause
    on closed: dwell_len += 1                       # re-emit held_btn unchanged
    return held_btn
```

The emitted button (not the raw argmax) is what goes to `env.step` / `fleet.submit_actions`. The
saccade `dx,dy` are applied **every** step regardless (gaze clock is independent). Re-emitting
`held_btn` is a *real, dynamics-correct hold* via the existing sticky/edge-read `PokeEnv` model:
a d-pad button stays pressed (continuous walk), a face button re-taps every `_TAP_GAP` frames
(dialogue mash / confirm), NOOP stays released (sustained idle) — the three dwell semantics fall out
of the current input model, so **`env.py` needs no change**.

### 3.3 Why the interrupts are trivial in this formulation
Salience, saccade, and reflex-margin are all just **force-open drives on the gate** — a boolean OR at
threshold time. In a fixed-duration-count design each would be an awkward special-case that aborts a
countdown; here they are one vectorized `|=`. This is exactly why gate+TC beats a duration head:
closed-loop, natively interruptible, reflex-floored for free, reusing the committed timing substrate.

### 3.4 Salience signal (pre-magno)
- **Foveal (Phase-0, shipping for [RUN]):** reuse the encoder's existing **motion block**. Verified
  layout in `fleet.py`: foveal obs = `periph[0:144] | fovea[144:288] | motion[288:432] | proprio | ram`;
  the motion sheet is `((periph_t − periph_{t−1}) + 1)/2` — a **gaze-invariant, whole-screen 12×12
  frame-difference** already written into the obs the parent holds. So the raw motion scalar is
  `s = mean(|X[:, 288:432] − 0.5|)` (`_o_motion:_o_proprio`). **Zero extra emulator/GPU work,
  no `prev_periph` state to keep or reset.** Do **not** recompute a separate periphery diff parent-side
  (Commit-Gate's redundant path — the motion block already *is* that signal).
- **Self-calibrating reflex (§11b, implemented).** The reflex does **not** compare `s` to a fixed
  magnitude. `MotorClock` keeps a **per-env** running EMA of the motion mean `μ_i` and variance `var_i`
  (Welford-style, decay `ac.salience_ema_decay`, μ seeded with the first sample) and fires the salience
  break when `(s − μ_i) > ac.salience_z · sqrt(var_i)` — "this frame's motion is `z` std above THIS
  env's recent motion" — gated on a warmup (`sal_steps_i ≥ ac.salience_warmup`) and a live variance
  (`var_i > 0`). Dimensionless `z`, game-agnostic, no magic constant. **Per-env is load-bearing for
  engine-parity:** each env's EMA depends only on its own salience sequence (identical serial-vs-furnace);
  a global/cross-env EMA would read the batch order and break parity. The EMA is **not** reset by
  `reset(i)` (μ/σ are a game-level estimate, not per-episode state).
- **Retina (deferred):** no motion sheet in the 14134-d obs / 102-d controller latent. Fallback =
  `surprise = ||z_t − z_{t−1}||_1` over the controller latent (keep `prev_lat[i]`, n×102 — cheaper
  than an n×7056 periph84 diff). On the first post-reset step `prev_lat` is unset → `surprise = 0`
  (no spurious break; correct).
- **Later:** when the real magno motion-energy channel lands, drop it straight into the same
  `surprise` hook — no other change.

### 3.5 Optional deterministic slow-seed for the gate (`ac.seed_gate_slow`)
To avoid the cold-start risk that the gate is seeded fast (`alpha≈1`) while `R_resp` entropy could
penalize early holds, optionally bias the gate into the slow band. Implemented **rng-free** in
`train()` by overwriting the initial genomes' gate-node `alpha ← ac.gate_seed_alpha` (default `0.3`)
**after** `make_genome`. Because the gate is the last output, its `_seed_alpha` draw already happened
last in the output loop — overwriting the resulting value does **not** shift any subsequent rng draw,
so this needs **no `genome.py` edit and preserves the rng stream exactly**. Default **off** (keep the
standard-node path); flip on if gen-0 dwell fails to emerge.

---

## 4. Determinism plan (the hard constraint)

**AC introduces NO new gene type and NO new mutation operator, so there is no new rng-stream position
to guard.** The gate is an ordinary appended output node; its fan-in weights mutate inside the existing
`perturb_weights` / `_fast_perturb` loop over `g.conns.values()` (same order in `ops.py` and `loop.py`),
and its `alpha` mutates inside the existing **[TC] `mutate_tau`** guarded call — already matched between
`ops.mutate_genome` (`ops.py:248`) and `_fast_mutate` (`loop.py:2665`). `_fast_mutate` / `_fast_crossover`
never hard-code `n_out`; they iterate whatever nodes/conns a genome has. Therefore **`fast_reproduce`
stays rng-stream-identical to `ops.reproduce` for `N_OUT=12` exactly as it does for `N_OUT=11`** — the
identity contract is over the same `Genome` objects in both implementations, independent of `N_OUT`
(this is already proven by the 2 existing saccade outputs).

**OFF switch = bit-identical legacy.** `config.ac.enable = False` (default):
- `train()` derives `N_OUT = int(config.evo.n_out)` — no gate node, no gate edges.
- `MotorClock` is **never constructed**; every eval site emits the plain argmax.
- Every `ac.*` value is inert (never read into an rng-drawing or behavior path).
- ⇒ **zero extra rng, byte-for-byte identical to the current legacy loop** for the same seed
  (same genomes, same rng stream, same actions).
- The `MotorClock.decide()` threshold draws **zero rng**, so even AC-ON does not perturb reproduction
  or rollout determinism.

**Honest caveat (AC-ON is a distinct gen-0 baseline).** `make_genome`'s `full` connect loop is
`for src: for out:`, so appending output 11 draws ~`fan_in` extra `N(0,std)` init weights and (under
TC) one extra `_seed_alpha` draw for the 12th output. The alpha loop is sequential over outputs, so the
gate's alpha draws **last** — legacy outputs' alpha draws are untouched. This is a one-time
**init-stream** shift confined to its own run (exactly as the saccade head's `+2` outputs already are);
it is **not** a reproduce-path violation. The contract that must hold — *AC-OFF == legacy* and
*fast_reproduce == ops.reproduce* — is preserved.

**Resume guard.** The existing guard (`loop.py:3208`, keyed on `(n_in, N_OUT)`) already refuses a
mixed OFF/ON checkpoint resume (11 ≠ 12), so runs cannot silently cross-contaminate.

**Reserved future gene (documented, NOT built in v1).** If a per-genome cadence-bias scalar is ever
wanted, its guarded rng-stream slot is **immediately after the `mutate_tau` block** in **both**
`ops.mutate_genome` and `loop._fast_mutate`, as
`if rates.ac_mutate_gain and rng.random() < rates.ac_mutate_gain: mutate_gain(...)` — default `0.0`,
so Python's `and` short-circuits and draws zero rng, exactly mirroring the `mutate_act`/`mutate_tau`
precedent. v1 adds no such gene and therefore no `ops.py` edit.

---

## 5. Throughput

The forward already runs **every** step (required for gate + salience — AC deliberately saves no
forward compute). Per ready step, parent-side added cost is O(n):
- one gate-column slice + threshold compare,
- one `mean(|·−0.5|)` over ~144 resident float32 (foveal motion block) — microseconds,
- one reflex-margin gather over `out[:, :9]` (argmax already computed),
- an integer latch/gather over the ready `idx` subset.

Tensor width grows by **1 output column** (~`fan_in` extra edges): a sub-percent pack/compile bump. No
extra emulator steps, no extra GPU passes, no new worker/IPC code. The ~10k steps/s furnace budget is
untouched.

---

## 6. File-by-file implementation plan

### `pokeio/config/__init__.py` — **edit** (add section + field)
Add an `@dataclass ACConfig` and wire it into `Config` as `ac`:
```python
@dataclass
class ACConfig:
    """[AC] Adaptive cadence — learned dwell via the commit-gate output (docs/specs/adaptive-cadence.md)."""
    enable: bool = False            # single OFF switch; drives N_OUT (11 off, 12 on)
    commit_thresh: float = 0.0      # gate_raw >= thresh -> re-decide (open)
    min_dwell: int = 1              # reflex floor reachable; >1 forces a minimum hold
    max_dwell: int = 64             # hard liveness cap (0 = uncapped) — no-stall guarantee
    salience_interrupt: bool = True  # master switch for the self-calibrating reflex
    # §11b self-calibrating salience: fire when this frame's motion is salience_z std
    # ABOVE this env's OWN recent motion (per-env EMA-z) — dimensionless, no fixed
    # game-specific magnitude.  Replaces the deprecated fixed salience_thresh=0.08.
    salience_z: float = 1.5         # z-score (std above the per-env motion baseline)
    salience_ema_decay: float = 0.99  # per-env EMA decay for the motion mean/var baseline
    salience_warmup: int = 16       # min per-env steps before a salience break can fire
    reflex_margin: float = 0.0      # §11b: default OFF (gate learns "a better button appeared"); logit[argmax]-logit[held] > margin -> re-decide
    saccade_interrupt: bool = False # optional gaze->motor coupling (eye jump -> re-decide hand)
    saccade_deadband: float = 0.05  # |saccade| beyond this counts as "moved"
    seed_gate_slow: bool = True     # §11b: seed gen-0 gate alpha into the slow band so the LEARNED α is the primary dwell driver (rng-free, AC-gated)
    gate_seed_alpha: float = 0.3
```
Add `ac: ACConfig = field(default_factory=ACConfig)` to `Config` and to `__all__`. **`evo.n_out`
stays 11**; the effective `N_OUT` is derived in `train()`.

> **Self-tuning form (§11b, implemented).** The behavior scalars above are the
> self-calibrated / default-off resolution of the §11b mandate, **not** hand-tuned
> magnitudes: `salience_z` / `salience_ema_decay` / `salience_warmup` parameterize a
> per-env EMA-z of the motion signal itself (surprise = "more motion than THIS env's
> context recently showed"), `reflex_margin` defaults **off** (the gate learns the
> "better button appeared" signal from the same inputs), and `R_resp` is reformulated
> to the salience↔commit correlation (see §11b) so `reward.w_resp` is a fitness-mix
> weight, never a per-frame dwell knob.

### `pokeio/train/loop.py` — **edit** (the whole feature lives here + config)
1. **`train()` N_OUT derivation** (replace `loop.py:2826`):
   `N_OUT = int(config.evo.n_out) + (1 if config.ac.enable else 0)`. Define `GATE_IDX = N_BUTTONS + 2`.
   Do **not** hardcode `11`.
2. **Retina assert** (`loop.py:2859`): relax `int(N_OUT) == 11` → `int(N_OUT) in (11, 12)` (mode-aware),
   so a future retina+AC run fails loudly on a real mismatch, not silently. (Foveal-first; retina AC is
   deferred — see §7.)
3. **`_split_head`** (`loop.py:82`): when `outv.shape[1] == 12` also return the gate column
   `outv[:, GATE_IDX]`; width `11` keeps the legacy 3-tuple return. All callers that don't need the gate
   ignore the extra value.
4. **`MotorClock(n)` helper** (new): per-env `held_btn`, `dwell_len`, `prev_lat` (retina); `reset(i=None)`;
   `decide(...)` per §3.2 (vectorized over a ready subset); a `break_cause` counter
   (gate / salience / reflex_margin / saccade / cap) for telemetry.
5. **Wire into ALL eval engines** — replace the emitted button with `clock.decide(...)` immediately
   after each `_split_head` and before the env step / `submit_actions`:
   - `evaluate_wave` (serial) — split at `loop.py:1022`, step at `1036`.
   - `evaluate_wave_parallel` (barrier) — split at `loop.py:1184`, step via `fleet.step_all` at `1207`.
   - `evaluate_wave_async` (furnace) — split at `loop.py:1676`, submit via `fleet.submit_actions` at
     `1690` (clock only the ready `idx` subset, matching the one-obs-step-per-submit invariant).
   - **Champion showcase replay** — split at `loop.py:2224`, step at `2228` (use `MotorClock(1)`), so the
     wall matches training. The **CUDA-graph** furnace path returns raw all-`N_OUT` `outv`, so it needs no
     change beyond the `_split_head`/`decide` wiring.
   - **Boot-eval** (`policy_eval`) inherits automatically — it reuses the three engines above.
6. **Reset wiring** — call `clock.reset(i)` at **every** boundary where env `i`'s per-episode state is
   cleared (episode end / Go-Explore restore), co-located with the existing `encoder.reset(i)` /
   `retina_pipe.reset(i)` calls, plus a full `clock.reset()` at wave start. Known anchors:
   `evaluate_wave` `loop.py:965`; `evaluate_wave_parallel` `1141`; `evaluate_wave_async` `1409`; showcase
   `2200`. **Care-point:** all engines must route through the same `MotorClock` or dwell diverges across
   engines (this is the single biggest implementation hazard — see risks).
7. **Optional slow-seed** (§3.5): if `config.ac.enable and config.ac.seed_gate_slow`, after building the
   initial `pop_size` genomes overwrite each genome's gate-node (`id = n_in + 1 + 11`) `alpha ←
   config.ac.gate_seed_alpha`. Rng-free; does not shift the stream.
8. **Telemetry** (§8): `_champion_cadence_terms()` mirroring `_champion_alpha_terms` (`loop.py:752`);
   surface the gate node's `alpha`, the dwell-length distribution, and the break-cause histogram.
9. **CRITICAL:** do **not** touch `fast_reproduce` / `_fast_mutate` / `_fast_crossover` — leaving them
   untouched is what preserves the identity contract.

### `pokeio/emu/fleet.py` — **no functional change** (optional accessor)
The foveal motion sheet is already in the obs; its offsets `_o_motion` (288) / `_o_proprio` (432) already
exist on `FovealEncoder`. Optionally expose them as public attributes so the parent can slice the
salience block without reaching into privates. **No worker/encoder change.**

### `pokeio/emu/env.py` — **no change**
The sticky/edge-read input model already realizes a held/re-tapped/released button across repeated
agent-steps, so re-emitting `held_btn` is a dynamics-correct hold. (Listed as the site for a *future*
sub-agent-step frame-level hold; not needed now.)

### `pokeio/evo/genome.py`, `pokeio/evo/forward.py`, `pokeio/evo/ops.py` — **no change**
The gate is a standard output node produced from `n_out=12`; its alpha seeding, `mutate_tau`,
`perturb_weights`, `add_conn`/`add_node`/`toggle`, and crossover are all already generic over output
count and applied in identical order by `ops.reproduce` and `loop.fast_reproduce`. **This is the
headline: no evolutionary-core edit.**

### `tests/` — **add**
See §9.

---

## 7. Foveal-first; retina deferred
Phase-0 foveal is the spine shipping for [RUN], and it carries the **true** gaze-invariant motion sheet,
so AC ships and is validated there. Retina AC is deferred: it needs (a) the `N_OUT in {11,12}` assert
bump (done in §6.2), (b) the controller-latent L1 surprise hook + `prev_lat` state, and (c) — later — the
swap to the real magno channel. Retina also currently lacks the E1 controller-latent transform hook, so
retina AC lands with that work, not before.

---

## 8. `--live` UI acceptance metric

Champion **"Cadence"** panel on the wall (built like the [TC] `_champion_alpha_terms` alpha panel).
Cadence is proven adaptive when **all** hold for the *same* champion within one episode:

1. **State-dependent, bimodal dwell (headline).** Mean/median dwell **conditioned on binned salience**
   (a curve, not a bare histogram — DwellHead's legibility graft): **long** dwells (`dwell ≫ 1`) in
   low-motion states (menus / dialogue, where salience stays low), **collapsing toward `dwell ≈ 1`**
   during high-motion overworld play.
2. **Reflex floor reachable.** `dwell = 1` stretches appear, and the **break-cause histogram** shows
   nonzero **gate**, **salience**, and **reflex-margin** breaks (not only cap/expiry) — i.e. mid-dwell
   interruption fires and single-step re-decision is live.
3. **TC carries the clock.** The champion's commit-gate `alpha` lands in the **slow band** (via
   `_champion_alpha_terms`), confirming dwell rides the time-constant substrate, not a bolt-on counter.
4. **No stall.** No env freezes on one button for a whole episode (bounded by salience + reflex margin +
   `max_dwell`), and **boot-gauntlet depth does not regress** vs the AC-off baseline.

**Config note to check on the wall (`R_resp` antagonism).** Long menu/dialogue holds *lower* the
button-histogram entropy that `reward.w_resp` rewards, so `w_resp` could penalize legitimate dwells.
Measure dwell separately and confirm `w_resp` is not suppressing genuine holds; if it is, note it for a
`w_resp` retune (do not let the entropy reward fight AC).

---

## 9. Ordered, testable implementation task list

1. **Config** — add `ACConfig` + `Config.ac` + `__all__`. *Test:* round-trips through YAML; defaults
   load; `ac.enable` defaults `False`.
2. **N_OUT derivation + asserts** — `train()` derives `N_OUT`; `GATE_IDX = N_BUTTONS+2`; relax retina
   assert to `{11,12}`. *Test:* `ac.enable=False ⇒ N_OUT==11`; `True ⇒ 12`.
3. **Determinism gate (do this before any behavior)** — `ac.enable=False` yields a **bit-identical
   reproduce rng stream + identical genomes** vs a pre-AC baseline, and `fast_reproduce == ops.reproduce`
   at `N_OUT=12`. *Test:* seed-locked genome-hash equality (OFF==legacy; fast==ops for both 11 and 12).
4. **`_split_head`** — return the gate column at width 12; legacy 3-tuple at 11. *Test:* shape/branch
   unit.
5. **`MotorClock`** — implement `decide` + `reset` + break-cause counters. *Test (dwell-adaptivity
   unit):* a **slow-alpha** gate produces long holds; a **fast-alpha** (`alpha=1`) gate re-decides
   ≈ every step. *Test (interrupt):* a salience spike **and** a reflex-margin gap each force a commit
   mid-dwell. *Test (no-stall):* `max_dwell` force-commits a permanently-closed gate.
6. **Wire serial `evaluate_wave`** + reset. *Smoke:* emitted actions differ from raw argmax only inside
   holds; gaze cadence unchanged.
7. **Wire furnace `evaluate_wave_async` + barrier `evaluate_wave_parallel`** (ready-subset clocking) +
   resets. *Smoke:* dwell behavior matches serial for the same genome/seed (engine-parity check — the
   care-point).
8. **Wire champion showcase replay** (`MotorClock(1)`) + reset; boot-eval inherits. *Smoke:* the wall
   replay matches training dwell.
9. **Optional gate slow-seed** (`seed_gate_slow`) — rng-free overwrite in `train()`. *Test:* rng stream
   unchanged with the flag on; gen-0 gate `alpha == gate_seed_alpha`.
10. **Telemetry "Cadence" panel** — dwell-vs-binned-salience curve, break-cause histogram, gate-alpha.
11. **`--live` smoke on the wall** — verify §8 acceptance (bimodal state-dependent dwell; reflex floor
    reachable; gate-alpha slow; no stall; boot-gauntlet depth holds; `w_resp` not suppressing holds).
    Then commit.

---

## 10. Risks & mitigations

1. **Degenerate always-closed gate (freeze on one button).** Mitigated by three independent break paths
   (salience, reflex margin, `saccade_interrupt`) + the `max_dwell` hard cap — a closed gate is never a
   true lock. `max_dwell=64` default is the liveness floor that does **not** depend on evolution wiring
   the salience path.
2. **Always-open gate.** AC degrades to legacy every-step re-decide (safe no-op); evolution should favor
   holds only where they pay.
3. **Engine divergence.** All of serial / barrier / furnace / boot-eval / showcase **must** route through
   the same `MotorClock`, or dwell behavior differs across engines — the single biggest implementation
   hazard. The engine-parity smoke (task 7) guards it.
4. **~~`salience_thresh` tuning~~ (RESOLVED, §11b).** No fixed magnitude ships: the reflex fires on a
   per-env EMA-z of the motion signal itself (`salience_z`/`salience_ema_decay`/`salience_warmup`), so it
   self-calibrates to each scene's own motion statistics. `salience_z` too low → perpetual interrupts →
   degrades to legacy (harmless); too high → relies on gate + cap. Dimensionless, game-agnostic.
5. **~~`R_resp` entropy reward fights long holds~~ (RESOLVED, §11b).** R_resp is reformulated to the
   salience↔commit correlation for foveal+AC envs (§3/§11b), so a legitimate low-salience holder is
   rewarded as correct conditioning; `w_resp` is a fitness-mix weight, not a per-frame dwell knob. (Legacy
   §8 note) — measure dwell separately; retune `w_resp` if
   it suppresses legitimate menu/dialogue holds.
6. **TC-off weakens the mechanism** (§1) — run AC with `time_constants=True` (the default).
7. **Retina surprise proxy** (latent L1) is weaker than true magno — only bites the deferred retina spine;
   swap in magno when it lands.

---

## 11b. Self-tuning mandate (BINDING — supersedes any hand-set threshold above)

**Directive (user, 2026-07-17): no hand-tuned *behavior* knobs. Cadence must be a robust
self-tuning system — learned, evolved, or self-calibrated from the observation stream — never a value
a human balances.** The hardcoded scalars in §3/§6 (`salience_thresh=0.08`, `reflex_margin=0.5`, and
the `reward.w_resp` balance) are **scaffolding to eliminate**, not the mechanism. This section is the
design of record where it conflicts with the literal defaults above.

**Why the mechanism is already self-tuning.** The foveal obs the controller reads *already contains the
whole-screen motion sheet* (`X[:, 288:432]`, §3.4). So the commit-gate neuron conditions its open/hold
decision on on-screen change through its ordinary evolved fan-in — no external threshold needed.
Evolution tunes cadence on two axes with zero human input: the gate's **fan-in weights** (state-dependent
*when* to commit, reading motion directly) and its **TC `alpha`** (the hysteresis / *how long* to hold).
That learned gate **is** the self-tuning cadence controller; everything else is a safety floor.

**Knob-by-knob resolution:**
- `commit_thresh` — keep the literal boundary at `0.0`; it is *not* a tuning knob (the gate's evolved
  **bias** sets the effective commit point). Self-tuning already.
- `salience_thresh` — **do not ship a fixed 0.08.** The gate learns its motion response from the obs.
  Any residual reflex-safety interrupt fires on a **running percentile / EMA-z-score of the motion
  signal itself** (surprise = "more motion than this context recently showed"), so it is game-agnostic
  and self-calibrating (mirror the existing WRAM churn-mask / novelty-rarity EMA machinery). No magic
  constant in a behavior path.
- `reflex_margin` — **default OFF.** "A better button appeared" is learnable by the gate (same inputs
  as the button head). Keep the code path only as an optional, self-normalized (by running logit-gap
  scale) safety, not a default-on tuned value.
- `max_dwell` — reframe as a **liveness watchdog, not a behavior knob**: a generous "never freeze
  forever" backstop, self-scaled to `k ×` the running median dwell where practical.
- `reward.w_resp` **antagonism — dissolve it at the source.** Reformulate responsiveness from raw
  action/entropy (which a masher maxes and a legitimate holder fails) to **response-to-change**: the
  (self-normalizing) correlation between per-step salience and per-step gate-opens over an episode.
  Masher → ~0 (no conditioning), freezer → ~0, state-appropriate agent → high. It lives in [−1, 1],
  uses only already-computed signals (motion sheet + MotorClock break events), and needs **no weight
  balanced against dwell** — low-salience holds are *rewarded as correct conditioning*, not penalized.
  (Guard the degenerate zero-variance case → 0.)

**Build order for this mandate:** the v1 skeleton (learned gate + MotorClock + determinism gate +
engine-parity wiring + telemetry) is correct as-is and lands first — the gate+α is the self-tuning
core. Then a focused self-tuning pass: (1) EMA/percentile self-calibration of the salience reflex,
(2) reflex-margin default-off, (3) the `R_resp` → salience↔commit-correlation reformulation. Acceptance
(§8) is unchanged except: **verify no fixed behavior threshold is load-bearing** — perturbing the
motion statistics (different game/scene) must not require re-tuning any constant for cadence to remain
state-appropriate.

## 11. Fallback (documented, not built)
If NEAT proves unable to credit-assign the closed-loop gate (gen-0 dwell never emerges even with
`seed_gate_slow`), the escape hatch is DwellHead's **explicit integer duration head** (a second appended
output → `N_OUT=13`, `D = 1 + round(clamp01(tanh)·(dwell_max−1))`, TC used to de-jitter `D`). It is more
legible but less neural-native, costs a `make_genome` edit, and creates a two-head search problem — kept
here only as a labeled fallback, not the design of record.
