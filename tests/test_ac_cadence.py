"""[AC] Adaptive cadence — learned motor dwell via the commit-gate output.

Spec: ``docs/specs/adaptive-cadence.md``.  Dwell is the leaky-integrated
activation of ONE appended OUTPUT neuron (the commit gate); that node's evolved
time-constant ``alpha`` (spec [TC]) IS the dwell clock, and a parent-side
``MotorClock`` re-emits the held button until the gate opens or an interrupt
fires.  The whole feature lives in ``config`` + ``loop.py`` — NO genome.py /
forward.py / ops.py / env.py edit — which is what keeps the fast_reproduce
determinism contract intact.

Covered here:
* **config** round-trips through YAML; ``ac.enable`` defaults False; N_OUT
  derivation (11 off, 12 on) and the gate id / GATE_IDX layout.
* **DETERMINISM GATE (load-bearing):** ``ac.enable=False`` is byte-identical to
  legacy (same genomes, same rng stream), and ``fast_reproduce`` stays
  rng-stream-identical to ``ops.reproduce`` at N_OUT=12 exactly as at 11
  (offspring genome-hash + rng bit-generator-state equality).
* **_split_head** returns the gate column at width 12, the legacy 3-tuple at 11.
* **MotorClock**: a closed (slow) gate holds; an always-open (fast/alpha=1) gate
  re-decides every step; a salience spike AND a reflex-margin gap each force a
  mid-dwell commit; ``max_dwell`` force-commits a permanently-closed gate (no
  stall); and decide() draws zero rng.
* **engine parity**: the clock's per-env dwell is invariant to the ready-subset
  interleaving, so furnace (async ready subset) == serial/barrier (all ready)
  for a fixed per-env stream — the single biggest implementation hazard (§10.3).
* **slow-seed** (§3.5): the rng-free gate-alpha overwrite leaves the rng stream
  unchanged.

CPU-safe (no emulator / GPU): the MotorClock is pure numpy, and the determinism
proof is over the same ``Genome`` objects both reproduce paths consume.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from pokeio.config import ACConfig, Config
from pokeio.evo import ops
from pokeio.evo.genome import OUTPUT, InnovationTracker, make_genome
from pokeio.train.loop import (
    GATE_IDX,
    N_BUTTONS,
    MotorClock,
    _foveal_salience,
    _resp_scores,
    _split_head,
    fast_reproduce,
)


# ==========================================================================
# 1. config: defaults + roundtrip + N_OUT derivation
# ==========================================================================
def test_ac_config_defaults_and_roundtrip():
    c = Config()
    assert c.ac.enable is False  # single OFF switch defaults off (legacy)
    assert c.ac.min_dwell == 1  # reflex floor reachable
    assert c.ac.max_dwell == 64  # liveness cap
    assert c.ac.commit_thresh == 0.0
    # §11b self-tuning defaults: reflex margin OFF, salience is a dimensionless
    # per-env EMA-z (no fixed magnitude), slow-seed ON (learned α is the driver).
    assert c.ac.reflex_margin == 0.0  # §11b: the gate learns "a better button appeared"
    assert c.ac.salience_z == 1.5
    assert c.ac.salience_ema_decay == 0.99
    assert c.ac.salience_warmup == 16
    assert c.ac.seed_gate_slow is True  # §11b: α-driven gen-0 dwell
    assert not hasattr(c.ac, "salience_thresh")  # the fixed magnitude is gone
    # round-trips through the dict/YAML serializer
    d = c.to_dict()
    assert "ac" in d and d["ac"]["enable"] is False
    c2 = Config.from_dict(d)
    assert c2.ac == c.ac


def test_ac_config_all_fields_present():
    ac = ACConfig()
    for f in ("enable", "commit_thresh", "min_dwell", "max_dwell",
              "salience_interrupt", "salience_z", "salience_ema_decay",
              "salience_warmup", "reflex_margin", "saccade_interrupt",
              "saccade_deadband", "seed_gate_slow", "gate_seed_alpha"):
        assert hasattr(ac, f), f"ACConfig missing {f}"


def _n_out(cfg: Config) -> int:
    """The exact N_OUT derivation train() applies (spec §6.2)."""
    return int(cfg.evo.n_out) + (1 if cfg.ac.enable else 0)


def test_n_out_derivation():
    c = Config()
    assert _n_out(c) == 11  # ac off -> legacy 11 (9 buttons + 2 saccade)
    c.ac.enable = True
    assert _n_out(c) == 12  # ac on -> append the commit gate
    # GATE_IDX is the appended column: constant N_BUTTONS+2, independent of N_OUT.
    assert GATE_IDX == N_BUTTONS + 2 == 11


def test_gate_is_highest_output_id():
    """The gate is appended at the highest OUTPUT id (spec §3.1), so make_genome
    at n_out=12 grows exactly one extra output and the loop's slow-seed target
    (``max(output_ids())``) lands on it."""
    rng = np.random.default_rng(0)
    t11 = InnovationTracker(3, 11)
    t12 = InnovationTracker(3, 12)
    g11 = make_genome(3, 11, t11, rng, connect="full")
    g12 = make_genome(3, 12, t12, rng, connect="full")
    o11 = [n for n in g11.nodes.values() if n.type == OUTPUT]
    o12 = [n for n in g12.nodes.values() if n.type == OUTPUT]
    assert len(o11) == 11 and len(o12) == 12
    # the extra output is the new highest id (a pure append)
    assert max(g12.output_ids()) == max(g11.output_ids()) + 1


# ==========================================================================
# 2. _split_head: gate column at width 12, legacy 3-tuple at width 11
# ==========================================================================
def test_split_head_width11_is_legacy_3tuple():
    outv = np.zeros((4, 11), dtype=np.float32)
    outv[np.arange(4), [0, 3, 8, 5]] = 1.0
    out = _split_head(outv)
    assert len(out) == 3  # legacy 3-tuple, no gate
    buttons, dx, dy = out
    assert list(buttons) == [0, 3, 8, 5]


def test_split_head_width12_returns_gate_column():
    outv = np.zeros((3, 12), dtype=np.float32)
    outv[np.arange(3), [1, 2, 4]] = 2.0
    outv[:, GATE_IDX] = np.array([-0.7, 0.0, 0.9], dtype=np.float32)
    out = _split_head(outv)
    assert len(out) == 4  # buttons, dx, dy, gate
    buttons, dx, dy, gate = out
    assert list(buttons) == [1, 2, 4]
    assert np.allclose(gate, [-0.7, 0.0, 0.9])
    assert gate.shape == (3,)


# ==========================================================================
# 3. DETERMINISM GATE — the load-bearing invariant (do before behavior)
# ==========================================================================
def _genome_sig(g):
    """Structural fingerprint incl. alpha (mirrors test_fast_reproduce)."""
    nodes = tuple(sorted(
        (n.id, n.type, n.act, round(float(n.bias), 12), round(float(n.alpha), 12))
        for n in g.nodes.values()
    ))
    conns = tuple(sorted(
        (c.in_id, c.out_id, round(float(c.weight), 12), bool(c.enabled), c.innov)
        for c in g.conns.values()
    ))
    return (g.n_in, g.n_out, g.age, nodes, conns)


def _sigs(gs):
    return [_genome_sig(g) for g in gs]


def _build(seed, npop, *, n_in=4, n_out=11, time_constants=True, threshold=1e9):
    rng = np.random.default_rng(seed)
    tracker = InnovationTracker(n_in, n_out)
    genomes = [
        make_genome(n_in, n_out, tracker, rng, connect="full",
                    time_constants=time_constants)
        for _ in range(npop)
    ]
    for g in genomes:
        g.fitness = float(rng.random())
    species = ops.Speciation(threshold=threshold).assign(genomes, rng)
    return genomes, species, tracker


def _run_both(genomes, species, tracker, *, pop_size, seed=1234, rates=None):
    rates = rates or ops.MutationRates()
    rng_ops = np.random.default_rng(seed)
    rng_fast = np.random.default_rng(seed)
    t_ops = copy.deepcopy(tracker)
    t_fast = copy.deepcopy(tracker)
    sp = {k: list(v) for k, v in species.items()}
    ko = ops.reproduce(genomes, sp, t_ops, rng_ops, rates, pop_size=pop_size)
    kf = fast_reproduce(genomes, sp, t_fast, rng_fast, rates, pop_size=pop_size)
    return ko, rng_ops, kf, rng_fast


@pytest.mark.parametrize("n_out", [11, 12])
def test_fast_reproduce_identical_to_ops_at_both_widths(n_out):
    """fast_reproduce stays rng-stream-identical to ops.reproduce at N_OUT=12
    EXACTLY as at 11 (spec §4): the gate is an ordinary appended output node, so
    reproduce iterates the same nodes/conns in the same order and draws the same
    rng in both paths — proven here with tau LIVE (mutate_tau=0.5)."""
    genomes, species, tracker = _build(21, 20, n_out=n_out, time_constants=True)
    # sanity: seeding actually produced slow integrators, so alpha is exercised
    assert any(float(n.alpha) < 1.0 for g in genomes for n in g.nodes.values())
    rates = ops.MutationRates(mutate_tau=0.5, tau_perturb_sigma=0.2)
    ko, ro, kf, rf = _run_both(genomes, species, tracker, pop_size=20, rates=rates)
    assert len(ko) == len(kf) == 20
    assert _sigs(ko) == _sigs(kf), f"offspring diverged at n_out={n_out}"
    # identical final rng bit-state => identical draw count + order
    assert ro.bit_generator.state == rf.bit_generator.state


def test_ac_off_is_byte_identical_to_legacy():
    """OFF switch == bit-identical legacy (spec §4): with ac.enable=False the
    derived N_OUT is 11 and NOTHING in make_genome / reproduce changes, so a
    seed-locked build + reproduce is byte-identical (genomes + rng bit-state) to
    a baseline that never knew about AC.  We prove it by building the exact same
    n_out=11 population twice from one seed and requiring identical genome hashes
    AND identical post-reproduce rng bit-state."""
    cfg = Config()
    assert cfg.ac.enable is False and _n_out(cfg) == 11
    ga, spa, tra = _build(7, 16, n_out=_n_out(cfg), time_constants=True)
    gb, spb, trb = _build(7, 16, n_out=11, time_constants=True)
    assert _sigs(ga) == _sigs(gb), "AC-off gen-0 genomes differ from legacy"
    koa, roa, kfa, rfa = _run_both(ga, spa, tra, pop_size=16)
    kob, rob, kfb, rfb = _run_both(gb, spb, trb, pop_size=16)
    assert _sigs(kfa) == _sigs(kfb)
    assert rfa.bit_generator.state == rfb.bit_generator.state


def test_ac_slow_seed_is_rng_free(gate_alpha=0.3):
    """The optional gate slow-seed (§3.5) overwrites the gate node's alpha AFTER
    make_genome — the gate's own _seed_alpha draw already happened last, so the
    overwrite shifts NO subsequent draw and preserves the rng stream exactly."""
    seed = 99
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    t_a = InnovationTracker(4, 12)
    t_b = InnovationTracker(4, 12)
    n = 8
    gs_a = [make_genome(4, 12, t_a, rng_a, connect="full", time_constants=True)
            for _ in range(n)]
    gs_b = [make_genome(4, 12, t_b, rng_b, connect="full", time_constants=True)
            for _ in range(n)]
    # apply the loop's rng-free slow-seed to the B build only
    for g in gs_b:
        g.nodes[max(g.output_ids())].alpha = gate_alpha
    # rng bit-state must be identical (the overwrite drew nothing)
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    # and the gate alpha landed on the slow band
    for g in gs_b:
        assert g.nodes[max(g.output_ids())].alpha == gate_alpha


# ==========================================================================
# 4. MotorClock: dwell adaptivity, interrupts, no-stall, zero rng
# ==========================================================================
def _ac(**kw):
    ac = ACConfig(enable=True)
    for k, v in kw.items():
        setattr(ac, k, v)
    return ac


def _logits(n, argmax, hi=3.0):
    """(n, 9) logits with a clear argmax per row (all others 0)."""
    lg = np.zeros((n, N_BUTTONS), dtype=np.float32)
    lg[np.arange(n), argmax] = hi
    return lg


def test_open_gate_re_decides_every_step():
    """A fast (alpha=1) gate reads open every step -> the clock re-decides every
    step (dwell collapses to 1): emitted follows the fresh argmax (spec §1)."""
    ac = _ac(reflex_margin=0.0, salience_interrupt=False, max_dwell=0)
    clock = MotorClock(1, ac)
    argmaxes = [4, 2, 7, 0, 5]
    for a in argmaxes:
        amax = np.array([a], dtype=np.int32)
        gate = np.array([1.0], dtype=np.float32)  # >= commit_thresh -> open
        emitted = clock.decide(amax, gate, _logits(1, amax), None, None)
        assert int(emitted[0]) == a  # committed the fresh argmax this step
    assert clock.break_cause["gate"] == len(argmaxes)
    assert all(d == 1 for d in clock.dwells)  # every hold length is 1


def test_closed_gate_holds_the_button():
    """A slow (small-alpha) gate stays closed -> once a button is committed it is
    re-emitted (held) across many steps regardless of the fresh argmax (§1)."""
    ac = _ac(reflex_margin=0.0, salience_interrupt=False, max_dwell=0)
    clock = MotorClock(1, ac)
    # step 0: gate open -> commit button 3
    e = clock.decide(np.array([3], np.int32), np.array([1.0], np.float32),
                     _logits(1, np.array([3])), None, None)
    assert int(e[0]) == 3
    # steps 1..K: gate closed, argmax keeps changing -> button 3 is HELD
    for k, a in enumerate([6, 1, 8, 2, 0, 7]):
        amax = np.array([a], np.int32)
        e = clock.decide(amax, np.array([-1.0], np.float32),
                         _logits(1, amax), None, None)
        assert int(e[0]) == 3, f"held button broke at step {k+1}"
        assert int(clock.dwell_len[0]) == k + 1  # dwell grows every closed step


def _warm(clock, s_stream, argmax=3):
    """Feed a per-env salience stream (n, T) through the clock (gate held closed)."""
    n, T = s_stream.shape
    for t in range(T):
        amax = np.full(n, argmax, np.int32)
        clock.decide(amax, np.full(n, -1.0, np.float32), _logits(n, amax),
                     s_stream[:, t].astype(np.float32), None)


# --------------------------------------------------------------------------
# §11b self-calibrating salience: the "no fixed magnitude" proof
# --------------------------------------------------------------------------
def test_salience_constant_high_calibrates_and_stops_firing():
    """A HIGH but CONSTANT motion level (0.3): once warmed up the per-env EMA-z
    calibrates to it (var stays 0) and NO salience break fires — a self-tuning
    reflex.  A FIXED threshold (the deprecated 0.08) would fire every single step
    at this level.  This is the core "no game-specific magnitude" property (§11b)."""
    ac = _ac(salience_interrupt=True, salience_z=1.5, salience_ema_decay=0.9,
             salience_warmup=8, reflex_margin=0.0, max_dwell=0)
    clock = MotorClock(1, ac)
    _warm(clock, np.full((1, 60), 0.3, np.float32))
    assert clock.break_cause["salience"] == 0  # constant motion is never a surprise


def test_salience_two_baselines_fire_on_same_relative_spike():
    """THE HEADLINE PROOF (§11b): two envs at DIFFERENT absolute motion baselines
    (A≈0.10, B≈0.50) both fire the salience break on the SAME relative spike
    (+0.2 above each env's own baseline) — even though env B's steady baseline
    (0.50) sits FAR ABOVE env A's spike level (0.30).  No single absolute-magnitude
    threshold could fire A's 0.30 spike yet stay silent through B's 0.50 baseline,
    so the reflex must be calibrated to each env's OWN recent motion, not a fixed
    magnitude.  (Warmup + tiny alternating jitter seed a live per-env variance.)"""
    ac = _ac(salience_interrupt=True, salience_z=1.5, salience_ema_decay=0.9,
             salience_warmup=8, reflex_margin=0.0, max_dwell=0)
    clock = MotorClock(2, ac)
    base = np.array([0.10, 0.50])
    jit = np.array([[0.005 if t % 2 == 0 else -0.005 for t in range(40)]] * 2)
    _warm(clock, (base[:, None] + jit))
    assert clock.break_cause["salience"] == 0  # neither steady baseline fires
    # a +0.2 RELATIVE spike above each env's own baseline, in a single step
    before = clock.break_cause["salience"]
    spike = (base + 0.2).astype(np.float32)
    clock.decide(np.array([3, 3], np.int32), np.full(2, -1.0, np.float32),
                 _logits(2, np.array([3, 3])), spike, None)
    assert clock.break_cause["salience"] - before == 2  # BOTH fired on their spike
    # the proof, stated as values: B's steady 0.50 (silent) > A's spike 0.30 (fired)
    assert base[1] > base[0] + 0.2


def test_salience_warmup_suppresses_early_breaks():
    """No salience break fires until an env has seen ``salience_warmup`` steps: a
    huge spike inside the warmup window is suppressed, the same spike after it
    fires (§11b warmup guard lets the EMA baseline settle first)."""
    ac = _ac(salience_interrupt=True, salience_z=1.5, salience_ema_decay=0.9,
             salience_warmup=8, reflex_margin=0.0, max_dwell=0)
    clock = MotorClock(1, ac)
    jit = np.array([[0.1 + (0.003 if t % 2 else -0.003) for t in range(5)]])
    _warm(clock, jit)  # 5 steps < warmup
    clock.decide(np.array([3], np.int32), np.array([-1.0], np.float32),
                 _logits(1, np.array([3])), np.array([0.9], np.float32), None)
    assert clock.break_cause["salience"] == 0  # step 6 < warmup=8 -> suppressed
    # settle past warmup on the baseline, then the same spike fires
    jit2 = np.array([[0.1 + (0.003 if t % 2 else -0.003) for t in range(6)]])
    _warm(clock, jit2)
    clock.decide(np.array([3], np.int32), np.array([-1.0], np.float32),
                 _logits(1, np.array([3])), np.array([0.9], np.float32), None)
    assert clock.break_cause["salience"] == 1  # now warmed -> the spike fires


def test_reflex_margin_gap_forces_mid_dwell_commit():
    ac = _ac(reflex_margin=0.5, salience_interrupt=False, max_dwell=0)
    clock = MotorClock(1, ac)
    # commit button 3 (held button)
    clock.decide(np.array([3], np.int32), np.array([1.0], np.float32),
                 _logits(1, np.array([3])), None, None)
    # closed gate, but logit[argmax=6] - logit[held=3] = 3.0 > 0.5 -> force commit
    lg = np.zeros((1, N_BUTTONS), np.float32)
    lg[0, 6] = 3.0  # big drive on a different button than the held (3)
    e = clock.decide(np.array([6], np.int32), np.array([-1.0], np.float32),
                     lg, None, None)
    assert int(e[0]) == 6
    assert clock.break_cause["reflex_margin"] == 1


def test_max_dwell_force_commits_permanently_closed_gate():
    """No-stall guarantee (§10.1): a gate that never opens still re-decides every
    ``max_dwell`` steps via the liveness cap; dwell never exceeds max_dwell."""
    ac = _ac(reflex_margin=0.0, salience_interrupt=False, max_dwell=4)
    clock = MotorClock(1, ac)
    committed_at = []
    for t in range(13):
        amax = np.array([3], np.int32)
        e = clock.decide(amax, np.array([-5.0], np.float32),  # gate stuck closed
                         _logits(1, amax), np.array([0.0], np.float32), None)
        if int(clock.dwell_len[0]) == 0:  # a commit reset the dwell this step
            committed_at.append(t)
        assert int(clock.dwell_len[0]) <= ac.max_dwell  # never exceeds the cap
    assert committed_at == [4, 9]  # a re-decide every max_dwell steps (no freeze)
    assert clock.break_cause["cap"] == len(committed_at) == 2  # every commit = cap


def test_min_dwell_enforces_minimum_hold():
    """min_dwell > 1 blocks EVERY re-open (even the first) until the dwell floor
    is reached (spec §3.2: ``open &= dwell_len >= min_dwell``): the current hold
    (NOOP at episode start) is held for at least min_dwell steps."""
    ac = _ac(reflex_margin=0.0, salience_interrupt=False, min_dwell=3, max_dwell=0)
    clock = MotorClock(1, ac)
    # gate is OPEN every step, but min_dwell=3 forbids a commit until dwell hits 3
    for _ in range(3):
        e = clock.decide(np.array([7], np.int32), np.array([1.0], np.float32),
                         _logits(1, np.array([7])), None, None)
        assert int(e[0]) == 8  # held NOOP (below the min_dwell floor)
    # dwell_len is now 3 >= min_dwell, so the open gate finally commits
    e = clock.decide(np.array([7], np.int32), np.array([1.0], np.float32),
                     _logits(1, np.array([7])), None, None)
    assert int(e[0]) == 7


def test_decide_draws_zero_rng():
    """MotorClock.decide is a pure inference-time threshold — it must not touch
    numpy's global rng (spec §4: no reproduce/rollout determinism perturbation)."""
    ac = _ac()
    clock = MotorClock(3, ac)
    st0 = np.random.get_state()
    for _ in range(20):
        amax = np.array([1, 2, 3], np.int32)
        clock.decide(amax, np.array([0.1, -0.1, 0.5], np.float32),
                     _logits(3, amax), np.array([0.2, 0.0, 0.9], np.float32),
                     np.array([False, True, False]))
    st1 = np.random.get_state()
    assert st0[0] == st1[0] and np.array_equal(st0[1], st1[1])


# ==========================================================================
# 5. engine parity: dwell is invariant to ready-subset interleaving (§10.3)
# ==========================================================================
def _make_streams(n_envs, T, seed):
    """Per-env deterministic (argmax, gate, salience) streams of length T."""
    rng = np.random.default_rng(seed)
    argmax = rng.integers(0, N_BUTTONS, size=(n_envs, T)).astype(np.int32)
    gate = rng.standard_normal((n_envs, T)).astype(np.float32)  # +/- around thresh
    sal = rng.random((n_envs, T)).astype(np.float32) * 0.2
    return argmax, gate, sal


def _run_serial(ac, argmax, gate, sal):
    """All envs ready every step (serial / barrier lockstep)."""
    n_envs, T = argmax.shape
    clock = MotorClock(n_envs, ac)
    emitted_hist = np.zeros((n_envs, T), dtype=np.int32)
    for t in range(T):
        lg = _logits(n_envs, argmax[:, t])
        e = clock.decide(argmax[:, t], gate[:, t], lg, sal[:, t], None)
        emitted_hist[:, t] = e
    return emitted_hist, clock


def _run_furnace(ac, argmax, gate, sal, sched_seed):
    """Envs processed in a scrambled ready-subset schedule (furnace): each env
    still advances through ITS stream in order, but interleaved with the others.
    Records each env's emitted button at each of its own stream positions."""
    n_envs, T = argmax.shape
    clock = MotorClock(n_envs, ac)
    emitted_hist = np.zeros((n_envs, T), dtype=np.int32)
    ptr = np.zeros(n_envs, dtype=np.int64)  # next stream index per env
    srng = np.random.default_rng(sched_seed)
    # build the full-width per-cycle input by gathering each ready env's ptr slot
    while np.any(ptr < T):
        avail = np.nonzero(ptr < T)[0]
        # a random, out-of-order ready subset (the furnace batches whatever is ready)
        k = int(srng.integers(1, len(avail) + 1))
        idx = np.sort(srng.permutation(avail)[:k])
        am = np.zeros(n_envs, np.int32)
        gt = np.zeros(n_envs, np.float32)
        sl = np.zeros(n_envs, np.float32)
        for i in idx:
            am[i] = argmax[i, ptr[i]]
            gt[i] = gate[i, ptr[i]]
            sl[i] = sal[i, ptr[i]]
        lg = _logits(n_envs, am)
        e = clock.decide(am, gt, lg, sl, None, ready_idx=idx)
        for i in idx:
            emitted_hist[i, ptr[i]] = e[i]
            ptr[i] += 1
    return emitted_hist, clock


def test_engine_parity_serial_vs_furnace_dwell_identical():
    """Same per-env stream => identical dwell behavior whether every env clocks
    every step (serial/barrier) or an out-of-order ready subset clocks each cycle
    (furnace).  This is the care-point: any engine bypassing the MotorClock, or a
    clock that ticked non-ready envs, would diverge here."""
    ac = _ac(reflex_margin=0.5, salience_interrupt=True, salience_z=1.5,
             salience_warmup=8, max_dwell=8)
    argmax, gate, sal = _make_streams(n_envs=6, T=40, seed=3)
    serial_hist, serial_clock = _run_serial(ac, argmax, gate, sal)
    for sched in (0, 1, 2):
        furnace_hist, furnace_clock = _run_furnace(ac, argmax, gate, sal, sched)
        assert np.array_equal(serial_hist, furnace_hist), (
            f"emitted-button history diverged (schedule {sched})"
        )
        # the committed dwell multiset must match too (order differs by interleave)
        assert sorted(serial_clock.dwells) == sorted(furnace_clock.dwells)
        assert serial_clock.break_cause == furnace_clock.break_cause
        # the §11b per-env EMA-z salience and R_resp correlation are per-env, so they
        # too are invariant to the ready-subset interleaving (a global EMA would not).
        np.testing.assert_allclose(
            serial_clock.resp_correlation(), furnace_clock.resp_correlation(),
            equal_nan=True,
        )


# ==========================================================================
# 6. salience helper: motion-block mean-abs-dev; None offsets -> zeros (retina)
# ==========================================================================
def test_foveal_salience_motion_block():
    # foveal obs at periph_grid=12: motion block is [288:432]
    X = np.full((2, 454), 0.5, np.float32)  # no motion baseline -> surprise 0
    assert np.allclose(_foveal_salience(X, 288, 432), 0.0)
    X[0, 288:432] = 1.0  # full motion on row 0
    s = _foveal_salience(X, 288, 432)
    assert s[0] == pytest.approx(0.5) and s[1] == pytest.approx(0.0)
    # None offsets (retina / fovea_static): salience disabled -> zeros
    assert np.array_equal(_foveal_salience(X, None, None), np.zeros(2, np.float32))


# ==========================================================================
# 7. §11b R_resp reformulation: salience<->commit correlation
# ==========================================================================
def _drive(ac, gate_stream, sal_stream, argmax_stream=None):
    """Run a single-env clock over per-step (gate, salience[, argmax]) streams and
    return its resp_correlation() value."""
    T = len(gate_stream)
    clock = MotorClock(1, ac)
    for t in range(T):
        a = int(argmax_stream[t]) if argmax_stream is not None else (t % N_BUTTONS)
        amax = np.array([a], np.int32)
        clock.decide(amax, np.array([gate_stream[t]], np.float32), _logits(1, amax),
                     np.array([sal_stream[t]], np.float32), None)
    return float(clock.resp_correlation()[0]), clock


def test_reflex_margin_defaults_off():
    """§11b: reflex_margin defaults to 0.0 (OFF) — the gate learns 'a better button
    appeared' from the same inputs, so it is not a default-on tuned value."""
    assert ACConfig().reflex_margin == 0.0


def test_resp_correlation_masher_is_zero():
    """A masher commits every step (gate always open) -> c≡1 -> zero commit
    variance -> correlation guarded to 0 (not rewarded for thrashing)."""
    sal = np.where(np.arange(60) % 3 == 0, 0.5, 0.05)  # salience has variance
    r, _ = _drive(_ac(salience_interrupt=False, reflex_margin=0.0, max_dwell=0),
                  np.full(60, 1.0), sal)
    assert r == 0.0


def test_resp_correlation_freezer_is_zero():
    """A freezer never opens the gate (no interrupts, no cap) -> c≡0 -> zero commit
    variance -> correlation guarded to 0 (a legitimate holder is NOT penalized)."""
    sal = np.where(np.arange(60) % 3 == 0, 0.5, 0.05)
    r, _ = _drive(_ac(salience_interrupt=False, reflex_margin=0.0, max_dwell=0),
                  np.full(60, -1.0), sal)
    assert r == 0.0


def test_resp_correlation_salience_tracker_is_strongly_positive():
    """A state-appropriate agent that commits IFF motion is high -> c tracks s ->
    strongly positive correlation (the reformulated responsiveness reward)."""
    hi = np.arange(60) % 3 == 0
    sal = np.where(hi, 0.5, 0.05)
    gate = np.where(hi, 1.0, -1.0)  # open exactly on the high-salience steps
    r, _ = _drive(_ac(salience_interrupt=False, reflex_margin=0.0, max_dwell=0),
                  gate, sal)
    assert r > 0.9  # commits and motion move together


def test_resp_correlation_constant_salience_guarded_to_zero():
    """Zero-variance salience (s≡const) -> zero salience variance -> guarded to 0
    even though the gate (commits) varies."""
    gate = np.where(np.arange(60) % 2 == 0, 1.0, -1.0)
    r, _ = _drive(_ac(salience_interrupt=False, reflex_margin=0.0, max_dwell=0),
                  gate, np.full(60, 0.3))
    assert r == 0.0


def test_resp_correlation_retina_all_zero_salience_is_nan_sentinel():
    """Retina (no motion sheet -> all-zero salience) never sets sal_seen, so
    resp_correlation returns a NaN sentinel -> the caller falls back to legacy
    R_resp for those envs (AC-off/retina keep entropy+var; only foveal+AC swaps)."""
    gate = np.where(np.arange(60) % 2 == 0, 1.0, -1.0)
    r, clock = _drive(_ac(salience_interrupt=False, reflex_margin=0.0, max_dwell=0),
                      gate, np.zeros(60))
    assert np.isnan(r)
    assert not clock.rc_sal_seen[0]  # salience was never meaningfully non-zero


def test_resp_correlation_mixed_foveal_and_frozen_env():
    """Per-env sentinel: a moving (foveal) env gets a finite correlation while a
    frozen (all-zero salience) env in the SAME batch returns NaN -> the caller
    replaces only the finite rows, leaving the frozen row on legacy R_resp."""
    ac = _ac(salience_interrupt=False, reflex_margin=0.0, max_dwell=0)
    clock = MotorClock(2, ac)
    hi = np.arange(60) % 3 == 0
    for t in range(60):
        amax = np.array([t % N_BUTTONS, t % N_BUTTONS], np.int32)
        gate = np.array([1.0 if hi[t] else -1.0, -1.0], np.float32)  # env1 frozen
        sal = np.array([0.5 if hi[t] else 0.05, 0.0], np.float32)  # env1 no motion
        clock.decide(amax, gate, _logits(2, amax), sal, None)
    r = clock.resp_correlation()
    assert r[0] > 0.9 and np.isnan(r[1])


def test_resp_scores_legacy_unchanged():
    """AC-off path: _resp_scores still returns exactly button-entropy + variance
    (the reformulation adds a fallback at the CALL sites; the legacy function is
    byte-for-byte unchanged, so AC-off R_resp is identical to pre-AC)."""
    # a uniform 9-button histogram -> Shannon entropy == log2(9); a constant output
    # -> zero variance term.  Verified against the closed-form value.
    btn_hist = np.full((1, N_BUTTONS), 4, dtype=np.int64)  # uniform over 9 buttons
    o_sum = np.zeros((1, 11), dtype=np.float64)  # constant (all-zero) output
    o_sq = np.zeros(1, dtype=np.float64)
    o_cnt = np.full(1, 36, dtype=np.int64)
    resp = _resp_scores(btn_hist, o_sum, o_sq, o_cnt, w_resp_var=0.5)
    assert resp[0] == pytest.approx(np.log2(9))  # uniform entropy, zero variance
    # a single-button (degenerate) histogram -> zero entropy
    bh2 = np.zeros((1, N_BUTTONS), dtype=np.int64)
    bh2[0, 3] = 36
    resp2 = _resp_scores(bh2, o_sum, o_sq, o_cnt, w_resp_var=0.5)
    assert resp2[0] == pytest.approx(0.0)
