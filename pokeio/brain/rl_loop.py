"""System-1 RL loop — PPO actor-critic + backward-robustification (#32/#34).

Ties the System-1 core together on the existing furnace: obs (FovealEncoder) ->
ActorCritic -> button -> fleet step -> dense from-boot reward (read off the fleet's
full-WRAM channel) -> GAE -> PPO update, with episodes started from the
backward-robustification curriculum (start near the demo's goal, recede to boot).

Design choices (v1, gate-focused):
  * Engine = BarrierFleet (clean synchronous ``step_all``; bit-identical to the
    async furnace, shares FovealEncoder + goexplore restore). Port to AsyncFleet for
    throughput AFTER the gate (#36).
  * Action = the 9-way button (Discrete PPO). Gaze held at 0 for v1 (learned gaze is
    the post-gate stage #11). Motor timing reuses ``frame_skip`` (premotor, #34).
  * Reward reads ``fleet.arr["wram"]`` (the opt-in full-WRAM channel) per env per
    step — NEVER the obs, so progress can't leak into the percept.
  * Episodes: one rollout = reset_all(restore=curriculum depths) then a horizon sized
    to the frontier (short near the goal, growing toward boot). Per-env "reached the
    milestone" (party>0) is reported back to the curriculum to recede the frontier.
  * Neuromodulation (inline, #40 seed): the entropy coefficient self-calibrates off
    the curriculum's success EMA (NE -> explore-temp): explore more when stalled,
    exploit when winning. No hand-set schedule.

The premotor cadence / AC commit-gate and the async port are follow-ons; this is the
smallest correct loop that can answer the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from pokeio.brain.actor_critic import ActorCritic
from pokeio.brain.coupling import coupling_report
from pokeio.brain.replay import BackwardCurriculum, DemoTrajectory, ForwardCurriculum
from pokeio.brain.reward import ProgressReward


@dataclass
class BrainConfig:
    n_envs: int = 64
    gamma: float = 0.999          # long-horizon: the milestone is ~500 steps out
    lam: float = 0.95             # GAE
    clip: float = 0.2             # PPO clip
    vf_coef: float = 0.5
    ent_coef_base: float = 0.02   # entropy BONUS, scaled by (1 - success_ema): explore when stalled
    decisiveness: float = 0.01    # entropy PENALTY, scaled by success_ema: commit when winning
                                  # (closes the greedy gap — a diffuse policy that only solves
                                  #  stochastically; self-tuned by the EMA, no hard schedule)
    lr: float = 2.5e-4
    epochs: int = 4               # PPO epochs per rollout
    minibatches: int = 4
    max_grad_norm: float = 0.5
    horizon_margin: int = 96      # steps beyond (len - frontier) to allow reaching goal
    horizon_min: int = 48
    horizon_max: int = 896        # compute cap; at boot this is the EXPLORE horizon
                                  # (past the demo endpoint; bounded by the bigger canvas obs)
    eval_every: int = 20          # iterations between from-boot gate evals
    eval_horizon: int = 896
    seed: int = 0
    # -- forward-progression (brain6) archive bounds: RESOURCE caps, not behaviour
    # knobs, mirroring the NEAT loop's defaults (loop.py --goexplore-capacity /
    # --goexplore-caps-per-round). capacity bounds blob RAM (the only heavy
    # payload); caps_per_round meters the measured ~47 ms worker-side save_states
    # (the realtime CPU-burn cause) via the archive's existing token bucket.
    goexplore_capacity: int = 16384
    goexplore_caps_per_round: float = 4.0


def _gae(rewards, values, last_value, gamma, lam):
    """Generalized advantage estimation over a (T, N) rollout (no mid-episode dones:
    one rollout = one episode, bootstrapped by ``last_value`` (N,))."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)
    nxt = last_value
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * nxt - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        nxt = values[t]
    return adv, adv + values


class BrainTrainer:
    """The System-1 trainer. Owns the policy/optimizer/reward/curriculum; drives a
    (already-constructed) BarrierFleet that was created with ``expose_wram=True`` and
    ``goexplore=True``. Call :meth:`train(iterations)`."""

    def __init__(self, fleet, *, grid: int = 12, fovea_grid: int | None = None,
                 mem_grid: int = 0, reflex_gaze: bool = True, learned_gaze: bool = False,
                 forward: bool = False, device="cuda:1",
                 demo: DemoTrajectory | None = None, cfg: BrainConfig | None = None):
        self.fleet = fleet
        self.cfg = cfg or BrainConfig()
        self.device = torch.device(device)
        self.n_envs = int(fleet.n_envs)
        self.obs_dim = int(fleet.obs_dim)
        self.grid = int(grid)
        self.fovea_grid = int(fovea_grid) if fovea_grid is not None else int(grid)
        self.mem_grid = int(mem_grid)   # >0 => the AI acts on the persisted-vision CANVAS
        torch.manual_seed(self.cfg.seed)

        # Reflex saccade (bottom-up, motion x staleness, self-calibrated) fills the
        # canvas over saccades. A reference encoder (matching the fleet's layout — the
        # memory buffer shifts the proprio offset) gives o_proprio + reflex params;
        # ReflexGaze turns the target (in the obs proprio) into the per-step gaze delta.
        self._reflex = None
        self._o_proprio = 0
        if reflex_gaze:
            from pokeio.emu.fleet import FovealEncoder, ReflexGaze
            ref = FovealEncoder(1, periph_grid=grid, fovea_native_px=48,
                                fovea_grid=self.fovea_grid, n_ram=8, reflex_gaze=True,
                                foveal_memory=self.mem_grid > 0, mem_grid=max(1, self.mem_grid))
            self._reflex = ReflexGaze.maybe(ref, self.n_envs)
            self._o_proprio = int(ref.o_proprio)

        # Path B (#11): a LEARNED gaze-delta action ADDED to the reflex, trained by the
        # joint PPO objective (critic-baselined REINFORCE).  Off => reflex-only gaze,
        # byte-identical to before.
        self._learned_gaze = bool(learned_gaze)
        self.policy = ActorCritic(self.obs_dim, periph_grid=grid,
                                  fovea_grid=self.fovea_grid,
                                  canvas_grid=self.mem_grid,
                                  learned_gaze=self._learned_gaze).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.reward = ProgressReward(self.n_envs)
        self.demo = demo if demo is not None else DemoTrajectory()
        # Forward-progression regime (brain6): episodes spawn from a SELF-TUNING
        # mix of {boot, demo depth, Go-Explore archive frontier} so no open-loop
        # script solves the start distribution and vision becomes necessary. The
        # capture half of the fleet's goexplore plumbing (dormant in the backward
        # path) is switched on: worker cell keys feed a parent NoveltyArchive
        # (bookkeeping only — keys are computed worker-side with the fleet's own
        # geometry) and promising cells are captured into a bounded GoExplore
        # archive whose states seed later spawns. Off => byte-identical backward.
        self._forward = bool(forward)
        if self._forward:
            from pokeio.reward.archive import NoveltyArchive
            from pokeio.reward.goexplore import GoExplore

            self.curriculum = ForwardCurriculum(len(self.demo), seed=self.cfg.seed)
            # The parent novelty set must hold the SAME (churn-masked) keys the
            # workers mint: reuse the fleet's archive kwargs (brain_loop's
            # build_fleet calibrates the WRAM churn-mask, the loop.py:3647-3686
            # idiom). Unmasked keys land in the tile-map buffer that churns on
            # every camera scroll — the seen-set would grow without bound and
            # the capture budget would be spent on scroll/animation noise.
            self.archive = NoveltyArchive(
                **dict(getattr(fleet, "archive_kwargs", None) or {}))
            self.goexplore = GoExplore(
                capacity=self.cfg.goexplore_capacity,
                caps_per_round=self.cfg.goexplore_caps_per_round,
                rng=np.random.default_rng(self.cfg.seed + 1))
        else:
            self.curriculum = BackwardCurriculum(len(self.demo), seed=self.cfg.seed)
        # RAM block slice of the foveal obs (for the coupling ablation probe).
        self._ram_slice = (self.obs_dim - int(getattr(fleet, "obs_ram", 8)), self.obs_dim)
        self.iter = 0
        self.history: list[dict] = []

    # -- helpers -----------------------------------------------------------
    def _wram_rows(self):
        w = self.fleet.arr["wram"]
        return [w[i] for i in range(self.n_envs)]

    def _horizon(self, spawns=None) -> int:
        # At boot (frontier receded to 0) use the full EXPLORE horizon, so from-boot
        # episodes run long PAST the demo's endpoint and the dense from-boot reward
        # can pull the policy toward the next milestones (deeper into the game).
        if spawns is not None:
            # Forward mix: the SAME rule generalized per spawn, no new constant.
            # A demo spawn has a known distance-to-goal, so it gets the backward
            # formula with the batch's shallowest sampled depth standing in for
            # the frontier (the frontier IS where backward's depths cluster). A
            # boot/archive spawn has no known distance-to-goal — exactly the
            # existing frontier<=0 case — so it needs the horizon_max explore/
            # compute cap. One rollout shares one H: take the max over the batch.
            demo_depths = [d for s, d in spawns if s == "demo" and d > 0]
            if len(demo_depths) < len(spawns):
                return self.cfg.horizon_max
            played = len(self.demo) - min(demo_depths)
            return int(np.clip(played + self.cfg.horizon_margin,
                               self.cfg.horizon_min, self.cfg.horizon_max))
        if self.curriculum.frontier <= 0:
            return self.cfg.horizon_max
        played = len(self.demo) - self.curriculum.frontier
        h = int(played + self.cfg.horizon_margin)
        return int(np.clip(h, self.cfg.horizon_min, self.cfg.horizon_max))

    @torch.no_grad()
    def _act(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        out = self.policy.act(obs)
        gaze = out["gaze"].cpu().numpy().astype(np.float32) if "gaze" in out else None
        return (out["buttons"].cpu().numpy().astype(np.int32),
                out["logp"].cpu().numpy(), out["value"].cpu().numpy(), gaze)

    def _gaze(self, obs, n: int, learned=None):
        """Per-env saccade command to submit = self-calibrated REFLEX (bottom-up,
        toward motion x staleness from the obs proprio target) + the LEARNED top-down
        delta (Path B, #11).  ``learned`` is the sampled ``(n,2)`` gaze action (or None
        for reflex-only, the byte-identical default).  ReflexGaze.blend adds the reflex
        ON TOP of the learned seed, so ``gaze_delta = reflex + learned`` — the net can
        follow, nudge, or override the reflex (tanh in update_gaze saturates)."""
        z = np.zeros(n, np.float32)
        ldx = z if learned is None else np.ascontiguousarray(learned[:, 0], np.float32)
        ldy = z if learned is None else np.ascontiguousarray(learned[:, 1], np.float32)
        if self._reflex is None:
            return ldx, ldy                       # learned-only when reflex is off
        gdx, gdy = self._reflex.blend(np.asarray(obs, np.float32), self._o_proprio,
                                      ldx.copy(), ldy.copy())
        return gdx.astype(np.float32), gdy.astype(np.float32)

    # -- forward-progression spawn/capture plumbing (brain6) ----------------
    def _demo_history(self, depth: int):
        """Spawning-trajectory progress for a demo-depth restore (``None`` at
        boot, and for demo stand-ins that expose no ``history_at``)."""
        if depth <= 0:
            return None
        fn = getattr(self.demo, "history_at", None)
        return fn(depth) if fn else None

    def _forward_spawns(self):
        """Draw this rollout's spawn mix and build the fleet restore map.

        Returns ``(spawns, restore, base_depth, histories)``. Demo blobs come
        from the demo capture ladder; archive blobs from Go-Explore
        ``sample_many`` (exactly the NEAT restore idiom, loop.py).
        ``base_depth`` is each env's CUMULATIVE distance-from-newgame (demo
        depth, or the restored cell's chained depth) so cells discovered this
        rollout store honest frontier depths. ``histories`` maps env -> the
        spawning trajectory's progress snapshot (demo prefix / capturing cell's
        reward state) for the teleport-decoupled reward baseline."""
        N = self.n_envs
        spawns = self.curriculum.sample_spawns(N, self.goexplore.size)
        base_depth = np.zeros(N, dtype=np.int64)
        histories: dict[int, dict] = {}
        restore = self.demo.restore_map(
            {i: d for i, (s, d) in enumerate(spawns) if s == "demo"})
        for i, (s, d) in enumerate(spawns):
            if s == "demo":
                base_depth[i] = d
                h = self._demo_history(d)
                if h:
                    histories[i] = h
        arch = [i for i, (s, _) in enumerate(spawns) if s == "archive"]
        if arch:
            for i, entry in zip(arch, self.goexplore.sample_many(len(arch))):
                restore[i] = entry.state
                base_depth[i] = entry.depth
                if entry.progress:
                    histories[i] = entry.progress
            self.goexplore.n_restores += len(arch)
        return spawns, restore, base_depth, histories

    def _absorb_captures(self, keys, captured, cap_flags, pending, base_depth, t):
        """One barrier round of parent-side Go-Explore bookkeeping (deferred
        capture semantics — mirrors the NEAT barrier loop, train/loop.py):
        fulfil the flags set LAST round (``captured`` holds the state each env
        was still in), then pick this round's candidates — globally-new cells
        plus still-rare re-captures of evicted ones — under the token budget
        that meters the ~47 ms worker-side save_states."""
        for i, (key, depth) in pending.items():
            blob = captured.get(i)
            if blob is not None:
                # The reward has absorbed exactly through LAST round's step —
                # the same state the worker's deferred capture serialized — so
                # this snapshot is the blob's own progress history (running
                # maxes + maps traversed), stored with it for teleport-decoupled
                # re-baselining when the cell later seeds a spawn.
                self.goexplore.store_captured(key, blob, depth,
                                              progress=self.reward.snapshot(i))
        pending.clear()
        cap_flags[:] = 0
        cand: list[tuple[int, bytes]] = []
        for i in range(self.n_envs):
            key = keys[i].tobytes()
            globally_new = self.archive.add(key)
            prior = self.archive.visit(key)
            if not self.goexplore.revisit(key) and (globally_new or prior < 3):
                cand.append((i, key))
        self.goexplore.feed_capture_budget(1.0)
        off = t % len(cand) if cand else 0
        for j in range(len(cand)):
            if not self.goexplore.admit_capture():
                self.goexplore.n_throttled += len(cand) - 1 - j
                break
            i, key = cand[(off + j) % len(cand)]
            cap_flags[i] = 1
            pending[i] = (key, int(base_depth[i]) + t)

    # -- one curriculum rollout (backward demo-recede, or forward spawn-mix) --
    def rollout(self) -> dict:
        cfg, N = self.cfg, self.n_envs
        if self._forward:
            spawns, restore, base_depth, histories = self._forward_spawns()
            self.goexplore.begin_generation(self.iter)  # recency clock = iters
        else:
            spawns = None
            depths = self.curriculum.sample_depths(N)
            restore = self.demo.restore_map({i: depths[i] for i in range(N)})
            histories = {}
            for i in range(N):
                h = self._demo_history(depths[i])
                if h:
                    histories[i] = h
        obs = self.fleet.reset_all(restore=restore).copy()
        # TELEPORT-DECOUPLING INVARIANT: re-baseline the reward from the RESTORED
        # state's WRAM (the reset round published it) PLUS the spawning
        # trajectory's own progress history BEFORE any reward step — progress a
        # spawn was HANDED (running maxes AND the maps its trajectory already
        # traversed, which the WRAM alone cannot show) is never paid, for demo
        # AND archive spawns alike.
        self.reward.reset_many(range(N), self._wram_rows(),
                               [histories.get(i) for i in range(N)])
        H = self._horizon(spawns)

        obs_buf = np.zeros((H, N, self.obs_dim), np.float32)
        act_buf = np.zeros((H, N), np.int64)
        logp_buf = np.zeros((H, N), np.float32)
        val_buf = np.zeros((H, N), np.float32)
        rew_buf = np.zeros((H, N), np.float32)
        gaze_buf = np.zeros((H, N, 2), np.float32) if self._learned_gaze else None
        reached = np.zeros(N, dtype=bool)
        cap_flags = np.zeros(N, np.uint8) if self._forward else None
        pending: dict[int, tuple[bytes, int]] = {}

        for t in range(H):
            buttons, logp, value, gaze = self._act(obs)
            obs_buf[t] = obs
            act_buf[t] = buttons
            logp_buf[t] = logp           # JOINT (button + gaze) log-prob when learned_gaze
            val_buf[t] = value
            if gaze_buf is not None:
                gaze_buf[t] = gaze
            gdx, gdy = self._gaze(obs, N, learned=gaze)
            if self._forward:
                obs2, keys, _dones, captured = self.fleet.step_all(
                    buttons, capture_flags=cap_flags, gaze_dx=gdx, gaze_dy=gdy)
                self._absorb_captures(keys, captured, cap_flags, pending,
                                      base_depth, t)
            else:
                obs2, _keys, _dones, _cap = self.fleet.step_all(buttons, gaze_dx=gdx, gaze_dy=gdy)
            rew_buf[t] = self.reward.step_many(range(N), self._wram_rows())
            obs = obs2.copy()
            reached |= np.array([self.reward.started(i) for i in range(N)])

        with torch.no_grad():
            last_v = self.policy(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            )[1].cpu().numpy()
        adv, ret = _gae(rew_buf, val_buf, last_v, cfg.gamma, cfg.lam)
        if self._forward:
            # Outcome = spawn-relative MILESTONE crossing (earned_milestone):
            # a tier-0 spawn succeeds only by EARNING started() (party>0/badge
            # — the same real milestone the BackwardCurriculum receded on); a
            # spawn handed that milestone reports against the NEXT unearned
            # badge/event tier. Immune to the started() leak (handed party>0
            # is in the spawn baseline, not a success) AND not trivially
            # satisfiable by +w_map map transitions — 'walk out the door' must
            # not recede the frontier, flip ent_coef into its decisiveness
            # penalty, or saturate the source EMAs. ``reached`` stays a pure
            # metric here.
            progressed = [self.reward.earned_milestone(i) for i in range(N)]
            self.curriculum.report_spawns([s for s, _ in spawns], progressed)
        else:
            self.curriculum.report_many(reached.tolist())
        return {
            "obs": obs_buf.reshape(H * N, self.obs_dim),
            "act": act_buf.reshape(H * N),
            "logp": logp_buf.reshape(H * N),
            "gaze": gaze_buf.reshape(H * N, 2) if gaze_buf is not None else None,
            "adv": adv.reshape(H * N),
            "ret": ret.reshape(H * N),
            "H": H, "reached": float(reached.mean()),
            "rew_sum_mean": float(rew_buf.sum(0).mean()),
        }

    # -- PPO update --------------------------------------------------------
    def update(self, batch) -> dict:
        cfg = self.cfg
        dev = self.device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=dev)
        act = torch.as_tensor(batch["act"], dtype=torch.long, device=dev)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=dev)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=dev)
        adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Path B: the sampled gaze delta is part of the action; re-evaluate its JOINT
        # log-prob so PPO trains it (critic-baselined REINFORCE).  None => button-only.
        gaze = (torch.as_tensor(batch["gaze"], dtype=torch.float32, device=dev)
                if batch.get("gaze") is not None else None)

        # NE neuromodulation: explore when stalled (bonus), COMMIT when winning
        # (penalty) — both self-scaled by the success EMA, so a saturated policy is
        # pushed to be decisive (greedy-workable), not left diffuse.
        ema = self.curriculum.success_ema
        ent_coef = cfg.ent_coef_base * (1.0 - ema) - cfg.decisiveness * ema
        n = obs.shape[0]
        idx = np.arange(n)
        mb = max(1, n // cfg.minibatches)
        stats = {"pi_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "clipfrac": 0.0, "kl": 0.0}
        n_upd = 0
        for _ in range(cfg.epochs):
            self._rng().shuffle(idx)
            for s in range(0, n, mb):
                j = idx[s:s + mb]
                logp, ent, val = self.policy.evaluate_actions(
                    obs[j], act[j], gaze[j] if gaze is not None else None)
                ratio = torch.exp(logp - old_logp[j])
                a = adv[j]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a
                pi_loss = -torch.min(unclipped, clipped).mean()
                vf_loss = 0.5 * (val - ret[j]).pow(2).mean()
                ent_mean = ent.mean()
                loss = pi_loss + cfg.vf_coef * vf_loss - ent_coef * ent_mean
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pi_loss"] += pi_loss.item()
                    stats["vf_loss"] += vf_loss.item()
                    stats["entropy"] += ent_mean.item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    stats["kl"] += (old_logp[j] - logp).mean().item()
                n_upd += 1
        for k in stats:
            stats[k] /= max(1, n_upd)
        stats["ent_coef"] = ent_coef
        return stats

    def _rng(self):
        if not hasattr(self, "_np_rng"):
            self._np_rng = np.random.default_rng(self.cfg.seed)
        return self._np_rng

    # -- from-boot gate eval ----------------------------------------------
    @torch.no_grad()
    def _gate_pass(self, greedy: bool, H: int) -> dict:
        """One from-boot (no restore) eval pass; returns reached-frac, progress, the
        final per-env map-id (for the greedy-stall diagnostic), and the last obs."""
        N = self.n_envs
        obs = self.fleet.reset_all(restore=None).copy()
        self.reward.reset_many(range(N), self._wram_rows())
        reached = np.zeros(N, dtype=bool)
        last_obs = obs
        for _ in range(H):
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            out = self.policy.act(t, greedy=greedy)
            buttons = out["buttons"].cpu().numpy().astype(np.int32)
            gaze = out["gaze"].cpu().numpy().astype(np.float32) if "gaze" in out else None
            gdx, gdy = self._gaze(obs, N, learned=gaze)
            obs, _k, _d, _c = self.fleet.step_all(buttons, gaze_dx=gdx, gaze_dy=gdy)
            obs = obs.copy()
            self.reward.step_many(range(N), self._wram_rows())
            reached |= np.array([self.reward.started(i) for i in range(N)])
            last_obs = obs
        progress = np.array([self.reward.progress(i)["phi"] for i in range(N)])
        wram = self.fleet.arr["wram"]
        final_maps = [int(wram[i][0xD35D - 0xC000]) for i in range(N)]
        return {"reached_frac": float(reached.mean()),
                "progress_mean": float(progress.mean()),
                "progress_max": float(progress.max()),
                "final_maps": final_maps, "last_obs": last_obs}

    @torch.no_grad()
    def evaluate_gate(self, horizon: int | None = None) -> dict:
        """From-boot (no restore) gate eval — BOTH stochastic and greedy (#35/#44).

        The HONEST gate is stochastic: the policy's own sampled actions from a cold
        boot reach party>0. Greedy (argmax) is reported alongside with a stall-map
        histogram (where argmax ends up), because a policy that solves stochastically
        but not greedily is capable-but-not-decisive — and the map histogram localizes
        where the deterministic run gets stuck. Plus the permanent coupling probes.
        """
        from collections import Counter

        H = int(horizon or self.cfg.eval_horizon)
        greedy = self._gate_pass(greedy=True, H=H)
        stoch = self._gate_pass(greedy=False, H=H)
        cp = coupling_report(self.policy.numpy_policy_fn(self.device), greedy["last_obs"],
                             ram_slice=self._ram_slice)
        return {
            "gate_reached_frac": stoch["reached_frac"],       # HONEST gate (stochastic)
            "gate_reached_greedy": greedy["reached_frac"],
            "progress_mean": stoch["progress_mean"],
            "progress_max": stoch["progress_max"],
            "greedy_stall_maps": dict(Counter(greedy["final_maps"]).most_common(5)),
            **cp,
        }

    # -- driver ------------------------------------------------------------
    def train(self, iterations: int, log_every: int = 1, on_log=None) -> list[dict]:
        for _ in range(iterations):
            self.iter += 1
            batch = self.rollout()
            stats = self.update(batch)
            rec = {"iter": self.iter, **{k: batch[k] for k in ("H", "reached", "rew_sum_mean")},
                   **stats, **self.curriculum.state()}
            if self._forward:
                # Seen-set growth is the churn-mask's health metric (an unmasked
                # key stream grows this without bound); goexplore_states shows
                # whether the capture budget is buying real frontier cells.
                rec["archive_cells"] = self.archive.size
                rec["goexplore_states"] = self.goexplore.size
            if self.cfg.eval_every and self.iter % self.cfg.eval_every == 0:
                rec["eval"] = self.evaluate_gate()
            self.history.append(rec)
            if on_log and self.iter % log_every == 0:
                on_log(rec)
        return self.history


__all__ = ["BrainTrainer", "BrainConfig"]
