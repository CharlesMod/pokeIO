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
from pokeio.brain.replay import BackwardCurriculum, DemoTrajectory
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
    horizon_max: int = 768        # compute cap on a single rollout
    eval_every: int = 20          # iterations between from-boot gate evals
    eval_horizon: int = 768
    seed: int = 0


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
                 reflex_gaze: bool = True, device="cuda:1",
                 demo: DemoTrajectory | None = None, cfg: BrainConfig | None = None):
        self.fleet = fleet
        self.cfg = cfg or BrainConfig()
        self.device = torch.device(device)
        self.n_envs = int(fleet.n_envs)
        self.obs_dim = int(fleet.obs_dim)
        self.grid = int(grid)
        self.fovea_grid = int(fovea_grid) if fovea_grid is not None else int(grid)
        torch.manual_seed(self.cfg.seed)

        # Reflex saccade (bottom-up, motion x staleness, self-calibrated): a reference
        # encoder gives the proprio offset + reflex params; ReflexGaze turns the target
        # (surfaced in the obs proprio by the worker encoders) into the per-step gaze
        # delta we submit, so the SHARP fovea follows motion. None => centered fovea.
        self._reflex = None
        self._o_proprio = 0
        if reflex_gaze:
            from pokeio.emu.fleet import FovealEncoder, ReflexGaze
            ref = FovealEncoder(1, periph_grid=grid, fovea_native_px=48,
                                fovea_grid=self.fovea_grid, n_ram=8, reflex_gaze=True)
            self._reflex = ReflexGaze.maybe(ref, self.n_envs)
            self._o_proprio = int(ref.o_proprio)

        self.policy = ActorCritic(self.obs_dim, periph_grid=grid,
                                  fovea_grid=self.fovea_grid).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.reward = ProgressReward(self.n_envs)
        self.demo = demo if demo is not None else DemoTrajectory()
        self.curriculum = BackwardCurriculum(len(self.demo), seed=self.cfg.seed)
        # RAM block slice of the foveal obs (for the coupling ablation probe).
        self._ram_slice = (self.obs_dim - int(getattr(fleet, "obs_ram", 8)), self.obs_dim)
        self.iter = 0
        self.history: list[dict] = []

    # -- helpers -----------------------------------------------------------
    def _wram_rows(self):
        w = self.fleet.arr["wram"]
        return [w[i] for i in range(self.n_envs)]

    def _horizon(self) -> int:
        played = len(self.demo) - self.curriculum.frontier
        h = int(played + self.cfg.horizon_margin)
        return int(np.clip(h, self.cfg.horizon_min, self.cfg.horizon_max))

    @torch.no_grad()
    def _act(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        out = self.policy.act(obs)
        return (out["buttons"].cpu().numpy().astype(np.int32),
                out["logp"].cpu().numpy(), out["value"].cpu().numpy())

    def _gaze(self, obs, n: int):
        """Per-env saccade to submit: the self-calibrated REFLEX command (bottom-up,
        toward motion x staleness, read from the obs proprio target), or zeros when
        reflex is off (centered fovea). Learned top-down gaze (#11) blends in here."""
        z = np.zeros(n, np.float32)
        if self._reflex is None:
            return z, z
        gdx, gdy = self._reflex.blend(np.asarray(obs, np.float32), self._o_proprio,
                                      z.copy(), z.copy())
        return gdx.astype(np.float32), gdy.astype(np.float32)

    # -- one backward-robustification rollout ------------------------------
    def rollout(self) -> dict:
        cfg, N = self.cfg, self.n_envs
        depths = self.curriculum.sample_depths(N)
        restore = self.demo.restore_map({i: depths[i] for i in range(N)})
        obs = self.fleet.reset_all(restore=restore).copy()
        self.reward.reset_many(range(N), self._wram_rows())
        H = self._horizon()

        obs_buf = np.zeros((H, N, self.obs_dim), np.float32)
        act_buf = np.zeros((H, N), np.int64)
        logp_buf = np.zeros((H, N), np.float32)
        val_buf = np.zeros((H, N), np.float32)
        rew_buf = np.zeros((H, N), np.float32)
        reached = np.zeros(N, dtype=bool)

        for t in range(H):
            buttons, logp, value = self._act(obs)
            obs_buf[t] = obs
            act_buf[t] = buttons
            logp_buf[t] = logp
            val_buf[t] = value
            gdx, gdy = self._gaze(obs, N)
            obs2, _keys, _dones, _cap = self.fleet.step_all(buttons, gaze_dx=gdx, gaze_dy=gdy)
            rew_buf[t] = self.reward.step_many(range(N), self._wram_rows())
            obs = obs2.copy()
            reached |= np.array([self.reward.started(i) for i in range(N)])

        with torch.no_grad():
            last_v = self.policy(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            )[1].cpu().numpy()
        adv, ret = _gae(rew_buf, val_buf, last_v, cfg.gamma, cfg.lam)
        self.curriculum.report_many(reached.tolist())
        return {
            "obs": obs_buf.reshape(H * N, self.obs_dim),
            "act": act_buf.reshape(H * N),
            "logp": logp_buf.reshape(H * N),
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
                logp, ent, val = self.policy.evaluate_actions(obs[j], act[j])
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
            buttons = self.policy.act(t, greedy=greedy)["buttons"].cpu().numpy().astype(np.int32)
            gdx, gdy = self._gaze(obs, N)
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
            if self.cfg.eval_every and self.iter % self.cfg.eval_every == 0:
                rec["eval"] = self.evaluate_gate()
            self.history.append(rec)
            if on_log and self.iter % log_every == 0:
                on_log(rec)
        return self.history


__all__ = ["BrainTrainer", "BrainConfig"]
