"""Recurrent PPO over an asynchronous env pool.

Rollout layout: every env fills its own row of ``rollout_len`` steps, whatever
order the pool returns workers in. Workers whose rows are full are held: their
latest observation is both the GAE bootstrap and the first observation of the
next rollout, so no transitions are wasted.

Differences from pokemonred_puffer's CleanPuffeRL, each fixing a known issue:
  * LSTM state is stored at the start of every BPTT chunk, and training replays
    each chunk from that stored state. puffer started each minibatch from zeros
    and carried state across unrelated envs.
  * The LSTM state is zeroed at episode starts and swarm migrations.
  * GAE bootstraps correctly at row ends (puffer: "TODO: bootstrap between segment bounds").
  * Value targets are normalized (ValueNorm), because Pokémon-style rewards span
    ~0.005 per tile up to 10+ per badge.
  * Transitions whose action was never executed (swarm load) are masked out.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from pokeio.config import Config
from pokeio.env import POLICY_KEYS
from pokeio.policy import Policy
from pokeio.spec import GameSpec
from pokeio.swarm import SwarmCoordinator
from pokeio.vec import Batch, make_vec


class ValueNorm:
    """Running mean/std of value targets (EMA with debiasing, as in MAPPO)."""

    def __init__(self, beta: float = 0.99, eps: float = 1e-5, device="cpu"):
        self.beta, self.eps = beta, eps
        self.mean = torch.zeros((), device=device, dtype=torch.float64)
        self.sq = torch.zeros((), device=device, dtype=torch.float64)
        self.debias = torch.zeros((), device=device, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        x = x.double()
        self.mean.mul_(self.beta).add_(x.mean() * (1 - self.beta))
        self.sq.mul_(self.beta).add_((x * x).mean() * (1 - self.beta))
        self.debias.mul_(self.beta).add_(1 - self.beta)

    def _stats(self):
        d = self.debias.clamp(min=self.eps)
        mean = self.mean / d
        var = (self.sq / d - mean * mean).clamp(min=1e-4)
        return mean.float(), var.sqrt().float()

    def normalize(self, x):
        m, s = self._stats()
        return (x - m) / s

    def denormalize(self, x):
        m, s = self._stats()
        return x * s + m

    def state_dict(self):
        return {"mean": self.mean, "sq": self.sq, "debias": self.debias}

    def load_state_dict(self, d):
        self.mean.copy_(d["mean"])
        self.sq.copy_(d["sq"])
        self.debias.copy_(d["debias"])


def _subset(b: Batch, mask: np.ndarray) -> Batch:
    return Batch(
        b.env_ids[mask],
        {k: v[mask] for k, v in b.obs.items()},
        b.rewards[mask],
        b.dones[mask],
        [],
    )


class Logger:
    def __init__(self, run_dir: Path):
        self.f = open(run_dir / "metrics.jsonl", "a")
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(run_dir / "tb"))
        except Exception:
            pass

    def log(self, step: int, data: dict) -> None:
        self.f.write(json.dumps({"global_step": step, **data}) + "\n")
        self.f.flush()
        if self.tb is not None:
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        self.f.close()
        if self.tb is not None:
            self.tb.close()


class Trainer:
    def __init__(self, cfg: Config, spec: GameSpec):
        self.cfg, self.spec = cfg, spec
        tc = cfg.train
        torch.manual_seed(tc.seed)
        np.random.seed(tc.seed)
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.device = torch.device(tc.device if torch.cuda.is_available() or tc.device == "cpu" else "cpu")
        if tc.rollout_len % tc.bptt:
            raise ValueError("rollout_len must be a multiple of bptt")

        name = tc.run_name or time.strftime(f"{spec.name}-%Y%m%d-%H%M%S")
        self.run_dir = Path(tc.run_dir) / name
        (self.run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

        self.vec = make_vec(spec, cfg.vec, seed=tc.seed)
        self.obs_spec = self.vec.obs_spec
        self.N = self.vec.num_envs
        self.T = tc.rollout_len
        self.policy = Policy(self.obs_spec, cfg.policy).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=tc.lr, eps=tc.adam_eps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=tc.amp and self.device.type == "cuda")
        self.vnorm = ValueNorm(device=self.device)
        # Inference step; torch.compile needs Inductor (sm_70+), so it is opt-in.
        self._step = torch.compile(self.policy.step) if tc.compile else self.policy.step

        T, N, dev = self.T, self.N, self.device
        self.buf_obs = {
            k: torch.zeros((T, N, *shape), dtype=getattr(torch, np.dtype(dt).name), device=dev)
            for k, (shape, dt) in self.obs_spec.arrays().items()
            if k in POLICY_KEYS
        }
        z = lambda dt=torch.float32: torch.zeros((T, N), dtype=dt, device=dev)  # noqa: E731
        self.buf_act, self.buf_logp, self.buf_val = z(torch.long), z(), z()
        self.buf_rew, self.buf_done, self.buf_start, self.buf_valid = z(), z(torch.uint8), z(), z()
        H = cfg.policy.hidden
        self.h0 = torch.zeros((T // tc.bptt, N, H), device=dev)
        self.c0 = torch.zeros_like(self.h0)
        self.H, self.C = self.policy.initial_state(N, dev)
        self.next_start = np.ones(N, dtype=np.float32)
        self.held: list[Batch] = []

        self.swarm = SwarmCoordinator(spec.swarm, N, self.run_dir / "frontier", seed=tc.seed)
        self.global_step = 0
        self.update_i = 0
        self.ep_stats: dict[str, list] = defaultdict(list)
        self.milestone_first: dict[str, int] = {}
        self.milestone_counts: dict[str, int] = defaultdict(int)
        self.logger = Logger(self.run_dir)
        self.hub = None
        if cfg.dash.enabled:
            from pokeio.dash import LiveHub, start_server

            d = cfg.dash
            self.hub = LiveHub(spec, self.obs_spec, self.N, self.run_dir, d.wall_size, d.hero_history,
                               d.saliency_every_s)
            self.dash_server = start_server(self.hub, d.host, d.port)
        if tc.resume:
            self._resume(tc.resume)
        self.vec.reset()

    # ------------------------------------------------------------------ infos
    def _handle_infos(self, infos: list[dict]) -> None:
        for info in infos:
            self.swarm.observe(info)
            for m in info.get("milestones", {}):
                self.milestone_counts[m] += 1
                self.milestone_first.setdefault(m, self.global_step)
            ep = info.get("episode")
            if ep:
                for k in ("return", "length", "cells", "rooms", "score"):
                    self.ep_stats[k].append(ep[k])
                for k, v in ep.get("parts", {}).items():
                    self.ep_stats["r/" + k].append(v)

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect(self) -> None:
        T, bptt, dev = self.T, self.cfg.train.bptt, self.device
        ptr = np.zeros(self.N, dtype=np.int64)
        awaiting = np.zeros(self.N, dtype=bool)
        queue, self.held = self.held, []
        self.buf_valid.zero_()
        amp = self.cfg.train.amp and dev.type == "cuda"
        while True:
            if queue:
                b, fresh = queue.pop(), False
            elif (ptr < T).any():
                b, fresh = self.vec.recv(), True
            else:
                break
            ids = b.env_ids
            if fresh:
                aw = awaiting[ids]
                if aw.any():
                    e = ids[aw]
                    t = torch.from_numpy(ptr[e] - 1).to(dev)
                    et = torch.from_numpy(e).to(dev)
                    self.buf_rew[t, et] = torch.from_numpy(b.rewards[aw]).to(dev)
                    self.buf_done[t, et] = torch.from_numpy(b.dones[aw]).to(dev)
                    inv = b.dones[aw] == 2
                    if inv.any():
                        invt = torch.from_numpy(inv).to(dev)
                        self.buf_valid[t[invt], et[invt]] = 0.0
                    self.global_step += int(aw.sum())
                awaiting[ids] = False
                self.next_start[ids] = (b.dones != 0).astype(np.float32)
                self._handle_infos(b.infos)
                if self.hub is not None:
                    self.hub.on_infos(ids, b.obs, b.infos)

            full = ptr[ids] >= T
            if full.any():
                self.held.append(_subset(b, full))
                if full.all():
                    continue
                b = _subset(b, ~full)
                ids = b.env_ids

            t_np = ptr[ids]
            t = torch.from_numpy(t_np).to(dev)
            idt = torch.from_numpy(ids).to(dev)
            obs = {k: torch.from_numpy(b.obs[k]).to(dev, non_blocking=True) for k in POLICY_KEYS}
            st = torch.from_numpy(self.next_start[ids]).to(dev)
            cs = (t_np % bptt) == 0
            if cs.any():
                csi = torch.from_numpy(cs).to(dev)
                self.h0[t[csi] // bptt, idt[csi]] = self.H[idt[csi]]
                self.c0[t[csi] // bptt, idt[csi]] = self.C[idt[csi]]
            if self.hub is not None:
                j = self.hub.want_saliency(ids)
                if j is not None:
                    sl = slice(j, j + 1)
                    with torch.enable_grad():
                        sal, _ = self.policy.saliency({k: v[sl] for k, v in obs.items()},
                                                      (self.H[idt[sl]], self.C[idt[sl]]), st[sl])
                    self.hub.on_saliency(int(ids[j]), sal[0].float().cpu().numpy())
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, value, (h, c) = self._step(obs, (self.H[idt], self.C[idt]), st)
            dist = Categorical(logits=logits.float())
            a = dist.sample()
            if self.hub is not None:
                shown = self.vnorm.denormalize(value.float()) if self.cfg.train.value_norm else value.float()
                self.hub.on_step(ids, b.obs, a.cpu().numpy(), dist.probs.cpu().numpy(),
                                 shown.cpu().numpy(), b.rewards, b.dones, self.global_step)
            self.H[idt], self.C[idt] = h.float(), c.float()
            for k, v in obs.items():
                self.buf_obs[k][t, idt] = v
            self.buf_act[t, idt] = a
            self.buf_logp[t, idt] = dist.log_prob(a)
            self.buf_val[t, idt] = value.float()
            self.buf_start[t, idt] = st
            self.buf_valid[t, idt] = 1.0
            ptr[ids] += 1
            awaiting[ids] = True
            self.next_start[ids] = 0.0
            self.vec.send(a.cpu().numpy().astype(np.int32), ids)

        # bootstrap values from the held observations (state not committed)
        self.next_value = torch.zeros(self.N, device=dev)
        for b in self.held:
            idt = torch.from_numpy(b.env_ids).to(dev)
            obs = {k: torch.from_numpy(b.obs[k]).to(dev) for k in POLICY_KEYS}
            st = torch.from_numpy(self.next_start[b.env_ids]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, v, _ = self._step(obs, (self.H[idt], self.C[idt]), st)
            self.next_value[idt] = v.float()

    # ------------------------------------------------------------------ learning
    @torch.no_grad()
    def advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        tc = self.cfg.train
        V = self.vnorm.denormalize(self.buf_val) if tc.value_norm else self.buf_val
        nextV = self.vnorm.denormalize(self.next_value) if tc.value_norm else self.next_value
        nonterm = (self.buf_done == 0).float()
        adv = torch.zeros_like(self.buf_rew)
        last = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            nv = nextV if t == self.T - 1 else V[t + 1]
            delta = self.buf_rew[t] + tc.gamma * nv * nonterm[t] - V[t]
            last = delta + tc.gamma * tc.gae_lambda * nonterm[t] * last
            adv[t] = last
        adv *= self.buf_valid
        returns = adv + V
        return adv, returns

    def update(self) -> dict[str, float]:
        tc = self.cfg.train
        adv, returns = self.advantages()
        valid = self.buf_valid
        if tc.value_norm:
            self.vnorm.update(returns[valid > 0])
            targets = self.vnorm.normalize(returns)
        else:
            targets = returns
        if tc.anneal_lr:
            frac = max(0.0, 1.0 - self.global_step / tc.total_steps)
            for g in self.opt.param_groups:
                g["lr"] = tc.lr * frac

        S = self.T // tc.bptt
        n_seq = S * self.N
        M = min(tc.minibatch_seqs, n_seq)
        ar = torch.arange(tc.bptt, device=self.device)
        stats = defaultdict(float)
        n_mb = 0
        amp = tc.amp and self.device.type == "cuda"
        stop = False
        for _epoch in range(tc.epochs):
            perm = torch.randperm(n_seq, device=self.device)
            for i in range(0, n_seq - M + 1, M):
                sel = perm[i : i + M]
                k, n = sel // self.N, sel % self.N
                tt = (k[None, :] * tc.bptt + ar[:, None])  # [bptt, M]
                nn_ = n[None, :].expand_as(tt)
                obs = {key: buf[tt, nn_] for key, buf in self.buf_obs.items()}
                w = valid[tt, nn_]
                wsum = w.sum().clamp(min=1.0)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits, newv = self.policy.sequence(obs, (self.h0[k, n], self.c0[k, n]), self.buf_start[tt, nn_])
                dist = Categorical(logits=logits.float())
                newlogp = dist.log_prob(self.buf_act[tt, nn_])
                logratio = newlogp - self.buf_logp[tt, nn_]
                ratio = logratio.exp()
                a = adv[tt, nn_]
                am = (a * w).sum() / wsum
                astd = (((a - am) ** 2 * w).sum() / wsum).sqrt()
                a = (a - am) / (astd + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - tc.clip, 1 + tc.clip))
                pg_loss = (pg * w).sum() / wsum

                newv = newv.float()
                tgt = targets[tt, nn_]
                if tc.vf_clip > 0:
                    oldv = self.buf_val[tt, nn_]
                    vclip = oldv + (newv - oldv).clamp(-tc.vf_clip, tc.vf_clip)
                    vl = torch.max((newv - tgt) ** 2, (vclip - tgt) ** 2)
                else:
                    vl = (newv - tgt) ** 2
                v_loss = 0.5 * (vl * w).sum() / wsum
                ent = (dist.entropy() * w).sum() / wsum
                loss = pg_loss + tc.vf_coef * v_loss - tc.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), tc.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

                with torch.no_grad():
                    kl = ((((ratio - 1) - logratio) * w).sum() / wsum).item()
                    stats["policy_loss"] += pg_loss.item()
                    stats["value_loss"] += v_loss.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += kl
                    stats["clipfrac"] += ((((ratio - 1).abs() > tc.clip).float() * w).sum() / wsum).item()
                    stats["grad_norm"] += float(gn)
                n_mb += 1
                if tc.target_kl > 0 and kl > tc.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: v / max(n_mb, 1) for k, v in stats.items()}
        with torch.no_grad():
            m = valid > 0
            y, yhat = returns[m], (self.vnorm.denormalize(self.buf_val) if tc.value_norm else self.buf_val)[m]
            var = y.var()
            out["explained_variance"] = float(1 - (y - yhat).var() / var) if var > 0 else float("nan")
            out["reward_mean"] = float(self.buf_rew[m].mean())
        return out

    # ------------------------------------------------------------------ loop
    def train(self) -> None:
        tc = self.cfg.train
        t0, s0 = time.time(), self.global_step
        try:
            while self.global_step < tc.total_steps:
                tc_start = time.time()
                self.collect()
                t_collect = time.time() - tc_start
                assign = self.swarm.maybe_migrate(self.global_step)
                if assign:
                    self.vec.load_states(assign)
                    if self.hub is not None:
                        self.hub.on_migration(self.swarm.best_score, self.swarm.last_src, len(assign))
                    print(f"[swarm] migration #{self.swarm.migrations}: {len(assign)} envs -> "
                          f"score {self.swarm.best_score:.2f}", flush=True)
                tu = time.time()
                stats = self.update()
                t_update = time.time() - tu
                self.update_i += 1
                if self.update_i % tc.log_every == 0:
                    data = self._log(stats, t0, s0, t_collect, t_update)
                    if self.hub is not None:
                        acts = torch.bincount(self.buf_act[self.buf_valid > 0],
                                              minlength=self.obs_spec.n_actions).tolist()
                        self.hub.on_update(self.update_i, self.global_step, data["sps"], data, acts)
                        self.hub.save()
                if self.update_i % tc.checkpoint_every == 0:
                    path = self.save()
                    if self.hub is not None:
                        self.hub.on_checkpoint(str(path))
                if self.hub is not None and self.hub.stop_requested:
                    print("[dash] stop requested from the wall", flush=True)
                    break
        finally:
            self.save()
            if self.hub is not None:
                self.hub.status = "finished"
                self.hub.save()
            self.vec.close()
            self.logger.close()

    def _log(self, stats, t0, s0, t_collect, t_update) -> dict:
        sps = (self.global_step - s0) / max(time.time() - t0, 1e-6)
        data = {"sps": sps, "update": self.update_i, "time_collect": t_collect,
                "time_update": t_update, **{f"loss/{k}": v for k, v in stats.items()},
                "swarm/best_score": self.swarm.best_score if self.swarm.best_state else 0.0,
                "swarm/migrations": self.swarm.migrations}
        for k, v in self.ep_stats.items():
            if v:
                data[f"episode/{k}"] = float(np.mean(v))
        for m, n in self.milestone_counts.items():
            data[f"milestone/{m}"] = n
        self.ep_stats.clear()
        self.logger.log(self.global_step, data)
        firsts = ", ".join(f"{m}@{s:.2e}" for m, s in sorted(self.milestone_first.items(), key=lambda x: x[1]))
        print(
            f"upd {self.update_i:5d} | step {self.global_step:.3e} | sps {sps:7.0f} | "
            f"ret {data.get('episode/return', float('nan')):8.2f} | cells {data.get('episode/cells', float('nan')):6.0f} | "
            f"kl {stats.get('approx_kl', 0):.4f} | ev {stats.get('explained_variance', 0):+.2f} | "
            f"swarm {self.swarm.migrations} | {firsts}",
            flush=True,
        )
        return data

    # ------------------------------------------------------------------ checkpoints
    def save(self) -> Path:
        ck = {
            "policy": self.policy.state_dict(),
            "optimizer": self.opt.state_dict(),
            "vnorm": self.vnorm.state_dict(),
            "global_step": self.global_step,
            "update": self.update_i,
            "swarm": self.swarm.state_dict(),
            "milestone_first": self.milestone_first,
            "config": self.cfg.to_dict(),
            "spec_fingerprint": self.spec.fingerprint,
            "obs_spec": asdict(self.obs_spec),
        }
        path = self.run_dir / "ckpt" / f"ckpt_{self.update_i:06d}.pt"
        torch.save(ck, path)
        latest = self.run_dir / "ckpt" / "latest.pt"
        tmp = latest.with_suffix(".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, latest)
        return path

    def _resume(self, path: str) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "ckpt" / "latest.pt" if (p / "ckpt").exists() else p / "latest.pt"
        ck = torch.load(p, map_location=self.device, weights_only=False)
        if ck.get("spec_fingerprint") != self.spec.fingerprint:
            raise ValueError("checkpoint was trained with a spec of different obs/action shape")
        self.policy.load_state_dict(ck["policy"])
        self.opt.load_state_dict(ck["optimizer"])
        self.vnorm.load_state_dict(ck["vnorm"])
        self.global_step = ck["global_step"]
        self.update_i = ck["update"]
        self.milestone_first = ck.get("milestone_first", {})
        self.swarm.load_state_dict(ck.get("swarm", {}))
        assign = self.swarm.resume_assignments()
        if assign:
            self.vec.load_states(assign)
        print(f"resumed from {p} at step {self.global_step:.3e}", flush=True)


def load_policy(path: str | Path, device: str = "cpu"):
    """Rebuild a policy from a checkpoint (for eval / play)."""
    from pokeio.config import PolicyConfig
    from pokeio.env import ObsSpec

    ck = torch.load(path, map_location=device, weights_only=False)
    os_ = ck["obs_spec"]
    obs_spec = ObsSpec(
        pixels=tuple(os_["pixels"]), pixel_width=os_["pixel_width"], pixel_bpp=os_["pixel_bpp"],
        levels=os_["levels"], bits=os_["bits"], scalars=os_["scalars"],
        cats=[tuple(c) for c in os_["cats"]], n_actions=os_["n_actions"],
    )
    pc = PolicyConfig(**ck["config"]["policy"])
    policy = Policy(obs_spec, pc).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck
