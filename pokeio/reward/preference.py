r"""[MF] Frozen LLM preference-potential Φ (docs/specs/manifest-reward.md §3/§5).

A **Motif-style frozen preference potential** distilled from an offline LLM's
*pairwise* progress-preferences over short clips, then entering selection later as
Ng-1999 potential-based shaping. This module is the reward MODEL half of [MF]; it
imports nothing from :mod:`pokeio.train.loop` and needs no GPU or live LLM.

Three pieces, mirroring the retina idioms this repo already validated:

  * :class:`PrefModel` — a Bradley-Terry MLP ``in -> 128 -> 64 -> 1`` (SiLU +
    LayerNorm) whose scalar output is Φ(s). Trained by
    ``P(A≻B) = σ(Φ(A) − Φ(B))`` with confidence-weighted soft-label BCE and a
    held-out split that yields the accuracy gate.
  * :class:`PrefScorer` — a FROZEN inference wrapper (``snapshot()`` /
    ``score_batch`` mirroring :meth:`Retina.snapshot` / :meth:`Retina.encode_np`)
    plus a per-player **telescoping potential-return accumulator** (the
    anti-hacking credit, subsample-safe — NOT max-over-trajectory).
  * :func:`_pref_build_ckpt` / :func:`_pref_load_ckpt` — CPU-portable checkpoint
    round-trip mirroring :func:`_retina_build_ckpt`.

The INPUT WALL (spec §2/§9). Φ consumes ONLY the pixel-derived controller-latent
slice ``[z_periph(48) | z_fovea(32)]`` (80-d) plus the one-step latent delta
``(z_t − z_{t−1})`` and, optionally, generic proprio(14). The game-specific
``ram(8)`` tail at controller-latent ``[94:102]`` is **sliced off** and never
reaches :meth:`PrefModel.forward`. In foveal mode (no retina latent) the input is
the engine-agnostic screen-digest descriptor instead (built in
:mod:`pokeio.reward.pairs`). kNN-over-anchors is a warm-start SUPERVISION signal
only — anchors are never a model input, preserving the wall.

P100 / sm_60 note: fp32 throughout, tiny net (~few×10k params), CPU-portable. No
AMP. Distillation runs off the hot path on card0; the population queries only a
frozen snapshot on card1.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- controller-latent geometry (loop.py RetinaObsPipe layout) --------------
# The 102-d controller latent is [z_periph(48) | z_fovea(32) | proprio(14) | ram(8)]:
#   pixel slice  = [0:80]   -> the ONLY pixel content Φ may read
#   proprio      = [80:94]  -> optional generic efference copy (game-agnostic)
#   ram tail     = [94:102] -> game-specific; the INPUT WALL slices this off.
Z_PIX = 80          # z_periph(48) + z_fovea(32)
PROPRIO_LO = 80
PROPRIO_HI = 94     # proprio(14) spans [80:94]
RAM_LO = 94
RAM_HI = 102        # ram(8) spans [94:102] — NEVER an input to Φ
PROPRIO_DIM = PROPRIO_HI - PROPRIO_LO  # 14


def preferred_device() -> torch.device:
    """cuda:1 if present (card 0 hosts the GLM server / retina learner), else cpu.

    Mirrors :func:`pokeio.evo.retina.preferred_device` — Φ inference rides card1.
    """
    if torch.cuda.is_available():
        idx = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


# --------------------------------------------------------------------------- io
def retina_features(latent: np.ndarray, prev_latent: np.ndarray | None = None,
                    *, use_proprio: bool = True) -> np.ndarray:
    """Build Φ's model input from a raw controller latent — the INPUT WALL.

    ``latent`` is a single ``(D,)`` or batched ``(B,D)`` controller latent with
    ``D >= 80`` (the retina obs pipe emits ``D == 102``). The returned feature is
    ``[z_pix(80) | Δz_pix(80) ( | proprio(14) )]`` — the pixel slice ``[0:80]``,
    its one-step delta ``z_t − z_{t−1}`` (zeros when ``prev_latent`` is None), and
    optionally the proprio block ``[80:94]``. The game-specific ram tail
    ``[94:102]`` is **never referenced**, so no RAM value can enter
    :meth:`PrefModel.forward`.
    """
    lat = np.asarray(latent, dtype=np.float32)
    single = lat.ndim == 1
    if single:
        lat = lat[None]
    z = lat[:, :Z_PIX]                                     # pixel slice ONLY
    if prev_latent is None:
        dz = np.zeros_like(z)
    else:
        prev = np.asarray(prev_latent, dtype=np.float32)
        if prev.ndim == 1:
            prev = prev[None]
        dz = z - prev[:, :Z_PIX]
    parts = [z, dz]
    if use_proprio and lat.shape[1] >= PROPRIO_HI:
        parts.append(lat[:, PROPRIO_LO:PROPRIO_HI])       # proprio, NOT ram
    feat = np.concatenate(parts, axis=1).astype(np.float32)
    return feat[0] if single else feat


def retina_in_dim(use_proprio: bool = True) -> int:
    """Feature width :func:`retina_features` produces for the retina path."""
    return 2 * Z_PIX + (PROPRIO_DIM if use_proprio else 0)


# ------------------------------------------------ self-tuning weight (§5 gate)
def acc_reliability(val_acc: float) -> float:
    """Dimensionless held-out reliability ``c_acc = clamp((acc−0.5)/0.5, 0, 1)``.

    The model's own held-out pairwise accuracy, mapped so chance (0.5) -> 0 and a
    perfect model -> 1 (spec §5). No reward magnitude here — just reliability.
    """
    return float(np.clip((float(val_acc) - 0.5) / 0.5, 0.0, 1.0))


def pref_effective_weight(val_acc: float, w_pref_ref: float) -> float:
    """Self-tuning pref weight ``w_pref_eff = c_acc · w_pref_ref`` (spec §5).

    ``w_pref_ref`` inherits the loop's existing add-on regime (defaults to
    ``reward.w_resp``); a chance-level Φ contributes exactly 0, so no new operating
    magnitude is introduced and a gamed/unreliable model self-zeroes.
    """
    return acc_reliability(val_acc) * float(w_pref_ref)


def digest_features(digest: np.ndarray, prev_digest: np.ndarray | None = None) -> np.ndarray:
    """Foveal-path features from the engine-agnostic screen-digest descriptor.

    Same "progress is a transition" shape as :func:`retina_features`: the digest
    concatenated with its one-step delta. The descriptor itself (16×14 quantized
    screen ⊕ cell-depth) is built by :func:`pokeio.reward.pairs.screen_digest`, so
    the foveal path carries no retina latent yet stays game-agnostic.
    """
    d = np.asarray(digest, dtype=np.float32)
    single = d.ndim == 1
    if single:
        d = d[None]
    if prev_digest is None:
        dd = np.zeros_like(d)
    else:
        p = np.asarray(prev_digest, dtype=np.float32)
        if p.ndim == 1:
            p = p[None]
        dd = d - p
    feat = np.concatenate([d, dd], axis=1).astype(np.float32)
    return feat[0] if single else feat


# ------------------------------------------------------------------ the network
class PrefModel(nn.Module):
    """Bradley-Terry preference potential Φ: an MLP ``in -> 128 -> 64 -> 1``.

    SiLU + LayerNorm; the single scalar output is Φ(s). A clip's score is the mean
    Φ over its frames. Because BT constrains only *differences* of Φ, the absolute
    scale is unidentified — which is exactly why the downstream selection blend
    must use rank/quantile, never the raw value.

    The model is agnostic to how features were built; the INPUT WALL is enforced
    by :func:`retina_features` (retina path) or :func:`digest_features` (foveal),
    which never emit the ram tail. :meth:`score_latent` wires the retina builder
    in so ``forward`` provably cannot see ``latent[94:102]``.
    """

    def __init__(self, in_dim: int, *, hidden: tuple[int, int] = (128, 64),
                 mode: str = "retina", use_proprio: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = (int(hidden[0]), int(hidden[1]))
        self.mode = str(mode)               # "retina" | "foveal"
        self.use_proprio = bool(use_proprio)
        h1, h2 = self.hidden
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, h1), nn.LayerNorm(h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.SiLU(),
            nn.Linear(h2, 1),
        )
        # kNN-over-anchors warm-start table (supervision only; never a model input).
        self._anchor_feat: torch.Tensor | None = None
        self._anchor_phi: torch.Tensor | None = None

    # -- factories ----------------------------------------------------------
    @classmethod
    def for_retina(cls, *, use_proprio: bool = True, **kw) -> "PrefModel":
        return cls(retina_in_dim(use_proprio), mode="retina",
                   use_proprio=use_proprio, **kw)

    @classmethod
    def for_foveal(cls, digest_dim: int, **kw) -> "PrefModel":
        """Foveal path: features are ``[digest | Δdigest]`` (twice the descriptor)."""
        return cls(2 * int(digest_dim), mode="foveal", use_proprio=False, **kw)

    # -- device / tensor helpers -------------------------------------------
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _t(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return x.to(self._device(), dtype=torch.float32)

    # -- core forward -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_dim) -> (B,)`` scalar potential Φ. Accepts numpy too."""
        x = self._t(x)
        squeeze = x.dim() == 1
        if squeeze:
            x = x[None]
        phi = self.net(x).squeeze(-1)
        return phi[0] if squeeze else phi

    def clip_score(self, feats: torch.Tensor) -> torch.Tensor:
        """Mean Φ over a clip's per-frame features ``(T, in_dim) -> ()``."""
        return self.forward(feats).mean()

    # -- INPUT-WALL-enforcing latent entry points --------------------------
    def score_latent(self, latent, prev_latent=None) -> torch.Tensor:
        """Score a raw controller latent through :func:`retina_features`.

        The ram tail ``[94:102]`` is sliced off before the tensor ever reaches
        ``forward`` — the load-bearing input-wall guarantee (spec §2, test).
        """
        feat = retina_features(latent, prev_latent, use_proprio=self.use_proprio)
        return self.forward(feat)

    @torch.no_grad()
    def score_latent_np(self, latent, prev_latent=None) -> np.ndarray:
        self.eval()
        out = self.score_latent(latent, prev_latent)
        return out.detach().cpu().numpy().astype(np.float32)

    # -- kNN-over-anchors warm-start (graft; supervision ONLY) -------------
    def set_anchors(self, anchor_feats: np.ndarray, anchor_phi: np.ndarray) -> None:
        """Register (feature, DAG-order-fraction) anchors for the warm-start.

        Anchors bound pathological MLP extrapolation off-distribution when few
        labels exist; ``anchor_phi`` is each anchor's fraction along the mined
        milestone DAG. Stored as buffers, they are used only by
        :meth:`knn_phi` / :meth:`warm_start` — never concatenated into an input.
        """
        af = torch.as_tensor(np.asarray(anchor_feats, np.float32))
        ap = torch.as_tensor(np.asarray(anchor_phi, np.float32)).reshape(-1)
        if af.ndim != 2 or af.shape[0] != ap.shape[0]:
            raise ValueError("anchor_feats must be (N,in_dim) matching anchor_phi (N,)")
        self._anchor_feat = af
        self._anchor_phi = ap

    def knn_phi(self, feats: np.ndarray, k: int = 1) -> np.ndarray:
        """Φ target = mean order-fraction of the ``k`` nearest satisfied anchors."""
        if self._anchor_feat is None or self._anchor_phi is None:
            raise RuntimeError("no anchors set; call set_anchors first")
        x = torch.as_tensor(np.asarray(feats, np.float32))
        single = x.dim() == 1
        if single:
            x = x[None]
        d = torch.cdist(x, self._anchor_feat)             # (B, N)
        kk = min(int(k), self._anchor_feat.shape[0])
        idx = torch.topk(d, kk, dim=1, largest=False).indices
        phi = self._anchor_phi[idx].mean(dim=1).numpy().astype(np.float32)
        return phi[0] if single else phi

    def warm_start(self, *, steps: int = 100, lr: float = 1e-3,
                   k: int = 1, weight_decay: float = 1e-4) -> None:
        """Regress Φ toward the kNN-anchor order fraction (cold-start bootstrap).

        Bounds extrapolation before real labels exist; a no-op when no anchors are
        registered. Anchors supervise the OUTPUT only; the wall is untouched.
        """
        if self._anchor_feat is None or steps <= 0:
            return
        self.train()
        feats = self._anchor_feat.to(self._device())
        target = torch.as_tensor(
            self.knn_phi(self._anchor_feat.numpy(), k=k)).to(self._device())
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            pred = self.forward(feats)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

    # -- Bradley-Terry distillation ----------------------------------------
    def train_bt(self, labels, encode_fn, *, steps: int = 300, lr: float = 1e-3,
                 val_frac: float = 0.2, weight_decay: float = 1e-4,
                 batch_size: int = 64, rng: np.random.Generator | None = None,
                 device: torch.device | None = None) -> dict:
        r"""Distill Φ from pairwise labels; return held-out accuracy + stats.

        ``labels`` is a sequence of records ``{"a", "b", "winner", "confidence"}``
        where ``winner ∈ {"A","B","tie"}`` and ``confidence ∈ [0,1]``. ``a``/``b``
        are RAW clips (game-agnostic pixels/frames); ``encode_fn(clip)`` returns
        per-frame MODEL features ``(T, in_dim)`` under the CURRENT frozen retina
        snapshot (buffer-alignment, spec §3a). An optional ``"weight"`` (from the
        manifest weak-label cross-check) scales the pair's loss; default 1.

        Objective: ``P(A≻B) = σ(Φ(A) − Φ(B))`` with a confidence-weighted
        soft-label BCE — LLM confidence sets label smoothing (tie -> 0.5 target):

            soft_target = 0.5 + (hard − 0.5) · confidence

        so a high-confidence "A" targets ~1, a low-confidence one relaxes toward
        0.5, and a tie targets exactly 0.5 regardless. AdamW; a held-out split
        gives the pairwise accuracy that drives the §5 accuracy gate.
        """
        device = device or self._device()
        self.to(device)
        rng = rng if rng is not None else np.random.default_rng()

        # -- precompute per-clip feature tensors once (encode_fn is frozen) ----
        # Each row is (featA, featB, soft_target, weight, hard, conf) with the
        # per-frame features already on-device, so the training loop needs no
        # numpy->torch conversion and runs ONE batched forward per step.
        rows = []
        for lab in labels:
            fa = np.asarray(encode_fn(lab["a"]), dtype=np.float32)
            fb = np.asarray(encode_fn(lab["b"]), dtype=np.float32)
            if fa.ndim == 1:
                fa = fa[None]
            if fb.ndim == 1:
                fb = fb[None]
            ta = torch.as_tensor(fa, dtype=torch.float32, device=device)
            tb = torch.as_tensor(fb, dtype=torch.float32, device=device)
            winner = str(lab.get("winner", "tie")).upper()
            hard = 1.0 if winner == "A" else (0.0 if winner == "B" else 0.5)
            conf = float(np.clip(lab.get("confidence", 1.0), 0.0, 1.0))
            soft = 0.5 + (hard - 0.5) * conf
            weight = float(lab.get("weight", 1.0))
            rows.append((ta, tb, soft, weight, hard, conf))
        n = len(rows)
        if n == 0:
            return {"val_acc": 0.5, "n": 0, "n_val": 0, "loss": float("nan")}

        # -- deterministic held-out split (rng is the dedicated pref stream) --
        perm = rng.permutation(n)
        n_val = int(round(val_frac * n))
        n_val = min(max(n_val, 0), n - 1) if n > 1 else 0
        val_idx = set(perm[:n_val].tolist())
        train_rows = [rows[i] for i in range(n) if i not in val_idx]
        val_rows = [rows[i] for i in range(n) if i in val_idx]
        if not train_rows:                     # tiny corpora: train on all
            train_rows = rows

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        last_loss = float("nan")
        m = len(train_rows)
        for _ in range(max(0, int(steps))):
            self.train()
            sel = rng.integers(0, m, size=min(batch_size, m)).tolist()
            feats, seg_lens, targets, weights = [], [], [], []
            for j in sel:
                ta, tb, soft, weight, _hard, _conf = train_rows[int(j)]
                feats.append(ta); seg_lens.append(int(ta.shape[0]))     # A then B
                feats.append(tb); seg_lens.append(int(tb.shape[0]))
                targets.append(soft)
                weights.append(weight)
            phi_all = self.net(torch.cat(feats, dim=0)).squeeze(-1)     # ONE forward
            scores = torch.stack([s.mean() for s in torch.split(phi_all, seg_lens)])
            logit_t = scores[0::2] - scores[1::2]                       # BT logit P(A≻B)
            target_t = torch.as_tensor(targets, dtype=torch.float32, device=device)
            weight_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
            loss = (bce(logit_t, target_t) * weight_t).sum() / weight_t.sum().clamp_min(1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())

        val_acc = self._pairwise_accuracy(val_rows if val_rows else train_rows)
        return {
            "val_acc": float(val_acc),
            "n": n,
            "n_val": len(val_rows),
            "loss": last_loss,
        }

    @torch.no_grad()
    def _pairwise_accuracy(self, rows) -> float:
        """Fraction of NON-tie pairs whose Φ ordering matches the LLM winner.

        Ties are excluded from the denominator (they carry no order to score). A
        chance/gamed label stream drives this to ~0.5 -> the §5 gate self-zeroes.
        """
        self.eval()
        correct = 0
        total = 0
        for fa, fb, _soft, _weight, hard, _conf in rows:
            if hard == 0.5:                    # tie: no order to check
                continue
            diff = float(self.clip_score(fa) - self.clip_score(fb))
            pred_a = diff > 0.0
            want_a = hard == 1.0
            correct += int(pred_a == want_a)
            total += 1
        return correct / total if total else 0.5

    # -- freeze / snapshot (mirror Retina.snapshot) -------------------------
    def snapshot(self, device: torch.device | None = None) -> "PrefModel":
        """Frozen (eval, no-grad) deep copy for population inference on ``infer_card``."""
        self.eval()
        snap = copy.deepcopy(self)
        if device is not None:
            snap.to(device)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        return snap

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------ telescoping potential return
class TelescopeAccumulator:
    """Per-player TRUE telescoping potential-shaping return (spec §4).

    Accumulates ``Σ_t ( γ·Φ(s_{t+dt}) − Φ(s_t) )`` over the (subsampled) Φ values
    a player visits — the decisive anti-hacking formulation, NOT max-over-
    trajectory. Each :meth:`update` contributes exactly one telescoping term
    ``γ·φ_new − φ_prev`` regardless of the spacing ``dt`` between samples, so it is
    subsample-safe. With ``γ = 1`` the sum telescopes to ``φ_last − φ_first`` and a
    returning cycle (``φ_last == φ_first``) nets exactly 0 — the canonical
    "farm the boundary" exploit is impossible by construction (Ng-1999).
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = float(gamma)
        self._prev: dict = {}
        self._acc: dict = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._prev.clear()
            self._acc.clear()
        else:
            self._prev.pop(player, None)
            self._acc.pop(player, None)

    def update(self, player, phi: float) -> None:
        """Feed one (subsampled) potential for ``player``; add the telescoping term."""
        phi = float(phi)
        if player in self._prev:
            self._acc[player] = self._acc.get(player, 0.0) + self.gamma * phi - self._prev[player]
        else:
            self._acc.setdefault(player, 0.0)
        self._prev[player] = phi

    def update_batch(self, players, phis) -> None:
        for p, ph in zip(players, phis):
            self.update(p, float(ph))

    def value(self, player) -> float:
        return float(self._acc.get(player, 0.0))

    def values(self) -> dict:
        return dict(self._acc)


# --------------------------------------------------------- frozen scorer
@dataclass
class PrefScorer:
    """Frozen Φ inference + per-player telescoping accumulator (spec §3b).

    Mirrors the retina freeze discipline: holds a FROZEN :class:`PrefModel`
    snapshot and exposes ``score_batch`` (the analog of ``Retina.encode_np``) plus
    a :class:`TelescopeAccumulator`. Built via :meth:`from_model` so the live model
    keeps training while the population is fed a stationary snapshot swapped only
    at generation boundaries.
    """

    model: PrefModel
    gamma: float = 0.99
    accum: TelescopeAccumulator | None = None

    def __post_init__(self) -> None:
        if self.accum is None:
            self.accum = TelescopeAccumulator(self.gamma)

    @classmethod
    def from_model(cls, model: PrefModel, *, gamma: float = 0.99,
                   device: torch.device | None = None) -> "PrefScorer":
        return cls(model=model.snapshot(device=device), gamma=gamma)

    def set_snapshot(self, model: PrefModel, *, device: torch.device | None = None) -> None:
        """Swap the frozen snapshot at a freeze-and-swap boundary (accumulator kept)."""
        self.model = model.snapshot(device=device)

    @torch.no_grad()
    def score_batch(self, feats: np.ndarray) -> np.ndarray:
        """Φ over a batch of MODEL features ``(B, in_dim) -> (B,)`` numpy."""
        self.model.eval()
        out = self.model.forward(np.asarray(feats, dtype=np.float32))
        arr = out.detach().cpu().numpy().astype(np.float32)
        return np.atleast_1d(arr)

    @torch.no_grad()
    def score_latent_batch(self, latent: np.ndarray,
                           prev_latent: np.ndarray | None = None) -> np.ndarray:
        """Φ over a batch of RAW controller latents (input wall applied)."""
        self.model.eval()
        out = self.model.score_latent(latent, prev_latent)
        arr = out.detach().cpu().numpy().astype(np.float32)
        return np.atleast_1d(arr)

    # -- telescoping credit passthrough ------------------------------------
    def observe(self, player, phi: float) -> None:
        self.accum.update(player, phi)

    def observe_batch(self, players, phis) -> None:
        self.accum.update_batch(players, phis)

    def pref_return(self, player) -> float:
        return self.accum.value(player)

    def reset(self, player: int | None = None) -> None:
        self.accum.reset(player)


# --------------------------------------------------------- checkpoint round-trip
def _pref_ckpt_to_cpu(obj):
    """Recursively move tensors in a state_dict / optimizer-state tree to CPU so
    the pickle checkpoint is device-portable (mirror :func:`_retina_ckpt_to_cpu`)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _pref_ckpt_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_pref_ckpt_to_cpu(v) for v in obj)
    return obj


def _pref_build_ckpt(model: PrefModel, opt, scorer: PrefScorer | None) -> dict:
    """Serializable pref-learner state for the pickle checkpoint.

    Persists the learner + optimizer + the CURRENTLY-frozen inference snapshot
    (``scorer.model`` — the exact Φ the live population is fed), all on CPU, plus
    the shape needed to rebuild the module. Mirrors :func:`_retina_build_ckpt` so
    resume continues the SAME Φ instead of re-warming a fresh one."""
    snap_sd = None
    if scorer is not None and getattr(scorer, "model", None) is not None:
        snap_sd = _pref_ckpt_to_cpu(scorer.model.state_dict())
    return {
        "learner": _pref_ckpt_to_cpu(model.state_dict()),
        "optimizer": _pref_ckpt_to_cpu(opt.state_dict()) if opt is not None else None,
        "snapshot": snap_sd,
        "config": {
            "in_dim": int(model.in_dim),
            "hidden": list(model.hidden),
            "mode": str(model.mode),
            "use_proprio": bool(model.use_proprio),
            "gamma": float(scorer.gamma) if scorer is not None else 0.99,
        },
    }


def _pref_load_ckpt(ckpt: dict, *, device: torch.device | None = None,
                    lr: float = 1e-3, weight_decay: float = 1e-4):
    """Rebuild ``(model, optimizer, scorer)`` from a :func:`_pref_build_ckpt` dict.

    Missing / None ckpt -> ``(None, None, None)`` so a caller can safely re-warm.
    The frozen snapshot round-trips into a fresh :class:`PrefScorer` so resume is
    continuation, not a re-warm."""
    if not ckpt:
        return None, None, None
    device = device or torch.device("cpu")
    cfg = ckpt.get("config", {})
    model = PrefModel(
        int(cfg.get("in_dim", retina_in_dim(True))),
        hidden=tuple(cfg.get("hidden", (128, 64))),
        mode=str(cfg.get("mode", "retina")),
        use_proprio=bool(cfg.get("use_proprio", True)),
    ).to(device)
    if ckpt.get("learner") is not None:
        model.load_state_dict(ckpt["learner"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if ckpt.get("optimizer") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            pass  # optimizer state is best-effort; a fresh opt still resumes Φ
    gamma = float(cfg.get("gamma", 0.99))
    scorer = None
    if ckpt.get("snapshot") is not None:
        snap = PrefModel(
            int(cfg.get("in_dim", retina_in_dim(True))),
            hidden=tuple(cfg.get("hidden", (128, 64))),
            mode=str(cfg.get("mode", "retina")),
            use_proprio=bool(cfg.get("use_proprio", True)),
        )
        snap.load_state_dict(ckpt["snapshot"])
        scorer = PrefScorer.from_model(snap, gamma=gamma, device=device)
    return model, opt, scorer


__all__ = [
    "Z_PIX",
    "PROPRIO_LO",
    "PROPRIO_HI",
    "RAM_LO",
    "RAM_HI",
    "PROPRIO_DIM",
    "preferred_device",
    "retina_features",
    "retina_in_dim",
    "acc_reliability",
    "pref_effective_weight",
    "digest_features",
    "PrefModel",
    "TelescopeAccumulator",
    "PrefScorer",
    "_pref_build_ckpt",
    "_pref_load_ckpt",
]
