"""Champion export/import — brain (phenotype) + genome (genotype) as separate files.

A trained System-1 champion (for a session or a whole game) exports as TWO artifacts,
so the runnable and the heritable representations get purpose-fit formats:

  * the **BRAIN** (``.pt``) — the phenotype: weights + architecture + the obs/encoder
    spec + provenance. Load it to REPLAY / re-eval the champion standalone, or to
    resume/seed training. Self-contained.
  * the **GENOME** (``.npz``) — the genotype: the flat parameter vector + a
    reconstruction descriptor + lineage. The evolutionary outer loop (#41) and
    cross-game population seeding (the developmental ladder) mutate/cross THESE.

For a gradient net the genotype IS the weight vector, so "breeding" is weight-space
evolution (ERL/ES/PBT-style :func:`mutate` + :func:`crossover`), not NEAT topological
crossover. :func:`brain_from_genome` rebuilds the runnable phenotype from a genome, so
a bred child is immediately playable. The obs/encoder spec travels with the champion
so a bred champion stays percept-compatible (the transfer-stable-obs constraint the
ladder depends on — foveal ✓).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch

from pokeio.brain.actor_critic import ActorCritic

FORMAT = "pokeio.brain.champion/v1"


# --------------------------------------------------------------------------- meta
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def arch_of(policy: ActorCritic) -> dict:
    """The ActorCritic constructor args needed to rebuild it."""
    return {"obs_dim": int(policy.obs_dim), "grid": int(policy.G),
            "fovea_grid": int(policy.FG), "hidden": int(policy.trunk[0].out_features),
            "n_buttons": int(policy.n_buttons)}


def _build(arch: dict) -> ActorCritic:
    return ActorCritic(int(arch["obs_dim"]), periph_grid=int(arch["grid"]),
                       fovea_grid=int(arch.get("fovea_grid", arch["grid"])),
                       hidden=int(arch["hidden"]), n_buttons=int(arch["n_buttons"]))


# ------------------------------------------------------------------------- brain io
def export_brain(policy: ActorCritic, path: str, *, obs_spec: dict | None = None,
                 meta: dict | None = None) -> dict:
    """Write the runnable champion (phenotype). Returns the artifact dict."""
    art = {
        "format": FORMAT,
        "kind": "brain",
        "arch": arch_of(policy),
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "obs_spec": dict(obs_spec or {}),
        "meta": {"git_sha": _git_sha(), "created": _now(), **(meta or {})},
    }
    torch.save(art, str(path))
    return art


def load_brain(path: str, device="cpu") -> tuple[ActorCritic, dict]:
    """Rebuild the champion policy from a brain file. Returns (policy, artifact)."""
    art = torch.load(str(path), map_location=device, weights_only=False)
    policy = _build(art["arch"]).to(device)
    policy.load_state_dict(art["state_dict"])
    policy.eval()
    return policy, art


# ------------------------------------------------------------------------ genome io
def to_genome(policy: ActorCritic, *, meta: dict | None = None) -> dict:
    """The heritable genotype: flat weight vector + reconstruction descriptor."""
    sd = policy.state_dict()
    names = list(sd.keys())
    shapes = [tuple(sd[n].shape) for n in names]
    flat = np.concatenate([sd[n].detach().cpu().numpy().ravel() for n in names]).astype(np.float32)
    return {
        "format": FORMAT,
        "kind": "genome",
        "arch": arch_of(policy),
        "names": names,
        "shapes": shapes,
        "flat": flat,
        "meta": {"git_sha": _git_sha(), "created": _now(), **(meta or {})},
    }


def export_genome(policy: ActorCritic, path: str, *, meta: dict | None = None) -> dict:
    g = to_genome(policy, meta=meta)
    np.savez(str(path), flat=g["flat"],
             descriptor=np.frombuffer(json.dumps({
                 "format": g["format"], "kind": g["kind"], "arch": g["arch"],
                 "names": g["names"], "shapes": [list(s) for s in g["shapes"]],
                 "meta": g["meta"],
             }).encode(), dtype=np.uint8))
    return g


def load_genome(path: str) -> dict:
    z = np.load(str(path), allow_pickle=False)
    d = json.loads(bytes(z["descriptor"]).decode())
    d["flat"] = z["flat"].astype(np.float32)
    d["shapes"] = [tuple(s) for s in d["shapes"]]
    return d


def brain_from_genome(genome: dict, device="cpu") -> ActorCritic:
    """Rebuild a runnable policy (phenotype) from a genome (genotype)."""
    policy = _build(genome["arch"]).to(device)
    flat = np.asarray(genome["flat"], dtype=np.float32)
    sd, off = {}, 0
    for name, shape in zip(genome["names"], genome["shapes"]):
        n = int(np.prod(shape)) if shape else 1
        sd[name] = torch.from_numpy(flat[off:off + n].reshape(shape).copy())
        off += n
    if off != flat.size:
        raise ValueError(f"genome flat size {flat.size} != sum of shapes {off}")
    policy.load_state_dict(sd)
    policy.eval()
    return policy


# ------------------------------------------------------------------------- champion
def export_champion(policy: ActorCritic, brain_path: str, genome_path: str, *,
                    obs_spec: dict | None = None, meta: dict | None = None) -> None:
    """Export BOTH artifacts (phenotype + genotype) for one champion."""
    export_brain(policy, brain_path, obs_spec=obs_spec, meta=meta)
    export_genome(policy, genome_path, meta={**(meta or {}), "brain": str(brain_path)})


# ------------------------------------------------------------------------- breeding
def _segments(genome: dict):
    """Yield (name, shape, slice) over the flat vector."""
    off = 0
    for name, shape in zip(genome["names"], genome["shapes"]):
        n = int(np.prod(shape)) if shape else 1
        yield name, shape, slice(off, off + n)
        off += n


def _child_meta(parents: list[dict], op: str) -> dict:
    lineage = [p.get("meta", {}).get("id") or p.get("meta", {}).get("git_sha", "?")
               for p in parents]
    return {"git_sha": _git_sha(), "created": _now(), "op": op, "parents": lineage,
            "arch": parents[0]["arch"]}


def mutate(genome: dict, sigma: float = 0.02, rng=None) -> dict:
    """Weight-space Gaussian mutation (ES/ERL). Noise is scaled PER-TENSOR by that
    tensor's std, so every layer is perturbed proportionally (no hand-tuned per-layer
    knob). Returns a NEW genome (phenotype rebuildable via brain_from_genome)."""
    rng = rng or np.random.default_rng()
    flat = np.array(genome["flat"], dtype=np.float32, copy=True)
    for _name, _shape, sl in _segments(genome):
        seg = flat[sl]
        scale = float(sigma) * (float(seg.std()) + 1e-8)
        seg += rng.normal(0.0, scale, size=seg.shape).astype(np.float32)
    return {**genome, "flat": flat, "meta": _child_meta([genome], f"mutate(sigma={sigma})")}


def crossover(g1: dict, g2: dict, *, mode: str = "uniform", rng=None) -> dict:
    """Weight-space crossover of two same-architecture genomes -> a child genome.

    ``uniform`` = per-element 50/50 mask; ``mean`` = elementwise average; ``layer`` =
    per-tensor swap. Raises if the genomes are not shape-compatible.
    """
    if g1["names"] != g2["names"] or g1["shapes"] != g2["shapes"]:
        raise ValueError("crossover requires identical architecture (names/shapes)")
    rng = rng or np.random.default_rng()
    a = np.asarray(g1["flat"], np.float32)
    b = np.asarray(g2["flat"], np.float32)
    if mode == "mean":
        child = (a + b) * 0.5
    elif mode == "uniform":
        mask = rng.integers(0, 2, size=a.shape).astype(bool)
        child = np.where(mask, a, b).astype(np.float32)
    elif mode == "layer":
        child = a.copy()
        for _name, _shape, sl in _segments(g1):
            if rng.integers(0, 2):
                child[sl] = b[sl]
    else:
        raise ValueError(f"unknown crossover mode {mode!r}")
    return {**g1, "flat": child.astype(np.float32),
            "meta": _child_meta([g1, g2], f"crossover({mode})")}


__all__ = [
    "export_brain", "load_brain", "to_genome", "export_genome", "load_genome",
    "brain_from_genome", "export_champion", "mutate", "crossover", "arch_of", "FORMAT",
]
