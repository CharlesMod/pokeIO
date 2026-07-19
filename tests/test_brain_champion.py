"""Champion export/import + breeding tests (pokeio.brain.champion)."""

from __future__ import annotations

import numpy as np
import torch

from pokeio.brain.actor_critic import ActorCritic
from pokeio.brain.champion import (
    brain_from_genome,
    crossover,
    export_champion,
    export_genome,
    load_brain,
    load_genome,
    mutate,
    to_genome,
)

OBS = 454


def _policy(seed=0):
    torch.manual_seed(seed)
    return ActorCritic(OBS, grid=12).eval()


def _logits(policy, x):
    with torch.no_grad():
        return policy(x)[0].numpy()


# ------------------------------------------------------------ brain round-trip
def test_brain_export_load_reproduces_policy(tmp_path):
    from pokeio.brain.champion import export_brain
    p = _policy(1)
    x = torch.rand(5, OBS)
    l0 = _logits(p, x)
    bp = tmp_path / "champ_brain.pt"
    export_brain(p, str(bp), obs_spec={"dim": OBS}, meta={"game": "Pokemon Yellow"})
    p2, artifact = load_brain(str(bp))
    assert np.allclose(l0, _logits(p2, x), atol=1e-6)
    assert artifact["arch"]["obs_dim"] == OBS
    assert artifact["meta"]["game"] == "Pokemon Yellow"
    assert "git_sha" in artifact["meta"] and "created" in artifact["meta"]
    assert artifact["obs_spec"]["dim"] == OBS


# ----------------------------------------------------------- genome round-trip
def test_genome_roundtrip_reproduces_policy(tmp_path):
    p = _policy(2)
    x = torch.rand(4, OBS)
    l0 = _logits(p, x)
    gp = tmp_path / "champ_genome.npz"
    export_genome(p, str(gp), meta={"game": "Pokemon Yellow"})
    g = load_genome(str(gp))
    assert g["arch"]["obs_dim"] == OBS
    p2 = brain_from_genome(g)
    assert np.allclose(l0, _logits(p2, x), atol=1e-6)   # phenotype rebuilt exactly


def test_to_genome_flat_size_matches_params():
    p = _policy(3)
    g = to_genome(p)
    n_params = sum(int(np.prod(s)) for s in g["shapes"])
    assert g["flat"].size == n_params
    assert g["flat"].size == sum(v.numel() for v in p.state_dict().values())


# --------------------------------------------------------------- export both
def test_export_champion_writes_both_files(tmp_path):
    p = _policy(4)
    bp, gp = tmp_path / "b.pt", tmp_path / "g.npz"
    export_champion(p, str(bp), str(gp), obs_spec={"dim": OBS},
                    meta={"game": "Pokemon Yellow", "iter": 400})
    assert bp.exists() and gp.exists()
    p2, _ = load_brain(str(bp))
    g = load_genome(str(gp))
    x = torch.rand(3, OBS)
    assert np.allclose(_logits(p, x), _logits(p2, x), atol=1e-6)
    assert np.allclose(_logits(p, x), _logits(brain_from_genome(g), x), atol=1e-6)
    assert g["meta"]["brain"].endswith("b.pt")


# ------------------------------------------------------------------- breeding
def test_mutate_perturbs_but_stays_valid():
    p = _policy(5)
    g = to_genome(p)
    child = mutate(g, sigma=0.05, rng=np.random.default_rng(0))
    assert child["flat"].shape == g["flat"].shape
    assert not np.allclose(child["flat"], g["flat"])       # actually mutated
    assert child["arch"] == g["arch"]                       # architecture preserved
    assert child["meta"]["op"].startswith("mutate")
    # phenotype rebuildable + runs
    out = brain_from_genome(child)(torch.rand(2, OBS))[0]
    assert out.shape == (2, 9)


def test_mutate_scale_is_proportional_zero_sigma_is_identity():
    p = _policy(6)
    g = to_genome(p)
    same = mutate(g, sigma=0.0)
    assert np.allclose(same["flat"], g["flat"])            # sigma 0 -> no change


def test_crossover_modes_produce_valid_children():
    ga = to_genome(_policy(7))
    gb = to_genome(_policy(8))
    for mode in ("uniform", "mean", "layer"):
        child = crossover(ga, gb, mode=mode, rng=np.random.default_rng(1))
        assert child["flat"].shape == ga["flat"].shape
        assert child["meta"]["op"].startswith("crossover")
        brain_from_genome(child)(torch.rand(1, OBS))       # runs
    # mean is the elementwise average
    m = crossover(ga, gb, mode="mean")
    assert np.allclose(m["flat"], 0.5 * (ga["flat"] + gb["flat"]))


def test_crossover_rejects_mismatched_arch():
    ga = to_genome(ActorCritic(OBS, grid=12))
    gb = to_genome(ActorCritic(454, grid=12, hidden=128))  # different hidden -> shapes differ
    import pytest
    with pytest.raises(ValueError):
        crossover(ga, gb)
