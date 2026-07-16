"""NEAT operators: mutation, crossover, speciation, and selection.

These run once per generation on Python :class:`~pokeio.evo.genome.Genome`
objects (the batched GPU work is the *forward* pass, in ``forward.py``).  All
randomness flows through a passed-in ``numpy`` ``Generator`` so runs are seedable
and reproducible (a Guiding Invariant).

Coefficients / rates come from :class:`~pokeio.config.EvoConfig`, but every
function also takes explicit overrides so the module is testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokeio.evo.forward import RELU, SIGMOID, SIN, TANH
from pokeio.evo.genome import (
    BIAS,
    HIDDEN,
    INPUT,
    OUTPUT,
    ConnGene,
    Genome,
    InnovationTracker,
    NodeGene,
)

# activations a mutation may assign to a hidden/output node.
_HIDDEN_ACTS = (TANH, RELU, SIGMOID, SIN)


@dataclass
class MutationRates:
    add_node: float = 0.03
    add_conn: float = 0.05
    weight: float = 0.8  # prob a genome has its weights perturbed
    weight_perturb_sigma: float = 0.5
    weight_reset_prob: float = 0.1  # within a perturbed genome, per-gene reset
    weight_reset_scale: float = 1.0
    toggle: float = 0.01
    mutate_act: float = 0.0
    feedforward: bool = True  # forbid cycles on add-connection


# --------------------------------------------------------------------------
# mutation
# --------------------------------------------------------------------------
def perturb_weights(g: Genome, rng: np.random.Generator, rates: MutationRates) -> None:
    for c in g.conns.values():
        if rng.random() < rates.weight_reset_prob:
            c.weight = float(rng.normal(0.0, rates.weight_reset_scale))
        else:
            c.weight += float(rng.normal(0.0, rates.weight_perturb_sigma))


def _reachable(g: Genome, src: int, dst: int) -> bool:
    """Is ``dst`` reachable from ``src`` over enabled edges by a path of length
    >= 1?  (So ``_reachable(g, n, n)`` is True iff ``n`` sits on a cycle.)"""
    adj: dict[int, list[int]] = {}
    for c in g.conns.values():
        if c.enabled:
            adj.setdefault(c.in_id, []).append(c.out_id)
    stack = list(adj.get(src, ()))  # start one hop out, so length >= 1
    seen = set(stack)
    while stack:
        u = stack.pop()
        if u == dst:
            return True
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def mutate_add_connection(
    g: Genome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    rates: MutationRates,
    weight_scale: float = 1.0,
    tries: int = 20,
) -> bool:
    """Add a new enabled connection between two currently-unlinked nodes."""
    # sources: anything that can emit (not, by convention, into an input/bias)
    src_pool = [nid for nid, ng in g.nodes.items() if ng.type != OUTPUT] + [
        nid for nid, ng in g.nodes.items() if ng.type == OUTPUT and not rates.feedforward
    ]
    dst_pool = [nid for nid, ng in g.nodes.items() if ng.type in (HIDDEN, OUTPUT)]
    if not src_pool or not dst_pool:
        return False

    existing = {(c.in_id, c.out_id) for c in g.conns.values()}
    for _ in range(tries):
        s = int(rng.choice(src_pool))
        d = int(rng.choice(dst_pool))
        if s == d or (s, d) in existing:
            continue
        if g.nodes[d].type in (INPUT, BIAS):
            continue
        if rates.feedforward and _reachable(g, d, s):
            continue  # would create a cycle
        innov = tracker.conn_innov(s, d)
        g.conns[innov] = ConnGene(
            s, d, float(rng.normal(0.0, weight_scale)), True, innov
        )
        return True
    return False


def mutate_add_node(
    g: Genome, tracker: InnovationTracker, rng: np.random.Generator
) -> bool:
    """Split an enabled connection: disable it, insert a hidden node, add
    ``in->new`` (weight 1) and ``new->out`` (old weight)."""
    enabled = [c for c in g.conns.values() if c.enabled]
    if not enabled:
        return False
    c = enabled[int(rng.integers(len(enabled)))]
    c.enabled = False

    new_id = tracker.split_node(c.innov)
    act = _HIDDEN_ACTS[int(rng.integers(len(_HIDDEN_ACTS)))]
    g.nodes[new_id] = NodeGene(id=new_id, type=HIDDEN, act=act, bias=0.0)

    i1 = tracker.conn_innov(c.in_id, new_id)
    g.conns[i1] = ConnGene(c.in_id, new_id, 1.0, True, i1)
    i2 = tracker.conn_innov(new_id, c.out_id)
    g.conns[i2] = ConnGene(new_id, c.out_id, c.weight, True, i2)
    return True


def mutate_toggle(g: Genome, rng: np.random.Generator) -> None:
    if not g.conns:
        return
    c = list(g.conns.values())[int(rng.integers(len(g.conns)))]
    c.enabled = not c.enabled


def mutate_genome(
    g: Genome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    rates: MutationRates,
    weight_scale: float = 1.0,
) -> Genome:
    """Apply the full mutation cocktail in place, returning ``g``."""
    if rng.random() < rates.weight:
        perturb_weights(g, rng, rates)
    if rng.random() < rates.add_conn:
        mutate_add_connection(g, tracker, rng, rates, weight_scale)
    if rng.random() < rates.add_node:
        mutate_add_node(g, tracker, rng)
    if rng.random() < rates.toggle:
        mutate_toggle(g, rng)
    if rates.mutate_act and rng.random() < rates.mutate_act:
        hids = [nid for nid, ng in g.nodes.items() if ng.type in (HIDDEN, OUTPUT)]
        if hids:
            nid = int(rng.choice(hids))
            g.nodes[nid].act = _HIDDEN_ACTS[int(rng.integers(len(_HIDDEN_ACTS)))]
    return g


# --------------------------------------------------------------------------
# crossover (align by innovation number)
# --------------------------------------------------------------------------
def crossover(
    p1: Genome,
    p2: Genome,
    rng: np.random.Generator,
    disabled_inherit_prob: float = 0.75,
) -> Genome:
    """Recombine two parents; disjoint/excess genes come from the fitter parent.

    Ties (equal fitness) fall back to the smaller genome for excess/disjoint."""
    if p2.fitness > p1.fitness:
        p1, p2 = p2, p1
    elif p1.fitness == p2.fitness and len(p2.conns) < len(p1.conns):
        p1, p2 = p2, p1
    # p1 is now the "fitter" (or smaller-on-tie) parent.

    child = Genome(n_in=p1.n_in, n_out=p1.n_out)
    for innov, c1 in p1.conns.items():
        c2 = p2.conns.get(innov)
        if c2 is not None:
            # matching gene -> pick a parent's weight at random
            src = c1 if rng.random() < 0.5 else c2
            gene = ConnGene(src.in_id, src.out_id, src.weight, True, innov)
            if (not c1.enabled or not c2.enabled) and rng.random() < disabled_inherit_prob:
                gene.enabled = False
            child.conns[innov] = gene
        else:
            # disjoint/excess -> inherit from fitter parent
            child.conns[innov] = ConnGene(c1.in_id, c1.out_id, c1.weight, c1.enabled, innov)

    # collect the nodes the child needs; take gene attrs from p1, else p2.
    needed: set[int] = set()
    for c in child.conns.values():
        needed.add(c.in_id)
        needed.add(c.out_id)
    for nid in list(p1.input_ids()) + [p1.bias_id] + list(p1.output_ids()):
        needed.add(nid)
    for nid in needed:
        ng = p1.nodes.get(nid) or p2.nodes.get(nid)
        if ng is not None:
            child.nodes[nid] = NodeGene(ng.id, ng.type, ng.act, ng.bias)
    return child


# --------------------------------------------------------------------------
# speciation (compatibility distance)
# --------------------------------------------------------------------------
def compatibility_distance(
    g1: Genome,
    g2: Genome,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
    normalize_min: int = 20,
) -> float:
    """δ = c1·E/N + c2·D/N + c3·W̄  (excess, disjoint, mean matching weight diff).

    Following Stanley's NEAT, ``N`` (the larger gene count) is only used as a
    normaliser for large genomes; for small genomes (< ``normalize_min`` genes)
    ``N`` is set to 1 so speciation stays discriminative early on."""
    k1 = set(g1.conns)
    k2 = set(g2.conns)
    if not k1 and not k2:
        return 0.0
    max1 = max(k1) if k1 else -1
    max2 = max(k2) if k2 else -1
    cutoff = min(max1, max2)

    matching = k1 & k2
    only = k1 ^ k2
    excess = sum(1 for k in only if k > cutoff)
    disjoint = len(only) - excess

    if matching:
        wbar = float(
            np.mean([abs(g1.conns[k].weight - g2.conns[k].weight) for k in matching])
        )
    else:
        wbar = 0.0

    n = max(len(k1), len(k2))
    n = 1 if n < normalize_min else n
    return c1 * excess / n + c2 * disjoint / n + c3 * wbar


class Speciation:
    """Persistent species assignment via compatibility distance to a per-species
    representative (classic NEAT).  Representatives carry across generations."""

    def __init__(
        self,
        threshold: float = 3.0,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
    ) -> None:
        self.threshold = threshold
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.reps: dict[int, Genome] = {}
        self._next = 0

    def assign(
        self, genomes: list[Genome], rng: np.random.Generator
    ) -> dict[int, list[Genome]]:
        species: dict[int, list[Genome]] = {}
        for g in genomes:
            placed = False
            for sid, rep in self.reps.items():
                d = compatibility_distance(g, rep, self.c1, self.c2, self.c3)
                if d < self.threshold:
                    species.setdefault(sid, []).append(g)
                    g.species_id = sid
                    placed = True
                    break
            if not placed:
                sid = self._next
                self._next += 1
                self.reps[sid] = g.copy()
                species.setdefault(sid, []).append(g)
                g.species_id = sid
        # refresh reps to a random current member; drop empty species.
        self.reps = {
            sid: members[int(rng.integers(len(members)))].copy()
            for sid, members in species.items()
        }
        return species


# --------------------------------------------------------------------------
# selection / reproduction
# --------------------------------------------------------------------------
def tournament_select(
    pool: list[Genome], k: int, rng: np.random.Generator
) -> Genome:
    """Pick the fittest of ``k`` random contestants (age as a tiebreaker: prefer
    younger — the aging / regularized-evolution pressure)."""
    k = min(k, len(pool))
    idx = rng.choice(len(pool), size=k, replace=False)
    best = pool[int(idx[0])]
    for j in idx[1:]:
        cand = pool[int(j)]
        if cand.fitness > best.fitness or (
            cand.fitness == best.fitness and cand.age < best.age
        ):
            best = cand
    return best


def reproduce(
    genomes: list[Genome],
    species: dict[int, list[Genome]],
    tracker: InnovationTracker,
    rng: np.random.Generator,
    rates: MutationRates,
    pop_size: int,
    tournament_size: int = 3,
    elitism: int = 1,
    crossover_rate: float = 0.75,
    fitness_sharing: bool = True,
    weight_scale: float = 1.0,
    survival_threshold: float = 0.4,
    max_stagnation: int | None = None,
    c3: float = 0.4,
) -> list[Genome]:
    """One generational step: fitness-share, allocate offspring per species,
    then tournament-select parents, crossover, and mutate.

    ``fitness`` fields must already be set on every genome (>= 0 recommended)."""
    # --- shift fitnesses to be non-negative for proportional allocation ---
    min_fit = min(g.fitness for g in genomes)
    shift = -min_fit if min_fit < 0 else 0.0

    # adjusted (shared) fitness = raw / species_size
    species_adj: dict[int, float] = {}
    for sid, members in species.items():
        size = len(members)
        adj = sum((g.fitness + shift) / (size if fitness_sharing else 1) for g in members)
        species_adj[sid] = adj

    total_adj = sum(species_adj.values())
    if total_adj <= 0:
        total_adj = 1.0
        for sid in species_adj:
            species_adj[sid] = len(species[sid])

    # --- allocate integer offspring counts summing to pop_size ------------
    raw_alloc = {sid: species_adj[sid] / total_adj * pop_size for sid in species}
    alloc = {sid: int(np.floor(v)) for sid, v in raw_alloc.items()}
    remainder = pop_size - sum(alloc.values())
    # hand out leftover slots to the species with the largest fractional parts
    frac_order = sorted(species, key=lambda s: raw_alloc[s] - alloc[s], reverse=True)
    for i in range(remainder):
        alloc[frac_order[i % len(frac_order)]] += 1

    new: list[Genome] = []
    for sid, members in species.items():
        n_off = alloc[sid]
        if n_off <= 0:
            continue
        ranked = sorted(members, key=lambda g: g.fitness, reverse=True)

        # elitism: carry the species champion over unchanged.
        n_keep = min(elitism, n_off)
        for e in range(n_keep):
            if e < len(ranked):
                champ = ranked[e].copy()
                champ.age += 1
                new.append(champ)
        n_off -= n_keep

        # breeding pool: the top survival_threshold fraction.
        cut = max(1, int(np.ceil(len(ranked) * survival_threshold)))
        pool = ranked[:cut]

        for _ in range(n_off):
            if len(pool) > 1 and rng.random() < crossover_rate:
                p1 = tournament_select(pool, tournament_size, rng)
                p2 = tournament_select(pool, tournament_size, rng)
                child = crossover(p1, p2, rng)
            else:
                child = tournament_select(pool, tournament_size, rng).copy()
            mutate_genome(child, tracker, rng, rates, weight_scale)
            child.fitness = 0.0
            child.age = 0
            new.append(child)

    # exact population size (rounding / empty-species guards)
    while len(new) < pop_size:
        base = tournament_select(genomes, tournament_size, rng).copy()
        mutate_genome(base, tracker, rng, rates, weight_scale)
        base.fitness = 0.0
        base.age = 0
        new.append(base)
    return new[:pop_size]


__all__ = [
    "MutationRates",
    "perturb_weights",
    "mutate_add_connection",
    "mutate_add_node",
    "mutate_toggle",
    "mutate_genome",
    "crossover",
    "compatibility_distance",
    "Speciation",
    "tournament_select",
    "reproduce",
]
