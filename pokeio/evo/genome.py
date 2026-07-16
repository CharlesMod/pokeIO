"""NEAT genome representation: per-genome gene lists + a batched, padded-tensor
``Population`` that compiles to the GPU forward pass.

Two representations, on purpose:

* :class:`Genome` — a light Python object (dict of nodes, dict of connections
  keyed by innovation number).  This is what mutation / crossover / speciation
  operate on; it happens **once per generation** so Python-side bookkeeping is
  cheap and keeps the innovation logic obvious.
* :class:`Population` — the whole population packed into **padded tensors** with
  per-genome masks (topologies differ, so we pad to a max-node / max-conn
  budget).  :meth:`Population.compile` turns this into a
  :class:`~pokeio.evo.forward.CompiledPopulation` for the batched GPU forward.

Node-id convention (global, stable across genomes for historical marking)::

    inputs  : ids 0 .. n_in-1
    bias    : id  n_in
    outputs : ids n_in+1 .. n_in+n_out
    hidden  : ids >= n_in+n_out+1   (assigned by the InnovationTracker)

Because I/O ids are the smallest and every genome contains them, storing each
genome's nodes **sorted by id** puts the I/O nodes at fixed front slots in the
adjacency matrix — no per-genome I/O slot bookkeeping needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from pokeio.evo.forward import IDENTITY, SIGMOID, TANH, CompiledPopulation

# node types
INPUT: int = 0
BIAS: int = 1
OUTPUT: int = 2
HIDDEN: int = 3

# sentinel for an empty node slot in the padded id tensor (must sort *last*, so
# it has to be larger than any real node id).
_EMPTY_ID: int = (1 << 60)


# --------------------------------------------------------------------------
# per-genome gene objects
# --------------------------------------------------------------------------
@dataclass
class NodeGene:
    id: int
    type: int  # INPUT / BIAS / OUTPUT / HIDDEN
    act: int  # activation index (see forward.ACT_NAMES)
    bias: float = 0.0


@dataclass
class ConnGene:
    in_id: int
    out_id: int
    weight: float
    enabled: bool
    innov: int


@dataclass
class Genome:
    n_in: int
    n_out: int
    nodes: dict[int, NodeGene] = field(default_factory=dict)
    conns: dict[int, ConnGene] = field(default_factory=dict)  # keyed by innov
    fitness: float = 0.0
    age: int = 0
    species_id: int = -1

    # -- id helpers --------------------------------------------------------
    @property
    def bias_id(self) -> int:
        return self.n_in

    def input_ids(self) -> range:
        return range(0, self.n_in)

    def output_ids(self) -> range:
        return range(self.n_in + 1, self.n_in + 1 + self.n_out)

    def hidden_ids(self) -> list[int]:
        return [nid for nid, ng in self.nodes.items() if ng.type == HIDDEN]

    def copy(self) -> "Genome":
        return Genome(
            n_in=self.n_in,
            n_out=self.n_out,
            nodes={k: replace(v) for k, v in self.nodes.items()},
            conns={k: replace(v) for k, v in self.conns.items()},
            fitness=self.fitness,
            age=self.age,
            species_id=self.species_id,
        )


# --------------------------------------------------------------------------
# innovation bookkeeping (historical marking)
# --------------------------------------------------------------------------
class InnovationTracker:
    """Global counters so the same structural mutation gets the same id/innov,
    which is what makes crossover alignment and speciation meaningful."""

    def __init__(self, n_in: int, n_out: int) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.n_io = n_in + 1 + n_out
        self._next_node = self.n_io  # first free hidden id
        self._next_innov = 0
        self._conn_innov: dict[tuple[int, int], int] = {}
        self._node_split: dict[int, int] = {}  # split-conn innov -> new hidden id

    def conn_innov(self, in_id: int, out_id: int) -> int:
        key = (in_id, out_id)
        innov = self._conn_innov.get(key)
        if innov is None:
            innov = self._next_innov
            self._conn_innov[key] = innov
            self._next_innov += 1
        return innov

    def split_node(self, split_innov: int) -> int:
        """New hidden-node id for splitting the connection ``split_innov``.

        Keyed by the split connection so identical add-node mutations across the
        population share a node id."""
        nid = self._node_split.get(split_innov)
        if nid is None:
            nid = self._next_node
            self._node_split[split_innov] = nid
            self._next_node += 1
        return nid


# --------------------------------------------------------------------------
# genome factory
# --------------------------------------------------------------------------
def make_genome(
    n_in: int,
    n_out: int,
    tracker: InnovationTracker,
    rng,
    connect: str = "full",
    weight_scale: float = 1.0,
    hidden_act: int = TANH,
    output_act: int = SIGMOID,
) -> Genome:
    """Create a minimal genome (inputs + bias + outputs).

    ``connect='full'`` wires every input and the bias to every output with
    random weights; ``connect='none'`` leaves it unconnected.
    """
    g = Genome(n_in=n_in, n_out=n_out)
    for i in g.input_ids():
        g.nodes[i] = NodeGene(id=i, type=INPUT, act=IDENTITY, bias=0.0)
    g.nodes[g.bias_id] = NodeGene(id=g.bias_id, type=BIAS, act=IDENTITY, bias=0.0)
    for o in g.output_ids():
        g.nodes[o] = NodeGene(id=o, type=OUTPUT, act=output_act, bias=0.0)

    if connect == "full":
        src_ids = list(g.input_ids()) + [g.bias_id]
        for s in src_ids:
            for o in g.output_ids():
                innov = tracker.conn_innov(s, o)
                w = float(rng.normal(0.0, weight_scale))
                g.conns[innov] = ConnGene(s, o, w, True, innov)
    return g


# --------------------------------------------------------------------------
# batched, padded-tensor population
# --------------------------------------------------------------------------
@dataclass
class Population:
    """N genomes packed into padded tensors with per-genome masks."""

    genomes: list[Genome]
    n_in: int
    n_out: int
    M: int  # max-node budget (padding bound)
    C: int  # max-conn budget

    node_id: Tensor  # (N, M) long   (sorted asc; _EMPTY_ID pads)
    node_type: Tensor  # (N, M) long   (-1 pads)
    node_act: Tensor  # (N, M) long
    node_bias: Tensor  # (N, M) float
    node_mask: Tensor  # (N, M) bool

    conn_in: Tensor  # (N, C) long   (global src id; -1 pads)
    conn_out: Tensor  # (N, C) long   (global dst id; -1 pads)
    conn_weight: Tensor  # (N, C) float
    conn_enabled: Tensor  # (N, C) bool
    conn_innov: Tensor  # (N, C) long   (-1 pads)
    conn_mask: Tensor  # (N, C) bool

    @property
    def n(self) -> int:
        return len(self.genomes)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_genomes(
        cls,
        genomes: list[Genome],
        max_nodes: int | None = None,
        max_conns: int | None = None,
    ) -> "Population":
        assert genomes, "empty population"
        n_in = genomes[0].n_in
        n_out = genomes[0].n_out
        N = len(genomes)

        node_counts = [len(g.nodes) for g in genomes]
        conn_counts = [len(g.conns) for g in genomes]
        M = max_nodes or max(node_counts)
        C = max_conns or max(1, max(conn_counts))
        if max(node_counts) > M:
            raise ValueError(f"genome has {max(node_counts)} nodes > budget {M}")
        if max(conn_counts) > C:
            raise ValueError(f"genome has {max(conn_counts)} conns > budget {C}")

        node_id = torch.full((N, M), _EMPTY_ID, dtype=torch.long)
        node_type = torch.full((N, M), -1, dtype=torch.long)
        node_act = torch.zeros((N, M), dtype=torch.long)
        node_bias = torch.zeros((N, M), dtype=torch.float32)
        node_mask = torch.zeros((N, M), dtype=torch.bool)

        conn_in = torch.full((N, C), -1, dtype=torch.long)
        conn_out = torch.full((N, C), -1, dtype=torch.long)
        conn_weight = torch.zeros((N, C), dtype=torch.float32)
        conn_enabled = torch.zeros((N, C), dtype=torch.bool)
        conn_innov = torch.full((N, C), -1, dtype=torch.long)
        conn_mask = torch.zeros((N, C), dtype=torch.bool)

        for i, g in enumerate(genomes):
            # nodes sorted by id -> I/O land at fixed front slots
            for s, nid in enumerate(sorted(g.nodes)):
                ng = g.nodes[nid]
                node_id[i, s] = nid
                node_type[i, s] = ng.type
                node_act[i, s] = ng.act
                node_bias[i, s] = ng.bias
                node_mask[i, s] = True
            for s, (innov, c) in enumerate(g.conns.items()):
                conn_in[i, s] = c.in_id
                conn_out[i, s] = c.out_id
                conn_weight[i, s] = c.weight
                conn_enabled[i, s] = c.enabled
                conn_innov[i, s] = innov
                conn_mask[i, s] = True

        return cls(
            genomes=genomes,
            n_in=n_in,
            n_out=n_out,
            M=M,
            C=C,
            node_id=node_id,
            node_type=node_type,
            node_act=node_act,
            node_bias=node_bias,
            node_mask=node_mask,
            conn_in=conn_in,
            conn_out=conn_out,
            conn_weight=conn_weight,
            conn_enabled=conn_enabled,
            conn_innov=conn_innov,
            conn_mask=conn_mask,
        )

    # -- compile to the GPU forward view -----------------------------------
    def compile(self, device: str | torch.device = "cpu") -> CompiledPopulation:
        """Map global node ids -> adjacency slots and move to ``device``.

        ``node_id`` rows are sorted ascending, so ``searchsorted`` gives each
        connection's source/dest slot in one batched call."""
        dev = torch.device(device)
        node_id = self.node_id.to(dev)

        cin = self.conn_in.to(dev)
        cout = self.conn_out.to(dev)
        # clamp negatives (padding) so searchsorted is well-defined; masked out.
        in_slot = torch.searchsorted(node_id, cin.clamp(min=0)).clamp(max=self.M - 1)
        out_slot = torch.searchsorted(node_id, cout.clamp(min=0)).clamp(max=self.M - 1)

        valid = self.conn_mask.to(dev) & self.conn_enabled.to(dev)
        return CompiledPopulation(
            node_act=self.node_act.to(dev),
            node_bias=self.node_bias.to(dev),
            conn_in_slot=in_slot,
            conn_out_slot=out_slot,
            conn_weight=self.conn_weight.to(dev),
            conn_valid=valid,
            n_in=self.n_in,
            n_out=self.n_out,
            M=self.M,
        )


__all__ = [
    "INPUT",
    "BIAS",
    "OUTPUT",
    "HIDDEN",
    "NodeGene",
    "ConnGene",
    "Genome",
    "InnovationTracker",
    "make_genome",
    "Population",
]
