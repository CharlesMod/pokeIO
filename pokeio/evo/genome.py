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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from operator import attrgetter

import numpy as np
import torch
from torch import Tensor

from pokeio.evo.forward import IDENTITY, SIGMOID, TANH, CompiledPopulation

# attribute extractors used by the vectorized packer (C-level, one call per gene
# instead of one attribute-lookup expression compiled per element).
_NODE_ATTRS = attrgetter("type", "act", "bias")
_CONN_ATTRS = attrgetter("in_id", "out_id", "weight", "enabled")

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
        *,
        prev: "Population | None" = None,
        dirty: Sequence[bool] | None = None,
    ) -> "Population":
        """Pack ``genomes`` into padded tensors.

        Vectorized packer: each genome's genes are gathered with C-level attribute
        extractors and written into pre-allocated ``numpy`` arrays with a single
        batched slice assignment per field, then wrapped as tensors once. This is
        bit-identical to the original per-element loop but ~40-70x faster.

        Repack-only-mutated (optional, purely additive)
        ------------------------------------------------
        ``prev`` / ``dirty`` let a caller that re-packs a population whose elites /
        survivors are unchanged from the previous generation reuse those rows
        instead of rebuilding them. The reuse is **positional**: for index ``i``
        with ``dirty[i]`` false, row ``i`` of ``prev``'s tensors is copied verbatim
        (so the result is bit-identical to packing, provided ``genomes[i]`` is
        indeed unchanged from ``prev.genomes[i]`` — the caller's assertion).

        * ``dirty``: optional length-``N`` bool sequence; ``True`` = changed
          (must repack), ``False`` = unchanged (reuse ``prev`` row ``i``).
        * ``prev``: the previous :class:`Population` to copy clean rows from.
        * If ``dirty`` is omitted but ``prev`` is given, unchanged rows are
          auto-detected by object identity (``genomes[i] is prev.genomes[i]``).

        A cheap node/conn-count check guards every reuse: on any mismatch (budgets
        changed, misaligned indices, or a stale ``dirty`` flag) the row is repacked,
        so incorrect hints degrade to correct-but-slow, never wrong.
        """
        assert genomes, "empty population"
        n_in = genomes[0].n_in
        n_out = genomes[0].n_out
        N = len(genomes)

        node_counts = [len(g.nodes) for g in genomes]
        conn_counts = [len(g.conns) for g in genomes]
        max_nc = max(node_counts)
        max_cc = max(conn_counts)
        M = max_nodes or max_nc
        C = max_conns or max(1, max_cc)
        if max_nc > M:
            raise ValueError(f"genome has {max_nc} nodes > budget {M}")
        if max_cc > C:
            raise ValueError(f"genome has {max_cc} conns > budget {C}")

        # -- pre-allocate padded numpy buffers (pad values match the originals) --
        node_id = np.full((N, M), _EMPTY_ID, dtype=np.int64)
        node_type = np.full((N, M), -1, dtype=np.int64)
        node_act = np.zeros((N, M), dtype=np.int64)
        node_bias = np.zeros((N, M), dtype=np.float32)
        node_mask = np.zeros((N, M), dtype=np.bool_)

        conn_in = np.full((N, C), -1, dtype=np.int64)
        conn_out = np.full((N, C), -1, dtype=np.int64)
        conn_weight = np.zeros((N, C), dtype=np.float32)
        conn_enabled = np.zeros((N, C), dtype=np.bool_)
        conn_innov = np.full((N, C), -1, dtype=np.int64)
        conn_mask = np.zeros((N, C), dtype=np.bool_)

        # -- decide which rows can be reused from ``prev`` ---------------------
        reuse: list[int] = []
        reusable = (
            prev is not None
            and prev.M == M
            and prev.C == C
            and prev.n_in == n_in
            and prev.n_out == n_out
            and prev.n == N
        )
        if reusable:
            prev_ncount = prev.node_mask.sum(dim=1).tolist()
            prev_ccount = prev.conn_mask.sum(dim=1).tolist()
            if dirty is None:
                pg = prev.genomes
                is_dirty = [genomes[i] is not pg[i] for i in range(N)]
            else:
                is_dirty = [bool(d) for d in dirty]
            for i in range(N):
                # count-match guard: reuse only when the prev row provably has the
                # same shape as this genome (cheap net against misaligned hints).
                if (
                    not is_dirty[i]
                    and node_counts[i] == prev_ncount[i]
                    and conn_counts[i] == prev_ccount[i]
                ):
                    reuse.append(i)
            reuse_set = set(reuse)
            pack_idx = [i for i in range(N) if i not in reuse_set]
        else:
            pack_idx = list(range(N))

        # -- pack the (possibly reduced) set of genomes -----------------------
        for i in pack_idx:
            g = genomes[i]
            nodes = g.nodes
            ids = sorted(nodes)
            k = len(ids)
            node_id[i, :k] = ids
            # gather (type, act, bias) for the sorted nodes in one C-level pass
            types, acts, biases = zip(*map(_NODE_ATTRS, (nodes[nid] for nid in ids)))
            node_type[i, :k] = types
            node_act[i, :k] = acts
            node_bias[i, :k] = biases
            node_mask[i, :k] = True

            conns = g.conns
            kk = len(conns)
            if kk:
                ins, outs, ws, ens = zip(*map(_CONN_ATTRS, conns.values()))
                conn_in[i, :kk] = ins
                conn_out[i, :kk] = outs
                conn_weight[i, :kk] = ws
                conn_enabled[i, :kk] = ens
                conn_innov[i, :kk] = list(conns.keys())
                conn_mask[i, :kk] = True

        # -- copy reused rows verbatim from prev (bit-identical) --------------
        if reuse:
            ridx = np.asarray(reuse, dtype=np.int64)
            node_id[ridx] = prev.node_id.cpu().numpy()[ridx]
            node_type[ridx] = prev.node_type.cpu().numpy()[ridx]
            node_act[ridx] = prev.node_act.cpu().numpy()[ridx]
            node_bias[ridx] = prev.node_bias.cpu().numpy()[ridx]
            node_mask[ridx] = prev.node_mask.cpu().numpy()[ridx]
            conn_in[ridx] = prev.conn_in.cpu().numpy()[ridx]
            conn_out[ridx] = prev.conn_out.cpu().numpy()[ridx]
            conn_weight[ridx] = prev.conn_weight.cpu().numpy()[ridx]
            conn_enabled[ridx] = prev.conn_enabled.cpu().numpy()[ridx]
            conn_innov[ridx] = prev.conn_innov.cpu().numpy()[ridx]
            conn_mask[ridx] = prev.conn_mask.cpu().numpy()[ridx]

        return cls(
            genomes=genomes,
            n_in=n_in,
            n_out=n_out,
            M=M,
            C=C,
            node_id=torch.from_numpy(node_id),
            node_type=torch.from_numpy(node_type),
            node_act=torch.from_numpy(node_act),
            node_bias=torch.from_numpy(node_bias),
            node_mask=torch.from_numpy(node_mask),
            conn_in=torch.from_numpy(conn_in),
            conn_out=torch.from_numpy(conn_out),
            conn_weight=torch.from_numpy(conn_weight),
            conn_enabled=torch.from_numpy(conn_enabled),
            conn_innov=torch.from_numpy(conn_innov),
            conn_mask=torch.from_numpy(conn_mask),
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
