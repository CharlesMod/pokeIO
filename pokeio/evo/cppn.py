"""CPPN substrate-painter for ES-HyperNEAT (Lane B vision encoder).

A CPPN is an ordinary NEAT genome (:class:`pokeio.evo.genome.Genome`) with
``n_in = 4`` (the connection coordinates ``x1, y1, x2, y2``) and ``n_out = 1``
(the painted connection weight).  The standard NEAT bias node supplies the fifth
canonical input, so the CPPN realises ``CPPN(x1, y1, x2, y2, bias)`` exactly as
the project principle requires — no egocentric priors are ever fed in.

Two forward passes, do not confuse them
---------------------------------------
1. **CPPN forward** (this module, :func:`query_cppn`) — a *tiny* network run over
   a large batch of coordinate pairs to paint weights.  It needs a ``gauss``
   activation (for receptive-field bumps) that the evo core deliberately omits,
   so this module carries its OWN activation set (:func:`cppn_apply_activation`)
   and its own dense propagate.  Nothing in ``forward.py`` is touched.
2. **Phenotype forward** (the evo core) — the wide, sparse substrate network the
   CPPN painted, run with :func:`pokeio.evo.forward.population_forward_sparse`.
   Its node activations stay inside the core's set (identity / tanh).

Pipeline
--------
``CPPN genome  --query-->  weights over every substrate (src,tgt) pair
              --threshold-->  sparse edge list  -->  CompiledPopulation
              -->  population_forward_sparse  (card 1)``
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from pokeio.evo.forward import (
    IDENTITY,
    RELU,
    SIGMOID,
    SIN,
    TANH,
    CompiledPopulation,
    build_weight_matrix,
)
from pokeio.evo.genome import (
    BIAS,
    HIDDEN,
    INPUT,
    OUTPUT,
    ConnGene,
    Genome,
    InnovationTracker,
    NodeGene,
    Population,
)
from pokeio.evo.substrate import Substrate

# -- CPPN activation registry ----------------------------------------------
# Indices 0..4 are shared with forward.py; GAUSS/ABS/COS extend it *here only*
# (a CPPN needs a gaussian-like function for symmetric receptive fields).
GAUSS: int = 5
ABS: int = 6
COS: int = 7

CPPN_ACT_NAMES: dict[str, int] = {
    "identity": IDENTITY,
    "tanh": TANH,
    "relu": RELU,
    "sigmoid": SIGMOID,
    "sin": SIN,
    "gauss": GAUSS,
    "abs": ABS,
    "cos": COS,
}
CPPN_ACT_INDEX_TO_NAME: dict[int, str] = {v: k for k, v in CPPN_ACT_NAMES.items()}

# The natural activation palette to mutate a CPPN over (bias/identity for I/O).
CPPN_HIDDEN_ACTS: tuple[int, ...] = (SIN, GAUSS, TANH, SIGMOID, ABS, IDENTITY)

CPPN_N_IN: int = 4   # x1, y1, x2, y2  (bias node supplies the 5th input)
CPPN_N_OUT: int = 1  # the painted weight


def cppn_apply_activation(z: Tensor, act: Tensor) -> Tensor:
    """Per-node activation for the CPPN forward (extends forward's set w/ gauss).

    ``z``   : ``(N, B, M)`` pre-activations.
    ``act`` : ``(N, M)`` long activation indices.
    """
    a = act.unsqueeze(1)  # (N, 1, M)
    out = z  # IDENTITY default
    out = torch.where(a == TANH, torch.tanh(z), out)
    out = torch.where(a == RELU, torch.relu(z), out)
    out = torch.where(a == SIGMOID, torch.sigmoid(z), out)
    out = torch.where(a == SIN, torch.sin(z), out)
    out = torch.where(a == GAUSS, torch.exp(-(z * z)), out)
    out = torch.where(a == ABS, torch.abs(z), out)
    out = torch.where(a == COS, torch.cos(z), out)
    return out


def _cppn_propagate(
    W: Tensor,
    node_act: Tensor,
    node_bias: Tensor,
    n_in: int,
    n_out: int,
    X: Tensor,
    steps: int,
) -> Tensor:
    """Iterated dense propagation for the CPPN (gauss-aware).

    Mirrors :func:`pokeio.evo.forward.propagate` but uses the CPPN activation
    set.  ``W`` is ``(N, M, M)``, ``X`` is ``(N, B, n_in)``; returns
    ``(N, B, n_out)``.  ``steps`` must exceed the CPPN's feed-forward depth.
    """
    N, M, _ = W.shape
    B = X.shape[1]
    dev, dtype = W.device, W.dtype

    x = torch.zeros(N, B, M, device=dev, dtype=dtype)
    Xc = X.to(dev, dtype)
    x[:, :, 0:n_in] = Xc
    x[:, :, n_in] = 1.0  # bias node
    nbias = node_bias.unsqueeze(1)

    for _ in range(steps):
        z = torch.bmm(W, x.transpose(1, 2)).transpose(1, 2) + nbias
        x = cppn_apply_activation(z, node_act)
        x[:, :, 0:n_in] = Xc
        x[:, :, n_in] = 1.0

    out_start = n_in + 1
    return x[:, :, out_start : out_start + n_out]


def _as_population(cppn: Genome | list[Genome] | Population) -> Population:
    if isinstance(cppn, Population):
        return cppn
    genomes = [cppn] if isinstance(cppn, Genome) else list(cppn)
    return Population.from_genomes(genomes)


def query_cppn(
    cppn: Genome | list[Genome] | Population,
    coord4: Tensor | np.ndarray,
    device: str | torch.device = "cuda:1",
    steps: int = 8,
) -> Tensor:
    """Query one or many CPPNs over a batch of coordinate 4-tuples.

    ``coord4`` : ``(B, 4)`` array of ``(x1, y1, x2, y2)`` (shared across CPPNs).
    Returns    : ``(N, B)`` painted weights (``N`` = number of CPPN genomes).
    """
    dev = torch.device(device)
    pop = _as_population(cppn)
    cp = pop.compile(dev)

    if not torch.is_tensor(coord4):
        coord4 = torch.as_tensor(np.asarray(coord4), dtype=torch.float32)
    coord4 = coord4.to(dev, torch.float32)
    if coord4.dim() != 2 or coord4.shape[1] != CPPN_N_IN:
        raise ValueError(f"coord4 must be (B, {CPPN_N_IN}); got {tuple(coord4.shape)}")

    N = pop.n
    B = coord4.shape[0]
    X = coord4.unsqueeze(0).expand(N, B, CPPN_N_IN)  # (N, B, 4)
    W = build_weight_matrix(cp, 0, N)                # (N, M, M) — CPPN is tiny
    out = _cppn_propagate(
        W, cp.node_act, cp.node_bias, cp.n_in, cp.n_out, X, steps
    )  # (N, B, 1)
    return out[:, :, 0]


# ---------------------------------------------------------------------------
# Painting the substrate
# ---------------------------------------------------------------------------
def _enumerate_substrate(substrate: Substrate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate every transition's ``(in_slot, out_slot, coord4)``."""
    in_slots, out_slots, coords = [], [], []
    for t in substrate.transitions():
        i, o, c = t.pairs()
        in_slots.append(i)
        out_slots.append(o)
        coords.append(c)
    return (
        np.concatenate(in_slots),
        np.concatenate(out_slots),
        np.concatenate(coords, axis=0),
    )


def paint_substrate(
    cppn: Genome | list[Genome] | Population,
    substrate: Substrate,
    device: str | torch.device = "cuda:1",
    threshold: float = 0.05,
    weight_scale: float = 3.0,
    steps: int = 8,
) -> CompiledPopulation:
    """Paint a substrate with one/many CPPNs -> a phenotype ``CompiledPopulation``.

    The substrate topology (slot pairs) is fixed and shared across genomes; only
    the painted weights and the near-zero threshold mask differ per genome, so
    the returned edge list is efficient for :func:`population_forward_sparse`.

    ``threshold``    : |w| below this (post-scale) is pruned (``conn_valid=False``).
    ``weight_scale`` : painted CPPN outputs (in [-1,1] under tanh-ish nodes) are
                       multiplied by this to give usable synaptic strengths.
    Returns a ``CompiledPopulation`` on ``device`` ready for the sparse forward.
    """
    dev = torch.device(device)
    pop = _as_population(cppn)
    N = pop.n
    M = substrate.M

    in_slot_np, out_slot_np, coord4 = _enumerate_substrate(substrate)
    C = in_slot_np.shape[0]

    weights = query_cppn(pop, coord4, device=dev, steps=steps)  # (N, C)
    weights = weights * weight_scale
    valid = weights.abs() >= threshold  # (N, C) bool

    in_slot = torch.as_tensor(in_slot_np, dtype=torch.long, device=dev)
    out_slot = torch.as_tensor(out_slot_np, dtype=torch.long, device=dev)
    conn_in_slot = in_slot.unsqueeze(0).expand(N, C).contiguous()
    conn_out_slot = out_slot.unsqueeze(0).expand(N, C).contiguous()

    # phenotype node activations: identity for inputs+bias, tanh for hidden+out
    node_act = torch.full((N, M), TANH, dtype=torch.long, device=dev)
    node_act[:, 0 : substrate.n_in + 1] = IDENTITY  # inputs + bias
    node_bias = torch.zeros((N, M), dtype=torch.float32, device=dev)

    return CompiledPopulation(
        node_act=node_act,
        node_bias=node_bias,
        conn_in_slot=conn_in_slot,
        conn_out_slot=conn_out_slot,
        conn_weight=weights.to(torch.float32).contiguous(),
        conn_valid=valid.contiguous(),
        n_in=substrate.n_in,
        n_out=substrate.n_out,
        M=M,
    )


def paint_dense(
    cppn: Genome,
    substrate: Substrate,
    device: str | torch.device = "cuda:1",
    threshold: float = 0.0,
    weight_scale: float = 1.0,
    steps: int = 8,
) -> np.ndarray:
    """Paint a single CPPN into the dense ``(M, M)`` adjacency (edge j->i at [i,j]).

    For inspection / debugging only — the sparse path is what the loop uses.
    """
    dev = torch.device(device)
    in_slot, out_slot, coord4 = _enumerate_substrate(substrate)
    w = query_cppn(cppn, coord4, device=dev, steps=steps)[0] * weight_scale  # (C,)
    if threshold > 0.0:
        w = torch.where(w.abs() >= threshold, w, torch.zeros_like(w))
    M = substrate.M
    W = torch.zeros(M, M, device=dev, dtype=torch.float32)
    W[torch.as_tensor(out_slot, device=dev), torch.as_tensor(in_slot, device=dev)] = w
    return W.cpu().numpy()


def receptive_field(
    cppn: Genome,
    substrate: Substrate,
    target_xy: tuple[float, float] = (0.0, 0.0),
    device: str | torch.device = "cuda:1",
    steps: int = 8,
) -> np.ndarray:
    """Painted weight field from every screen pixel to a single target coord.

    Returns a ``(grid, grid)`` image = the receptive field of a phenotype node
    placed at ``target_xy``.  This is the direct visualisation of what spatial
    structure the CPPN has painted.
    """
    from pokeio.evo.substrate import grid_coords

    src = grid_coords(substrate.grid)  # (grid*grid, 2)
    tx, ty = target_xy
    tgt = np.broadcast_to(np.array([tx, ty], np.float32), src.shape)
    coord4 = np.concatenate([src, tgt], axis=1).astype(np.float32)
    w = query_cppn(cppn, coord4, device=device, steps=steps)[0].cpu().numpy()
    return w.reshape(substrate.grid, substrate.grid)


# ---------------------------------------------------------------------------
# A hand-built CPPN: a centre-surround (Difference-of-Gaussians) receptive field
# ---------------------------------------------------------------------------
def hand_center_surround_cppn(
    kn: float = 4.0,
    kw: float = 1.7,
    a_gain: float = 1.4,
    b_gain: float = 1.0,
) -> Genome:
    """A CPPN whose painted weight field is a centre-surround receptive field.

    Built ONLY from the canonical inputs ``(x1,y1,x2,y2,bias)`` and ``gauss``
    nodes — no egocentric distance/radius is fed in.  It computes a
    Difference-of-Gaussians:

    * ``bump_k = gauss( 2 - gauss(k*(x1-x2)) - gauss(k*(y1-y2)) )`` is a *radial*
      spot centred where source == target, assembled purely from separable 1-D
      gaussians plus summation (the classic CPPN way to get a 2-D bump without
      an explicit ``r``).
    * output ``= a_gain * bump_narrow - b_gain * bump_wide``  (identity node),
      i.e. a positive centre and a negative surround.

    Demonstrates that spatial structure is EXPRESSIBLE from the canonical inputs.
    """
    tr = InnovationTracker(n_in=CPPN_N_IN, n_out=CPPN_N_OUT)
    g = Genome(n_in=CPPN_N_IN, n_out=CPPN_N_OUT)

    # I/O nodes: inputs 0..3 (x1,y1,x2,y2), bias 4, output 5 (identity summation).
    for i in range(CPPN_N_IN):
        g.nodes[i] = NodeGene(id=i, type=INPUT, act=IDENTITY)
    g.nodes[4] = NodeGene(id=4, type=BIAS, act=IDENTITY)
    g.nodes[5] = NodeGene(id=5, type=OUTPUT, act=IDENTITY)

    def add_conn(s: int, d: int, w: float) -> None:
        innov = tr.conn_innov(s, d)
        g.conns[innov] = ConnGene(s, d, float(w), True, innov)

    def new_gauss() -> int:
        nid = tr._next_node
        tr._next_node += 1
        g.nodes[nid] = NodeGene(id=nid, type=HIDDEN, act=GAUSS)
        return nid

    def bump(k: float) -> int:
        """A radial gaussian spot of width set by ``k`` (larger k = narrower)."""
        hx = new_gauss()  # gauss(k*x1 - k*x2)
        add_conn(0, hx, k)
        add_conn(2, hx, -k)
        hy = new_gauss()  # gauss(k*y1 - k*y2)
        add_conn(1, hy, k)
        add_conn(3, hy, -k)
        spot = new_gauss()  # gauss(2 - hx - hy)  -> peaks where hx==hy==1
        add_conn(hx, spot, -1.0)
        add_conn(hy, spot, -1.0)
        add_conn(4, spot, 2.0)  # bias contributes +2
        return spot

    narrow = bump(kn)
    wide = bump(kw)
    add_conn(narrow, 5, a_gain)   # positive centre
    add_conn(wide, 5, -b_gain)    # negative surround
    return g


__all__ = [
    "GAUSS",
    "ABS",
    "COS",
    "CPPN_ACT_NAMES",
    "CPPN_ACT_INDEX_TO_NAME",
    "CPPN_HIDDEN_ACTS",
    "CPPN_N_IN",
    "CPPN_N_OUT",
    "cppn_apply_activation",
    "query_cppn",
    "paint_substrate",
    "paint_dense",
    "receptive_field",
    "hand_center_surround_cppn",
]
