"""Batched forward pass for a whole NEAT population, on the GPU (card 1).

Design
------
Every genome is materialised as a **weighted adjacency matrix** ``W`` over a
fixed *max-node budget* ``M`` with a connectivity mask (zeros where no edge).
``W[n, i, j]`` is the weight of the edge *from* node ``j`` *to* node ``i`` in
genome ``n``.  The network is evaluated by **iterated propagation** for ``T``
steps::

    x_{t+1} = act( W @ x_t + bias )      (inputs/bias re-clamped each step)

Iterated propagation is deliberate:

* it handles **variable depth** for free — after ``T`` steps a signal has
  travelled up to ``T`` hops, so a few extra steps cost nothing and cover any
  feed-forward depth that has evolved;
* it handles **recurrence** naturally — a cyclic ``W`` is just a discrete-time
  recurrent net unrolled ``T`` steps (evolved memory), no special casing;
* it is **one batched ``bmm`` per step** across the entire population.

The heavy state (``W``) is built per genome-chunk so a population with a large
input dimension does not need a single ``(N, M, M)`` allocation.

CPPN / HyperNEAT hook
---------------------
`propagate` takes a ready weight tensor ``W``.  A CPPN that *paints* a substrate
produces exactly such a ``(chunk, M, M)`` tensor — feed it straight to
`propagate` and skip `build_weight_matrix`.  Nothing here assumes the weights
came from an explicit connection list.

Activations: identity, tanh, relu, sigmoid, sin.  ``sin`` is included because it
matters for CPPNs later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# -- activation registry ----------------------------------------------------
IDENTITY: int = 0
TANH: int = 1
RELU: int = 2
SIGMOID: int = 3
SIN: int = 4

ACT_NAMES: dict[str, int] = {
    "identity": IDENTITY,
    "tanh": TANH,
    "relu": RELU,
    "sigmoid": SIGMOID,
    "sin": SIN,
}
ACT_INDEX_TO_NAME: dict[int, str] = {v: k for k, v in ACT_NAMES.items()}


def apply_activation(z: Tensor, act: Tensor) -> Tensor:
    """Apply a per-node activation.

    ``z``   : ``(N, B, M)`` pre-activations.
    ``act`` : ``(N, M)`` long activation indices (broadcast over the ``B`` axis).
    """
    a = act.unsqueeze(1)  # (N, 1, M)
    out = z  # IDENTITY is the default
    out = torch.where(a == TANH, torch.tanh(z), out)
    out = torch.where(a == RELU, torch.relu(z), out)
    out = torch.where(a == SIGMOID, torch.sigmoid(z), out)
    out = torch.where(a == SIN, torch.sin(z), out)
    return out


@dataclass
class CompiledPopulation:
    """Slot-space, device-resident view of a population, ready for forward.

    Node arrays are indexed by *slot* (the adjacency-matrix row/column).  By the
    genome's sorted-id convention the fixed I/O slots are::

        inputs  : slots [0, n_in)
        bias    : slot   n_in            (held at constant 1.0)
        outputs : slots [n_in+1, n_in+1+n_out)
        hidden  : slots [n_in+1+n_out, M)
    """

    node_act: Tensor  # (N, M) long
    node_bias: Tensor  # (N, M) float
    conn_in_slot: Tensor  # (N, C) long  (source slot)
    conn_out_slot: Tensor  # (N, C) long  (dest slot)
    conn_weight: Tensor  # (N, C) float
    conn_valid: Tensor  # (N, C) bool   (masked AND enabled)
    n_in: int
    n_out: int
    M: int

    @property
    def device(self) -> torch.device:
        return self.node_bias.device

    @property
    def n(self) -> int:
        return self.node_act.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.node_bias.dtype


def build_weight_matrix(cp: CompiledPopulation, a: int, b: int) -> Tensor:
    """Scatter genomes ``[a:b]`` into a dense ``(b-a, M, M)`` adjacency tensor."""
    nb = b - a
    M = cp.M
    dev = cp.device
    W = torch.zeros(nb, M, M, device=dev, dtype=cp.dtype)

    valid = cp.conn_valid[a:b]  # (nb, C)
    oi = cp.conn_out_slot[a:b]  # (nb, C)
    ii = cp.conn_in_slot[a:b]
    w = cp.conn_weight[a:b]

    rows = torch.arange(nb, device=dev).unsqueeze(1).expand_as(oi)
    flat = (rows * (M * M) + oi * M + ii)[valid]
    # index_add_ accumulates, which is safe if two genes ever share an edge.
    W.view(-1).index_add_(0, flat, w[valid].to(W.dtype))
    return W


def propagate(
    W: Tensor,
    node_act: Tensor,
    node_bias: Tensor,
    n_in: int,
    n_out: int,
    X: Tensor,
    steps: int,
) -> Tensor:
    """Iterated propagation over a ready weight tensor.

    ``W``         : ``(N, M, M)`` weights (edge j->i at ``W[n, i, j]``).  This is
                    the CPPN/HyperNEAT plug-in point — supply any painted tensor.
    ``node_act``  : ``(N, M)`` long.
    ``node_bias`` : ``(N, M)`` float.
    ``X``         : ``(N, B, n_in)`` inputs (``B`` = inputs evaluated per genome).
    returns       : ``(N, B, n_out)`` outputs.
    """
    N, M, _ = W.shape
    B = X.shape[1]
    dev = W.device
    dtype = W.dtype

    x = torch.zeros(N, B, M, device=dev, dtype=dtype)
    Xc = X.to(dev, dtype)
    x[:, :, 0:n_in] = Xc
    bias_slot = n_in
    x[:, :, bias_slot] = 1.0

    nbias = node_bias.unsqueeze(1)  # (N, 1, M)
    for _ in range(steps):
        # z[n,b,i] = sum_j W[n,i,j] x[n,b,j]
        z = torch.bmm(W, x.transpose(1, 2)).transpose(1, 2)  # (N, B, M)
        z = z + nbias
        x = apply_activation(z, node_act)
        # re-clamp the driven nodes (inputs + bias) every step
        x[:, :, 0:n_in] = Xc
        x[:, :, bias_slot] = 1.0

    out_start = n_in + 1
    return x[:, :, out_start : out_start + n_out]


def propagate_sparse(
    conn_in_slot: Tensor,
    conn_out_slot: Tensor,
    conn_weight: Tensor,
    conn_valid: Tensor,
    node_act: Tensor,
    node_bias: Tensor,
    n_in: int,
    n_out: int,
    M: int,
    X: Tensor,
    steps: int,
) -> Tensor:
    """Edge-list propagation that never materialises the dense ``(M, M)`` matrix.

    Cost is ``O(N·B·E)`` in the number of *edges* ``E`` rather than ``O(N·M²)``,
    so it is the right path for genomes with a large input dimension but sparse
    connectivity (e.g. per-pixel optical inputs).  Identical semantics to
    :func:`propagate`; also handles recurrence.

    Shapes: ``conn_*`` are ``(N, E)``; ``X`` is ``(N, B, n_in)``; returns
    ``(N, B, n_out)``.
    """
    N, E = conn_in_slot.shape
    B = X.shape[1]
    dev = conn_weight.device
    dtype = conn_weight.dtype

    w = (conn_weight * conn_valid.to(dtype)).unsqueeze(1)  # (N, 1, E)
    in_idx = conn_in_slot.unsqueeze(1).expand(N, B, E)  # (N, B, E)
    out_idx = conn_out_slot.unsqueeze(1).expand(N, B, E)

    x = torch.zeros(N, B, M, device=dev, dtype=dtype)
    Xc = X.to(dev, dtype)
    x[:, :, 0:n_in] = Xc
    x[:, :, n_in] = 1.0
    nbias = node_bias.unsqueeze(1)

    for _ in range(steps):
        x_in = torch.gather(x, 2, in_idx)  # (N, B, E)
        contrib = x_in * w
        z = torch.zeros(N, B, M, device=dev, dtype=dtype)
        z.scatter_add_(2, out_idx, contrib)
        z = z + nbias
        x = apply_activation(z, node_act)
        x[:, :, 0:n_in] = Xc
        x[:, :, n_in] = 1.0

    out_start = n_in + 1
    return x[:, :, out_start : out_start + n_out]


def population_forward_sparse(
    cp: "CompiledPopulation", X: Tensor, steps: int = 8
) -> Tensor:
    """Sparse-edge forward over a whole population (no dense adjacency)."""
    N = cp.n
    if X.dim() == 2:
        X = X.unsqueeze(0).expand(N, -1, -1)
    return propagate_sparse(
        cp.conn_in_slot,
        cp.conn_out_slot,
        cp.conn_weight,
        cp.conn_valid,
        cp.node_act,
        cp.node_bias,
        cp.n_in,
        cp.n_out,
        cp.M,
        X,
        steps,
    )


def population_forward(
    cp: CompiledPopulation,
    X: Tensor,
    steps: int = 8,
    chunk: int | None = None,
) -> Tensor:
    """Evaluate an entire population.

    ``X``     : ``(N, B, n_in)`` or ``(B, n_in)`` (shared across genomes).
    ``chunk`` : genomes per adjacency build (bounds peak memory); ``None`` = all.
    returns   : ``(N, B, n_out)``.
    """
    N = cp.n
    if X.dim() == 2:
        X = X.unsqueeze(0).expand(N, -1, -1)
    if chunk is None:
        chunk = N

    outs: list[Tensor] = []
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        W = build_weight_matrix(cp, a, b)
        out = propagate(
            W,
            cp.node_act[a:b],
            cp.node_bias[a:b],
            cp.n_in,
            cp.n_out,
            X[a:b],
            steps,
        )
        outs.append(out)
    return torch.cat(outs, dim=0)


__all__ = [
    "IDENTITY",
    "TANH",
    "RELU",
    "SIGMOID",
    "SIN",
    "ACT_NAMES",
    "ACT_INDEX_TO_NAME",
    "apply_activation",
    "CompiledPopulation",
    "build_weight_matrix",
    "propagate",
    "propagate_sparse",
    "population_forward",
    "population_forward_sparse",
]
