"""ES-HyperNEAT substrate layout for the pokeIO vision encoder (Lane B).

A *substrate* is the fixed geometric arrangement of the phenotype's neurons in a
low-dimensional coordinate space.  A CPPN (see :mod:`pokeio.evo.cppn`) is then
queried once per potential connection with the SOURCE and TARGET coordinates of
that connection and *paints* the connection's weight.  This module owns the
geometry only — it knows nothing about the CPPN or torch forward pass beyond
handing out ``(source, target)`` coordinate pairs and their adjacency slots.

Canonical CPPN contract (non-negotiable project principle)
----------------------------------------------------------
The CPPN is queried with ``CPPN(x1, y1, x2, y2, bias)`` — the raw source coord
``(x1, y1)`` and target coord ``(x2, y2)`` of the connection, plus the standard
NEAT bias input.  We deliberately do NOT hand the CPPN any egocentric prior
(no centre-distance, no radial ``r``, no phase, no hand-picked angle).  Radial /
symmetric / spatial structure is left for evolution to DISCOVER through the
CPPN's ``sin`` / ``gauss`` nodes.

Coordinate regions
------------------
* **Screen sheet** — a ``grid`` x ``grid`` retinotopic input sheet spanning the
  square ``[-1, 1]^2``.  Row-major, ``y`` descending (row 0 = top = ``y=+1``).
* **RAM-aux band** — the length-``ram_dim`` RAM aux vector lives in a *distinct*
  coordinate region: a horizontal strip ABOVE the screen sheet at ``y = +1.4``
  (outside ``[-1, 1]^2``).  Because it is geometrically separated the CPPN can
  learn to treat RAM inputs differently from screen pixels.
* **Hidden sheet** — an optional ``hidden`` x ``hidden`` sheet in ``[-1, 1]^2``.
* **Output layer** — the 8 Game Boy buttons at fixed, semantically arranged
  coordinates (d-pad cross + face buttons + start/select).

Adjacency-slot convention (matches :mod:`pokeio.evo.genome` / ``forward``)::

    inputs  : slots [0, n_in)            screen sheet (grid*grid) then RAM (ram_dim)
    bias    : slot   n_in                (phenotype bias node, held at 1.0)
    outputs : slots [n_in+1, n_in+1+8)   the 8 buttons
    hidden  : slots [n_in+1+8, M)        the hidden sheet (may be empty)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 8 Game Boy buttons, in env.py order: up, down, left, right, a, b, start, select
BUTTON_NAMES = ("up", "down", "left", "right", "a", "b", "start", "select")

# Fixed, meaningful 2D layout for the 8 outputs in [-1, 1]^2.  A d-pad cross on
# the left, the A/B face buttons on the right, start/select along the bottom.
_BUTTON_XY = np.array(
    [
        [-0.55, 0.55],   # up
        [-0.55, -0.55],  # down
        [-0.85, 0.0],    # left
        [-0.25, 0.0],    # right
        [0.85, 0.15],    # a
        [0.55, -0.15],   # b
        [0.10, -0.90],   # start
        [0.45, -0.90],   # select
    ],
    dtype=np.float32,
)


def grid_coords(n: int) -> np.ndarray:
    """``(n*n, 2)`` retinotopic coordinates in ``[-1, 1]^2``, row-major.

    Row 0 is the top of the screen (``y = +1``); column 0 is the left
    (``x = -1``).  With ``n == 1`` the single node sits at the origin.
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        axis = np.array([0.0], dtype=np.float32)
    else:
        axis = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    xs, ys = np.meshgrid(axis, axis[::-1])  # y descending down the rows
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)


def ram_coords(r: int, band_y: float = 1.4) -> np.ndarray:
    """``(r, 2)`` coordinates for the RAM-aux vector.

    A horizontal strip at ``y = band_y`` (default ``+1.4``, above the screen
    sheet) so it occupies a coordinate region DISTINCT from the pixel sheet.
    """
    if r <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if r == 1:
        xs = np.array([0.0], dtype=np.float32)
    else:
        xs = np.linspace(-1.0, 1.0, r, dtype=np.float32)
    ys = np.full(r, band_y, dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def button_coords() -> np.ndarray:
    """``(8, 2)`` fixed output-button coordinates."""
    return _BUTTON_XY.copy()


@dataclass(frozen=True)
class Transition:
    """One layer-to-layer block of potential connections to be painted.

    ``src_slots`` / ``tgt_slots`` are the adjacency-matrix slots of the source
    and target nodes.  ``src_xy`` / ``tgt_xy`` are their ``(., 2)`` coordinates.
    """

    name: str
    src_slots: np.ndarray  # (S,) long
    tgt_slots: np.ndarray  # (T,) long
    src_xy: np.ndarray     # (S, 2)
    tgt_xy: np.ndarray     # (T, 2)

    def pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enumerate every ``(source, target)`` pair in this transition.

        Returns ``(in_slot, out_slot, coord4)`` where each has length ``S*T``:

        * ``in_slot[k]``  — source adjacency slot of pair ``k``
        * ``out_slot[k]`` — target adjacency slot of pair ``k``
        * ``coord4[k]``   — ``(x1, y1, x2, y2)`` the CPPN is queried with

        Ordering is ``(target-major, source-minor)`` — i.e. all sources for
        target 0, then all for target 1, ...  (irrelevant to correctness; the
        slot arrays carry the mapping).
        """
        S = self.src_slots.shape[0]
        T = self.tgt_slots.shape[0]
        in_slot = np.tile(self.src_slots, T)                       # (T*S,)
        out_slot = np.repeat(self.tgt_slots, S)                    # (T*S,)
        src = np.tile(self.src_xy, (T, 1))                         # (T*S, 2)
        tgt = np.repeat(self.tgt_xy, S, axis=0)                    # (T*S, 2)
        coord4 = np.concatenate([src, tgt], axis=1).astype(np.float32)
        return in_slot.astype(np.int64), out_slot.astype(np.int64), coord4


@dataclass
class Substrate:
    """Full substrate geometry + adjacency-slot bookkeeping for the encoder.

    Parameters
    ----------
    grid : int
        Side length of the square retinotopic input sheet (default 24).  Set to
        32 to match the ``coarse`` obs sheet exactly.
    ram_dim : int
        Length of the RAM-aux vector (default 32, matches ``ObsBuilder``).
    hidden : int
        Side length of the square hidden sheet; ``0`` = no hidden layer (inputs
        wire straight to outputs).
    n_out : int
        Number of output buttons (fixed at 8 for pokeIO).
    direct_io : bool
        If ``True`` also enumerate a direct input->output transition alongside
        input->hidden->output (a skip pathway).  Ignored when ``hidden == 0``.
    """

    grid: int = 24
    ram_dim: int = 32
    hidden: int = 8
    n_out: int = 8
    direct_io: bool = False

    # filled in __post_init__
    input_xy: np.ndarray = field(init=False)
    hidden_xy: np.ndarray = field(init=False)
    output_xy: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        screen = grid_coords(self.grid)                 # (grid*grid, 2)
        ram = ram_coords(self.ram_dim)                  # (ram_dim, 2)
        self.input_xy = np.concatenate([screen, ram], axis=0).astype(np.float32)
        self.hidden_xy = grid_coords(self.hidden)       # (hidden*hidden, 2)
        self.output_xy = button_coords()[: self.n_out]  # (n_out, 2)

    # -- sizes -------------------------------------------------------------
    @property
    def n_screen(self) -> int:
        return self.grid * self.grid

    @property
    def n_in(self) -> int:
        return self.n_screen + self.ram_dim

    @property
    def n_hidden(self) -> int:
        return self.hidden * self.hidden

    @property
    def M(self) -> int:
        """Total adjacency slots: inputs + bias + outputs + hidden."""
        return self.n_in + 1 + self.n_out + self.n_hidden

    # -- slot ranges (match genome/forward convention) --------------------
    @property
    def input_slots(self) -> np.ndarray:
        return np.arange(0, self.n_in, dtype=np.int64)

    @property
    def bias_slot(self) -> int:
        return self.n_in

    @property
    def output_slots(self) -> np.ndarray:
        start = self.n_in + 1
        return np.arange(start, start + self.n_out, dtype=np.int64)

    @property
    def hidden_slots(self) -> np.ndarray:
        start = self.n_in + 1 + self.n_out
        return np.arange(start, start + self.n_hidden, dtype=np.int64)

    # -- transitions -------------------------------------------------------
    def transitions(self) -> list[Transition]:
        """Enumerate the feed-forward layer transitions to be painted.

        With a hidden sheet: ``input->hidden`` then ``hidden->output`` (plus an
        optional ``input->output`` skip if ``direct_io``).  Without a hidden
        sheet: a single ``input->output`` transition.
        """
        ts: list[Transition] = []
        if self.n_hidden > 0:
            ts.append(
                Transition(
                    "input->hidden",
                    self.input_slots, self.hidden_slots,
                    self.input_xy, self.hidden_xy,
                )
            )
            ts.append(
                Transition(
                    "hidden->output",
                    self.hidden_slots, self.output_slots,
                    self.hidden_xy, self.output_xy,
                )
            )
            if self.direct_io:
                ts.append(
                    Transition(
                        "input->output",
                        self.input_slots, self.output_slots,
                        self.input_xy, self.output_xy,
                    )
                )
        else:
            ts.append(
                Transition(
                    "input->output",
                    self.input_slots, self.output_slots,
                    self.input_xy, self.output_xy,
                )
            )
        return ts

    def n_potential_edges(self) -> int:
        return sum(t.src_slots.shape[0] * t.tgt_slots.shape[0] for t in self.transitions())


__all__ = [
    "BUTTON_NAMES",
    "grid_coords",
    "ram_coords",
    "button_coords",
    "Transition",
    "Substrate",
]
