"""Live streaming for the pokeIO dashboard.

Writes ``runs/<id>/live.json`` (atomically, ~3 Hz) so a dashboard can show the
REAL champion network actually playing plus a live sample of the training swarm.

Three moving parts:

* :class:`LiveWriter` — a throttled, atomic JSON writer (write ``.tmp`` then
  ``os.replace``).  ``due()`` gates the expensive payload build so we only pay
  base64/serialization cost ~3 times a second.
* :class:`ChampionShowcase` — a dedicated :class:`~pokeio.emu.env.PokeEnv` that
  continuously plays the best genome so far.  The genome is swapped in at each
  generation boundary; a few env steps are taken per live write and the real
  forward pass is run to capture live per-node activations.
* :class:`LiveStreamer` — bundles the two together and builds the full
  ``live.json`` payload (champion + swarm) matching the frontend contract.

Only stdlib + numpy + torch, and it imports the evo/emu packages read-only.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pokeio.evo.forward import apply_activation
from pokeio.evo.genome import HIDDEN as NODE_HIDDEN
from pokeio.evo.genome import OUTPUT as NODE_OUTPUT
from pokeio.evo.genome import Population

_SCREEN_H = 144
_SCREEN_W = 160
N_OUT = 8
WRAM_BASE = 0xC000

# Swarm downscale target (must divide the native screen evenly: 160/40, 144/36 = 4).
_SW_W = 40
_SW_H = 36

# A handful of known / interesting Pokemon Yellow WRAM bytes to tap for the UI.
INTERESTING_RAM = [
    0xD35E,  # current map id
    0xD361,  # player Y tile
    0xD362,  # player X tile
    0xD163,  # party count
    0xD356,  # event/flag byte
    0xD057,  # in-battle flag
    0xD347,  # money (low byte, BCD)
    0xD16B,  # first party mon current HP (hi)
]

# Payload budgets — keep live.json small and the wire cheap.
_CONN_CAP = 600
_NODE_CAP = 380


# --------------------------------------------------------------------------
# base64 / downscale helpers
# --------------------------------------------------------------------------
def b64_gray(gray) -> str:
    """base64 of raw grayscale bytes (1 byte/pixel, row-major, uint8)."""
    a = np.asarray(gray, dtype=np.uint8).ravel()
    return base64.b64encode(a.tobytes()).decode()


def downscale_swarm(screen: np.ndarray) -> np.ndarray:
    """Area-mean downscale a (144,160) uint8 screen to (36,40) uint8."""
    g = np.asarray(screen, dtype=np.float32)
    small = g.reshape(_SW_H, _SCREEN_H // _SW_H, _SW_W, _SCREEN_W // _SW_W).mean(
        axis=(1, 3)
    )
    return small.astype(np.uint8)


def build_swarm(
    screens: list[np.ndarray],
    fitness: np.ndarray,
    dead: list[bool],
    cap: int = 32,
    elite_k: int = 4,
) -> list[dict]:
    """Downscaled live frames of up to ``cap`` current-wave players."""
    n = min(len(screens), cap)
    if n == 0:
        return []
    fit = np.asarray(fitness, dtype=np.float64)[: len(screens)]
    order = np.argsort(-fit)
    elite = set(int(i) for i in order[:elite_k].tolist())
    out = []
    for i in range(n):
        out.append(
            {
                "id": i,
                "w": _SW_W,
                "h": _SW_H,
                "b64": b64_gray(downscale_swarm(screens[i])),
                "elite": bool(i in elite),
                "dead": bool(dead[i]) if dead is not None and i < len(dead) else False,
            }
        )
    return out


# --------------------------------------------------------------------------
# throttled atomic writer
# --------------------------------------------------------------------------
class LiveWriter:
    """Throttled (~``hz``) atomic writer for ``<run_dir>/live.json``."""

    def __init__(self, run_dir: str | Path, hz: float = 3.0) -> None:
        rd = Path(run_dir)
        rd.mkdir(parents=True, exist_ok=True)
        self.path = rd / "live.json"
        self.tmp = rd / "live.json.tmp"
        self.interval = 1.0 / max(0.1, hz)
        self._last = 0.0

    def due(self) -> bool:
        return (time.monotonic() - self._last) >= self.interval

    def write(self, payload: dict) -> int:
        """Serialize + atomically replace. Returns bytes written."""
        data = json.dumps(payload, separators=(",", ":")).encode()
        with open(self.tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(self.tmp, self.path)
        self._last = time.monotonic()
        return len(data)


# --------------------------------------------------------------------------
# single-genome forward with full activation capture
# --------------------------------------------------------------------------
def _forward_capture(cp, X: torch.Tensor, steps: int) -> torch.Tensor:
    """Sparse-edge propagation that returns the FULL (N,B,M) activation state.

    Semantics are identical to :func:`pokeio.evo.forward.propagate_sparse`; we
    just keep the whole node vector so we can read hidden/out activations for the
    live net view instead of only the output slice.
    """
    conn_in_slot = cp.conn_in_slot
    conn_out_slot = cp.conn_out_slot
    N, E = conn_in_slot.shape
    B = X.shape[1]
    M = cp.M
    dev = cp.conn_weight.device
    dtype = cp.conn_weight.dtype

    w = (cp.conn_weight * cp.conn_valid.to(dtype)).unsqueeze(1)  # (N,1,E)
    in_idx = conn_in_slot.unsqueeze(1).expand(N, B, E)
    out_idx = conn_out_slot.unsqueeze(1).expand(N, B, E)

    x = torch.zeros(N, B, M, device=dev, dtype=dtype)
    Xc = X.to(dev, dtype)
    x[:, :, 0 : cp.n_in] = Xc
    x[:, :, cp.n_in] = 1.0
    nbias = cp.node_bias.unsqueeze(1)

    for _ in range(steps):
        x_in = torch.gather(x, 2, in_idx)
        contrib = x_in * w
        z = torch.zeros(N, B, M, device=dev, dtype=dtype)
        z.scatter_add_(2, out_idx, contrib)
        z = z + nbias
        x = apply_activation(z, cp.node_act)
        x[:, :, 0 : cp.n_in] = Xc
        x[:, :, cp.n_in] = 1.0
    return x


def _node_kind(node_type: int) -> str:
    if node_type == NODE_OUTPUT:
        return "out"
    if node_type == NODE_HIDDEN:
        return "hidden"
    return "in"  # INPUT and BIAS both surface as inputs to the UI


# --------------------------------------------------------------------------
# champion showcase env
# --------------------------------------------------------------------------
class ChampionShowcase:
    """A dedicated env that continuously plays the best-genome-so-far."""

    def __init__(
        self,
        env,
        encoder,
        device: torch.device,
        reset_state: str,
        forward_steps: int,
        max_nodes: int,
        max_conns: int,
        ram_addrs: list[int] | None = None,
    ) -> None:
        self.env = env
        self.encoder = encoder
        self.device = device
        self.reset_state = reset_state
        self.forward_steps = int(forward_steps)
        self.max_nodes = int(max_nodes)
        self.max_conns = int(max_conns)
        self.ram_addrs = ram_addrs if ram_addrs is not None else list(INTERESTING_RAM)

        self.genome = None
        self.genome_id = "none"
        self.cp = None
        self._id_to_slot: dict[int, int] = {}
        self._net_nodes: list[dict] = []
        self._net_conns: list[dict] = []
        self._net_node_ids: set[int] = set()

        self.screen: np.ndarray | None = None
        self.wram: np.ndarray | None = None
        self.obs_vis: np.ndarray | None = None
        self.last_action = 0
        self._act: dict[str, float] = {}

    # -- champion swap -----------------------------------------------------
    def set_champion(self, genome, genome_id: str) -> None:
        """Install a new champion genome (keeps env state for continuity)."""
        self.genome = genome.copy()
        self.genome_id = str(genome_id)
        pop = Population.from_genomes(
            [self.genome], max_nodes=self.max_nodes, max_conns=self.max_conns
        )
        self.cp = pop.compile(self.device)
        self._prepare_net()
        # Reset only the first time so the showcase plays continuously afterwards.
        if self.screen is None:
            self.screen = self.env.reset(self.reset_state)
            self.wram = self.env.raw_wram()
            self.obs_vis = self._encode_vis(self.screen, self.wram)

    def _encode_vis(self, screen, wram) -> np.ndarray:
        res = self.encoder.res
        obs = self.encoder.encode(screen, wram)
        return obs[: res * res].reshape(res, res)

    def _prepare_net(self) -> None:
        """Pick a bounded, self-consistent subgraph (hidden+out + strong inputs)."""
        g = self.genome
        sorted_ids = sorted(g.nodes)
        self._id_to_slot = {nid: s for s, nid in enumerate(sorted_ids)}

        # always include hidden + output nodes
        included: set[int] = {
            nid for nid, ng in g.nodes.items() if ng.type in (NODE_OUTPUT, NODE_HIDDEN)
        }

        enabled = [(innov, c) for innov, c in g.conns.items() if c.enabled]
        enabled.sort(key=lambda kc: abs(kc[1].weight), reverse=True)

        conns: list[dict] = []
        for _innov, c in enabled:
            if len(conns) >= _CONN_CAP:
                break
            new_nodes = {c.in_id, c.out_id} - included
            if new_nodes and len(included) + len(new_nodes) > _NODE_CAP:
                continue  # keep node budget; skip conns that would add new nodes
            included.update((c.in_id, c.out_id))
            conns.append(
                {
                    "from": int(c.in_id),
                    "to": int(c.out_id),
                    "w": round(float(c.weight), 4),
                    "en": True,
                }
            )
        self._net_conns = conns

        nodes = []
        for nid in sorted(included):
            ng = g.nodes.get(nid)
            if ng is None:
                continue
            nodes.append({"id": int(nid), "kind": _node_kind(ng.type)})
        self._net_nodes = nodes
        self._net_node_ids = {n["id"] for n in nodes}

    # -- stepping ----------------------------------------------------------
    def step(self, n_steps: int = 2) -> None:
        """Advance the showcase env ``n_steps`` and capture the live activations."""
        if self.cp is None or self.screen is None:
            return
        res = self.encoder.res
        for _ in range(max(1, n_steps)):
            x = self.encoder.encode(self.screen, self.wram)
            xt = torch.from_numpy(x[None, :]).to(self.device).unsqueeze(1)  # (1,1,dim)
            act_state = _forward_capture(self.cp, xt, self.forward_steps)  # (1,1,M)
            out = act_state[0, 0, self.cp.n_in + 1 : self.cp.n_in + 1 + N_OUT]
            self.last_action = int(out.argmax().item())
            self.screen, self.wram, _done, _info = self.env.step(self.last_action)

        # capture per-node activations from the LAST forward (in slot space)
        vec = act_state[0, 0].detach().to("cpu").numpy()
        self._act = {
            str(nid): round(float(vec[slot]), 4)
            for nid, slot in self._id_to_slot.items()
            if slot < vec.shape[0] and int(nid) in self._net_node_ids
        }
        self.obs_vis = self._encode_vis(self.screen, self.wram)

    # -- payload -----------------------------------------------------------
    def _ram_bytes(self) -> dict:
        out = {}
        if self.wram is None:
            return out
        for a in self.ram_addrs:
            idx = a - WRAM_BASE
            if 0 <= idx < self.wram.size:
                out[format(a, "04X")] = int(self.wram[idx])
        return out

    def payload(self) -> dict:
        buttons = [0] * N_OUT
        if 0 <= self.last_action < N_OUT:
            buttons[self.last_action] = 1
        obs_gray = (
            np.clip(self.obs_vis, 0.0, 1.0) * 255.0 if self.obs_vis is not None else None
        )
        return {
            "genome_id": self.genome_id,
            "frame_w": _SCREEN_W,
            "frame_h": _SCREEN_H,
            "frame_b64": b64_gray(self.screen) if self.screen is not None else "",
            "obs_res": int(self.encoder.res),
            "obs_b64": b64_gray(obs_gray) if obs_gray is not None else "",
            "action": int(self.last_action),
            "buttons": buttons,
            "ram": self._ram_bytes(),
            "net": {
                "n_in": int(self.encoder.dim),
                "n_out": N_OUT,
                "nodes": self._net_nodes,
                "conns": self._net_conns,
                "act": self._act,
            },
        }


# --------------------------------------------------------------------------
# top-level streamer
# --------------------------------------------------------------------------
class LiveStreamer:
    """Bundles the writer + showcase and emits the full live.json payload."""

    def __init__(
        self,
        run_dir: str | Path,
        showcase: ChampionShowcase,
        run_id: str,
        hz: float = 3.0,
        swarm_cap: int = 32,
        elite_k: int = 4,
        champ_steps: int = 2,
    ) -> None:
        self.writer = LiveWriter(run_dir, hz)
        self.showcase = showcase
        self.run_id = str(run_id)
        self.swarm_cap = int(swarm_cap)
        self.elite_k = int(elite_k)
        self.champ_steps = int(champ_steps)
        self.last_size = 0
        self._cache = None  # (screens, fitness, dead) from the most recent sample

    def set_champion(self, genome, genome_id: str) -> None:
        self.showcase.set_champion(genome, genome_id)

    def maybe_write(
        self,
        gen: int,
        screens: list[np.ndarray],
        fitness: np.ndarray,
        dead: list[bool] | None = None,
    ) -> bool:
        """Throttled write from inside the wave step loop."""
        self._cache = (screens, fitness, dead)
        if not self.writer.due():
            return False
        return self._emit(gen)

    def force_write(self, gen: int) -> bool:
        """Unconditional write (e.g. at a generation boundary)."""
        return self._emit(gen)

    def _emit(self, gen: int) -> bool:
        self.showcase.step(self.champ_steps)
        if self._cache is not None:
            screens, fitness, dead = self._cache
            swarm = build_swarm(
                screens, fitness, dead, self.swarm_cap, self.elite_k
            )
        else:
            swarm = []
        payload = {
            "t": time.time(),
            "gen": int(gen),
            "run": self.run_id,
            "champion": self.showcase.payload(),
            "swarm": swarm,
        }
        self.last_size = self.writer.write(payload)
        return True


__all__ = [
    "LiveWriter",
    "ChampionShowcase",
    "LiveStreamer",
    "build_swarm",
    "b64_gray",
    "downscale_swarm",
    "INTERESTING_RAM",
]
