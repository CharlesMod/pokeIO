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
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from pokeio.analytics.yellow import game_is_yellow
from pokeio.emu.fleet import FovealEncoder
from pokeio.evo.forward import apply_activation
from pokeio.evo.genome import BIAS as NODE_BIAS
from pokeio.evo.genome import HIDDEN as NODE_HIDDEN
from pokeio.evo.genome import INPUT as NODE_INPUT
from pokeio.evo.genome import OUTPUT as NODE_OUTPUT
from pokeio.evo.genome import Population
from pokeio.manifest.generate import ram_addresses_from_manifest

_SCREEN_H = 144
_SCREEN_W = 160
# Active-vision head (spec §3.1): the genome now emits 11 outputs — a Discrete-9
# button head (argmax, ACTIONS order) + 2 raw saccade commands out[9]/out[10].
# The dashboard renders the 9 button values; the saccade/gaze are exposed
# separately. The full output width is read from the compiled genome (cp.n_out).
N_BUTTONS = 9  # up down left right A B START SELECT NOOP (the argmax button slice)
WRAM_BASE = 0xC000

# Swarm downscale target (must divide the native screen evenly: 160/40, 144/36 = 4).
_SW_W = 40
_SW_H = 36

# A handful of known / interesting Pokemon Yellow WRAM bytes to tap for the UI.
# QUARANTINED: this is the game-specific FALLBACK, used only when no manifest is
# present (legacy default, keeps the dashboard working) or when a Yellow
# manifest declares no RAM taps. A non-Yellow manifest routes taps through
# ``ram_taps_from_manifest`` instead, so these Yellow addresses never leak.
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


def ram_taps_from_manifest(manifest) -> list[int]:
    """RAM tap addresses for the UI, routed through a loaded Progress Manifest.

    * ``manifest is None`` -> the legacy Yellow :data:`INTERESTING_RAM` table
      (the current default; keeps the existing dashboard unchanged).
    * manifest declares RAM taps -> those addresses (spatial + progress dims).
    * a Yellow manifest with no RAM taps -> :data:`INTERESTING_RAM`.
    * any other manifest with no RAM taps -> ``[]`` (no Yellow-label leak).
    """
    if manifest is None:
        return list(INTERESTING_RAM)
    taps = ram_addresses_from_manifest(manifest)
    if taps:
        return taps
    return list(INTERESTING_RAM) if game_is_yellow(manifest) else []

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


# --------------------------------------------------------------------------
# active-vision obs decoding (foveal block layout — spec §2.1)
# --------------------------------------------------------------------------
# A FovealEncoder obs vector is a contiguous [periph|fovea|motion|proprio|ram]
# layout with G = periph_grid: periph [0:G^2], fovea [G^2:2G^2], motion
# [2G^2:3G^2], proprio [3G^2:3G^2+14], ram trailing. These helpers pull the
# optical views + gaze back out of a raw obs vector so the spectator can SEE
# both the low-res periphery and the high-acuity fovea crop the agent looks at.
# NOTE: these decoders assume the LEGACY uniform-grid layout (fovea_grid == G).
# A sharp fovea (optical-frontend-v2 §2, fovea_grid > G) makes the fovea block
# FG^2 wide, shifting the motion/proprio offsets — generalizing this decode to
# read FG off the obs is the rendering increment (spec §8.7, task #12). Default
# runs (fovea_grid=0 => FG=G) are byte-identical, so this stays correct for them.
def _block2d(vec, grid: int, block: str) -> np.ndarray:
    """Extract a ``grid``x``grid`` optical block ('periph'|'fovea'|'motion')."""
    n = grid * grid
    off = {"periph": 0, "fovea": n, "motion": 2 * n}[block]
    a = np.asarray(vec[off : off + n], dtype=np.float32)
    if a.size < n:  # defensive: short/legacy vector -> pad
        a = np.concatenate([a, np.zeros(n - a.size, np.float32)])
    return a.reshape(grid, grid)


def _b64_block(block2d) -> str:
    """base64 grayscale of a [0,1] optical block (fovea/periphery/motion view)."""
    g = np.clip(np.asarray(block2d, dtype=np.float32), 0.0, 1.0) * 255.0
    return b64_gray(g)


def _gaze_from_proprio(vec, grid: int, screen_h: int, screen_w: int):
    """Recover the (gy, gx) screen-px fovea centre from a foveal obs vector.

    proprio[0]=gx*2/W-1, proprio[1]=gy*2/H-1 (spec §2.1), so the gaze a stored
    obs was cropped at is recoverable without the encoder that built it (used to
    show where a focus swarm-agent was looking)."""
    off = 3 * grid * grid
    if off + 1 >= len(vec):
        return float(screen_h) / 2.0, float(screen_w) / 2.0
    gx = (float(vec[off]) + 1.0) * 0.5 * screen_w
    gy = (float(vec[off + 1]) + 1.0) * 0.5 * screen_h
    return gy, gx


def _gaze_payload(gy, gx, screen_h, screen_w, fovea_px) -> dict:
    """Attention-box telemetry: fovea centre (px + normalised) + box size (frac).

    ``x01/y01`` are the normalised centre and ``w01/h01`` the box size as a
    fraction of the frame, so the dashboard can draw the fixation rectangle over
    the full-res frame directly."""
    return {
        "gy": round(float(gy), 2),
        "gx": round(float(gx), 2),
        "y01": round(float(gy) / screen_h, 4),
        "x01": round(float(gx) / screen_w, 4),
        "w01": round(float(fovea_px) / screen_w, 4),
        "h01": round(float(fovea_px) / screen_h, 4),
    }


def _saccade_payload(dx, dy) -> dict:
    """Raw (pre-tanh) saccade command out[9]/out[10]; update_gaze applies tanh."""
    return {"dx": round(float(dx), 4), "dy": round(float(dy), 4)}


def _split_head(out_np: np.ndarray):
    """Split a raw (N_out,) genome output into (button, dx, dy) — spec §3.1.

    Replicated inline (NOT imported from ``train.loop`` — that would be a circular
    import: ``loop`` imports this module). ``out[:9].argmax()`` is the button
    (ACTIONS order); ``out[9]``/``out[10]`` are the RAW saccade commands — the
    gaze integrator (``FovealEncoder.update_gaze``) applies tanh, so no tanh here.
    """
    a = np.asarray(out_np, dtype=np.float32).ravel()
    button = int(a[:N_BUTTONS].argmax())
    dx = float(a[N_BUTTONS]) if a.size > N_BUTTONS else 0.0
    dy = float(a[N_BUTTONS + 1]) if a.size > N_BUTTONS + 1 else 0.0
    return button, dx, dy


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
    """Throttled (~``hz``) atomic writer for ``<run_dir>/live.json``.

    The serialize + disk write runs on a dedicated background thread so a slow
    filesystem never stalls the training barrier: ``write`` just hands off the
    (already plain-python) payload and returns immediately.  ``os.replace`` is
    atomic on its own, so the previous inline ``os.fsync`` (~21 ms per call at
    3 Hz — ~6% of a loop core) is dropped; a reader always sees a complete file.
    Only the newest payload is kept — if the writer falls behind, stale frames
    are discarded rather than queued.
    """

    def __init__(self, run_dir: str | Path, hz: float = 3.0) -> None:
        rd = Path(run_dir)
        rd.mkdir(parents=True, exist_ok=True)
        self.path = rd / "live.json"
        self.tmp = rd / "live.json.tmp"
        self.interval = 1.0 / max(0.1, hz)
        self._last = 0.0
        self._last_size = 0
        # single-slot hand-off to the writer thread
        self._pending: dict | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="live-writer", daemon=True)
        self._thread.start()

    def due(self) -> bool:
        return (time.monotonic() - self._last) >= self.interval

    def write(self, payload: dict) -> int:
        """Hand the payload to the writer thread (non-blocking). Returns last size."""
        with self._lock:
            self._pending = payload
        self._last = time.monotonic()
        self._wake.set()
        return self._last_size

    def _run(self) -> None:
        while True:
            self._wake.wait()
            if self._stop and self._pending is None:
                return
            with self._lock:
                payload = self._pending
                self._pending = None
                self._wake.clear()
            if payload is None:
                if self._stop:
                    return
                continue
            try:
                data = json.dumps(payload, separators=(",", ":")).encode()
                with open(self.tmp, "wb") as fh:
                    fh.write(data)
                os.replace(self.tmp, self.path)
                self._last_size = len(data)
            except Exception:
                pass

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        try:
            self._thread.join(timeout=2)
        except Exception:
            pass


# --------------------------------------------------------------------------
# single-genome forward with full activation capture
# --------------------------------------------------------------------------
def _forward_capture(
    cp, X: torch.Tensor, steps: int, state: torch.Tensor | None = None
) -> torch.Tensor:
    """Sparse-edge propagation that returns the FULL (N,B,M) activation state.

    Semantics are identical to :func:`pokeio.evo.forward.propagate_sparse`; we
    just keep the whole node vector so we can read hidden/out activations for the
    live net view instead of only the output slice.  ``state`` seeds the node
    activations (previous step's returned x) so the cage reflects the same
    cross-step recurrent memory the champion had in training; ``None`` = zeros.
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

    if state is None:
        x = torch.zeros(N, B, M, device=dev, dtype=dtype)
    else:
        x = state.to(dev, dtype).clone()
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


# Genotype-view node kinds (focus.genes) — distinct from the net-view kinds.
_GENE_KIND = {
    NODE_INPUT: "in",
    NODE_BIAS: "bias",
    NODE_OUTPUT: "out",
    NODE_HIDDEN: "hid",
}


def build_genes(genome) -> dict:
    """Raw Lane-A genotype straight off the genome object.

    ``nodes``: ``[[node_id, kind, act_fn_idx], ...]`` (id-sorted;
    kind in ``in|hid|out|bias``).  ``conns``: innovation-ordered
    ``[[innov, in_id, out_id, weight, enabled], ...]``.
    """
    nodes = [
        [int(nid), _GENE_KIND.get(ng.type, "in"), int(ng.act)]
        for nid, ng in sorted(genome.nodes.items())
    ]
    conns = [
        [int(innov), int(c.in_id), int(c.out_id), round(float(c.weight), 4),
         bool(c.enabled)]
        for innov, c in sorted(genome.conns.items())
    ]
    return {"nodes": nodes, "conns": conns}


def build_net_view(genome) -> tuple[list[dict], list[dict], dict[int, int], set[int]]:
    """Bounded, self-consistent net subgraph (hidden+out + strongest conns).

    Returns ``(nodes, conns, id_to_slot, included_node_ids)`` where ``nodes`` /
    ``conns`` use the champion.net schema and ``id_to_slot`` maps node id ->
    compiled slot (node ids sorted, matching Population.from_genomes packing).
    """
    sorted_ids = sorted(genome.nodes)
    id_to_slot = {nid: s for s, nid in enumerate(sorted_ids)}

    # always include hidden + output nodes
    included: set[int] = {
        nid for nid, ng in genome.nodes.items()
        if ng.type in (NODE_OUTPUT, NODE_HIDDEN)
    }

    enabled = [(innov, c) for innov, c in genome.conns.items() if c.enabled]
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

    nodes = []
    for nid in sorted(included):
        ng = genome.nodes.get(nid)
        if ng is None:
            continue
        nodes.append({"id": int(nid), "kind": _node_kind(ng.type)})
    node_ids = {n["id"] for n in nodes}
    return nodes, conns, id_to_slot, node_ids


# --------------------------------------------------------------------------
# champion showcase env
# --------------------------------------------------------------------------
class ChampionShowcase:
    """A dedicated env replaying the current champion from its eval spawn.

    Champions that earned their fitness from a Go-Explore frontier restore are
    replayed from that same restored state; a deterministic policy diverges
    wildly from a different start (a dialogue-masher restored mid-text looks
    like a couch potato from the silent newgame bedroom), so replaying from
    newgame would misrepresent almost every restored champion.
    """

    STUCK_STEPS = 40  # looping frames before the run restarts (fixed point)

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
        manifest=None,
    ) -> None:
        self.env = env
        self.device = device
        self.reset_state = reset_state
        self.forward_steps = int(forward_steps)
        self.max_nodes = int(max_nodes)
        self.max_conns = int(max_conns)
        self.manifest = manifest
        # Active-vision spine (§2/§3): the showcase drives a movable fovea, so it
        # needs its OWN stateful gaze/motion. The passed ``encoder`` is the
        # PARENT's FovealEncoder (its env slot 0 is used concurrently by the
        # solo champion/miner replay), so we build a PRIVATE single-env encoder
        # from the same vision knobs rather than share slot 0 and stomp its
        # gaze/motion/step state. Taps are synced off the parent each install.
        self._parent_encoder = encoder
        self.encoder = FovealEncoder(
            1,
            periph_grid=int(getattr(encoder, "G", 12)),
            fovea_native_px=int(getattr(encoder, "F", 48)),
            fovea_grid=int(getattr(encoder, "FG", getattr(encoder, "G", 12))),
            n_ram=int(getattr(encoder, "n_ram", 8)),
            saccade_gain=float(getattr(encoder, "gain", 32.0)),
            saccade_every_k=int(getattr(encoder, "every_k", 1)),
            screen_h=int(getattr(encoder, "H", _SCREEN_H)),
            screen_w=int(getattr(encoder, "W", _SCREEN_W)),
            shades=int(getattr(encoder, "shades", 4)),
            episode_steps=int(getattr(encoder, "episode_steps", 1024)),
        )
        self._G = int(self.encoder.G)
        self._F = int(self.encoder.F)
        # Explicit ram_addrs win; else route the UI taps through the manifest
        # (falling back to the quarantined Yellow table only when appropriate).
        if ram_addrs is not None:
            self.ram_addrs = list(ram_addrs)
        else:
            self.ram_addrs = ram_taps_from_manifest(manifest)
        self._sync_taps()

        self.genome = None
        self.genome_id = "none"
        self.cp = None
        self.n_out = N_BUTTONS  # full genome output width (set from cp on install)
        self._id_to_slot: dict[int, int] = {}
        self._net_nodes: list[dict] = []
        self._net_conns: list[dict] = []
        self._net_node_ids: set[int] = set()

        self.screen: np.ndarray | None = None
        self.wram: np.ndarray | None = None
        self._last_obs: np.ndarray | None = None  # last 454-d obs (block decode)
        self.obs_vis: np.ndarray | None = None     # periphery view (GxG, [0,1])
        self.fovea_vis: np.ndarray | None = None    # fovea crop view (GxG, [0,1])
        self.motion_vis: np.ndarray | None = None   # motion view (GxG, [0,1])
        self.last_action = 8  # NOOP until the first action lands (efference copy)
        self.last_probs: list[float] = [0.0] * N_BUTTONS  # 9 button values
        self.saccade: tuple[float, float] = (0.0, 0.0)  # raw (dx,dy) out[9]/out[10]
        self._gaze_disp: tuple[float, float] = (
            float(self.encoder.H) / 2.0,
            float(self.encoder.W) / 2.0,
        )  # (gy,gx) that cropped the displayed fovea
        # recent frame signatures: catches both dead-still fixed points and
        # short pixel loops (wall-bump animation is a 2-frame cycle).
        self._still_ring: deque = deque(maxlen=4)
        self._still_n = 0
        self._fis = 0  # frames into the current agent-step (advance() path)
        self._state = None  # recurrent node-state carried across cage steps
        self.spawn_state: bytes | None = None
        self._act: dict[str, float] = {}
        self._genes: dict = {"nodes": [], "conns": []}

    def _sync_taps(self) -> None:
        """Point the private encoder's RAM tail at the parent's live mined taps.

        The parent keeps its encoder tap-synced with the progress miner's counter
        addresses (``encoder.set_taps`` each miner cycle); mirror them so the
        showcase obs's ram block matches what the champion actually trained on.
        Falls back to the UI ``ram_addrs`` (manifest / quarantined Yellow) when
        the parent has none set yet."""
        taps = list(getattr(self._parent_encoder, "tap_addrs", []) or [])
        if not taps:
            taps = list(self.ram_addrs)
        try:
            self.encoder.set_taps(taps)
        except Exception:
            pass

    # -- champion swap -----------------------------------------------------
    def set_champion(
        self, genome, genome_id: str, spawn_state: bytes | None = None
    ) -> None:
        """Install a new champion genome and replay it from its eval spawn.

        ``spawn_state`` is the serialized emulator state the champion actually
        started its winning episode from (a Go-Explore frontier restore), or
        None for the canonical newgame state.
        """
        self.genome = genome.copy()
        self.genome_id = str(genome_id)
        self.spawn_state = spawn_state
        pop = Population.from_genomes(
            [self.genome], max_nodes=self.max_nodes, max_conns=self.max_conns
        )
        self.cp = pop.compile(self.device)
        self.n_out = int(self.cp.n_out)  # 11 (9 button + 2 saccade) for the spine
        self._sync_taps()  # pick up any taps the miner added since the last swap
        self._prepare_net()
        # Every install replays from the champion's own spawn. A deterministic
        # argmax policy in a deterministic env converges to a fixed point
        # within seconds of "continuous play", which just freezes the cage —
        # each install is a fresh honest attempt instead.
        self._reset_run()

    def _reset_run(self) -> None:
        self.encoder.reset(0)  # gaze -> centre (72,80), motion -> 0.5 (new episode)
        self.last_action = 8   # NOOP: no button applied yet (efference copy)
        if self.spawn_state is not None:
            self.env.load_state(self.spawn_state)
            self.screen = self.env.reset(None)  # clear held input + settle a frame
        else:
            self.screen = self.env.reset(self.reset_state)
        self.wram = self.env.raw_wram()
        # Seed a valid first-frame obs view (motion -> 0.5 on frame 1 by design).
        x0 = self.encoder.encode(0, self.screen, self.wram, button=self.last_action)
        self._set_obs_views(x0)
        self.saccade = (0.0, 0.0)
        self._gaze_disp = self.encoder.gaze(0)
        self._still_ring.clear()
        self._still_n = 0
        self._fis = 0  # frames into the current agent-step (advance() path)
        self._state = None  # new episode → clear recurrent memory

    def _set_obs_views(self, obs_vec) -> None:
        """Decode the periphery / fovea / motion optical views from a foveal obs."""
        self._last_obs = obs_vec
        g = self._G
        self.obs_vis = _block2d(obs_vec, g, "periph")
        self.fovea_vis = _block2d(obs_vec, g, "fovea")
        self.motion_vis = _block2d(obs_vec, g, "motion")

    def _prepare_net(self) -> None:
        """Pick a bounded, self-consistent subgraph (hidden+out + strong inputs)."""
        nodes, conns, id_to_slot, node_ids = build_net_view(self.genome)
        self._net_nodes = nodes
        self._net_conns = conns
        self._id_to_slot = id_to_slot
        self._net_node_ids = node_ids
        self._genes = build_genes(self.genome)

    # -- stepping ----------------------------------------------------------
    def _forward_choose(self):
        """One captured forward pass on the current obs; drives the movable fovea.

        Active-vision spine (§3): foveal-encode the current screen (gaze +
        efference copy of the last applied button), forward the champion to an
        11-d output, split into the Discrete-9 button + raw (dx,dy) saccade, then
        steer the NEXT step's fovea via ``update_gaze`` (the one-tick efference
        delay is intentional — the saccade from obs t crops the fovea of t+1).
        Returns ``(act_state, out)`` for the caller to publish probs/acts.
        """
        x = self.encoder.encode(0, self.screen, self.wram, button=int(self.last_action))
        self._set_obs_views(x)
        xt = torch.from_numpy(x[None, :]).to(self.device).unsqueeze(1)  # (1,1,dim)
        # seed with the previous step's node state so the cage carries the same
        # cross-step recurrent memory the champion had in training
        act_state = _forward_capture(
            self.cp, xt, self.forward_steps, state=self._state
        )  # (1,1,M)
        self._state = act_state
        out = act_state[0, 0, self.cp.n_in + 1 : self.cp.n_in + 1 + self.n_out]
        outv = out.detach().to("cpu").numpy()
        button, dx, dy = _split_head(outv)
        self.last_action = int(button)
        self.saccade = (dx, dy)
        # gaze that cropped THIS obs (captured before update moves it for t+1)
        self._gaze_disp = self.encoder.gaze(0)
        self.encoder.update_gaze(0, dx, dy)  # steer next fovea (tanh applied inside)
        return act_state, out

    def _publish_forward(self, act_state, out) -> None:
        """Expose the last forward's button probs + per-node activations (slots)."""
        # Render the 9 button values only (out[:9]); the 2 saccade outputs are
        # surfaced separately (payload ``saccade`` / gaze), never as buttons.
        self.last_probs = [round(float(v), 4) for v in out[:N_BUTTONS].tolist()]
        vec = act_state[0, 0].detach().to("cpu").numpy()
        self._act = {
            str(nid): round(float(vec[slot]), 4)
            for nid, slot in self._id_to_slot.items()
            if slot < vec.shape[0] and int(nid) in self._net_node_ids
        }

    def _stuck_check(self) -> bool:
        """Fixed-point detector, run once per completed agent-step.

        A deterministic policy that wedges itself into a wall produces
        pixel-identical frames — or a tight 2-frame bump loop — forever.
        After STUCK_STEPS agent-steps whose frames all recur within the last
        few signatures, restart the run so the cage stays alive (~16 s at
        realtime pace, ~7 s at max). Genuine play (walking, text advancing)
        mints fresh signatures and keeps resetting the counter.
        Returns True iff the run was restarted.
        """
        sig = self.screen[::8, ::8].tobytes() if self.screen is not None else None
        if sig is not None and sig in self._still_ring:
            self._still_n += 1
            if self._still_n >= self.STUCK_STEPS:
                self._reset_run()
                return True
        else:
            self._still_n = 0
        if sig is not None:
            self._still_ring.append(sig)
        return False

    def step(self, n_steps: int = 2) -> None:
        """Advance the showcase env ``n_steps`` full agent-steps (max pace)."""
        if self.cp is None or self.screen is None:
            return
        for _ in range(max(1, n_steps)):
            act_state, out = self._forward_choose()
            self.screen, self.wram, _done, _info = self.env.step(self.last_action)
        self._fis = 0  # step() always ends on an agent-step boundary
        if self._stuck_check():
            return
        self._publish_forward(act_state, out)
        # obs/fovea views + gaze are set inside _forward_choose (from the exact
        # obs the champion acted on) — no stateful re-encode here (that would
        # corrupt the encoder's motion/prev-periphery state).

    def advance(self, n_frames: int) -> None:
        """Advance ``n_frames`` game frames (realtime pace: 60/s of wall clock).

        The net still decides once per ``env.frame_skip`` frames — identical
        dynamics to :meth:`step` — but the screen is rendered at every call,
        so a 10 Hz emit cadence shows 10 fresh frames/s instead of one
        24-frame gulp every 400 ms (the cage played at 2.5 fps before this).
        """
        if self.cp is None or self.screen is None or n_frames <= 0:
            return
        fs = max(1, int(getattr(self.env, "frame_skip", 24)))
        act_state = out = None
        while n_frames > 0:
            if self._fis == 0:
                act_state, out = self._forward_choose()
                # hold() may consume frames re-tapping an edge-read button;
                # count them into the step so dynamics match training.
                used = int(self.env.hold(self.last_action) or 0)
                if used:
                    self._fis = min(used, fs - 1)
                    n_frames -= min(used, n_frames)
                    if n_frames <= 0:
                        break
            run = min(n_frames, fs - self._fis)
            self.screen = self.env.tick_frames(run)
            self.wram = self.env.raw_wram()
            self._fis = (self._fis + run) % fs
            n_frames -= run
            if self._fis == 0 and self._stuck_check():
                return
        if out is not None:
            self._publish_forward(act_state, out)
        # obs/fovea views + gaze track the last forward (set in _forward_choose);
        # within an agent-step the gaze is fixed, so the fovea box holds steady
        # while realtime frames render, then jumps at the next step boundary.

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
        buttons = [0] * N_BUTTONS
        if 0 <= self.last_action < N_BUTTONS:
            buttons[self.last_action] = 1
        obs_gray = (
            np.clip(self.obs_vis, 0.0, 1.0) * 255.0 if self.obs_vis is not None else None
        )
        gy, gx = self._gaze_disp
        dx, dy = self.saccade
        return {
            "genome_id": self.genome_id,
            "spawn": "frontier" if self.spawn_state is not None else "newgame",
            "frame_w": _SCREEN_W,
            "frame_h": _SCREEN_H,
            "frame_b64": b64_gray(self.screen) if self.screen is not None else "",
            # optical view: the low-res periphery the agent actually sees (GxG).
            "obs_res": int(self._G),
            "obs_b64": b64_gray(obs_gray) if obs_gray is not None else "",
            # active vision: the high-acuity fovea crop + where it is looking +
            # the raw saccade command that will move it next step.
            "fovea_res": int(self._G),
            "fovea_px": int(self._F),
            "fovea_b64": _b64_block(self.fovea_vis) if self.fovea_vis is not None else "",
            "motion_b64": _b64_block(self.motion_vis) if self.motion_vis is not None else "",
            "gaze": _gaze_payload(gy, gx, _SCREEN_H, _SCREEN_W, self._F),
            "saccade": _saccade_payload(dx, dy),
            "action": int(self.last_action),
            "buttons": buttons,
            "ram": self._ram_bytes(),
            "net": {
                "n_in": int(self.encoder.dim),
                "n_out": int(self.n_out),
                "nodes": self._net_nodes,
                "conns": self._net_conns,
                "act": self._act,
            },
        }


# --------------------------------------------------------------------------
# top-level streamer
# --------------------------------------------------------------------------
class LiveStreamer:
    """Bundles the writer + showcase and emits the full live.json payload.

    Beyond the champion + swarm sample, it implements the *focus agent*
    protocol: the dashboard writes ``runs/<id>/select.json`` (via
    ``/api/select``) naming a player slot; each emit the streamer cheaply
    stat-polls that file and streams the REAL brain of the focused agent — a
    capture forward with that genome on that agent's current obs (activations
    + output probs), its full-res frame from the shm screens block, and its
    raw genotype.  All focus work runs at emit rate (~3 Hz), never per round.

    The feed NEVER freezes: during waves the loop thread emits between
    barrier rounds (throttled to ~``hz``) exactly as before; a dedicated
    *pump thread* watches for gaps and takes over whenever no emit has landed
    for a full interval — i.e. through generation boundaries (speciation /
    reproduction / frontier restore), where the champion showcase keeps
    playing since it owns its own env, independent of the fleet.  The
    training loop marks where it is via
    :meth:`set_phase`; the payload carries top-level ``phase`` ("wave" |
    "evolving") and ``phase_detail`` fields for the dashboard.  ``_emit`` and
    champion swaps are serialized by an RLock; the wave-loop hooks
    (:meth:`maybe_write`, :meth:`begin_wave`, :meth:`set_phase`) only do
    atomic reference swaps and never block on an in-flight emit.
    """

    def __init__(
        self,
        run_dir: str | Path,
        showcase: ChampionShowcase,
        run_id: str,
        hz: float = 3.0,
        swarm_cap: int = 32,
        elite_k: int = 4,
        champ_steps: int = 2,
        archive=None,
        goexplore=None,
        champ_dwell: float = 5.0,
    ) -> None:
        self.writer = LiveWriter(run_dir, hz)
        self.showcase = showcase
        self.run_id = str(run_id)
        self.swarm_cap = int(swarm_cap)
        self.elite_k = int(elite_k)
        self.champ_steps = int(champ_steps)
        self.last_size = 0
        # -- pace (spectate mode): emit cadence + payload fields --------------
        self._hz_max = float(hz)  # base cadence in "max" mode (~3 Hz)
        self._hz_realtime = 10.0  # cadence in "realtime" spectate mode
        self._pace = "max"
        self._eta_s: float | None = None  # loop-fed seconds to next breeding
        self._round_hz = 0.0  # loop-fed measured round rate
        self._n_waves = 1
        # At 10 Hz the heavy focus work (capture forward + genes) runs on a
        # ~3 Hz subcadence; the cached payload is reused between rebuilds.
        self._focus_min_dt = 1.0 / 3.0
        self._focus_last = 0.0
        self._focus_payload: dict | None = None
        # Realtime: the champion showcase env is paced to authentic GB speed —
        # frame-accurate (60 game frames per wall second, rendered per emit)
        # instead of one 24-frame agent-step gulp every 400 ms.
        self._champ_fs = int(getattr(getattr(showcase, "env", None), "frame_skip", 24))
        self._champ_step_ts = 0.0
        # -- champion dwell: a freshly-installed champ is displayed >= champ_dwell
        # seconds before a newer one may replace it (latest pending wins).
        self.champ_dwell = float(champ_dwell)
        self._champ_since: float | None = None  # unix ts of current install
        self._champ_gen = -1
        self._champ_fitness = 0.0
        self._pending_champ: tuple | None = None  # (genome, id, gen, fitness)
        # (screens, fitness, dead, obs, actions) from the most recent sample
        self._cache = None
        # -- side panels (loop-provided at generation boundaries) ----------
        self.archive = archive  # NoveltyArchive (read-only)
        self.goexplore = goexplore  # GoExplore or None (read-only)
        self._reward_terms: dict = {}
        self._species: list = []
        # -- focus selection ------------------------------------------------
        self._select_path = Path(run_dir) / "select.json"
        self._select_sig = None  # (mtime_ns, size) of last parsed select.json
        self._select_idx = -1  # -1 = champion (default)
        # -- current wave context --------------------------------------------
        self._wave_genomes: list = []
        self._wave_gen = 0
        self._wave_offset = 0  # index of slot 0 within the population
        self._wave_idx = 0  # wave counter within the generation
        self._round = 0  # round counter within the episode
        self._focus_cache: dict[int, dict] = {}  # slot -> compiled focus entry
        # -- pump thread: emits at ~hz THROUGH generation boundaries ----------
        self._gen = 0
        self._play_seconds = 0.0  # cumulative game-time played across all agents
        self._phase = "wave"
        self._phase_detail = ""
        self._lock = threading.RLock()  # serializes _emit / champion swaps
        self._pump_stop = threading.Event()
        self._pump = threading.Thread(
            target=self._pump_loop, name="live-pump", daemon=True
        )
        self._pump.start()

    def set_champion(
        self, genome, genome_id: str, gen: int = -1, fitness: float = 0.0,
        spawn_state: bytes | None = None,
    ) -> None:
        """Offer a new champion; installed now or after the dwell period.

        The current champion keeps the showcase for at least ``champ_dwell``
        seconds.  Offers arriving inside the dwell window are stashed (the
        LATEST offer wins — intermediates are skipped) and promoted by the next
        emit once the dwell has elapsed.  ``spawn_state`` is the emulator state
        the champion's winning episode actually started from (a Go-Explore
        frontier restore), or None for newgame.
        """
        now = time.time()
        if (
            self._champ_since is None
            or (now - self._champ_since) >= self.champ_dwell
        ):
            self._install_champion(genome, genome_id, gen, fitness, now, spawn_state)
        else:
            # copy: the loop's genome object may be recycled by reproduce()
            self._pending_champ = (genome.copy(), str(genome_id), int(gen),
                                   float(fitness), spawn_state)

    def _install_champion(
        self, genome, genome_id: str, gen: int, fitness: float, now: float,
        spawn_state: bytes | None = None,
    ) -> None:
        with self._lock:  # never swap the showcase net mid-emit
            self.showcase.set_champion(genome, genome_id, spawn_state)
            self._champ_since = now
            self._champ_gen = int(gen)
            self._champ_fitness = float(fitness)
            self._pending_champ = None

    def _maybe_promote_pending(self) -> None:
        """Install the stashed (latest) champion once the dwell has elapsed."""
        if self._pending_champ is None:
            return
        now = time.time()
        if (
            self._champ_since is None
            or (now - self._champ_since) >= self.champ_dwell
        ):
            g, gid, gen, fit, spawn = self._pending_champ
            self._install_champion(g, gid, gen, fit, now, spawn)

    # -- loop-facing context hooks ------------------------------------------
    def begin_wave(
        self, genomes, gen: int, offset: int, wave_idx: int, n_waves: int = 1
    ) -> None:
        """New wave: slot -> genome mapping changes, drop compiled focus state."""
        self._wave_genomes = list(genomes)
        self._wave_gen = int(gen)
        self._wave_offset = int(offset)
        self._wave_idx = int(wave_idx)
        self._n_waves = max(1, int(n_waves))
        self._round = 0
        self._focus_cache.clear()
        self._focus_payload = None  # stale slot mapping: force a focus rebuild

    def set_pace(self, mode: str) -> None:
        """Live pace flip from the loop's PaceController (thread-safe: plain
        attribute/float writes).  realtime -> ~10 Hz emits; max -> base ~3 Hz.
        The pump thread re-reads the interval every tick, so the cadence
        adapts without a restart."""
        self._pace = "realtime" if mode == "realtime" else "max"
        hz = self._hz_realtime if self._pace == "realtime" else self._hz_max
        self.writer.interval = 1.0 / max(0.1, hz)

    def set_play_seconds(self, secs: float) -> None:
        """Cumulative game-time played across ALL agents (for the wall's
        'Play Years' tile). One agent-step = frame_skip/60 game-seconds;
        summed over every parallel agent and every generation."""
        self._play_seconds = float(secs)

    def set_eta(self, eta_s: float, round_hz: float) -> None:
        """Per-round countdown feed from the loop (cheap attribute writes):
        seconds until the next breeding event + measured round rate."""
        self._eta_s = float(eta_s)
        self._round_hz = float(round_hz)

    def set_side_stats(self, reward_terms: dict, species: list) -> None:
        """Generation-boundary side-panel payloads (already plain python)."""
        self._reward_terms = dict(reward_terms)
        self._species = list(species)

    def set_phase(self, phase: str, detail: str = "") -> None:
        """Mark the loop's current phase ("wave" | "evolving") for the feed."""
        self._phase = str(phase)
        self._phase_detail = str(detail)

    def maybe_write(
        self,
        gen: int,
        screens: list[np.ndarray],
        fitness: np.ndarray,
        dead: list[bool] | None = None,
        obs: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        round_t: int = 0,
    ) -> bool:
        """Cache the freshest wave sample for the pump thread (non-blocking).

        ``obs`` is the (n, obs_dim) batch the wave's actions were computed
        from and ``actions`` the resulting per-slot action ints — both are
        what the focus capture replays for the selected agent.  During waves
        the emit happens right here on the loop thread (throttled to ~hz,
        exactly the pre-pump behaviour); the pump thread only fills the gaps
        when the loop stops calling (generation boundaries).
        """
        self._cache = (screens, fitness, dead, obs, actions)
        self._round = int(round_t)
        self._gen = int(gen)
        if not self.writer.due():
            return False
        return self._emit(gen)

    def force_write(self, gen: int) -> bool:
        """Immediate synchronous write (e.g. at a generation boundary)."""
        self._gen = int(gen)
        return self._emit(gen, force=True)

    # -- pump thread --------------------------------------------------------
    def _pump_loop(self) -> None:
        """Keep the feed alive through generation boundaries.

        Polls at a fraction of the emit interval; whenever no emit has landed
        for a full interval (the wave loop normally beats it to the punch),
        emits one itself.  So during waves this thread is idle and the cadence
        is the loop thread's ~hz; during boundaries (speciation, reproduction,
        frontier restore) it takes over and the champion showcase — which owns
        its env, independent of the fleet — keeps playing.
        """
        warned = False
        # tick re-read each cycle: the pace toggle changes writer.interval live
        while not self._pump_stop.wait(self.writer.interval / 3.0):
            if not self.writer.due():
                continue
            try:
                self._emit(self._gen)
            except Exception:
                # The pump must never die mid-boundary; report the first hit.
                if not warned:
                    warned = True
                    import traceback

                    print("[live] pump emit failed (feed continues):", flush=True)
                    traceback.print_exc()

    def close(self) -> None:
        """Stop the pump + writer threads (flushes the last payload)."""
        self._pump_stop.set()
        try:
            self._pump.join(timeout=2)
        except Exception:
            pass
        self.writer.close()

    # -- select.json polling ----------------------------------------------
    def _poll_select(self) -> None:
        """Cheap stat each emit; parse only when the file changed."""
        try:
            st = self._select_path.stat()
        except OSError:
            self._select_idx = -1
            self._select_sig = None
            return
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._select_sig:
            return
        try:
            data = json.loads(self._select_path.read_bytes())
            self._select_idx = int(data.get("idx", -1))
            self._select_sig = sig
        except Exception:
            # mid-write/corrupt: keep the previous selection, retry next emit
            pass

    # -- focus payloads -----------------------------------------------------
    def _focus_champion(self) -> dict:
        sc = self.showcase
        buttons = [0] * N_BUTTONS
        if 0 <= sc.last_action < N_BUTTONS:
            buttons[sc.last_action] = 1
        obs_gray = (
            np.clip(sc.obs_vis, 0.0, 1.0) * 255.0 if sc.obs_vis is not None else None
        )
        gy, gx = sc._gaze_disp
        dx, dy = sc.saccade
        return {
            "idx": -1,
            "genome_id": sc.genome_id,
            "frame_b64": b64_gray(sc.screen) if sc.screen is not None else "",
            "obs_res": int(sc._G),
            "obs_b64": b64_gray(obs_gray) if obs_gray is not None else "",
            "fovea_res": int(sc._G),
            "fovea_px": int(sc._F),
            "fovea_b64": _b64_block(sc.fovea_vis) if sc.fovea_vis is not None else "",
            "motion_b64": _b64_block(sc.motion_vis) if sc.motion_vis is not None else "",
            "gaze": _gaze_payload(gy, gx, _SCREEN_H, _SCREEN_W, sc._F),
            "saccade": _saccade_payload(dx, dy),
            "action": int(sc.last_action),
            "buttons": buttons,
            "probs": list(sc.last_probs),
            "net": {
                "n_in": int(sc.encoder.dim),
                "n_out": int(sc.n_out),
                "nodes": sc._net_nodes,
                "conns": sc._net_conns,
                "act": sc._act,
            },
            "genes": sc._genes,
        }

    def _focus_entry(self, idx: int) -> dict:
        """Compiled single-genome forward state for slot ``idx`` (cached per wave)."""
        entry = self._focus_cache.get(idx)
        if entry is not None:
            return entry
        sc = self.showcase
        g = self._wave_genomes[idx]
        pop = Population.from_genomes(
            [g], max_nodes=sc.max_nodes, max_conns=sc.max_conns
        )
        nodes, conns, id_to_slot, node_ids = build_net_view(g)
        entry = {
            "cp": pop.compile(sc.device),
            "nodes": nodes,
            "conns": conns,
            "id_to_slot": id_to_slot,
            "node_ids": node_ids,
            "genes": build_genes(g),
            "genome_id": f"gen{self._wave_gen}_g{self._wave_offset + idx}",
        }
        self._focus_cache[idx] = entry
        return entry

    def _focus_agent(self, idx: int, screens, obs, actions) -> dict:
        sc = self.showcase
        entry = self._focus_entry(idx)
        cp = entry["cp"]

        # Capture forward: THAT genome on THAT agent's current obs (~3 Hz only).
        x = np.ascontiguousarray(obs[idx], dtype=np.float32)
        xt = torch.from_numpy(x[None, :]).to(sc.device).unsqueeze(1)  # (1,1,dim)
        act_state = _forward_capture(cp, xt, sc.forward_steps)  # (1,1,M)
        out = act_state[0, 0, cp.n_in + 1 : cp.n_in + 1 + cp.n_out]
        outv = out.detach().to("cpu").numpy()
        probs = [round(float(v), 4) for v in outv[:N_BUTTONS]]  # 9 button values
        _btn, gdx, gdy = _split_head(outv)  # this genome's saccade command
        vec = act_state[0, 0].detach().to("cpu").numpy()
        node_ids = entry["node_ids"]
        act = {
            str(nid): round(float(vec[slot]), 4)
            for nid, slot in entry["id_to_slot"].items()
            if slot < vec.shape[0] and int(nid) in node_ids
        }

        action = int(actions[idx])  # the button the fleet actually applied
        buttons = [0] * N_BUTTONS
        if 0 <= action < N_BUTTONS:
            buttons[action] = 1
        g = sc._G
        # Optical views decoded straight from the stored 454-d obs the fleet fed
        # this agent; gaze recovered from its proprio block (where it was looking).
        obs_img = _block2d(obs[idx], g, "periph")
        gy, gx = _gaze_from_proprio(obs[idx], g, sc.encoder.H, sc.encoder.W)
        return {
            "idx": int(idx),
            "genome_id": entry["genome_id"],
            "frame_b64": b64_gray(screens[idx]),
            "obs_res": int(g),
            "obs_b64": _b64_block(obs_img),
            "fovea_res": int(g),
            "fovea_px": int(sc._F),
            "fovea_b64": _b64_block(_block2d(obs[idx], g, "fovea")),
            "motion_b64": _b64_block(_block2d(obs[idx], g, "motion")),
            "gaze": _gaze_payload(gy, gx, _SCREEN_H, _SCREEN_W, sc._F),
            "saccade": _saccade_payload(gdx, gdy),
            "action": action,
            "buttons": buttons,
            "probs": probs,
            "net": {
                "n_in": int(sc.encoder.dim),
                "n_out": int(cp.n_out),
                "nodes": entry["nodes"],
                "conns": entry["conns"],
                "act": act,
            },
            "genes": entry["genes"],
        }

    def _build_focus(self, screens, obs, actions) -> dict:
        idx = self._select_idx
        if (
            idx is None
            or idx < 0
            or idx >= len(self._wave_genomes)
            or screens is None
            or obs is None
            or actions is None
            or idx >= len(screens)
            or idx >= len(obs)
            or idx >= len(actions)
        ):
            return self._focus_champion()
        try:
            return self._focus_agent(idx, screens, obs, actions)
        except Exception:
            return self._focus_champion()

    # -- side panels ----------------------------------------------------------
    def _reward_terms_payload(self, fitness) -> dict:
        terms = dict(self._reward_terms)
        if fitness is not None and len(fitness):
            terms["novelty"] = round(float(np.max(fitness)), 3)
        if not terms:
            terms = {"novelty": 0.0}
        return terms

    def _archive_payload(self) -> dict:
        a = self.archive
        g = self.goexplore
        out = {
            "cells": int(a.size) if a is not None else 0,
            "gen_delta": int(a.generation_delta) if a is not None else 0,
            "restores": 0,
            "captured": 0,
            "max_depth": 0,
        }
        if g is not None:
            out["restores"] = int(g.n_restores)
            out["captured"] = int(g.n_captured)
            out["max_depth"] = int(max((e.depth for e in g.cells.values()), default=0))
        return out

    # -- emit -----------------------------------------------------------------
    def _emit(self, gen: int, force: bool = False) -> bool:
        _prof = os.environ.get("POKEIO_LIVE_PROF") == "1"
        _t0 = time.perf_counter() if _prof else 0.0
        with self._lock:
            # Loop thread + pump thread can race to the same due() window; the
            # loser rechecks under the lock and skips the duplicate emit.
            if not force and not self.writer.due():
                return False
            self._maybe_promote_pending()
            if self._pace == "realtime":
                # Spectate mode: the champion cage plays at authentic GB speed,
                # frame-accurately — advance however many 60 Hz game frames of
                # wall clock have elapsed and render, so every emit carries
                # fresh pixels (smooth ~10 fps) instead of one 24-frame gulp
                # every 400 ms (2.5 fps).
                _now = time.monotonic()
                frames = int((_now - self._champ_step_ts) * 60.0)
                cap = 2 * self._champ_fs  # bound catch-up after a stall/boundary
                if frames > cap:
                    frames = cap
                    self._champ_step_ts = _now - frames / 60.0
                if frames > 0:
                    self.showcase.advance(frames)
                    self._champ_step_ts += frames / 60.0
            else:
                self.showcase.step(self.champ_steps)
            _t1 = time.perf_counter() if _prof else 0.0
            self._poll_select()
            # snapshot mutable refs once (the loop thread swaps them atomically)
            cache = self._cache
            phase = self._phase
            phase_detail = self._phase_detail
            screens = fitness = dead = obs = actions = None
            if cache is not None:
                screens, fitness, dead, obs, actions = cache
                swarm = build_swarm(
                    screens, fitness, dead, self.swarm_cap, self.elite_k
                )
            else:
                swarm = []
            champ = self.showcase.payload()
            champ["champ_since"] = self._champ_since  # unix ts of install (UI tenure)
            champ["champ_gen"] = int(self._champ_gen)
            champ["fitness"] = float(self._champ_fitness)
            # Heavy focus (capture forward + genes) at a ~3 Hz subcadence when
            # emitting at 10 Hz; swarm/champion stay per-tick.
            _now = time.monotonic()
            if (
                self._pace == "realtime"
                and self._focus_payload is not None
                and (_now - self._focus_last) < self._focus_min_dt
            ):
                focus = self._focus_payload
            else:
                focus = self._build_focus(screens, obs, actions)
                self._focus_payload = focus
                self._focus_last = _now
            # eta to the next breeding event: 0 while the boundary itself runs
            # (phase="evolving" covers it in the UI).
            if phase == "evolving" or self._eta_s is None:
                eta_s = 0.0
            else:
                eta_s = max(0.0, float(self._eta_s))
            payload = {
                "t": time.time(),
                "gen": int(gen),
                "run": self.run_id,
                "phase": phase,
                "phase_detail": phase_detail,
                "pace": self._pace,
                "eta_s": round(eta_s, 2),
                "round_hz": round(float(self._round_hz), 3),
                "waves": int(self._n_waves),
                "play_seconds": round(float(self._play_seconds), 1),
                "champion": champ,
                "swarm": swarm,
                "focus": focus,
                "reward_terms": self._reward_terms_payload(fitness),
                "archive": self._archive_payload(),
                "species": self._species,
                "wave": int(self._wave_idx),
                "round": int(self._round),
            }
        self.last_size = self.writer.write(payload)
        if _prof:
            _t2 = time.perf_counter()
            print(
                f"[live-prof] emit={_t2 - _t0:.4f}s "
                f"(showcase={_t1 - _t0:.4f}s focus+payload={_t2 - _t1:.4f}s) "
                f"idx={self._select_idx}",
                flush=True,
            )
        return True


__all__ = [
    "LiveWriter",
    "ChampionShowcase",
    "LiveStreamer",
    "build_swarm",
    "build_genes",
    "build_net_view",
    "b64_gray",
    "downscale_swarm",
    "INTERESTING_RAM",
    "ram_taps_from_manifest",
]
