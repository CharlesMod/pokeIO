"""Lane B sanity check: a CPPN can EXPRESS spatial structure, and the full
CPPN -> painted weights -> sparse phenotype forward path runs on the P100.

Run::

    cd /home/cmod/pokeIO && . .venv/bin/activate
    PYTHONPATH=/home/cmod/pokeIO python scripts/test_cppn.py

What it does
------------
1. Hand-builds a centre-surround (Difference-of-Gaussians) CPPN from ONLY the
   canonical inputs (x1,y1,x2,y2,bias) + gauss nodes (no egocentric prior).
2. Paints the weight field from every screen pixel to a target node and renders
   it as an ASCII heatmap (positive centre, negative surround) — proving the
   spatial receptive field is expressible.  Saves a PNG if matplotlib is present.
3. Paints the whole encoder substrate and runs population_forward_sparse for a
   batch of CPPN genomes on cuda:1, confirming the full path executes on-GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from pokeio.evo.cppn import (
    hand_center_surround_cppn,
    paint_dense,
    paint_substrate,
    receptive_field,
)
from pokeio.evo.forward import population_forward_sparse
from pokeio.evo.substrate import BUTTON_NAMES, Substrate

# A diverging ASCII ramp: strong-negative ... near-zero ... strong-positive.
_NEG = "WX#=c-,·"   # magnitude high -> low  (negative lobe)
_POS = "·.:+o*%@"   # magnitude low  -> high (positive lobe)


def ascii_heat(img: np.ndarray) -> str:
    """Diverging ASCII heatmap of a signed 2-D field (normalised by max |.|)."""
    m = float(np.max(np.abs(img))) or 1.0
    lines = []
    for row in img:
        chars = []
        for v in row:
            t = float(v) / m  # in [-1, 1]
            if t >= 0:
                idx = min(len(_POS) - 1, int(t * len(_POS)))
                chars.append(_POS[idx])
            else:
                idx = min(len(_NEG) - 1, int((-t) * len(_NEG)))
                chars.append(_NEG[len(_NEG) - 1 - idx])
        lines.append(" ".join(chars))
    return "\n".join(lines)


def sign_map(img: np.ndarray, frac: float = 0.12) -> str:
    """Coarse sign map: '+' centre lobe, '-' surround lobe, ' ' near zero."""
    m = float(np.max(np.abs(img))) or 1.0
    thr = frac * m
    rows = []
    for row in img:
        rows.append(" ".join("+" if v > thr else "-" if v < -thr else "." for v in row))
    return "\n".join(rows)


def maybe_save_png(img: np.ndarray, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = float(np.max(np.abs(img))) or 1.0
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(img, cmap="RdBu_r", vmin=-m, vmax=m, extent=(-1, 1, -1, 1))
        ax.set_title("CPPN-painted centre-surround\nreceptive field")
        ax.set_xlabel("x1")
        ax.set_ylabel("y1")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception as exc:  # matplotlib optional
        print(f"[png] skipped ({type(exc).__name__}: {exc})")
        return None


def pick_device(prefer: str = "cuda:1") -> torch.device:
    if torch.cuda.is_available():
        idx = int(prefer.split(":")[1]) if ":" in prefer else 0
        return torch.device(prefer if idx < torch.cuda.device_count() else "cuda:0")
    return torch.device("cpu")


def main() -> None:
    device = pick_device("cuda:1")
    print(f"[env] torch {torch.__version__}  device={device} "
          f"({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")

    # --- substrate --------------------------------------------------------
    sub = Substrate(grid=24, ram_dim=32, hidden=8, n_out=8)
    print(
        f"\n[substrate] grid={sub.grid}x{sub.grid} (screen={sub.n_screen}) "
        f"+ ram_aux={sub.ram_dim}  ->  n_in={sub.n_in}\n"
        f"            hidden={sub.hidden}x{sub.hidden} ({sub.n_hidden})  "
        f"outputs={sub.n_out} {BUTTON_NAMES}\n"
        f"            M(slots)={sub.M}  potential edges={sub.n_potential_edges():,}"
    )
    for t in sub.transitions():
        print(f"            transition {t.name:16s}: "
              f"{t.src_slots.shape[0]:>5d} x {t.tgt_slots.shape[0]:<4d} pairs")

    # --- 1. spatial expressivity: centre-surround receptive field ---------
    cppn = hand_center_surround_cppn(kn=4.0, kw=1.7, a_gain=1.4, b_gain=1.0)
    print(f"\n[cppn] hand-built centre-surround: "
          f"{len(cppn.nodes)} nodes, {len(cppn.conns)} conns "
          f"(n_in={cppn.n_in} -> n_out={cppn.n_out})")

    rf = receptive_field(cppn, sub, target_xy=(0.0, 0.0), device=device)
    print(f"\n[receptive field @ target=(0,0)]  "
          f"min={rf.min():+.3f} max={rf.max():+.3f} "
          f"centre={rf[sub.grid // 2, sub.grid // 2]:+.3f}\n")
    print(ascii_heat(rf))
    print("\n[sign map]  '+' centre lobe   '-' surround lobe   '.' ~zero\n")
    print(sign_map(rf))

    # centre-surround assertions
    centre = rf[sub.grid // 2, sub.grid // 2]
    corner = rf[0, 0]
    ring = rf[sub.grid // 2, sub.grid // 2 + 4]  # a few cells off-centre
    print(f"\n[check] centre={centre:+.3f} (want >0)   "
          f"ring={ring:+.3f} (want <0)   corner={corner:+.3f} (want ~0)")
    assert centre > 0 and ring < 0, "centre-surround structure did not emerge!"
    print("[check] PASS: positive centre + negative surround => centre-surround emerged.")

    # translation covariance: move the target, the spot follows it
    rf_off = receptive_field(cppn, sub, target_xy=(0.5, -0.5), device=device)
    peak_r, peak_c = np.unravel_index(int(np.argmax(rf_off)), rf_off.shape)
    px = -1 + 2 * peak_c / (sub.grid - 1)
    py = 1 - 2 * peak_r / (sub.grid - 1)
    print(f"[check] target moved to (0.5,-0.5): painted peak at "
          f"(x={px:+.2f}, y={py:+.2f}) -> field is translation-covariant.")

    png = maybe_save_png(rf, "/tmp/claude-1000/-home-cmod-pokeIO/"
                             "f3e4f4ac-87ed-47a6-a1c4-e9b7e87097ca/scratchpad/rf.png")
    if png:
        print(f"[png] saved receptive field to {png}")

    # --- 2. full path: paint substrate -> sparse phenotype forward --------
    dense = paint_dense(cppn, sub, device=device, threshold=0.05, weight_scale=3.0)
    nnz = int((np.abs(dense) > 0).sum())
    print(f"\n[dense paint] W shape={dense.shape}  nonzero edges={nnz:,} "
          f"(of {sub.n_potential_edges():,} potential)")

    # a small population of CPPNs -> phenotypes.  Use a modest weight_scale: the
    # substrate has ~600-way fan-in, so a large scale saturates tanh (every node
    # -> +/-1).  Real integration wants fan-in normalisation (see TODO in report).
    n_pop = 8
    rng = np.random.default_rng(0)
    cppns = [
        hand_center_surround_cppn(
            kn=float(rng.uniform(3.0, 5.0)),
            kw=float(rng.uniform(1.3, 2.1)),
            a_gain=float(rng.uniform(1.1, 1.6)),
            b_gain=float(rng.uniform(0.8, 1.1)),
        )
        for _ in range(n_pop)
    ]
    cp = paint_substrate(cppns, sub, device=device, threshold=0.05,
                         weight_scale=0.15, steps=8)
    active = int(cp.conn_valid[0].sum())
    print(f"[paint_substrate] population={cp.n}  edges/genome={cp.conn_weight.shape[1]:,}  "
          f"active(after threshold)={active:,}  device={cp.device}")

    # random observations at the substrate input dim, then the sparse forward
    obs = torch.rand(n_pop, 1, sub.n_in, device=device)  # (N, B=1, n_in)
    out = population_forward_sparse(cp, obs, steps=6)     # (N, 1, n_out)
    print(f"[forward] population_forward_sparse OK on {out.device}  "
          f"out shape={tuple(out.shape)}  "
          f"range=[{out.min().item():+.3f}, {out.max().item():+.3f}]")
    assert out.shape == (n_pop, 1, sub.n_out)
    assert torch.isfinite(out).all(), "non-finite outputs from the phenotype forward!"
    argmax = out[:, 0, :].argmax(dim=1).tolist()
    print(f"[forward] per-genome argmax button: "
          f"{[BUTTON_NAMES[i] for i in argmax]}")
    print("\n[DONE] CPPN -> weight tensor -> population_forward_sparse ran on "
          f"{out.device} with no error.")


if __name__ == "__main__":
    main()
