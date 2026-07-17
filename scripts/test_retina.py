"""Lane C retina smoke — DECODER-FREE (reconstruction viewer retired).

The retina is no longer an MSE reconstruction autoencoder; it is a decoder-free
self-supervised encoder (SPR latent self-prediction + inverse dynamics + FSQ
latent) — see pokeio/evo/retina.py and docs/specs/active-vision-spine.md §4.
There is no reconstruction to visualize, so the old before/after PNG demo is gone.

The authoritative, self-contained smoke is the pytest:

    PYTHONPATH=/home/cmod/pokeIO python -m pytest tests/test_retina_spr.py -q

It trains the SPR encoder on a learnable synthetic task and asserts the cosine
loss drops, the latent does not collapse, inverse-dynamics beats chance, and the
FSQ codebook stays diverse. This script just forwards to it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    print(__doc__)
    print("Running tests/test_retina_spr.py ...\n")
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/test_retina_spr.py", "-q"],
        cwd=str(REPO),
    )


if __name__ == "__main__":
    raise SystemExit(main())
