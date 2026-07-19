"""Fetch libretro cores into assets/cores/ for the multi-console env (#36).

The stable-retro PyPI wheel bundles prebuilt, open-source libretro cores for the
whole developmental ladder (gambatte GB/GBC, mgba GBA, parallel_n64 N64, snes9x,
fceumm NES, genesis_plus_gx, …). We do NOT depend on stable-retro at runtime — we
just borrow its cores and drive them via pokeio.emu.libretro (raw ctypes). The
cores are gitignored (binaries); this script provisions them reproducibly.

Usage:
    .venv/bin/python scripts/fetch_cores.py                # gambatte (default)
    .venv/bin/python scripts/fetch_cores.py mgba parallel_n64
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORES_DIR = os.path.join(ROOT, "assets", "cores")
DEFAULT = ["gambatte"]


def _download_wheel(dest: str) -> str:
    subprocess.check_call([sys.executable, "-m", "pip", "download", "stable-retro",
                           "--no-deps", "-d", dest, "-q"])
    whl = [f for f in os.listdir(dest) if f.endswith(".whl")]
    if not whl:
        raise SystemExit("stable-retro wheel not found after pip download")
    return os.path.join(dest, whl[0])


def fetch(cores: list[str]) -> None:
    os.makedirs(CORES_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        whl = _download_wheel(tmp)
        with zipfile.ZipFile(whl) as z:
            names = z.namelist()
            for core in cores:
                want = f"{core}_libretro.so"
                member = next((n for n in names if n.endswith(f"cores/{want}")), None)
                if member is None:
                    print(f"  [skip] {want}: not in wheel")
                    continue
                data = z.read(member)
                out = os.path.join(CORES_DIR, want)
                with open(out, "wb") as fh:
                    fh.write(data)
                os.chmod(out, 0o755)
                sha = hashlib.sha256(data).hexdigest()[:16]
                print(f"  [ok]   {want}  ({len(data)} bytes, sha256:{sha}…) -> {out}")


if __name__ == "__main__":
    fetch(sys.argv[1:] or DEFAULT)
