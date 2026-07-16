"""pokeio.pipeline — Phase 1.5 high-throughput decoupled async runtime.

Modules
-------
* ``gpu_vision``  — batched on-GPU ObsBuilder replica (cuda:1).
* ``async_fleet`` — shared-memory, sense-reversing-barrier render-on fleet +
  stub batched inference. The Phase-1.5 transport (replaces pipe-based VecFleet
  for fs=1 throughput; ``emu.fleet.VecFleet`` remains the CPU fallback).
"""

from pokeio.pipeline.async_fleet import ACTIONS, AsyncFleet, StubPolicy
from pokeio.pipeline.gpu_vision import DMG_SHADES, GPUVision, build_shade_lut

__all__ = [
    "AsyncFleet",
    "StubPolicy",
    "ACTIONS",
    "GPUVision",
    "build_shade_lut",
    "DMG_SHADES",
]
