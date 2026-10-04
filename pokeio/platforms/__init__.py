"""Emulator platforms. A platform is the thinnest possible wrapper around one
emulator instance: buttons, frame ticks, screen, memory, save states.

Adding a console (NES, SNES, ...) means adding one module here that implements
``Platform``; nothing above this layer changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokeio.platforms.base import Platform

if TYPE_CHECKING:
    from pokeio.spec import GameSpec


def make_platform(spec: "GameSpec", headless: bool = True) -> Platform:
    if spec.platform == "gameboy":
        from pokeio.platforms.gameboy import GameBoy

        return GameBoy(spec, headless=headless)
    if spec.platform == "gridworld":
        from pokeio.platforms.gridworld import GridWorld

        return GridWorld(spec)
    raise ValueError(f"unknown platform {spec.platform!r}")


__all__ = ["Platform", "make_platform"]
