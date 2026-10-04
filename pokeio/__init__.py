"""pokeIO — a game-agnostic RL agent for console games.

The engine never names a game. Everything game-specific (RAM fields, buttons,
progress signals, milestones) lives in a declarative *game spec* under
``games/<name>/spec.yaml``. See docs/ARCHITECTURE.md.
"""

__version__ = "3.0.0.dev0"
