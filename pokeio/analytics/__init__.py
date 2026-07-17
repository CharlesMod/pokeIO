"""Trainer-facing meta-analytics for pokeIO runs.

This package is for US (the humans running the experiment) — progress reports,
run diaries, game-state decoding, entertainment. It is explicitly ALLOWED to be
game-specific (e.g. Pokémon Yellow RAM addresses and map names) because nothing
here feeds back into the evolutionary/learning system, which stays strictly
game-agnostic. Read-only over runs/<id>/ artifacts and checkpoints.
"""
