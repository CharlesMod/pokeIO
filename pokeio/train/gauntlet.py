"""Boot gauntlet — from-boot competence measurement for the champion.

Go-Explore restores mean a frontier cell at depth 40k was reached by a CHAIN
of episodes run by possibly different genomes; no single genome has ever
demonstrated newgame→frontier play, and nothing measured whether the champion
can still play from power-on. Every N generations this runs the champion solo
from the canonical newgame state at full speed and scores how deep into the
*currently archived* frontier it re-reaches — turning "we don't know" into a
per-run curve (the original Go-Explore "robustification" concern, instrumented
before paying for any robustification mechanism).

Metric caveat: ``boot_depth`` measures re-reaching currently ARCHIVED cells. A
champion that finds a genuinely novel route from boot scores low on
``boot_cells_known`` while being excellent — that's why ``boot_cells`` (raw
distinct cells) is logged alongside. Do NOT "fix" this by writing to the
archive: the gauntlet is STRICTLY read-only against both archives (the same
rule as the champion replay — audit REWARD#5), otherwise it would burn fresh
keys into ``seen`` with no state capture and poison the frontier accounting.

Full spec: docs/specs/boot-gauntlet.md.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch

from pokeio.evo.forward import population_forward_sparse
from pokeio.evo.genome import Population
from pokeio.reward.archive import NoveltyArchive
from pokeio.reward.goexplore import GoExplore

__all__ = ["boot_progress_from_keys", "run_boot_gauntlet"]


def boot_progress_from_keys(
    keys: Iterable[bytes],
    goexplore: GoExplore,
) -> tuple[int, int, int]:
    """Pure metric: ``(max_known_depth, n_distinct, n_known)``.

    ``max_known_depth`` = max ``CellEntry.depth`` among visited keys present in
    ``goexplore.cells`` (0 if none). ``n_distinct`` = distinct keys visited.
    ``n_known`` = distinct visited keys present in the Go-Explore archive.
    Membership probes only — never mutates the archive.
    """
    distinct = set(keys)
    max_depth = 0
    n_known = 0
    cells = goexplore.cells
    for k in distinct:
        entry = cells.get(k)
        if entry is not None:
            n_known += 1
            if entry.depth > max_depth:
                max_depth = int(entry.depth)
    return max_depth, len(distinct), n_known


def run_boot_gauntlet(
    genome,
    env,
    encoder,
    archive: NoveltyArchive,
    goexplore: GoExplore,
    device: torch.device,
    steps: int,
    max_nodes: int,
    max_conns: int,
    reset_state: str,
    forward_steps: int = 4,
    recurrent_memory: bool = True,
) -> dict[str, float]:
    """Run the champion solo from ``reset_state``; return the metric dict.

    Mirrors the champion-replay loop (compile a 1-genome population, then per
    step: encode → sparse forward → argmax → step) at full speed, collecting
    ``archive.cell_key`` per step. Key compatibility with the worker compact
    path is guaranteed by :meth:`NoveltyArchive.cell_key_compact` (verified
    byte-identical), so ``goexplore.cells`` membership is exact.

    STRICTLY READ-ONLY vs both archives: no add/observe/visit, no note, no
    captures — key construction and dict membership only.
    """
    pop = Population.from_genomes([genome], max_nodes=max_nodes, max_conns=max_conns)
    cp = pop.compile(device)
    screen = env.reset(reset_state)
    wram = env.raw_wram()
    seen_keys: set[bytes] = set()
    state = None  # recurrent node-state carried across the gauntlet episode
    n = 0
    for _t in range(int(steps)):
        x = encoder.encode(screen, wram)
        xt = torch.from_numpy(x[None, :]).to(device).unsqueeze(1)
        if recurrent_memory:
            out, state = population_forward_sparse(
                cp, xt, steps=forward_steps, state=state, return_state=True
            )
        else:
            out = population_forward_sparse(cp, xt, steps=forward_steps)
        action = int(out[0, 0, :].argmax().item())
        screen, wram, _done, _info = env.step(action)
        seen_keys.add(archive.cell_key(screen, wram))
        n += 1
        time.sleep(0)  # cooperative GIL handoff for the live pump thread

    max_depth, n_distinct, n_known = boot_progress_from_keys(seen_keys, goexplore)
    # same-snapshot frontier depth: eviction can shrink it between gens, so
    # the fraction must be computed against the archive as it is RIGHT NOW.
    frontier = max(
        (e.depth for e in goexplore.cells.values()), default=0
    )
    return {
        "boot_depth": float(max_depth),
        "boot_depth_frac": float(max_depth) / float(max(1, frontier)),
        "boot_cells": float(n_distinct),
        "boot_cells_known": float(n_known),
        "boot_steps": float(n),
    }
