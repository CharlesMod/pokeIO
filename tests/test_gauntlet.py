"""Boot-gauntlet tests (docs/specs/boot-gauntlet.md §Tests).

1. Pure metric on a synthetic archive (empty / no / partial / full overlap).
2. Read-only guarantee against real archives with a real env.
3. cell_key / cell_key_compact equality — the invariant the metric rests on.

The env-backed tests skip cleanly when the ROM is absent (CI without assets).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokeio.reward.archive import NoveltyArchive
from pokeio.reward.goexplore import CellEntry, GoExplore
from pokeio.train.gauntlet import boot_progress_from_keys, run_boot_gauntlet

ROM = Path("roms/pokemon_yellow.gb")
STATE = Path("roms/yellow_newgame.state")


def _go_with(depths: dict[bytes, int]) -> GoExplore:
    g = GoExplore(capacity=64, rng=np.random.default_rng(0))
    for key, depth in depths.items():
        g.cells[key] = CellEntry(
            key=key, state=b"", depth=depth, gen_added=0, gen_seen=0
        )
    return g


def test_boot_progress_pure_metric():
    go = _go_with({b"a": 10, b"b": 250, b"c": 3})

    # empty key set
    assert boot_progress_from_keys([], go) == (0, 0, 0)
    # no overlap with the archive
    assert boot_progress_from_keys([b"x", b"y"], go) == (0, 2, 0)
    # partial overlap: deepest known visited cell wins
    assert boot_progress_from_keys([b"a", b"x", b"c"], go) == (10, 3, 2)
    # full overlap (+ duplicates collapse to distinct)
    assert boot_progress_from_keys(
        [b"a", b"b", b"c", b"b", b"b"], go
    ) == (250, 3, 3)


@pytest.mark.skipif(not ROM.exists(), reason="ROM assets not present")
def test_gauntlet_foveal_path():
    """Port smoke: the gauntlet runs under the active-vision FovealEncoder /
    11-output head — produces a depth score, mutates NO archive counters, and
    the movable fovea actually moves (spec §2/§3, boot-gauntlet.md invariant 1).
    """
    import torch

    from pokeio.emu.env import PokeEnv
    from pokeio.emu.fleet import FovealEncoder
    from pokeio.evo.genome import TANH, InnovationTracker, make_genome

    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        # Foveal obs: 454-d (3*12^2 periph/fovea/motion + 14 proprio + 8 ram),
        # 11-output head (9 buttons + dx,dy saccade), TANH decisive head.
        encoder = FovealEncoder(
            1,
            periph_grid=12,
            fovea_native_px=48,
            n_ram=8,
            saccade_gain=32.0,
            saccade_every_k=1,
            episode_steps=64,
        )
        assert encoder.dim == 454
        n_out = 11

        archive = NoveltyArchive()
        go = _go_with({b"seed-cell": 42})
        # pre-populate the novelty archive so any mutation would be detectable
        archive.add(b"pre-existing")
        archive.visit(b"pre-existing")
        before = (
            archive.size,
            dict(archive._visits),
            len(go.cells),
            go.n_captured,
            go.n_restores,
        )

        rng = np.random.default_rng(0)
        tracker = InnovationTracker(n_in=encoder.dim, n_out=n_out)
        genome = make_genome(
            encoder.dim, n_out, tracker, rng,
            connect="sparse", sparse_k=6, output_act=TANH,
            n_ram=8, n_proprio=14,
        )

        # Record every integrated gaze position to prove the fovea moves.
        gaze_seen: list[tuple[float, float]] = []
        _real_update = encoder.update_gaze

        def _spy_update(env_idx, dx, dy):
            gy, gx = _real_update(env_idx, dx, dy)
            gaze_seen.append((gy, gx))
            return gy, gx

        encoder.update_gaze = _spy_update  # type: ignore[method-assign]

        steps = 24
        metrics = run_boot_gauntlet(
            genome, env, encoder, archive, go, torch.device("cpu"),
            steps=steps, max_nodes=encoder.dim + 64, max_conns=4096,
            reset_state=str(STATE),
        )

        # A real depth score came back over a well-formed metric dict.
        assert metrics["boot_steps"] == float(steps)
        assert metrics["boot_cells"] >= 1.0
        assert metrics["boot_depth"] >= 0.0
        assert 0.0 <= metrics["boot_depth_frac"] <= 1.0
        assert metrics["boot_cells_known"] >= 0.0

        # The fovea moved: at least one integrated gaze left the screen centre
        # (72, 80). update_gaze was called once per step.
        assert len(gaze_seen) == steps
        assert any(pos != (72.0, 80.0) for pos in gaze_seen), gaze_seen

        # STRICTLY read-only: nothing about either archive may change.
        after = (
            archive.size,
            dict(archive._visits),
            len(go.cells),
            go.n_captured,
            go.n_restores,
        )
        assert before == after
    finally:
        env.close()


@pytest.mark.skipif(not ROM.exists(), reason="ROM assets not present")
def test_cell_key_compact_equality():
    from pokeio.emu.env import PokeEnv

    env = PokeEnv(rom_path=str(ROM), frame_skip=24)
    try:
        archive = NoveltyArchive()
        screen = env.reset(str(STATE))
        wram = env.raw_wram()
        assert archive.cell_key(screen, wram) == archive.cell_key_compact(
            screen, wram[:: archive.wram_stride]
        )
        # and with a churn-mask installed (the production configuration)
        masked = NoveltyArchive(wram_mask=tuple(range(0, 120)))
        assert masked.cell_key(screen, wram) == masked.cell_key_compact(
            screen, wram[:: masked.wram_stride]
        )
    finally:
        env.close()
