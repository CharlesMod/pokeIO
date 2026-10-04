import numpy as np

from pokeio.exploration import CellMemory, InteractionNovelty, ScreenNovelty, VisitedMask
from pokeio.memory import Memory
from pokeio.progress import Progress
from pokeio.spec import Condition, MemoryField, Milestone, ProgressTerm

from .test_spec_memory_screen import FakeRAM


def test_binary_cell_memory():
    m = CellMemory(half_life_steps=0, floor=0.15)
    assert m.visit(0, 3, 4, t=0) == 1.0
    assert m.visit(0, 3, 4, t=10**6) == 0.0
    assert m.visit(1, 3, 4, t=1) == 1.0  # different room
    assert m.unique == 2
    m.wipe()
    assert m.visit(0, 3, 4, t=2) == 1.0


def test_decaying_cell_memory():
    m = CellMemory(half_life_steps=100, floor=0.15)
    m.visit(0, 1, 1, t=0)
    assert abs(m.visit(0, 1, 1, t=100) - 0.5) < 1e-9
    # never decays below the floor
    m.visit(0, 2, 2, t=0)
    assert abs(m.visit(0, 2, 2, t=10**6) - 0.85) < 1e-9


def test_grid_growth():
    m = CellMemory(0, 0.15)
    assert m.visit(0, 200, 150, 0) == 1.0
    assert m.rooms[0].t.shape[0] > 150 and m.rooms[0].t.shape[1] > 200


def test_visited_mask_follow_alignment():
    mem = CellMemory(0, 0.15)
    mem.visit(0, 10, 10, 0)  # the player's own cell
    mem.visit(0, 11, 10, 0)  # one cell to the right
    vm = VisitedMask(144, 160, 2, 16, (64, 64), 4)
    img = vm.render(mem, 0, 10, 10, 1)
    assert img.shape == (72, 80)
    # player cell occupies [32:40, 32:40] at downsample 2; right neighbour [32:40, 40:48]
    assert (img[32:40, 32:48] == 3).all()
    assert img[32:40, 24:32].max() == 0 and img[24:32, 32:40].max() == 0


def test_visited_mask_unaligned_origin():
    mem = CellMemory(0, 0.15)
    mem.visit(0, 5, 5, 0)
    vm = VisitedMask(144, 160, 2, 16, (64, 58), 4)
    img = vm.render(mem, 0, 5, 5, 1)
    assert img.shape == (72, 80)
    assert (img[29:37, 32:40] == 3).all() and img[28, 32] == 0 and img[37, 32] == 0


def test_visited_mask_fixed_camera():
    mem = CellMemory(0, 0.15)
    mem.visit(0, 2, 1, 0)
    vm = VisitedMask(144, 160, 2, 16, (0, 0), 4, camera="fixed")
    img = vm.render(mem, 0, 2, 1, 1)
    assert (img[8:16, 16:24] == 3).all() and img.sum() == 3 * 64


def test_interaction_novelty():
    n = InteractionNovelty(min_change=0.05)
    a = np.zeros((10, 10), np.uint8)
    b = a.copy()
    b[:2] = 3
    assert n.observe(0, 1, 1, 0, False, a, b) == 1.0
    assert n.observe(0, 1, 1, 0, False, a, b) == 0.0  # same key
    assert n.observe(0, 1, 1, 1, True, a, b) == 0.0  # moved
    assert n.observe(0, 1, 1, 2, False, a, a) == 0.0  # nothing changed


def test_screen_novelty_decreases():
    s = ScreenNovelty((3, 4), 4)
    img = np.zeros((72, 80), np.uint8)
    assert s.observe(img) == 1.0
    assert s.observe(img) < 1.0


def _progress(ram):
    fields = {
        "ev": MemoryField("ev", addr=0, length=2),
        "cnt": MemoryField("cnt", addr=2),
        "lv": MemoryField("lv", addr=3, count=3),
        "hp": MemoryField("hp", addr=6, count=3),
        "mhp": MemoryField("mhp", addr=9, count=3),
    }
    mem = Memory(ram, fields)
    terms = [
        ProgressTerm("events", "bitcount", "ev", weight=2.0),
        ProgressTerm("party", "value", "cnt", weight=1.0),
        ProgressTerm("levels", "sum_softcap", "lv", count_field="cnt", knee=10, slope=0.25, swarm=False),
        ProgressTerm("heal", "ratio_gain", "hp", denom="mhp", count_field="cnt", weight=1.0),
    ]
    ms = [Milestone("two_mons", [Condition("cnt", "ge", 2)])]
    return Progress(terms, ms, mem), mem


def test_progress_terms():
    ram = FakeRAM([0] * 16)
    ram.ram[2] = 1
    ram.ram[3] = 5
    ram.ram[6], ram.ram[9] = 5, 10
    p, mem = _progress(ram)
    p.rebase(0)
    r, parts, new = p.step(1)
    assert r == 0 and not new  # rebase: nothing for pre-existing progress

    ram.ram[0] = 0b11  # two events
    mem.invalidate()
    r, parts, _ = p.step(2)
    assert parts == {"events": 4.0}

    ram.ram[0] = 0  # toggled off: monotone -> no penalty
    mem.invalidate()
    r, parts, _ = p.step(3)
    assert r == 0

    ram.ram[3] = 13  # level sum 13 -> softcap 10 + 3*0.25
    mem.invalidate()
    _, parts, _ = p.step(4)
    assert abs(parts["levels"] - (10.75 - 5)) < 1e-9

    ram.ram[6] = 10  # healed 5/10 -> 10/10
    mem.invalidate()
    _, parts, _ = p.step(5)
    assert abs(parts["heal"] - 0.5) < 1e-9

    ram.ram[2] = 2  # caught a second mon: party +1, heal ignored when count changes
    ram.ram[4], ram.ram[7], ram.ram[10] = 1, 1, 10
    mem.invalidate()
    _, parts, new = p.step(6)
    assert parts.get("party") == 1.0 and "heal" not in parts and new == ["two_mons"]


def test_state_score_excludes_non_swarm_terms():
    ram = FakeRAM([0] * 16)
    ram.ram[0] = 0b111
    ram.ram[2] = 2
    ram.ram[3:5] = [50, 50]
    p, _ = _progress(ram)
    p.rebase(0)
    assert p.state_score() == 2.0 * 3 + 1.0 * 2  # levels has swarm=False
