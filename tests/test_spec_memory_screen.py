import numpy as np
import pytest
import yaml

from pokeio.memory import Memory, parse_sym_file
from pokeio.screen import ScreenProcessor, pack, quantize, unpack
from pokeio.spec import Condition, MemoryField, SpecError, load_spec


@pytest.mark.parametrize("name", ["pokemon_yellow", "pokemon_red", "generic_gb", "gridworld"])
def test_shipped_specs_load(root, name):
    spec = load_spec(root / f"games/{name}/spec.yaml", root)
    assert spec.controls.buttons
    assert len(spec.fingerprint) == 12


def test_unknown_field_reference_is_rejected(tmp_path, root):
    raw = yaml.safe_load((root / "games/gridworld/spec.yaml").read_text())
    raw["rewards"]["terms"][0]["field"] = "nope"
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(SpecError, match="nope"):
        load_spec(p, root)


class FakeRAM:
    def __init__(self, data):
        self.ram = np.array(data, dtype=np.uint8)

    def read(self, addr, n=1):
        return self.ram[addr : addr + n].copy()

    def write(self, addr, data):
        self.ram[addr : addr + len(data)] = data

    def symbol(self, name):
        return {"wThing": 3}.get(name)


def test_memory_types_and_conditions():
    ram = [0] * 64
    ram[0:2] = [0x01, 0x02]  # u16be = 258
    ram[2] = 0b1010_0000
    ram[3] = 7  # symbol wThing
    ram[10], ram[14] = 5, 9  # stride-4 array
    ram[20:23] = [0x12, 0x34, 0x56]  # bcd 123456
    fields = {
        "w": MemoryField("w", addr=0, type="u16be"),
        "flags": MemoryField("flags", addr=2),
        "hi": MemoryField("hi", addr=2, mask=0xF0, shift=4),
        "sym": MemoryField("sym", symbol="wThing"),
        "arr": MemoryField("arr", addr=10, count=2, stride=4),
        "money": MemoryField("money", addr=20, type="bcd", bcd_bytes=3),
        "blk": MemoryField("blk", addr=2, length=2),
        "missing": MemoryField("missing", symbol="wNope"),
    }
    m = Memory(FakeRAM(ram), fields)
    assert m.get("w") == 258
    assert m.get("hi") == 0b1010
    assert m.get("sym") == 7
    assert m.array("arr").tolist() == [5, 9]
    assert m.get("money") == 123456
    assert m.popcount("blk") == 2 + 3
    assert m.popcount("blk", ignore_bits=[(0, 7)]) == 4
    assert m.unresolved == ["missing"] and not m.has("missing")
    assert m.check(Condition("flags", "bit", index=7))
    assert not m.check(Condition("flags", "bit", index=6))
    assert m.check(Condition("blk", "bit", index=8 + 0)) is True  # byte 1 = 7 -> bit 0 set
    assert m.check(Condition("arr", "ge", value=5))


def test_sym_parser(tmp_path):
    p = tmp_path / "x.sym"
    p.write_text("; comment\n00:cd6b wJoyIgnore\n01:4000 SomeLabel.local\n")
    assert parse_sym_file(p) == {"wJoyIgnore": 0xCD6B, "SomeLabel.local": 0x4000}


def test_quantize_gb_shades():
    shades = np.array([[0, 85, 153, 170, 255]], dtype=np.uint8)
    assert quantize(shades, 4).tolist() == [[0, 1, 2, 2, 3]]


@pytest.mark.parametrize("bpp", [1, 2, 4, 8])
def test_pack_roundtrip(bpp):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 1 << min(bpp, 8), size=(9, 16), dtype=np.uint8)
    assert np.array_equal(unpack(pack(img, bpp), bpp), img)


def test_screen_processor_shapes():
    sp = ScreenProcessor(144, 160, 2, 4, "luma")
    assert (sp.h, sp.w, sp.bpp, sp.packed_w) == (72, 80, 2, 20)
    rgb = np.full((144, 160, 3), 255, dtype=np.uint8)
    lv = sp.levels_image(rgb)
    assert lv.shape == (72, 80) and lv.max() == 3
    assert sp.pack(lv).shape == (72, 20)
