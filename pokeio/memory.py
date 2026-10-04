"""Typed, cached reads of the spec's named memory fields.

Each field is resolved once (symbol -> address) and read at most once per
agent step: call ``invalidate()`` after advancing the emulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokeio.platforms.base import Platform
from pokeio.spec import Condition, MemoryField

_SIZES = {"u8": 1, "u16be": 2, "u16le": 2, "u24le": 3}


def parse_sym_file(path: Path) -> dict[str, int]:
    """Parse an RGBDS/no$gmb ``.sym`` file (``BB:AAAA Name`` per line)."""
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.split(";", 1)[0].strip()
            if not line or ":" not in line:
                continue
            loc, _, name = line.partition(" ")
            _bank, _, addr = loc.partition(":")
            try:
                out[name.strip()] = int(addr, 16)
            except ValueError:
                continue
    return out


class UnresolvedField(KeyError):
    pass


class Memory:
    def __init__(
        self,
        platform: Platform,
        fields: dict[str, MemoryField],
        symbols: dict[str, int] | None = None,
    ):
        self.p = platform
        self.fields = fields
        self.addr: dict[str, int] = {}
        self.unresolved: list[str] = []
        for name, f in fields.items():
            a = f.addr
            if a is None:
                a = (symbols or {}).get(f.symbol)
                if a is None:
                    a = platform.symbol(f.symbol)
                if a is not None:
                    a += f.offset
            if a is None:
                self.unresolved.append(name)
            else:
                self.addr[name] = a
        self._cache: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.addr

    def invalidate(self) -> None:
        self._cache.clear()

    def _raw(self, name: str) -> np.ndarray:
        """Bytes covering the whole field (all array elements / the block)."""
        key = "raw:" + name
        hit = self._cache.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        if name not in self.addr:
            raise UnresolvedField(name)
        f = self.fields[name]
        if f.is_block:
            n = f.length
        else:
            size = f.bcd_bytes if f.type == "bcd" else _SIZES[f.type]
            n = f.stride * (f.count - 1) + size
        data = self.p.read(self.addr[name], n)
        self._cache[key] = data
        return data

    def array(self, name: str) -> np.ndarray:
        """All ``count`` elements as int64 (blocks: the raw bytes)."""
        key = "arr:" + name
        hit = self._cache.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        f = self.fields[name]
        raw = self._raw(name).astype(np.int64)
        if f.is_block:
            out = raw
        else:
            idx = np.arange(f.count) * f.stride
            if f.type == "u8":
                out = (raw[idx] & f.mask) >> f.shift
            elif f.type == "u16be":
                out = (raw[idx] << 8) | raw[idx + 1]
            elif f.type == "u16le":
                out = raw[idx] | (raw[idx + 1] << 8)
            elif f.type == "u24le":
                out = raw[idx] | (raw[idx + 1] << 8) | (raw[idx + 2] << 16)
            else:  # bcd, big-endian digit pairs
                out = np.zeros(f.count, dtype=np.int64)
                for k in range(f.bcd_bytes):
                    b = raw[idx + k]
                    out = out * 100 + (b >> 4) * 10 + (b & 0xF)
        self._cache[key] = out
        return out

    def get(self, name: str, index: int = 0) -> int:
        return int(self.array(name)[index])

    def block(self, name: str) -> np.ndarray:
        """Raw uint8 bytes of the field (for bitfields)."""
        return self._raw(name)

    def popcount(self, name: str, ignore_bits: list[tuple[int, int]] | None = None) -> int:
        raw = self._raw(name)
        if self.fields[name].is_block:
            bits = np.unpackbits(raw)
            n = int(bits.sum())
            for byte, bit in ignore_bits or []:
                if byte < len(raw) and raw[byte] & (1 << bit):
                    n -= 1
            return n
        n = int(sum(int(v).bit_count() for v in self.array(name)))
        vals = self.array(name)
        for elem, bit in ignore_bits or []:
            if elem < len(vals) and int(vals[elem]) & (1 << bit):
                n -= 1
        return n

    def write(self, name: str, value: int) -> None:
        self.p.write(self.addr[name], [int(value) & 0xFF])
        self.invalidate()

    def check(self, c: Condition) -> bool:
        if c.op == "bit":
            if self.fields[c.field].is_block:
                raw = self._raw(c.field)
                return bool(raw[c.index // 8] & (1 << (c.index % 8)))
            return bool(self.get(c.field) & (1 << c.index))
        v = self.get(c.field, c.index)
        op = c.op
        if op == "nonzero":
            return v != 0
        if op == "zero":
            return v == 0
        if op == "any_bits":
            return (v & c.value) != 0
        if op == "eq":
            return v == c.value
        if op == "ne":
            return v != c.value
        if op == "gt":
            return v > c.value
        if op == "ge":
            return v >= c.value
        if op == "lt":
            return v < c.value
        if op == "le":
            return v <= c.value
        raise ValueError(op)

    def all(self, conds: list[Condition]) -> bool:
        return all(self.check(c) for c in conds)
