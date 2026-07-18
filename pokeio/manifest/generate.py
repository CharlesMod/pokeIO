"""Generate a Progress Manifest from mined WRAM candidates via the GLM LLM.

The game-agnostic miner (:mod:`pokeio.reward.miner`) discovers WRAM addresses
that *behave like* progress counters with ZERO game knowledge. This module is
where those raw, unlabeled addresses acquire human meaning: given the mined
candidates (+ the ROM name and optional disassembly symbols) it builds a GLM
prompt and asks the oversight LLM to label them into a validated
:class:`~pokeio.manifest.schema.Manifest`.

Design goals (mirrors the rest of pokeIO):

* **Game-agnostic** — no game constants are hardcoded here; the only labels
  come from the LLM (or, in the fallback, from the addresses themselves).
* **Never hard-crash** — :func:`fallback_manifest` builds a minimal *valid*
  manifest straight from the candidates, so a caller gets a usable manifest
  even with no LLM server reachable. :func:`generate_manifest` always returns a
  manifest that passes :meth:`Manifest.validate`.
* **Dependency-light** — stdlib + orjson; :class:`LlamaClient` imported lazily
  so importing this module never pulls the client (or a network dependency).

The consumers of a loaded manifest (the dashboard/analytics RAM taps) use
:func:`ram_addresses_from_manifest` to route their reads through it instead of
hardcoding game-specific addresses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pokeio.manifest.schema import Manifest, ProgressDimension

__all__ = [
    "MANIFEST_JSON_SCHEMA",
    "build_prompt",
    "fallback_manifest",
    "generate_manifest",
    "ram_addresses_from_manifest",
    "eventful_frames",
    "milestone_ordering_digest",
]

# WRAM column 0 maps to this GB address (kept in sync with reward.miner.WRAM_BASE).
WRAM_BASE = 0xC000


# JSON schema handed to LlamaClient.complete so the model returns the right
# shape. Manifest.validate() remains the authority for the finer rules
# (non-empty game/label, unique ids, ram|screen source); a reply that passes
# this schema but fails validate() falls back to the mined-only manifest.
MANIFEST_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["game", "milestone_label", "progress_dimensions"],
    "properties": {
        "game": {"type": "string"},
        "milestone_label": {"type": "string"},
        "spatial": {"type": "object"},
        "progress_dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "source", "dir"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "source": {"type": "object"},
                    "dir": {"type": "string", "enum": ["up", "down"]},
                    "icon": {"type": "string"},
                },
            },
        },
    },
}


# --------------------------------------------------------------------------- #
# candidate accessors (accept a miner.Candidate OR a plain dict)
# --------------------------------------------------------------------------- #
def _get(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _candidate_addr(candidate: Any) -> int | None:
    addr = _get(candidate, "address")
    return _parse_addr(addr)


def _candidate_width(candidate: Any) -> int:
    try:
        return int(_get(candidate, "width", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _candidate_direction(candidate: Any) -> str:
    d = str(_get(candidate, "direction", "increasing") or "increasing")
    return "down" if d.lower().startswith("dec") else "up"


def _parse_addr(value: Any) -> int | None:
    """Coerce ``0xD356`` / ``"0xD356"`` / ``54614`` into an int, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s, 16) if s.lower().startswith("0x") else int(s)
        except ValueError:
            return None
    return None


def _game_name(rom_name: str | None) -> str:
    """A human game name from a ROM path/name (stem, no directory/extension)."""
    if not rom_name:
        return "unknown"
    stem = Path(str(rom_name)).stem.strip()
    return stem or "unknown"


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
def build_prompt(
    rom_name: str, candidates: Any, symbols: dict[int, str] | None = None
) -> str:
    """Build the GLM prompt: label mined WRAM candidates into a manifest.

    ``candidates`` is a sequence of miner ``Candidate`` objects (or dicts) with
    ``address``/``width``/``direction``/``score``. ``symbols`` optionally maps a
    GB address -> disassembly symbol name, added inline as a hint.
    """
    lines = [
        "You are labeling raw Game Boy WRAM addresses that a game-agnostic miner",
        "flagged as progress-counter-like, for the ROM below. For each address,",
        "give a short human label and the direction of improvement. Group them",
        "into a single Progress Manifest JSON object. Do NOT invent addresses that",
        "are not listed. If you cannot label an address, omit it.",
        "",
        f"ROM: {rom_name}",
        "",
        "Mined candidates (best-first):",
    ]
    for c in candidates or []:
        addr = _candidate_addr(c)
        if addr is None:
            continue
        sym = ""
        if symbols and addr in symbols:
            sym = f"  symbol={symbols[addr]}"
        score = _get(c, "score", None)
        score_s = f" score={float(score):.3f}" if isinstance(score, (int, float)) else ""
        lines.append(
            f"  - addr=0x{addr:04X} width={_candidate_width(c)} "
            f"dir={_get(c, 'direction', 'increasing')}{score_s}{sym}"
        )
    lines += [
        "",
        "Return a JSON object with keys: game, milestone_label, spatial,",
        "progress_dimensions (each: id, label, source.ram.addr, dir in "
        "['up','down'], icon).",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# fallback (no server / invalid LLM reply)
# --------------------------------------------------------------------------- #
def fallback_manifest(candidates: Any, rom_name: str) -> Manifest:
    """Build a minimal, *valid* manifest straight from the mined candidates.

    No game knowledge and no LLM: each candidate becomes a generically-labeled
    progress dimension keyed by its address. Always returns a manifest that
    passes :meth:`Manifest.validate` (a placeholder dimension is synthesized
    when there are no usable candidates).
    """
    dims: list[ProgressDimension] = []
    seen: set[str] = set()
    for c in candidates or []:
        addr = _candidate_addr(c)
        if addr is None:
            continue
        cid = f"addr_{addr:04x}"
        if cid in seen:
            continue
        seen.add(cid)
        dims.append(
            ProgressDimension(
                id=cid,
                label=f"WRAM 0x{addr:04X}",
                source={
                    "ram": {
                        "addr": f"0x{addr:04X}",
                        "kind": "counter",
                        "width": _candidate_width(c),
                    }
                },
                dir=_candidate_direction(c),
            )
        )
    if not dims:
        # Never emit an invalid manifest: a single placeholder keeps validate()
        # happy so callers always have a usable (if content-free) manifest.
        dims.append(
            ProgressDimension(
                id="progress",
                label="Progress",
                source={"ram": {"addr": "0x0000", "kind": "counter"}},
                dir="up",
            )
        )
    return Manifest(
        game=_game_name(rom_name),
        milestone_label="Progress",
        progress_dimensions=dims,
    )


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def generate_manifest(
    candidates: Any,
    rom_name: str,
    *,
    client: Any = None,
    symbols: dict[int, str] | None = None,
    use_llm: bool = True,
    save_path: str | Path | None = None,
) -> Manifest:
    """Mined candidates -> a validated :class:`Manifest`.

    Asks the GLM oversight LLM (via :meth:`LlamaClient.complete` with a schema)
    to label the candidates; on any failure — no server, transport error,
    malformed reply, or a reply that fails :meth:`Manifest.validate` — falls
    back to :func:`fallback_manifest`. The returned manifest ALWAYS validates.

    ``client`` may be any object with a compatible ``.complete(prompt, schema=,
    system=)`` method (a real :class:`LlamaClient` or a test stub); when None a
    default client is constructed lazily. Set ``use_llm=False`` to skip the LLM
    and build the manifest from the candidates alone. ``save_path`` optionally
    writes the manifest JSON (best-effort).
    """
    manifest: Manifest | None = None
    if use_llm:
        try:
            manifest = _generate_via_llm(candidates, rom_name, client, symbols)
        except Exception:
            # Any failure (no server, transport, schema, parse) -> fall back.
            manifest = None
    if manifest is None or manifest.validate():
        manifest = fallback_manifest(candidates, rom_name)
    if save_path is not None:
        try:
            manifest.save_json(save_path)
        except OSError:
            pass  # persistence is best-effort; never fail generation over it
    return manifest


def _generate_via_llm(
    candidates: Any, rom_name: str, client: Any, symbols: dict[int, str] | None
) -> Manifest:
    if client is None:
        from pokeio.llm.client import LlamaClient  # lazy: no hard dep at import

        client = LlamaClient()
    prompt = build_prompt(rom_name, candidates, symbols)
    system = (
        "You label raw Game Boy WRAM progress-counter candidates into a "
        "game-progress manifest. Respond with a single JSON object only, no prose."
    )
    data = client.complete(prompt, schema=MANIFEST_JSON_SCHEMA, system=system)
    if not isinstance(data, dict):
        raise TypeError(f"LLM manifest reply was not a JSON object: {type(data)!r}")
    data.setdefault("game", _game_name(rom_name))
    return Manifest.from_dict(data)


# --------------------------------------------------------------------------- #
# consumption helper: manifest -> RAM tap addresses
# --------------------------------------------------------------------------- #
def ram_addresses_from_manifest(manifest: Any) -> list[int]:
    """Ordered, de-duplicated WRAM addresses declared by a manifest.

    Reads ``spatial.source.ram`` (map/x/y-style taps, listed first because they
    are the UI's core coordinates) then each ``progress_dimensions[*].source.
    ram.addr``. Values may be ``int`` or hex strings (``"0xD356"``). Returns
    ``[]`` when a manifest declares no RAM sources (e.g. a screen-only game).
    """
    out: list[int] = []
    seen: set[int] = set()

    def _add(value: Any) -> None:
        a = _parse_addr(value)
        if a is not None and a not in seen:
            seen.add(a)
            out.append(a)

    if manifest is None:
        return out

    spatial = getattr(manifest, "spatial", None) or {}
    if isinstance(spatial, dict):
        src = spatial.get("source", {})
        ram = src.get("ram", {}) if isinstance(src, dict) else {}
        if isinstance(ram, dict):
            for v in ram.values():
                _add(v)

    for pd in getattr(manifest, "progress_dimensions", []) or []:
        source = getattr(pd, "source", None)
        if source is None and isinstance(pd, dict):
            source = pd.get("source")
        source = source or {}
        ram = source.get("ram", {}) if isinstance(source, dict) else {}
        if isinstance(ram, dict):
            _add(ram.get("addr"))

    return out


# --------------------------------------------------------------------------- #
# [MF] consume helpers: eventful frames + anonymized milestone-ordering digest
# --------------------------------------------------------------------------- #
def eventful_frames(manifest: Any, wram_trace: Any) -> list[int]:
    """Frame indices where a manifest-declared counter moved (offline; spec §6).

    Reuses :func:`ram_addresses_from_manifest` to route reads through the manifest
    (no hardcoded addresses), then returns the ``t`` at which any declared WRAM
    counter changed between snapshot ``t-1`` and ``t``. ``wram_trace`` is a ``(T,
    N)`` array of raw WRAM snapshots (column ``i`` -> ``0xC000 + i``), matching the
    miner's input contract. Powers the active-learning "eventful frame" trigger in
    :mod:`pokeio.reward.pairs` (spec §2b.5). Game-specific labels/addresses stay
    OFFLINE — they only select which frames to sample, never enter Φ.
    """
    import numpy as np  # local: keep the module import-light

    addrs = ram_addresses_from_manifest(manifest)
    trace = np.asarray(wram_trace)
    if trace.ndim != 2 or trace.shape[0] < 2 or not addrs:
        return []
    cols = [a - WRAM_BASE for a in addrs if 0 <= (a - WRAM_BASE) < trace.shape[1]]
    if not cols:
        return []
    sub = trace[:, cols].astype(np.int64)
    moved = np.any(np.diff(sub, axis=0) != 0, axis=1)  # (T-1,)
    return [int(t + 1) for t in np.nonzero(moved)[0]]  # index where the move landed


def milestone_ordering_digest(manifest: Any) -> dict:
    """Anonymized milestone-ordering digest of a manifest (graft, MF-MM; spec §2a).

    Each progress dimension becomes an ANONYMIZED node — ``counter_0``,
    ``counter_1``, ... in manifest order — carrying only its direction and its
    fraction along the discovered DAG. The game name, real labels, and addresses
    are STRIPPED, so the digest is a game-agnostic ordering substrate for the
    text-only prompt and the kNN-anchor order fractions. Never a model input.
    """
    dims = getattr(manifest, "progress_dimensions", None) or []
    n = len(dims)
    nodes = []
    for i, pd in enumerate(dims):
        d = getattr(pd, "dir", None)
        if d is None and isinstance(pd, dict):
            d = pd.get("dir", "up")
        nodes.append({
            "id": f"counter_{i}",
            "dir": str(d or "up"),
            "order": i,
            "order_frac": (i + 1) / n if n else 0.0,
        })
    return {"n": n, "nodes": nodes}
