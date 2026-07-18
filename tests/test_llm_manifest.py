"""Offline tests for the LLM client hardening (A11) + manifest generate/consume
(A12) + honesty-pass quarantine (A13). No live server or GPU required.

Covered:
* extract_json / validate_schema / _mini_validate
* LlamaClient.from_config URL construction + model normalization
* cache key (version + served-model fingerprint) + TTL / version invalidation +
  the schema-reply caching gate
* manifest generate: fallback, no-LLM, stub-LLM success/failure/invalid-reply,
  ram_addresses_from_manifest
* Yellow-table quarantine behind manifest.game (decode + milestones)
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import orjson
import pytest

from pokeio.llm.client import (
    CACHE_VERSION,
    LlamaClient,
    SchemaError,
    _mini_validate,
    extract_json,
    validate_schema,
)
from pokeio.manifest.generate import (
    build_prompt,
    fallback_manifest,
    generate_manifest,
    ram_addresses_from_manifest,
)
from pokeio.manifest.schema import Manifest, ProgressDimension
from pokeio.reward.miner import Candidate

EXAMPLE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "pokeio"
    / "manifest"
    / "examples"
    / "pokemon_yellow.json"
)


# --------------------------------------------------------------------------- #
# A11 — extract_json
# --------------------------------------------------------------------------- #
def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_embedded_span():
    assert extract_json('sure, here: {"a": 1} done')["a"] == 1


def test_extract_json_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_failure_raises():
    with pytest.raises(SchemaError):
        extract_json("no json anywhere in here")


# --------------------------------------------------------------------------- #
# A11 — validate_schema / _mini_validate
# --------------------------------------------------------------------------- #
def test_mini_validate_ok():
    _mini_validate(
        {"a": 1},
        {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}},
    )


def test_mini_validate_missing_required():
    with pytest.raises(SchemaError):
        _mini_validate({}, {"type": "object", "required": ["a"]})


def test_mini_validate_type_mismatch():
    with pytest.raises(SchemaError):
        _mini_validate("x", {"type": "integer"})


def test_mini_validate_bool_is_not_integer():
    with pytest.raises(SchemaError):
        _mini_validate(True, {"type": "integer"})


def test_mini_validate_enum():
    _mini_validate("up", {"enum": ["up", "down"]})
    with pytest.raises(SchemaError):
        _mini_validate("sideways", {"enum": ["up", "down"]})


def test_mini_validate_nested_and_items():
    schema = {
        "type": "object",
        "properties": {"xs": {"type": "array", "items": {"type": "integer"}}},
    }
    _mini_validate({"xs": [1, 2, 3]}, schema)
    with pytest.raises(SchemaError):
        _mini_validate({"xs": [1, "a"]}, schema)


def test_validate_schema_valid_and_invalid():
    validate_schema({"a": 1}, {"type": "object", "required": ["a"]})
    with pytest.raises(SchemaError):
        validate_schema({}, {"type": "object", "required": ["a"]})


# --------------------------------------------------------------------------- #
# A11 — from_config URL construction + model normalization
# --------------------------------------------------------------------------- #
def test_from_config_strips_trailing_v1_and_lowercases_model():
    cfg = SimpleNamespace(
        base_url="http://127.0.0.1:8080/v1",
        model="GLM-4.7-Flash",
        timeout_s=42.0,
        max_retries=5,
        temperature=0.2,
        cache_dir="runs/llm_cache",
    )
    c = LlamaClient.from_config(cfg)
    assert c.base_url == "http://127.0.0.1:8080"
    assert c.model == "glm-4.7-flash"
    assert c.timeout == 42.0
    assert c.max_retries == 5
    # The URL the chat path will actually build must carry a single /v1.
    assert f"{c.base_url}/v1/chat/completions" == (
        "http://127.0.0.1:8080/v1/chat/completions"
    )


def test_from_config_no_v1_unchanged():
    cfg = SimpleNamespace(base_url="http://localhost:9292", model="x")
    assert LlamaClient.from_config(cfg).base_url == "http://localhost:9292"


def test_from_config_trailing_slash_and_v1():
    cfg = SimpleNamespace(base_url="http://h:8080/v1/", model="m")
    assert LlamaClient.from_config(cfg).base_url == "http://h:8080"


def test_from_config_defaults_when_attrs_missing():
    c = LlamaClient.from_config(SimpleNamespace())
    assert c.base_url.startswith("http")
    assert c.model  # non-empty


# --------------------------------------------------------------------------- #
# A11 — cache key (version + fingerprint) + TTL / version invalidation + gate
# --------------------------------------------------------------------------- #
def test_cache_key_stable_and_sensitive(tmp_path):
    c = LlamaClient(cache_dir=tmp_path)
    k1 = c._cache_key("p", None, None, 0.2, 128)
    assert k1 == c._cache_key("p", None, None, 0.2, 128)
    # different model -> different key
    assert LlamaClient(cache_dir=tmp_path, model="other")._cache_key(
        "p", None, None, 0.2, 128
    ) != k1
    # different served fingerprint -> different key
    assert LlamaClient(cache_dir=tmp_path, served_fingerprint="sha:dead")._cache_key(
        "p", None, None, 0.2, 128
    ) != k1


def test_cache_version_and_ttl_invalidation(tmp_path):
    c = LlamaClient(cache_dir=tmp_path)
    key = c._cache_key("p", None, None, 0.2, 128)
    c._cache_put(key, {"result": "hi", "raw": "hi"})
    assert c._cache_get(key)["result"] == "hi"

    path = c._cache_path(key)
    # stale CACHE_VERSION -> miss
    path.write_bytes(
        orjson.dumps({"result": "hi", "v": CACHE_VERSION + 99, "ts": time.time()})
    )
    assert c._cache_get(key) is None
    # pre-versioned record (no "v") -> miss
    path.write_bytes(orjson.dumps({"result": "hi"}))
    assert c._cache_get(key) is None

    # TTL expiry
    c_ttl = LlamaClient(cache_dir=tmp_path, cache_ttl=10.0)
    path.write_bytes(
        orjson.dumps({"result": "hi", "v": CACHE_VERSION, "ts": time.time() - 100})
    )
    assert c_ttl._cache_get(key) is None
    path.write_bytes(
        orjson.dumps({"result": "hi", "v": CACHE_VERSION, "ts": time.time()})
    )
    assert c_ttl._cache_get(key)["result"] == "hi"


def test_schema_reply_cache_gate(tmp_path):
    schema = {"type": "object"}
    c = LlamaClient(cache_dir=tmp_path, cache_schema_replies=False)
    # Offline: shortcut the transport with a fixed reply (no server).
    c._request = lambda prompt, system, temperature, max_tokens, images=None: '{"a": 1}'

    # gate OFF: schema reply is NOT cached
    assert c.complete("p", schema=schema) == {"a": 1}
    assert list(tmp_path.glob("*.json")) == []

    # gate ON: schema reply IS cached
    c.cache_schema_replies = True
    assert c.complete("p", schema=schema) == {"a": 1}
    assert len(list(tmp_path.glob("*.json"))) == 1

    # now served from cache: a changed transport reply is ignored
    c._request = lambda prompt, system, temperature, max_tokens, images=None: '{"z": 9}'
    assert c.complete("p", schema=schema) == {"a": 1}


# --------------------------------------------------------------------------- #
# A12 — manifest generate
# --------------------------------------------------------------------------- #
def _cands():
    return [
        Candidate(address=0xD347, width=3, score=0.9, direction="increasing"),
        Candidate(address=0xD356, width=1, score=0.8, direction="decreasing"),
    ]


def test_fallback_manifest_validates_and_labels():
    m = fallback_manifest(_cands(), "roms/pokemon_yellow.gb")
    assert m.validate() == []
    assert m.game == "pokemon_yellow"
    dirs = {pd.id: pd.dir for pd in m.progress_dimensions}
    assert dirs["addr_d347"] == "up"
    assert dirs["addr_d356"] == "down"


def test_fallback_manifest_empty_candidates_still_valid():
    m = fallback_manifest([], "tetris.gb")
    assert m.validate() == []
    assert len(m.progress_dimensions) >= 1


def test_generate_no_llm_uses_fallback():
    m = generate_manifest(_cands(), "roms/x.gb", use_llm=False)
    assert m.validate() == []
    assert m.game == "x"


class _StubClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, prompt, schema=None, system=None, **kw):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_generate_llm_path_uses_reply():
    reply = {
        "game": "Custom Game",
        "milestone_label": "Score",
        "progress_dimensions": [
            {"id": "s", "label": "Score", "source": {"ram": {"addr": "0xD000"}}, "dir": "up"}
        ],
    }
    stub = _StubClient(reply)
    m = generate_manifest(_cands(), "roms/x.gb", client=stub)
    assert stub.calls == 1
    assert m.game == "Custom Game"
    assert m.validate() == []


def test_generate_llm_failure_falls_back():
    stub = _StubClient(RuntimeError("no server reachable"))
    m = generate_manifest(_cands(), "roms/x.gb", client=stub)
    assert m.validate() == []
    assert m.game == "x"  # fallback derived from ROM name


def test_generate_llm_invalid_reply_falls_back():
    # Shape-plausible but fails Manifest.validate() (empty game/label, no dims).
    stub = _StubClient({"game": "", "milestone_label": "", "progress_dimensions": []})
    m = generate_manifest(_cands(), "roms/yellow.gb", client=stub)
    assert m.validate() == []
    assert m.game == "yellow"


def test_generate_saves_when_requested(tmp_path):
    p = tmp_path / "manifest.json"
    generate_manifest(_cands(), "roms/x.gb", use_llm=False, save_path=p)
    assert p.exists()
    assert Manifest.load_json(p).validate() == []


def test_build_prompt_lists_addresses():
    prompt = build_prompt("roms/x.gb", _cands(), symbols={0xD347: "wPlayerMoney"})
    assert "0xD347" in prompt
    assert "wPlayerMoney" in prompt


# --------------------------------------------------------------------------- #
# A12 — ram_addresses_from_manifest (consumption helper)
# --------------------------------------------------------------------------- #
def test_ram_addresses_from_example_manifest():
    m = Manifest.load_json(EXAMPLE_MANIFEST)
    addrs = ram_addresses_from_manifest(m)
    # spatial taps first, then progress-dimension addrs
    assert 0xD35E in addrs  # spatial map_id
    assert 0xD356 in addrs  # badges progress dim
    assert 0xD347 in addrs  # money progress dim
    assert len(addrs) == len(set(addrs))  # de-duplicated


def test_ram_addresses_none_and_screen_only():
    assert ram_addresses_from_manifest(None) == []
    screen_only = Manifest(
        game="Tetris",
        milestone_label="Lines",
        progress_dimensions=[
            ProgressDimension(
                id="l", label="Lines", source={"screen": {"region": [0, 0, 1, 1]}}, dir="up"
            )
        ],
    )
    assert ram_addresses_from_manifest(screen_only) == []


# --------------------------------------------------------------------------- #
# A13 — Yellow-table quarantine behind manifest.game
# --------------------------------------------------------------------------- #
def _wram_oaks_lab():
    a = np.zeros(0x2000, dtype=np.uint8)
    a[0xD35E - 0xC000] = 40  # map id 40 == "Oak's Lab" in the Yellow table
    return a


def test_game_is_yellow():
    from pokeio.analytics.yellow import game_is_yellow

    yellow = Manifest(game="Pokemon Yellow", milestone_label="B", progress_dimensions=[])
    tetris = Manifest(game="Tetris", milestone_label="L", progress_dimensions=[])
    assert game_is_yellow(yellow) is True
    assert game_is_yellow(tetris) is False
    assert game_is_yellow(None) is False


def test_decode_no_manifest_uses_yellow_table():
    from pokeio.analytics.yellow import decode

    assert decode(_wram_oaks_lab()).map_name == "Oak's Lab"


def test_decode_nonyellow_manifest_no_label_leak():
    from pokeio.analytics.yellow import decode

    tetris = Manifest(
        game="Tetris",
        milestone_label="Lines",
        progress_dimensions=[
            ProgressDimension(id="l", label="Lines", source={"screen": {}}, dir="up")
        ],
    )
    s = decode(_wram_oaks_lab(), manifest=tetris)
    assert s.map_name.startswith("map $")  # numeric, no Yellow label leak
    assert s.badges is None


def test_milestones_quarantine():
    from pokeio.analytics.milestones import furthest_milestone, milestones_reached
    from pokeio.analytics.yellow import decode

    s = decode(_wram_oaks_lab())
    yellow = Manifest(
        game="Pokemon Yellow",
        milestone_label="B",
        progress_dimensions=[
            ProgressDimension(id="b", label="B", source={"ram": {"addr": "0x1"}}, dir="up")
        ],
    )
    tetris = Manifest(
        game="Tetris",
        milestone_label="L",
        progress_dimensions=[
            ProgressDimension(id="l", label="L", source={"screen": {}}, dir="up")
        ],
    )
    assert milestones_reached([s]) != []  # no manifest -> Yellow ladder
    assert milestones_reached([s], yellow) != []  # Yellow manifest -> ladder
    assert milestones_reached([s], tetris) == []  # other game -> quarantined
    assert furthest_milestone([s], tetris) is None
