r"""[MF] Unit tests for offline LLM pairwise annotation (docs/specs/manifest-reward.md §2/§10).

Isolated — imports only :mod:`pokeio.llm.annotate`, no live server, no GPU. Covers
the enum-constrained :data:`PAIR_SCHEMA`; the fixed GENERIC rubric that names NO
game constant (the LABEL wall, §9); ``annotate_pair`` parsing + clamping of a valid
schema reply; and the never-crash discipline — an absent / raising / malformed
client degrades to ``None`` and the :class:`AnnotationWorker` keeps draining
without blocking or crashing the run (§1 constraint 1).
"""

from __future__ import annotations

from pokeio.llm.annotate import (
    PAIR_RUBRIC,
    PAIR_SCHEMA,
    PAIR_SYSTEM,
    AnnotationWorker,
    annotate_pair,
    build_pair_prompt,
    read_labels,
)


class _StubClient:
    """Duck-typed LlamaClient: returns a fixed reply (or raises if it is an Exception)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, prompt, schema=None, *, system=None, use_cache=True, **kw):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# Game-specific tokens that must NEVER appear in the rubric/prompt (LABEL wall §9).
_BANNED = [
    "pokemon", "pokémon", "yellow", "pikachu", "badge", "pokedex", "gym",
    "trainer", "pallet", "oak", "charmander", "bulbasaur", "squirtle",
    "cerulean", "viridian", "nintendo", "gameboy", "game boy", "tetris", "mario",
]


# --------------------------------------------------------------------------- #
# PAIR_SCHEMA structure
# --------------------------------------------------------------------------- #
def test_pair_schema_structure():
    assert PAIR_SCHEMA["type"] == "object"
    assert set(PAIR_SCHEMA["required"]) == {"winner", "confidence"}
    props = PAIR_SCHEMA["properties"]
    assert props["winner"]["enum"] == ["A", "B", "tie"]
    assert props["confidence"]["type"] == "number"


# --------------------------------------------------------------------------- #
# LABEL wall — no game constant in the rubric, system prompt, or built prompt
# --------------------------------------------------------------------------- #
def test_rubric_and_prompt_carry_no_game_constant():
    prompt = build_pair_prompt("[motion=0.10; c0:1]", "[motion=0.30; c0:3]")
    for text in (PAIR_RUBRIC, PAIR_SYSTEM, prompt):
        low = text.lower()
        for tok in _BANNED:
            assert tok not in low, f"game constant {tok!r} leaked into the prompt"


def test_build_pair_prompt_embeds_descriptions_and_contract():
    prompt = build_pair_prompt("DESC_A", "DESC_B")
    assert "DESC_A" in prompt and "DESC_B" in prompt
    assert '"winner"' in prompt and '"confidence"' in prompt


# --------------------------------------------------------------------------- #
# annotate_pair — parse a valid schema reply + clamp/coerce
# --------------------------------------------------------------------------- #
def test_annotate_pair_parses_valid_reply():
    client = _StubClient({"winner": "A", "confidence": 0.9})
    out = annotate_pair(client, "a", "b")
    assert out == {"winner": "A", "confidence": 0.9}
    assert client.calls == 1


def test_annotate_pair_clamps_and_coerces():
    # annotate_pair(client, desc_a, desc_b) — descriptions are required positionals.
    assert annotate_pair(_StubClient({"winner": "B", "confidence": 1.5}), "a", "b")["confidence"] == 1.0
    assert annotate_pair(_StubClient({"winner": "B", "confidence": -3}), "a", "b")["confidence"] == 0.0
    # illegal winner falls back to tie; a missing confidence defaults to 0.5.
    assert annotate_pair(_StubClient({"winner": "C"}), "a", "b")["winner"] == "tie"
    assert annotate_pair(_StubClient({"winner": "tie"}), "a", "b")["confidence"] == 0.5


def test_annotate_pair_arguments_use_stub_positionally():
    # descriptions default to empty when omitted; call signature stays stable.
    out = annotate_pair(_StubClient({"winner": "tie", "confidence": 0.5}), "x", "y")
    assert out["winner"] == "tie"


# --------------------------------------------------------------------------- #
# never-crash: absent / raising / malformed client -> None
# --------------------------------------------------------------------------- #
def test_annotate_pair_degrades_to_none():
    assert annotate_pair(None, "a", "b") is None                      # no client
    assert annotate_pair(_StubClient(RuntimeError("no server")), "a", "b") is None
    assert annotate_pair(_StubClient("not-a-dict"), "a", "b") is None  # malformed reply
    assert annotate_pair(_StubClient(None), "a", "b") is None


# --------------------------------------------------------------------------- #
# AnnotationWorker — synchronous drain, graceful degradation, no block
# --------------------------------------------------------------------------- #
def _item(i: int) -> dict:
    return {"desc_a": f"[a{i}]", "desc_b": f"[b{i}]", "pair_hash": f"h{i}",
            "a_hash": f"a{i}", "b_hash": f"b{i}", "strategy": "temporal",
            "weak_dir": "B"}


def test_worker_drain_writes_labels(tmp_path):
    out = tmp_path / "pref" / "labels.jsonl"
    w = AnnotationWorker(_StubClient({"winner": "A", "confidence": 0.8}), out)
    w.enqueue_many([_item(0), _item(1)])
    written = w.drain_now()
    assert written == 2 and w.n_written == 2 and w.n_failed == 0
    recs = read_labels(out)
    assert len(recs) == 2
    assert recs[0]["winner"] == "A" and recs[0]["confidence"] == 0.8
    assert recs[0]["pair_hash"] == "h0" and recs[0]["strategy"] == "temporal"


def test_worker_degrades_with_absent_client(tmp_path):
    out = tmp_path / "labels.jsonl"
    w = AnnotationWorker(None, out)             # no client at all
    w.enqueue_many([_item(0), _item(1)])
    written = w.drain_now()                      # must not raise or block
    assert written == 0 and w.n_written == 0 and w.n_failed == 2
    assert read_labels(out) == []               # nothing written, file absent


def test_worker_degrades_with_raising_client(tmp_path):
    out = tmp_path / "labels.jsonl"
    w = AnnotationWorker(_StubClient(RuntimeError("boom")), out)
    w.enqueue_many([_item(0)])
    assert w.drain_now() == 0                    # skipped, never crashes
    assert w.n_failed == 1


def test_worker_thread_drains_async_without_blocking(tmp_path):
    # Drive the background thread and prove it drains the queue without blocking the
    # producer, then shut it down via the public ``stop()`` lifecycle — which also
    # guards the fix for the ``self._stop`` / ``threading.Thread._stop`` shadowing
    # bug (a raw ``threading.Event`` on ``_stop`` used to make ``stop()``'s join
    # raise ``TypeError: 'Event' object is not callable``).
    out = tmp_path / "labels.jsonl"
    w = AnnotationWorker(_StubClient({"winner": "B", "confidence": 0.6}), out,
                         poll=0.01)
    w.start()
    try:
        w.enqueue_many([_item(0), _item(1), _item(2)])
        w.queue.join()                      # returns once every item is processed
        assert w.n_written == 3
        assert len(read_labels(out)) == 3
    finally:
        w.stop(timeout=3.0)                 # public lifecycle (fixed _stop shadowing): sentinel + join
    assert not w.is_alive()                 # clean shutdown, no TypeError


def test_read_labels_missing_file_is_empty(tmp_path):
    assert read_labels(tmp_path / "nope.jsonl") == []
