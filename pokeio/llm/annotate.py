r"""[MF] Offline LLM pairwise progress-annotation (spec §2 / §10).

Turns a queue of clip PAIRS into forced-choice progress verdicts from the GLM
oversight LLM, appended to ``runs/<id>/pref/labels.jsonl``. The LLM is a FROZEN
offline annotator here — it never scores rollouts live; a background
:class:`AnnotationWorker` drains the queue on card0 and only writes labels to
disk, so the CPU/card1 evolution loop never blocks on it (spec §1, constraint 1).

The rubric is a FIXED, GENERIC one — "which clip shows more progress toward
completing whatever game this is, or tie?" — that names NO game and NO mechanic
(the LABEL wall, spec §9). The clip descriptions handed in are already anonymized
(see :func:`pokeio.reward.pairs.describe_clip`), so no game-specific string reaches
the model. Every verdict flows through :meth:`LlamaClient.complete` with
:data:`PAIR_SCHEMA` (enum-enforced ``winner``, ``confidence`` 0..1); the client's
on-disk cache (keyed by the stable pair hash) makes re-runs free.

Never-crash discipline: absent server / LLMError / malformed reply all degrade to
``None`` (skipped, no label), never an exception into the loop.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import orjson

# JSON schema handed to LlamaClient.complete so the reply is enum-constrained: a
# forced-choice winner + a scalar confidence (prose is rejected, spec §2).
PAIR_SCHEMA: dict = {
    "type": "object",
    "required": ["winner", "confidence"],
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "confidence": {"type": "number"},
    },
}

# FIXED generic rubric — NO game name, NO mechanic, NO Yellow constant (spec §9
# LABEL wall). "whatever game this is" is deliberate: the model must judge PROGRESS
# from the anonymized clip features alone.
PAIR_RUBRIC = (
    "You are shown structured feature summaries of two short gameplay clips, A and "
    "B, from the same unknown game. Judge which clip shows MORE PROGRESS toward "
    "completing whatever game this is. Progress means advancing the game state "
    "(reaching new areas, increasing tracked counters, crossing milestones), not "
    "merely more on-screen motion. If the two clips show equal progress, answer "
    '"tie". Report your confidence in [0,1].'
)

PAIR_SYSTEM = (
    "You compare two short game clips and judge which shows more progress toward "
    "finishing the game. Respond with a single JSON object only, no prose."
)


def build_pair_prompt(desc_a: str, desc_b: str, *, rubric: str = PAIR_RUBRIC) -> str:
    """Build the pairwise-preference prompt from two ANONYMIZED clip descriptions.

    Contains only the fixed generic rubric + the two descriptions + the answer
    contract. Callers must pass game-agnostic descriptions (see
    :func:`pokeio.reward.pairs.describe_clip`); this function adds no game noun.
    """
    return (
        f"{rubric}\n\n"
        f"Clip A: {desc_a}\n"
        f"Clip B: {desc_b}\n\n"
        'Answer with JSON: {"winner": "A" | "B" | "tie", "confidence": <0..1>}.'
    )


def annotate_pair(client, desc_a: str, desc_b: str, *, use_cache: bool = True):
    """One forced-choice verdict, or ``None`` on any failure (never raises).

    ``client`` is a :class:`~pokeio.llm.client.LlamaClient` (or a duck-typed stub
    exposing ``complete(prompt, schema, system=, use_cache=)``). Degrades to
    ``None`` when ``client`` is None, the server is unreachable, or the reply is
    malformed — the norm in tests/headless.
    """
    if client is None:
        return None
    prompt = build_pair_prompt(desc_a, desc_b)
    try:
        data = client.complete(prompt, PAIR_SCHEMA, system=PAIR_SYSTEM, use_cache=use_cache)
    except Exception:  # noqa: BLE001 — LLMError / transport / schema all degrade to skip
        return None
    if not isinstance(data, dict):
        return None
    winner = str(data.get("winner", "tie"))
    if winner not in ("A", "B", "tie"):
        winner = "tie"
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = float(np.clip(conf, 0.0, 1.0))
    return {"winner": winner, "confidence": conf}


class AnnotationWorker(threading.Thread):
    """Background worker: drain a pair queue -> LLM verdicts -> ``labels.jsonl``.

    Runs on card0 (the LLM's card) and only appends label records to disk; the
    evolution loop enqueues pairs at generation boundaries and never blocks. Mirror
    of the retina freeze cadence (spec §7): annotation is fully asynchronous and
    absorbed by the on-disk cache + swap-every-N discipline.

    Robust by design: each item is handled in a try/except so a bad pair, an absent
    server, or a malformed reply increments ``n_failed`` and the worker keeps
    draining — it never crashes the run. Use :meth:`drain_now` for a synchronous,
    thread-free drain (deterministic in tests/headless).
    """

    _SENTINEL = None

    def __init__(self, client, out_path, *, use_cache: bool = True,
                 poll: float = 0.1, daemon: bool = True):
        super().__init__(daemon=daemon)
        self.client = client
        self.out_path = Path(out_path)
        self.queue: "queue.Queue" = queue.Queue()
        self.use_cache = bool(use_cache)
        self.poll = float(poll)
        self._stop = threading.Event()
        self.n_written = 0
        self.n_failed = 0

    # -- producer side ------------------------------------------------------
    def enqueue(self, item: dict) -> None:
        """Enqueue a pair item (see :func:`pokeio.reward.pairs.build_pair_item`)."""
        self.queue.put(item)

    def enqueue_many(self, items) -> None:
        for it in items:
            self.queue.put(it)

    # -- consumer side ------------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=self.poll)
            except queue.Empty:
                continue
            try:
                if item is self._SENTINEL:
                    break
                self._handle(item)
            except Exception:  # noqa: BLE001 — worker must never crash the run
                self.n_failed += 1
            finally:
                self.queue.task_done()

    def drain_now(self) -> int:
        """Process all currently-queued items inline (no background thread).

        Returns the number of labels written. Never blocks on an empty queue and
        never raises — the headless/test path."""
        written0 = self.n_written
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                if item is not self._SENTINEL:
                    self._handle(item)
            except Exception:  # noqa: BLE001
                self.n_failed += 1
            finally:
                self.queue.task_done()
        return self.n_written - written0

    def _handle(self, item: dict) -> None:
        verdict = annotate_pair(
            self.client, item.get("desc_a", ""), item.get("desc_b", ""),
            use_cache=self.use_cache)
        if verdict is None:
            self.n_failed += 1
            return
        rec = {
            "pair_hash": item.get("pair_hash"),
            "a_hash": item.get("a_hash"),
            "b_hash": item.get("b_hash"),
            "strategy": item.get("strategy"),
            "weak_dir": item.get("weak_dir"),
            "winner": verdict["winner"],
            "confidence": verdict["confidence"],
        }
        self._append(rec)

    def _append(self, rec: dict) -> None:
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_path, "ab") as fh:
                fh.write(orjson.dumps(rec))
                fh.write(b"\n")
            self.n_written += 1
        except OSError:
            self.n_failed += 1  # disk is best-effort; never fail the run over it

    # -- lifecycle ----------------------------------------------------------
    def stop(self, *, drain: bool = False, timeout: float | None = 5.0) -> None:
        """Signal the worker to finish. ``drain`` waits for the queue to empty first."""
        if drain:
            self.queue.join()
        self._stop.set()
        self.queue.put(self._SENTINEL)  # unblock a get() that is waiting
        if self.is_alive():
            self.join(timeout=timeout)


def read_labels(path) -> list[dict]:
    """Read a ``labels.jsonl`` file into a list of records (missing file -> [])."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_bytes().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(orjson.loads(line))
        except orjson.JSONDecodeError:
            continue
    return out


__all__ = [
    "PAIR_SCHEMA",
    "PAIR_RUBRIC",
    "PAIR_SYSTEM",
    "build_pair_prompt",
    "annotate_pair",
    "AnnotationWorker",
    "read_labels",
]
