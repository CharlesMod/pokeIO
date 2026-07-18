"""The dashboard STOP button end-to-end control-file contract: the server
(``/api/stop`` -> :func:`pokeio.dash.serve._stop`) writes ``runs/<id>/stop.json``
and the trainer (:func:`pokeio.train.loop._stop_requested`) reads AND consumes it.

The consume is load-bearing: a run that stopped writes a checkpoint; if the flag
persisted, a later ``--resume`` would see it and immediately re-stop. These pin
both halves + the round-trip so the two never drift apart.
"""

from __future__ import annotations

import json

import pokeio.dash.serve as serve
from pokeio.train.loop import _stop_requested


# --------------------------------------------------------------------------
# server side: /api/stop -> stop.json
# --------------------------------------------------------------------------
def test_server_stop_writes_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "RUNS_DIR", tmp_path)
    (tmp_path / "myrun").mkdir()
    r = serve._stop("myrun")
    assert r == {"ok": True, "stopping": "myrun"}
    f = tmp_path / "myrun" / serve.STOP_FILENAME
    assert f.exists()
    assert json.loads(f.read_text())["stop"] is True


def test_server_stop_rejects_bad_run(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "RUNS_DIR", tmp_path)
    assert serve._stop("does-not-exist") == {"ok": False}
    assert serve._stop(None) == {"ok": False}
    assert serve._stop("") == {"ok": False}


# --------------------------------------------------------------------------
# trainer side: _stop_requested reads AND consumes
# --------------------------------------------------------------------------
def test_stop_requested_false_when_absent(tmp_path):
    assert _stop_requested(tmp_path) is False


def test_stop_requested_true_then_consumes(tmp_path):
    (tmp_path / "stop.json").write_text(json.dumps({"stop": True, "ts": 0}))
    assert _stop_requested(tmp_path) is True
    # consumed -> a --resume must NOT immediately re-stop
    assert not (tmp_path / "stop.json").exists()
    assert _stop_requested(tmp_path) is False


# --------------------------------------------------------------------------
# the contract that must not drift: server writes what the trainer reads
# --------------------------------------------------------------------------
def test_stop_roundtrip_server_to_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "RUNS_DIR", tmp_path)
    run = tmp_path / "run1"
    run.mkdir()
    assert serve._stop("run1")["ok"] is True          # UI click
    assert _stop_requested(run) is True               # trainer poll sees + consumes it
    assert _stop_requested(run) is False              # gone (resume-safe)
