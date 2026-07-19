"""pokeIO training-wall server.

Serves the dashboard (``wall.html``) plus a small JSON API so the page can show a
LIVE run's telemetry with real host CPU/GPU stats. Bind ``0.0.0.0`` and reach it
over Tailscale (never localhost for this box).

    python -m pokeio.dash.serve --port 8600 [--run <run_id>]

HONESTY NOTE — what the served data actually reflects
-----------------------------------------------------
The live signal this server exposes (``/api/telemetry``, ``/api/live``) is the
REAL run: **novelty + mined-progress** reward terms and the champion network.
There is **no live "GLM judge", no co-evolved reward-genome population, and no
anti-Goodhart oversight loop** running — LLM oversight is NOT yet wired. Some
``wall.html`` panels (notably "Reward Genome" with its "co-evolved · unsupervised"
and "LLM judge" labels, and the "GLM judge" tag on GPU 0) are **static
placeholder mock-ups**, not backed by any endpoint here; treat them as
aspirational UI, not telemetry. Those strings live in ``wall.html`` (not owned by
this module); this server never fabricates a judge/co-evolution signal.

Endpoints
---------
* ``GET /``                      → wall.html
* ``GET /api/runs``              → {"runs": [{"id","generations","mtime"}, ...]}  newest first
* ``GET /api/telemetry?run=ID``  → {"run", "records":[...], "host":{cpu,gpu}}  (tail of records)
* ``GET /api/live?run=ID``       → parsed runs/<run-or-newest>/live.json (real frames + champion net) or {}
* ``GET /api/select?run=ID&idx=N`` → writes runs/<ID>/select.json (focus agent; -1 = champion)
* ``GET /api/pace?run=ID&mode=realtime|max`` → writes runs/<ID>/pace.json (live spectate toggle)
* ``GET /api/stop?run=ID``       → writes runs/<ID>/stop.json (graceful stop: checkpoint + free VRAM)
* ``GET /api/host``              → {"cpu_pct","cpu_per_core":[...],"gpu":[...]}

Dependency-light: stdlib + psutil (+ orjson via the telemetry reader). Never 500s
on a missing/partial run — returns empty/last-good instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import psutil

from pokeio.telemetry.schema import TELEMETRY_FILENAME, read_generations

DASH_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASH_DIR.parents[1]
RUNS_DIR = REPO_ROOT / "runs"
WALL_HTML = DASH_DIR / "wall.html"
BRAIN_HTML = DASH_DIR / "brain_wall.html"  # System-1 (gradient-RL) dashboard
LIVE_FILENAME = "live.json"  # real live-frame payload (frames + champion net), written ~3 Hz
MAX_RECORDS = 200  # tail length returned to the page

# prime psutil's per-interval counters so the first real call is non-blocking
psutil.cpu_percent(percpu=True)


# --------------------------------------------------------------------------- #
# host stats
# --------------------------------------------------------------------------- #
def _gpu_stats() -> list[dict]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        gpus = []
        for line in out.splitlines():
            idx, util, used, total = (p.strip() for p in line.split(","))
            gpus.append(
                {
                    "index": int(idx),
                    "util": float(util),
                    "mem_used_mb": float(used),
                    "mem_total_mb": float(total),
                }
            )
        return gpus
    except Exception:
        return []


def _host_stats() -> dict:
    try:
        per_core = psutil.cpu_percent(percpu=True)
        return {
            "cpu_pct": round(sum(per_core) / max(1, len(per_core)), 1),
            "cpu_per_core": [round(c, 1) for c in per_core],
            "gpu": _gpu_stats(),
            "t": time.time(),
        }
    except Exception:
        return {"cpu_pct": 0.0, "cpu_per_core": [], "gpu": [], "t": time.time()}


# --------------------------------------------------------------------------- #
# run discovery + telemetry
# --------------------------------------------------------------------------- #
def _run_dirs() -> list[Path]:
    if not RUNS_DIR.is_dir():
        return []
    dirs = [
        d
        for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / TELEMETRY_FILENAME).is_file()
    ]
    return sorted(dirs, key=lambda d: (d / TELEMETRY_FILENAME).stat().st_mtime, reverse=True)


def _brain_runs() -> dict:
    """System-1 runs (dirs with a ``kind:"system1"`` live.json) + a light summary for
    the dashboard run-rail — id, iter, phase, gate, reach, and whether it's live
    (live.json touched recently). No frames (the rail fetches those via /api/live)."""
    import time

    out = []
    if not RUNS_DIR.is_dir():
        return {"runs": []}
    dirs = sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        lj = d / LIVE_FILENAME
        if not d.is_dir() or not lj.is_file():
            continue
        try:
            live = json.loads(lj.read_text())
        except Exception:
            continue
        if live.get("kind") != "system1":
            continue
        m = live.get("metrics", {}) or {}
        mtime = lj.stat().st_mtime
        out.append({
            "id": d.name, "iter": live.get("iter", 0), "phase": live.get("phase", ""),
            "gate_stochastic": m.get("gate_stochastic"), "gate_greedy": m.get("gate_greedy"),
            "reach": m.get("reach"), "running": (time.time() - mtime) < 15, "mtime": mtime,
        })
    return {"runs": out}


def _list_runs() -> dict:
    runs = []
    for d in _run_dirs():
        tf = d / TELEMETRY_FILENAME
        try:
            n = sum(1 for _ in tf.open("rb"))
        except Exception:
            n = 0
        runs.append({"id": d.name, "generations": n, "mtime": tf.stat().st_mtime})
    return {"runs": runs}


def _resolve_run(run: str | None) -> Path | None:
    if run:
        d = RUNS_DIR / run
        if (d / TELEMETRY_FILENAME).is_file():
            return d
        return None
    dirs = _run_dirs()
    return dirs[0] if dirs else None


def _resolve_live_run(run: str | None) -> Path | None:
    """Resolve the run dir whose ``live.json`` we should serve.

    A live run may only have ``live.json`` (e.g. a synthetic sample with no
    telemetry yet), so this keys off ``LIVE_FILENAME`` rather than telemetry.
    """
    if run:
        d = RUNS_DIR / run
        return d if (d / LIVE_FILENAME).is_file() else None
    if not RUNS_DIR.is_dir():
        return None
    live_dirs = [
        d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / LIVE_FILENAME).is_file()
    ]
    if not live_dirs:
        return None
    return max(live_dirs, key=lambda d: (d / LIVE_FILENAME).stat().st_mtime)


def _live(run: str | None) -> dict:
    """Parsed ``runs/<run-or-newest>/live.json`` or ``{}`` if absent/partial.

    Never raises: a missing, truncated, or mid-write (invalid JSON) file yields
    ``{}`` so the page falls back to the demo animation instead of erroring.
    """
    d = _resolve_live_run(run)
    if d is None:
        return {}
    try:
        return json.loads((d / LIVE_FILENAME).read_bytes())
    except Exception:
        return {}


SELECT_FILENAME = "select.json"  # focus-agent selection, read by the trainer
PACE_FILENAME = "pace.json"  # spectate/max pace toggle, read by the trainer
STOP_FILENAME = "stop.json"  # graceful-stop request, read by the trainer
_PACE_MODES = ("realtime", "max")


def _pace(run: str | None, mode) -> dict:
    """Atomically write ``runs/<run>/pace.json`` = {"mode": m, "ts": now}.

    ``mode`` is "realtime" (whole swarm at authentic Game Boy speed) or "max"
    (flat-out training).  Missing run dir / bad mode -> {"ok": false} (always
    HTTP 200; never raises).  The RUNNING trainer stat-polls this file and
    switches live — no restart.
    """
    if not run or mode not in _PACE_MODES:
        return {"ok": False}
    d = RUNS_DIR / run
    if not d.is_dir():
        return {"ok": False}
    tmp = d / (PACE_FILENAME + ".tmp")
    try:
        tmp.write_text(json.dumps({"mode": mode, "ts": time.time()}))
        os.replace(tmp, d / PACE_FILENAME)
    except Exception:
        return {"ok": False}
    return {"ok": True, "mode": mode}


def _select(run: str | None, idx) -> dict:
    """Atomically write ``runs/<run>/select.json`` = {"idx": n, "ts": now}.

    ``idx`` is the player slot within the current wave (0-based) or -1 for the
    champion (the default state).  Missing run dir / bad args -> {"ok": false}
    (always HTTP 200; never raises).
    """
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return {"ok": False}
    if not run:
        return {"ok": False}
    d = RUNS_DIR / run
    if not d.is_dir():
        return {"ok": False}
    tmp = d / (SELECT_FILENAME + ".tmp")
    try:
        tmp.write_text(json.dumps({"idx": i, "ts": time.time()}))
        os.replace(tmp, d / SELECT_FILENAME)
    except Exception:
        return {"ok": False}
    return {"ok": True, "idx": i}


def _stop(run: str | None) -> dict:
    """Atomically write ``runs/<run>/stop.json`` = {"stop": true, "ts": now}.

    The RUNNING trainer stat-polls this at each generation boundary; on seeing it,
    it saves a checkpoint, tears down the fleet (freeing the worker procs + the
    inference GPU/VRAM), and exits cleanly — the run is resumable from the
    checkpoint. Missing run dir -> {"ok": false} (always HTTP 200; never raises).
    """
    if not run:
        return {"ok": False}
    d = RUNS_DIR / run
    if not d.is_dir():
        return {"ok": False}
    tmp = d / (STOP_FILENAME + ".tmp")
    try:
        tmp.write_text(json.dumps({"stop": True, "ts": time.time()}))
        os.replace(tmp, d / STOP_FILENAME)
    except Exception:
        return {"ok": False}
    return {"ok": True, "stopping": run}


def _telemetry(run: str | None) -> dict:
    d = _resolve_run(run)
    host = _host_stats()
    if d is None:
        return {"run": None, "records": [], "host": host}
    try:
        recs = read_generations(d)
        data = [r.to_dict() for r in recs[-MAX_RECORDS:]]
    except Exception:
        data = []
    return {"run": d.name, "records": data, "host": host}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "pokeIO-wall/0.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        try:
            if path in ("/", "/index.html", "/wall.html"):
                self._send(200, WALL_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path in ("/brain", "/brain.html", "/brain_wall.html"):
                self._send(200, BRAIN_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/runs":
                self._json(_list_runs())
            elif path == "/api/brain_runs":
                self._json(_brain_runs())
            elif path == "/api/telemetry":
                run = (parse_qs(u.query).get("run") or [None])[0]
                self._json(_telemetry(run))
            elif path == "/api/live":
                run = (parse_qs(u.query).get("run") or [None])[0]
                self._json(_live(run))
            elif path == "/api/select":
                q = parse_qs(u.query)
                run = (q.get("run") or [None])[0]
                idx = (q.get("idx") or [None])[0]
                self._json(_select(run, idx))
            elif path == "/api/pace":
                q = parse_qs(u.query)
                run = (q.get("run") or [None])[0]
                mode = (q.get("mode") or [None])[0]
                self._json(_pace(run, mode))
            elif path == "/api/stop":
                q = parse_qs(u.query)
                run = (q.get("run") or [None])[0]
                self._json(_stop(run))
            elif path == "/api/host":
                self._json(_host_stats())
            elif path == "/healthz":
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # never 500 hard
            self._json({"error": str(e)}, 200)

    def log_message(self, *a) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="pokeIO training-wall server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--run", default=None, help="pin a run id (default: newest)")
    args = ap.parse_args()

    global _PINNED_RUN
    _PINNED_RUN = args.run

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"pokeIO wall serving on http://{args.host}:{args.port}  (runs: {RUNS_DIR})")
    print(f"  Tailscale:  http://100.74.178.26:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
