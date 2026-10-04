"""Training-wall HTTP server (stdlib only).

Started in a daemon thread by the trainer (``dash.enabled``), or standalone to
browse a finished/running run's last persisted state::

    python -m pokeio dash --run runs/<name> [--port 8600]

Binds 0.0.0.0 by default so the wall is reachable from another machine (e.g.
over Tailscale). Endpoints:

    GET /                 the wall (wall.html)
    GET /api/state        charts, milestones, events, heatmaps, frontier, leaders
    GET /api/live?since=N wall frames + focused-agent frames newer than N + saliency
    GET /api/focus?env=N  focus an agent  (?follow=1 to follow the frontier again)
    GET /api/stop?confirm=1  graceful stop: checkpoint, then exit
    (focus/stop require the X-Pokeio-Token header that the served page embeds, so a
    random web page or <img> tag can't drive the run)
    GET /api/host         CPU per core / GPU utilisation (psutil / nvidia-smi if present)
"""

from __future__ import annotations

import json
import secrets
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"


class _Host:
    def __init__(self):
        self._cache: tuple[float, dict] = (0.0, {})

    def get(self) -> dict:
        t, d = self._cache
        if time.time() - t < 2.0:
            return d
        out: dict = {}
        try:
            import psutil

            out["cpu_per_core"] = psutil.cpu_percent(percpu=True)
            out["mem_pct"] = psutil.virtual_memory().percent
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            out["gpus"] = [
                dict(zip(("index", "name", "util", "mem_used", "mem_total", "temp"), [c.strip() for c in line.split(",")]))
                for line in r.stdout.strip().splitlines()
                if line.strip()
            ]
        except Exception:
            pass
        self._cache = (time.time(), out)
        return out


class _Offline:
    """Serves a run's persisted dash_state.json when no trainer is attached."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def snapshot_state(self) -> dict:
        try:
            d = json.loads((self.run_dir / "dash_state.json").read_text())
            d["status"] = "offline"
            return d
        except Exception:
            return {"status": "offline", "error": f"no dash_state.json in {self.run_dir}"}

    def snapshot_live(self, since: int = 0) -> dict:
        return {"status": "offline", "wall": {}, "hero_frames": []}

    def focus(self, env, follow) -> None:
        pass

    def request_stop(self) -> None:
        pass


def _handler(hub, host: _Host):
    token = secrets.token_hex(16)
    page = (STATIC / "wall.html").read_text().replace("__POKEIO_TOKEN__", token).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(200, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                control = u.path in ("/api/focus", "/api/stop")
                if control and self.headers.get("X-Pokeio-Token") != token:
                    self._send(403, b"missing or bad X-Pokeio-Token", "text/plain")
                    return
                if u.path in ("/", "/index.html", "/wall.html"):
                    self._send(200, page, "text/html; charset=utf-8")
                elif u.path == "/api/state":
                    self._json(hub.snapshot_state())
                elif u.path == "/api/live":
                    self._json(hub.snapshot_live(int(q.get("since", 0))))
                elif u.path == "/api/focus":
                    env = int(q["env"]) if "env" in q else None
                    follow = q.get("follow") == "1" if "follow" in q else None
                    hub.focus(env, follow)
                    self._json({"ok": True})
                elif u.path == "/api/stop":
                    if q.get("confirm") == "1":
                        hub.request_stop()
                    self._json({"ok": q.get("confirm") == "1"})
                elif u.path == "/api/host":
                    self._json(host.get())
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # never take the trainer down from the web thread
                try:
                    self._send(500, str(e).encode(), "text/plain")
                except Exception:
                    pass

    return H


def start_server(hub, host: str = "0.0.0.0", port: int = 8600) -> ThreadingHTTPServer | None:
    try:
        srv = ThreadingHTTPServer((host, port), _handler(hub, _Host()))
    except OSError as e:
        print(f"[dash] could not bind {host}:{port}: {e} (training continues without the wall)", flush=True)
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="pokeio-dash", daemon=True).start()
    print(f"[dash] training wall on http://{host}:{port}/", flush=True)
    return srv


def serve_offline(run_dir: str | Path, host: str = "0.0.0.0", port: int = 8600) -> None:
    srv = ThreadingHTTPServer((host, port), _handler(_Offline(Path(run_dir)), _Host()))
    print(f"[dash] offline wall for {run_dir} on http://{host}:{port}/", flush=True)
    srv.serve_forever()
