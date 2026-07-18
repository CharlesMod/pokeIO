"""Thin client for a llama.cpp server (the GLM-4.7-Flash oversight LLM).

Talks to a llama.cpp / llama-swap server over either the OpenAI-compatible
``/v1/chat/completions`` endpoint (default) or the native ``/completion`` one.

Design goals (mirrors the rest of pokeIO): dependency-light — stdlib + orjson
only. ``requests`` is *optional*; if it is importable we use it, otherwise we
fall back to :mod:`urllib`. JSON-schema validation uses :mod:`jsonschema` when
installed, else a small built-in validator covering the common keywords.

Features
--------
* Configurable ``base_url`` / ``model`` / endpoint (``api="chat"|"completion"``).
* :meth:`LlamaClient.complete` — one-shot text completion. Given a ``schema`` it
  parses the model's reply as JSON, validates it, and on failure re-prompts the
  model with a repair instruction (bounded retries) before giving up.
* Timeout + retry-with-exponential-backoff on transport / 5xx errors.
* On-disk response cache under ``runs/llm_cache/`` keyed by a hash of the full
  request (model, endpoint, prompt, system, schema, sampling params). Cache hits
  skip the network entirely; pass ``use_cache=False`` to bypass.

Run ``python -m pokeio.llm.client`` for a smoke test against a live server.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

# Optional, faster HTTP; fall back to urllib if absent.
try:  # pragma: no cover - trivial import guard
    import requests as _requests
except Exception:  # noqa: BLE001
    _requests = None

# Optional, richer schema validation; fall back to a built-in validator.
try:  # pragma: no cover - trivial import guard
    import jsonschema as _jsonschema
except Exception:  # noqa: BLE001
    _jsonschema = None


# Default points at the dedicated single-card GLM oversight server (card 0 only,
# 4k ctx) so card 1 stays free for evolutionary training. The dual-card llama-swap
# endpoint (http://localhost:9292) remains usable by passing base_url explicitly.
DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "glm-4.7-flash"
# runs/llm_cache/ relative to the repo root (…/pokeIO/pokeio/llm/client.py -> parents[2]).
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "runs" / "llm_cache"

# Bump to invalidate every on-disk cache entry when the request/reply contract
# changes (a prompt-format fix, a schema-handling change, …). Entries tagged
# with a different version are treated as cache misses. See _cache_key/_cache_get.
CACHE_VERSION = 1


def _normalize_base_url(url: str | None) -> str:
    """Normalize an LLM server base URL for this client.

    The client always appends its own path (``/v1/chat/completions`` for the
    chat API, ``/completion`` for the native one), so a ``base_url`` that
    already ends in ``/v1`` would produce a double ``/v1`` on the wire. Strip
    exactly one trailing ``/v1`` segment (plus any trailing slashes).
    """
    u = (url or DEFAULT_BASE_URL).strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u.rstrip("/")


def _normalize_model(model: str | None) -> str:
    """Normalize a model alias to the served (lowercase) form.

    llama-swap routes on a lowercase alias in this project (``glm-4.7-flash``)
    while the config default is mixed-case (``GLM-4.7-Flash``); lowercasing
    avoids a routing miss. Applied defensively, only via :meth:`from_config`.
    """
    return (model or DEFAULT_MODEL).strip().lower()


class LLMError(RuntimeError):
    """Raised when the server cannot be reached or returns an unusable reply."""


class SchemaError(LLMError):
    """Raised when a reply cannot be coerced into the requested JSON schema."""


# --------------------------------------------------------------------------- #
# Minimal JSON-schema validation (used only when `jsonschema` is unavailable).
# --------------------------------------------------------------------------- #
_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _mini_validate(instance: Any, schema: dict, path: str = "$") -> None:
    """Validate ``instance`` against a small subset of JSON Schema.

    Supports: ``type``, ``enum``, ``required``, ``properties``, ``items``.
    Raises :class:`SchemaError` on the first violation.
    """
    expected = schema.get("type")
    if expected is not None:
        py = _TYPE_MAP.get(expected)
        # bool is a subclass of int — reject it for integer/number.
        if expected in ("integer", "number") and isinstance(instance, bool):
            raise SchemaError(f"{path}: expected {expected}, got boolean")
        if py is not None and not isinstance(instance, py):
            raise SchemaError(f"{path}: expected {expected}, got {type(instance).__name__}")

    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaError(f"{path}: {instance!r} not in enum {schema['enum']}")

    if expected == "object" or isinstance(instance, dict):
        if isinstance(instance, dict):
            for key in schema.get("required", []):
                if key not in instance:
                    raise SchemaError(f"{path}: missing required property {key!r}")
            props = schema.get("properties", {})
            for key, subschema in props.items():
                if key in instance:
                    _mini_validate(instance[key], subschema, f"{path}.{key}")

    if (expected == "array" or isinstance(instance, list)) and "items" in schema:
        if isinstance(instance, list):
            for i, item in enumerate(instance):
                _mini_validate(item, schema["items"], f"{path}[{i}]")


def validate_schema(instance: Any, schema: dict) -> None:
    """Validate ``instance`` against ``schema``, raising :class:`SchemaError`."""
    if _jsonschema is not None:
        try:
            _jsonschema.validate(instance, schema)
        except _jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            raise SchemaError(str(exc).splitlines()[0]) from exc
    else:
        _mini_validate(instance, schema)


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from an LLM reply.

    Strips ``` fences and slices to the outermost ``{...}`` / ``[...]`` span
    before parsing. Raises :class:`SchemaError` if nothing parses.
    """
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) and any trailing fence.
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    # Try a straight parse first.
    try:
        return orjson.loads(s)
    except orjson.JSONDecodeError:
        pass
    # Fall back to the outermost brace/bracket span.
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    ends = [i for i in (s.rfind("}"), s.rfind("]")) if i != -1]
    if starts and ends:
        span = s[min(starts) : max(ends) + 1]
        try:
            return orjson.loads(span)
        except orjson.JSONDecodeError:
            pass
    raise SchemaError(f"no JSON value found in reply: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
@dataclass
class LlamaClient:
    """Thin client for a llama.cpp / llama-swap server.

    Parameters
    ----------
    base_url:
        Server root, e.g. ``http://localhost:9292`` (llama-swap) or a direct
        ``llama-server`` port such as ``http://localhost:8080``.
    model:
        Model / alias name sent in the request (used by llama-swap to route).
    api:
        ``"chat"`` for OpenAI-compatible ``/v1/chat/completions`` (default) or
        ``"completion"`` for the native llama.cpp ``/completion`` endpoint.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api: str = "chat"  # "chat" | "completion"
    api_key: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    backoff: float = 1.5  # seconds; multiplied by 2**attempt
    temperature: float = 0.7
    max_tokens: int = 1024
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    schema_retries: int = 2  # extra repair attempts when a schema is supplied
    # -- cache invalidation ------------------------------------------------- #
    served_fingerprint: str | None = None  # served-model id/hash mixed into keys
    cache_ttl: float | None = None  # seconds; older entries are treated as misses
    cache_schema_replies: bool = True  # gate: cache schema-validated replies?

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.cache_dir = Path(self.cache_dir)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Any) -> "LlamaClient":
        """Build a client from an :class:`~pokeio.config.LLMConfig`-like object.

        Normalizes defensively: strips a trailing ``/v1`` from ``base_url`` (the
        client appends its own) and lowercases the ``model`` alias, so a config
        whose ``base_url`` is ``http://host:8080/v1`` and whose ``model`` is
        ``GLM-4.7-Flash`` does not double the path or miss llama-swap routing.
        Maps ``timeout_s``/``max_retries``/``temperature``/``cache_dir`` across.
        Duck-typed: any object exposing those attributes works, so this module
        keeps zero import dependency on :mod:`pokeio.config`.
        """
        base_url = _normalize_base_url(getattr(cfg, "base_url", DEFAULT_BASE_URL))
        model = _normalize_model(getattr(cfg, "model", DEFAULT_MODEL))
        timeout = getattr(cfg, "timeout_s", None)
        if timeout is None:
            timeout = getattr(cfg, "timeout", 120.0)
        cache_dir = getattr(cfg, "cache_dir", None) or DEFAULT_CACHE_DIR
        return cls(
            base_url=base_url,
            model=model,
            timeout=float(timeout),
            max_retries=int(getattr(cfg, "max_retries", 3)),
            temperature=float(getattr(cfg, "temperature", 0.7)),
            cache_dir=Path(cache_dir),
        )

    # -- public API -------------------------------------------------------- #
    def complete(
        self,
        prompt: str,
        schema: dict | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
        images: list | None = None,
    ) -> Any:
        """Complete ``prompt``.

        Returns the reply *string* normally, or the parsed+validated JSON value
        when ``schema`` is given. On schema failure the model is re-prompted with
        a repair instruction up to ``schema_retries`` times before raising
        :class:`SchemaError`.

        ``images`` is an OPTIONAL list of base64 image parts (data URLs or raw
        base64 strings) for a future multimodal card0 server (spec §2a). It is sent
        only on the chat API; the text-only path is byte-unchanged when ``images``
        is omitted (the cache key and wire payload are identical to today).
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        key = self._cache_key(prompt, schema, system, temperature, max_tokens, images)
        # Schema replies can be plausible-but-wrong (validated once, still not
        # what we wanted); gate their caching behind cache_schema_replies so a
        # suspect reply can be forced to regenerate without touching free-text.
        cache_ok = use_cache and (schema is None or self.cache_schema_replies)
        if cache_ok:
            cached = self._cache_get(key)
            if cached is not None:
                return cached["result"]

        if schema is None:
            text = self._request(prompt, system, temperature, max_tokens, images)
            if cache_ok:
                self._cache_put(key, {"result": text, "raw": text})
            return text

        # Schema path: request, parse, validate; repair-loop on failure.
        repair_hint = (
            "\n\nRespond with ONLY a single JSON value that conforms to this JSON "
            "schema. No prose, no markdown fences.\nSchema:\n"
            + orjson.dumps(schema).decode()
        )
        last_err: Exception | None = None
        cur_prompt = prompt + repair_hint
        for attempt in range(self.schema_retries + 1):
            text = self._request(cur_prompt, system, temperature, max_tokens, images)
            try:
                value = extract_json(text)
                validate_schema(value, schema)
            except SchemaError as exc:
                last_err = exc
                cur_prompt = (
                    prompt
                    + repair_hint
                    + f"\n\nYour previous reply was invalid ({exc}). "
                    "Return corrected JSON only."
                )
                continue
            if cache_ok:
                self._cache_put(key, {"result": value, "raw": text})
            return value
        raise SchemaError(f"schema validation failed after retries: {last_err}")

    def ping(self) -> bool:
        """Return True if the server's ``/health`` endpoint reports ready."""
        try:
            status, body = self._http("GET", f"{self.base_url}/health", None)
        except LLMError:
            return False
        if status != 200:
            return False
        try:
            return orjson.loads(body).get("status") in (None, "ok", "loading")
        except orjson.JSONDecodeError:
            return True  # some builds return a bare 200

    # -- transport --------------------------------------------------------- #
    def _request(
        self, prompt: str, system: str | None, temperature: float, max_tokens: int,
        images: list | None = None,
    ) -> str:
        if self.api == "completion":
            url = f"{self.base_url}/completion"
            payload: dict[str, Any] = {
                "prompt": (f"{system}\n\n{prompt}" if system else prompt),
                "n_predict": max_tokens,
                "temperature": temperature,
                "cache_prompt": True,
            }
        else:
            url = f"{self.base_url}/v1/chat/completions"
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            # Text-only path (images is None) is byte-identical to before: a plain
            # string content. With images, use the OpenAI multimodal content-parts
            # shape so a future multimodal server can consume the rendered clips.
            if images:
                content: Any = [{"type": "text", "text": prompt}]
                for img in images:
                    url_val = img if str(img).startswith("data:") else f"data:image/png;base64,{img}"
                    content.append({"type": "image_url", "image_url": {"url": url_val}})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        status, body = self._http_retry("POST", url, payload)
        if status != 200:
            raise LLMError(f"{url} -> HTTP {status}: {body[:300]}")
        try:
            data = orjson.loads(body)
        except orjson.JSONDecodeError as exc:
            raise LLMError(f"non-JSON reply from {url}: {body[:200]!r}") from exc

        if self.api == "completion":
            return data.get("content", "")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected chat reply shape: {body[:300]!r}") from exc

    def _http_retry(self, method: str, url: str, payload: dict | None) -> tuple[int, str]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                status, body = self._http(method, url, payload)
            except LLMError as exc:
                last_err = exc
            else:
                if status < 500:
                    return status, body
                last_err = LLMError(f"HTTP {status}: {body[:200]}")
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff * (2**attempt))
        raise LLMError(f"request to {url} failed after {self.max_retries} tries: {last_err}")

    def _http(self, method: str, url: str, payload: dict | None) -> tuple[int, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = orjson.dumps(payload) if payload is not None else None

        if _requests is not None:
            try:
                resp = _requests.request(
                    method, url, data=data, headers=headers, timeout=self.timeout
                )
                return resp.status_code, resp.text
            except _requests.RequestException as exc:  # type: ignore[union-attr]
                raise LLMError(f"transport error: {exc}") from exc

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(f"transport error: {exc}") from exc

    # -- cache ------------------------------------------------------------- #
    def _cache_key(
        self,
        prompt: str,
        schema: dict | None,
        system: str | None,
        temperature: float,
        max_tokens: int,
        images: list | None = None,
    ) -> str:
        blob_dict = {
            # cache_version + served-model fingerprint invalidate stale
            # replies when the contract or the served weights change.
            "cache_version": CACHE_VERSION,
            "fingerprint": self.served_fingerprint or self.model,
            "base_url": self.base_url,
            "model": self.model,
            "api": self.api,
            "prompt": prompt,
            "system": system,
            "schema": schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Only mix images into the key when present, so text-only keys are
        # byte-identical to pre-multimodal ones (no cache invalidation).
        if images:
            blob_dict["images"] = [
                hashlib.sha256(str(img).encode()).hexdigest() for img in images
            ]
        blob = orjson.dumps(blob_dict, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            rec = orjson.loads(path.read_bytes())
        except (orjson.JSONDecodeError, OSError):
            return None
        if not isinstance(rec, dict):
            return None
        # Entries written by an older CACHE_VERSION (or the pre-versioned format,
        # which has no "v") are treated as misses so a bump invalidates cleanly.
        if rec.get("v") != CACHE_VERSION:
            return None
        if self.cache_ttl is not None:
            ts = rec.get("ts")
            if not isinstance(ts, (int, float)) or (time.time() - ts) > self.cache_ttl:
                return None
        return rec

    def _cache_put(self, key: str, value: dict) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            rec = dict(value)
            rec["v"] = CACHE_VERSION
            rec.setdefault("ts", time.time())
            self._cache_path(key).write_bytes(orjson.dumps(rec))
        except OSError:
            pass  # cache is best-effort; never fail a completion over it


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _main() -> None:
    client = LlamaClient()
    print(f"pokeio.llm.client — base_url={client.base_url} model={client.model}")
    if not client.ping():
        print("server not running")
        return
    print("server is up; sending a test prompt…")
    try:
        reply = client.complete(
            "Reply with a single short sentence confirming you are online.",
            max_tokens=64,
            use_cache=False,
        )
        print("reply:", reply.strip() if isinstance(reply, str) else reply)
    except LLMError as exc:
        print(f"server reachable but request failed: {exc}")


if __name__ == "__main__":
    _main()
