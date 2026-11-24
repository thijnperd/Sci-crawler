"""A tiny, polite HTTP layer: retries, per-host pacing, on-disk cache.

Everything here is stdlib so the tool installs with no dependencies and runs
inside restricted agent sandboxes.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import Settings


class FetchError(RuntimeError):
    def __init__(self, url: str, status: int | None, message: str):
        super().__init__(f"{status or 'ERR'} {message} [{url}]")
        self.url = url
        self.status = status
        self.message = message


class Blocked(FetchError):
    """The host refused us (403/451/captcha) -- common for Sci-Hub mirrors."""


_LOCK = threading.Lock()
_LAST_CALL: dict[str, float] = {}
# Minimum seconds between calls to the same host, to stay in everyone's good books.
_PACE = {
    "api.crossref.org": 0.35,
    "api.openalex.org": 0.25,
    "www.ebi.ac.uk": 0.2,
    "api.semanticscholar.org": 1.1,
    "export.arxiv.org": 1.0,
    "api.unpaywall.org": 0.25,
}
_DEFAULT_PACE = 0.6


def _pace_for(host: str) -> float:
    for key, delay in _PACE.items():
        if key in host:
            return delay
    return _DEFAULT_PACE


def _throttle(host: str) -> None:
    delay = _pace_for(host)
    while True:
        with _LOCK:
            last = _LAST_CALL.get(host, 0.0)
            now = time.monotonic()
            wait = delay - (now - last)
            if wait <= 0:
                _LAST_CALL[host] = now
                return
        time.sleep(min(wait, 2.0))


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return cache_dir / "http" / digest[:2] / f"{digest}.json"


def _cache_read(cache_dir: Path, url: str, ttl: float) -> bytes | None:
    path = _cache_path(cache_dir, url)
    try:
        stats = path.stat()
    except FileNotFoundError:
        return None
    if ttl <= 0 or (time.time() - stats.st_mtime) > ttl:
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
        if "body_b64" in payload:
            return base64.b64decode(payload["body_b64"].encode("ascii"), validate=True)
        return payload["body"].encode("utf-8")
    except (OSError, ValueError, KeyError, UnicodeError):
        return None


def _cache_write(cache_dir: Path, url: str, body: bytes) -> None:
    path = _cache_path(cache_dir, url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        encoded = base64.b64encode(body).decode("ascii")
        tmp.write_text(json.dumps({"url": url, "body_b64": encoded}), "utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def fetch(
    url: str,
    settings: Settings,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float | None = None,
    retries: int = 2,
    cache_ttl: float = 0,
    accept: str = "*/*",
) -> bytes:
    """GET/POST a URL and return the body.

    ``cache_ttl`` caches successful responses for that many seconds; the cache
    is skipped entirely when caching is disabled or the request has a body.
    """
    use_cache = settings.cache and cache_ttl > 0 and data is None
    if use_cache:
        cached = _cache_read(settings.cache_dir, url, cache_ttl)
        if cached is not None:
            return cached

    host = urllib.parse.urlsplit(url).netloc
    timeout = timeout or settings.timeout
    hdrs = {
        "User-Agent": settings.user_agent,
        "Accept": accept,
        "Accept-Encoding": "identity",
    }
    if headers:
        hdrs.update(headers)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        _throttle(host)
        request = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                encoding = (response.headers.get("Content-Encoding") or "").lower()
                if encoding == "gzip":
                    body = gzip.decompress(body)
                if use_cache and body:
                    _cache_write(settings.cache_dir, url, body)
                return body
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (403, 451):
                raise Blocked(url, exc.code, f"blocked: {exc.reason}") from None
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 1.6 * (attempt + 1)
                except ValueError:
                    delay = 1.6 * (attempt + 1)
                time.sleep(min(delay, 8.0))
                continue
            raise FetchError(url, exc.code, str(exc.reason) or f"HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.9 * (attempt + 1))
                continue
    reason = getattr(last_error, "reason", last_error)
    raise FetchError(url, None, str(reason))


def fetch_json(url: str, settings: Settings, *, cache_ttl: float = 0, **kwargs) -> dict:
    body = fetch(url, settings, cache_ttl=cache_ttl, accept="application/json", **kwargs)
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise FetchError(url, None, f"invalid JSON: {exc}") from None


def query_url(base: str, params: dict[str, object]) -> str:
    clean = {k: v for k, v in params.items() if v not in (None, "", [])}
    return f"{base}?{urllib.parse.urlencode(clean, doseq=True)}"
