"""Small on-disk caches for screening verdicts and resolved access.

Screening is a pure function of (paper text, parsed intent): given the same
paper and the same scope it always returns the same verdict. Re-running a
search -- or screening the same reference list tomorrow -- should not pay for
it twice, and with the optional LLM second opinion it can be expensive.

Access is different: it is true only for a while (mirrors get blocked, DOIs get
retracted, Unpaywall finds new copies), so it expires quickly.

Both caches live beside the HTTP cache, are keyed by content (never by query
string), and fail silently: a broken or unwritable cache must never break a
search.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Settings
from .models import ScopeSpec, Verdict

VERDICT_TTL = 30 * 24 * 3600.0
ACCESS_TTL = 6 * 3600.0
# Bump when the screening rules change: a cached verdict is only valid for the
# rules that produced it, so the version is part of the cache key.
SCREENING_VERSION = 3


def fingerprint(spec: ScopeSpec) -> str:
    """A stable hash of everything that can change a verdict."""
    payload = {
        "screening": SCREENING_VERSION,
        "intent": spec.intent,
        "facets": [(f.name, sorted(f.terms), f.weight) for f in spec.facets],
        "head": spec.head.name if spec.head else None,
        "exclude": spec.exclude,
        "years": [spec.year_from, spec.year_to],
        "min_scope": spec.min_scope,
        "required": [list(item) for item in spec.required],
        "require_field": spec.require_field,
        "anchors": spec.anchors,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class JsonCache:
    """A tiny, process-safe-enough JSON map with per-entry expiry."""

    def __init__(self, path: Path, *, ttl: float, limit: int = 5000, enabled: bool = True):
        self.path = path
        self.ttl = ttl
        self.limit = limit
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._data: dict[str, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is not None:
            return self._data
        try:
            self._data = json.loads(self.path.read_text("utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except (OSError, ValueError):
            self._data = {}
        return self._data

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._load().get(key)
            if not entry:
                self.misses += 1
                return None
            if self.ttl and (time.time() - entry.get("t", 0)) > self.ttl:
                self._load().pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return entry.get("v")

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            data = self._load()
            data[key] = {"t": time.time(), "v": value}
            if len(data) > self.limit:
                # Drop the oldest quarter; the cache is an optimisation, not a
                # store of record.
                ordered = sorted(data.items(), key=lambda item: item[1].get("t", 0))
                for old_key, _ in ordered[: max(1, self.limit // 4)]:
                    data.pop(old_key, None)
            self._data = data

    def flush(self) -> None:
        if not self.enabled or self._data is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data), "utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._data or {})}


class Caches:
    """The two caches the pipeline uses, created once per run."""

    def __init__(self, settings: Settings, *, enabled: bool | None = None):
        use = settings.cache if enabled is None else enabled
        base = settings.cache_dir / "verdicts"
        self.verdicts = JsonCache(base / "screening.json", ttl=VERDICT_TTL, enabled=use)
        self.access = JsonCache(base / "access.json", ttl=ACCESS_TTL, enabled=use)

    # -- verdicts --------------------------------------------------------- #

    def get_verdict(self, spec: ScopeSpec, paper_key: str) -> Verdict | None:
        raw = self.verdicts.get(hashlib.sha1(f"{fingerprint(spec)}|{paper_key}".encode()).hexdigest())
        if not isinstance(raw, dict):
            return None
        try:
            return Verdict(**raw)
        except TypeError:
            return None

    def put_verdict(self, spec: ScopeSpec, paper_key: str, verdict: Verdict) -> None:
        self.verdicts.put(
            hashlib.sha1(f"{fingerprint(spec)}|{paper_key}".encode()).hexdigest(),
            verdict.to_dict(),
        )

    # -- access ----------------------------------------------------------- #

    def _access_key(
        self,
        doi: str,
        *,
        include_oa: bool = True,
        probe: bool = True,
        mirrors: Sequence[str] = (),
    ) -> str:
        payload = {
            "doi": doi.lower(),
            "oa": bool(include_oa),
            "probe": bool(probe),
            "mirrors": list(mirrors),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get_access(
        self,
        doi: str,
        *,
        include_oa: bool = True,
        probe: bool = True,
        mirrors: Sequence[str] = (),
    ) -> dict[str, Any] | None:
        raw = self.access.get(
            self._access_key(doi, include_oa=include_oa, probe=probe, mirrors=mirrors)
        )
        return raw if isinstance(raw, dict) else None

    def put_access(
        self,
        doi: str,
        payload: dict[str, Any],
        *,
        include_oa: bool = True,
        probe: bool = True,
        mirrors: Sequence[str] = (),
    ) -> None:
        self.access.put(
            self._access_key(doi, include_oa=include_oa, probe=probe, mirrors=mirrors),
            payload,
        )

    # -- lifecycle -------------------------------------------------------- #

    def flush(self) -> None:
        self.verdicts.flush()
        self.access.flush()

    def stats(self) -> dict[str, Any]:
        return {
            "verdict_hits": self.verdicts.hits,
            "verdict_misses": self.verdicts.misses,
            "access_hits": self.access.hits,
            "access_misses": self.access.misses,
            "enabled": self.verdicts.enabled,
        }
