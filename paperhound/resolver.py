"""Resolving access: legal open-access links first, then Sci-Hub mirrors.

The original crawler rotated three Sci-Hub domains by hand, slept 20 seconds,
and clicked a button in Chrome. This module does the same job headlessly and
deterministically: it knows which mirrors answer from here, builds the canonical
``{mirror}/{doi}`` link, and prefers an open-access copy when one exists.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
from dataclasses import dataclass, field

from .config import Settings
from .models import Paper, norm_doi
from .net import Blocked, FetchError, fetch, fetch_json, query_url

_HEALTH_TTL = 600.0
_health_lock = threading.Lock()
_health_cache: dict[tuple[str, ...], tuple[float, list["MirrorStatus"]]] = {}


@dataclass
class MirrorStatus:
    mirror: str
    state: str  # "ok" | "blocked" | "unreachable" | "error"
    http_code: int | None = None
    latency_ms: int | None = None
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.state == "ok"

    def to_dict(self) -> dict:
        return {
            "mirror": self.mirror,
            "state": self.state,
            "http_code": self.http_code,
            "latency_ms": self.latency_ms,
            "note": self.note,
            "usable": self.usable,
        }


def doi_url(doi: str) -> str:
    return f"https://doi.org/{urllib.parse.quote(norm_doi(doi) or doi, safe='/')}"


def mirror_links(doi: str, mirrors: list[str]) -> list[dict]:
    """Build the Sci-Hub URL for every mirror (the crawler's core trick)."""
    clean = norm_doi(doi)
    if not clean:
        return []
    encoded = urllib.parse.quote(clean, safe="/")
    return [{"mirror": m.rstrip("/"), "url": f"{m.rstrip('/')}/{encoded}"} for m in mirrors]


def _probe(mirror: str, settings: Settings) -> MirrorStatus:
    started = time.monotonic()
    try:
        fetch(mirror.rstrip("/") + "/", settings, timeout=8, retries=0, cache_ttl=0)
    except Blocked as exc:
        return MirrorStatus(mirror, "blocked", exc.status, None, "refused by the host (403/451)")
    except FetchError as exc:
        state = "unreachable" if exc.status is None else "error"
        return MirrorStatus(mirror, state, exc.status, None, exc.message)
    except Exception as exc:  # never let a probe kill the run
        return MirrorStatus(mirror, "unreachable", None, None, type(exc).__name__)
    latency = int((time.monotonic() - started) * 1000)
    return MirrorStatus(mirror, "ok", 200, latency, "reachable")


def mirror_health(
    settings: Settings,
    mirrors: list[str] | None = None,
    *,
    force: bool = False,
) -> list[MirrorStatus]:
    """Probe each mirror concurrently, memoised for ten minutes."""
    mirrors = [m.rstrip("/") for m in (mirrors or settings.mirrors)]
    key = tuple(mirrors)
    with _health_lock:
        cached = _health_cache.get(key)
        if cached and not force and (time.time() - cached[0]) < _HEALTH_TTL:
            return cached[1]
    statuses = list(_probe_all(mirrors, settings))
    with _health_lock:
        _health_cache[key] = (time.time(), statuses)
    return statuses


def _probe_all(mirrors: list[str], settings: Settings) -> list[MirrorStatus]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(mirrors)))) as pool:
        return list(pool.map(lambda m: _probe(m, settings), mirrors))


@dataclass
class Access:
    """Everything needed to actually get the PDF."""

    doi: str | None = None
    doi_url: str | None = None
    open_access: list[dict] = field(default_factory=list)
    scihub: list[dict] = field(default_factory=list)
    best: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "doi_url": self.doi_url,
            "open_access": self.open_access,
            "scihub": self.scihub,
            "best": self.best,
            "notes": self.notes,
        }


def oa_candidates(paper: Paper, settings: Settings) -> tuple[list[dict], list[str]]:
    """Collect legal open-access copies: Unpaywall, PMC, arXiv, provided URL."""
    found: list[dict] = []
    notes: list[str] = []

    def add(url: str | None, kind: str, license_: str | None = None) -> None:
        if not url:
            return
        if any(item["url"] == url for item in found):
            return
        found.append({"url": url, "kind": kind, "license": license_})

    add(paper.oa_url, "oa")
    if paper.pmcid:
        # Europe PMC has no fullTextPDF REST endpoint (it 404s); this is the
        # public PDF view. PMC itself serves the PDF but blocks scripted clients,
        # so a returned PDF is verified rather than assumed.
        add(f"https://europepmc.org/articles/{paper.pmcid}?pdf=render", "pmc-pdf")
        add(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{paper.pmcid}/fullTextXML",
            "pmc-xml",
        )
        add(f"https://pmc.ncbi.nlm.nih.gov/articles/{paper.pmcid}/", "pmc")
    if paper.arxiv_id:
        add(f"https://arxiv.org/abs/{paper.arxiv_id}", "arxiv")
        add(f"https://arxiv.org/pdf/{paper.arxiv_id}", "arxiv-pdf")

    doi = norm_doi(paper.doi)
    if doi and settings.polite_email:
        url = query_url(
            f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='/')}",
            {"email": settings.polite_email},
        )
        try:
            data = fetch_json(url, settings, cache_ttl=86400)
            for location in data.get("oa_locations", []) or []:
                add(location.get("url_for_pdf"), "unpaywall-pdf", location.get("license"))
                add(location.get("url"), "unpaywall", location.get("license"))
        except FetchError as exc:
            notes.append(f"Unpaywall: {exc.status or exc.message}")
    elif doi:
        notes.append("Unpaywall skipped (set PAPERHOUND_MAILTO for open-access lookup)")
    found.sort(key=_download_rank)
    return found, notes


def _download_rank(item: dict) -> tuple[int, int]:
    """Prefer things that are directly fetchable as a PDF over landing pages."""
    url = item["url"].lower()
    kind = item["kind"]
    if kind in ("unpaywall-pdf", "arxiv-pdf", "pmc-pdf") or url.endswith(".pdf") or "/pdf" in url:
        return (0, 0)
    if kind == "pmc-xml":
        return (1, 0)
    return (2, 0)


def access_for(
    paper: Paper, settings: Settings, *, probe: bool = True, include_oa: bool = True
) -> Access:
    """Build the full access plan for one paper."""
    result = Access(doi=norm_doi(paper.doi))
    if result.doi:
        result.doi_url = doi_url(result.doi)

    if include_oa:
        oa, notes = oa_candidates(paper, settings)
        result.open_access = oa
        result.notes.extend(notes)
    else:
        result.notes.append("open-access lookup skipped")

    if result.doi:
        links = mirror_links(result.doi, settings.mirrors)
        if probe:
            statuses = {s.mirror: s for s in mirror_health(settings)}
            for link in links:
                status = statuses.get(link["mirror"])
                link["state"] = status.state if status else "unknown"
                link["latency_ms"] = status.latency_ms if status else None
            links.sort(key=lambda link: (link["state"] != "ok", link["latency_ms"] or 9999))
        else:
            for link in links:
                link["state"] = "unchecked"
        result.scihub = links

    if result.open_access:
        result.best = result.open_access[0]
    elif result.scihub:
        usable = [link for link in result.scihub if link.get("state") in ("ok", "unchecked")]
        result.best = usable[0] if usable else result.scihub[0]
    return result


def download_any(
    access: Access,
    destination: str,
    settings: Settings,
    *,
    include_scihub: bool = True,
) -> tuple[bool, str]:
    """Try every candidate until one really is a PDF.

    Landing pages, XML and captcha HTML all fail the magic-byte check, so this
    walks the list instead of trusting whichever link looked best.
    """
    candidates = [item["url"] for item in access.open_access]
    if include_scihub:
        candidates += [
            link["url"] for link in access.scihub if link.get("state") in ("ok", "unchecked")
        ]
    seen: set[str] = set()
    errors: list[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        ok, note = download(url, destination, settings)
        if ok:
            return True, note
        errors.append(f"{urllib.parse.urlsplit(url).netloc}: {note}")
    return False, "; ".join(errors[:3]) or "no downloadable source"


def download(url: str, destination: str, settings: Settings) -> tuple[bool, str]:
    """Fetch a PDF to disk and verify it really is a PDF."""
    try:
        raw = fetch(url, settings, cache_ttl=0, timeout=90)
    except FetchError as exc:
        return False, f"download failed: {exc.status or exc.message}"
    if not raw.startswith(b"%PDF"):
        return False, "response was not a PDF (host may require a browser or captcha)"
    try:
        with open(destination, "wb") as handle:
            handle.write(raw)
    except OSError as exc:
        return False, f"could not write {destination}: {exc}"
    return True, f"{len(raw) // 1024} KiB -> {destination}"
