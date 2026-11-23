"""Reading: fetch what each paper actually says before judging it.

Abstracts come with search results, so the default read is free. ``deep=True``
additionally pulls open-access full text (Europe PMC XML, arXiv/PMC PDFs) and
extracts the prose, which lets screening see the methods and framing rather
than a 250-word summary. PDF extraction needs the optional ``pypdf`` extra and
degrades to a note when it is missing.
"""

from __future__ import annotations

import re
import urllib.parse

from .config import Settings
from .models import Paper
from .net import FetchError, fetch, fetch_json, query_url

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SECTION_NOISE = re.compile(
    r"\b(?:references|acknowledg(?:e)?ments|author contributions|competing interests|"
    r"conflict of interest|supplementary material)\b",
    re.I,
)


def _strip_xml(xml: str) -> str:
    # Cut the reference list: it is long, and it is not the paper's argument.
    noise = _SECTION_NOISE.search(xml)
    if noise:
        xml = xml[: noise.start()]
    text = _TAG.sub(" ", xml)
    return _WS.sub(" ", text).strip()


def fetch_abstract(paper: Paper, settings: Settings) -> str | None:
    """Fill in a missing abstract from Europe PMC."""
    if paper.abstract:
        return paper.abstract
    if not (paper.pmid or paper.pmcid or paper.doi):
        return None
    query = f"EXT_ID:{paper.pmid}" if paper.pmid else f'DOI:"{paper.doi}"'
    url = query_url(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {"query": query, "format": "json", "pageSize": 1, "resultType": "core"},
    )
    try:
        items = fetch_json(url, settings, cache_ttl=86400).get("resultList", {}).get("result", [])
    except FetchError:
        return None
    if not items:
        return None
    abstract = items[0].get("abstractText")
    if abstract:
        paper.abstract = _WS.sub(" ", _TAG.sub(" ", abstract)).strip() or None
    if not paper.pmcid and items[0].get("pmcid"):
        paper.pmcid = items[0]["pmcid"]
    if not paper.pmid and items[0].get("pmid"):
        paper.pmid = items[0]["pmid"]
    return paper.abstract


def fetch_fulltext(paper: Paper, settings: Settings) -> tuple[str | None, str]:
    """Return (text, note) for open-access full text, best source first."""
    if paper.pmcid:
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/"
            f"{paper.pmcid}/fullTextXML"
        )
        try:
            body = fetch(url, settings, cache_ttl=86400).decode("utf-8", "replace")
            text = _strip_xml(body)
            if len(text) > 2000:
                return text, f"Europe PMC full text ({paper.pmcid})"
        except FetchError as exc:
            note = f"Europe PMC full text unavailable ({exc.status or exc.message})"
        else:
            note = "Europe PMC full text too short"
    else:
        note = "no PMC id"

    pdf_url = paper.oa_url or (f"https://arxiv.org/pdf/{paper.arxiv_id}" if paper.arxiv_id else None)
    if pdf_url:
        text, pdf_note = fetch_pdf_text(pdf_url, settings)
        if text:
            return text, pdf_note
        note = f"{note}; {pdf_note}"
    return None, note


def fetch_pdf_text(url: str, settings: Settings) -> tuple[str | None, str]:
    """Download a PDF and extract text; requires the optional pypdf extra."""
    try:
        import pypdf  # type: ignore
    except ImportError:
        return None, "PDF reader not installed (pip install 'paperhound[pdf]')"
    try:
        raw = fetch(url, settings, cache_ttl=86400)
    except FetchError as exc:
        return None, f"PDF download failed ({exc.status or exc.message})"
    if not raw.startswith(b"%PDF"):
        return None, "not a PDF (publisher may have blocked the request)"
    try:
        import io

        reader = pypdf.PdfReader(io.BytesIO(raw))
        chunks = []
        for page in reader.pages[:40]:
            chunks.append(page.extract_text() or "")
        text = _WS.sub(" ", " ".join(chunks)).strip()
    except Exception as exc:  # pypdf raises many types on malformed files
        return None, f"PDF parse failed ({type(exc).__name__})"
    if len(text) < 500:
        return None, "PDF yielded little text (likely scanned)"
    return text, f"PDF text from {urllib.parse.urlsplit(url).netloc}"


def read(paper: Paper, settings: Settings, *, deep: bool = False) -> Paper:
    """Enrich a paper in place with whatever text we can legitimately read."""
    fetch_abstract(paper, settings)
    if deep:
        text, note = fetch_fulltext(paper, settings)
        paper.read_note = note
        if text:
            paper.fulltext_chars = len(text)
            if not paper.abstract:
                paper.abstract = text[:2500]
            # Keep the prose separate from the abstract: screening reads both,
            # and reports stay readable.
            paper.excerpt = text[:6000]
    return paper
