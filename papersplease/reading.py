"""Reading: fetch what each paper actually says before judging it.

Abstracts come with search results, so the default read is free. ``deep=True``
additionally pulls open-access full text (Europe PMC XML, arXiv/PMC PDFs) and
extracts the prose, which lets screening see the methods and framing rather
than a 250-word summary. PDF extraction needs the optional ``pypdf`` extra and
degrades to a note when it is missing.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from typing import Sequence

from .config import Settings
from .models import Paper
from .net import FetchError, fetch, fetch_json, query_url

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SECTION_NOISE = re.compile(
    r"<(?:ref-list|ack|fn-group)\b|<title[^>]*>\s*(?:references|acknowledg(?:e)?ments|"
    r"author contributions|competing interests|conflict of interest|supplementary material)\s*</title>",
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
    if paper.pmid:
        query = f"EXT_ID:{paper.pmid}"
    elif paper.pmcid:
        query = f"PMCID:{paper.pmcid}"
    else:
        query = f'DOI:"{paper.doi}"'
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
        paper.abstract = _WS.sub(" ", html.unescape(_TAG.sub(" ", abstract))).strip() or None
    if not paper.pmcid and items[0].get("pmcid"):
        paper.pmcid = items[0]["pmcid"]
    if not paper.pmid and items[0].get("pmid"):
        paper.pmid = items[0]["pmid"]
    return paper.abstract


def batch_fetch_abstracts(papers: Sequence[Paper], settings: Settings) -> int:
    """Backfill missing abstracts in batches of 40 DOIs via OpenAlex."""
    from .sources import norm_doi, reconstruct_abstract

    targets = [p for p in papers if not p.abstract and p.doi]
    if not targets:
        return 0
    by_doi: dict[str, Paper] = {}
    for p in targets:
        nd = norm_doi(p.doi)
        if nd and nd not in by_doi:
            by_doi[nd] = p

    doi_list = list(by_doi.keys())
    filled = 0
    for start in range(0, len(doi_list), 40):
        chunk = doi_list[start:start + 40]
        doi_filter = "|".join(f"https://doi.org/{d}" for d in chunk)
        url = query_url(
            "https://api.openalex.org/works",
            {
                "filter": f"doi:{doi_filter}",
                "per-page": 50,
                "select": "doi,abstract_inverted_index,cited_by_count,best_oa_location,open_access",
                "mailto": settings.polite_email or None,
                "api_key": settings.openalex_key or None,
            },
        )
        try:
            data = fetch_json(url, settings, cache_ttl=86400, timeout=15)
        except FetchError:
            continue
        for item in data.get("results", []) or []:
            nd = norm_doi(item.get("doi"))
            paper = by_doi.get(nd or "")
            if paper is None:
                continue
            abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
            if abstract and not paper.abstract:
                paper.abstract = abstract
                filled += 1
            if not paper.cited_by and item.get("cited_by_count"):
                paper.cited_by = int(item["cited_by_count"])
            if not paper.is_oa and (item.get("open_access") or {}).get("is_oa"):
                paper.is_oa = True
            if not paper.oa_url:
                oa = item.get("best_oa_location") or {}
                pdf_or_landing = oa.get("pdf_url") or oa.get("landing_page_url")
                if pdf_or_landing:
                    paper.oa_url = pdf_or_landing
    return filled


def fetch_fulltext(paper: Paper, settings: Settings) -> tuple[str | None, str]:
    """Return (text, note) for open-access full text, best source first."""
    if paper.pmcid:
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/"
            f"{paper.pmcid}/fullTextXML"
        )
        try:
            body = fetch(url, settings, cache_ttl=86400, timeout=20, retries=0).decode("utf-8", "replace")
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
        return None, "PDF reader not installed (pip install 'papersplease[pdf]')"
    try:
        # Publishers answer 403 to a scripted client; retrying with a 90 s
        # timeout just makes a deep read feel hung. Fail fast, try the next
        # source.
        raw = fetch(url, settings, cache_ttl=86400, timeout=20, retries=0)
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
