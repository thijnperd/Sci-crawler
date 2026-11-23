"""Source adapters: Europe PMC, Crossref, arXiv, Semantic Scholar, OpenAlex.

Every adapter takes a parsed ScopeSpec, returns Papers and -- critically --
never raises for a single source failing. One API being down or rate-limited
degrades the result set and adds a warning; it does not end the run.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .config import Settings
from .models import Paper, ScopeSpec, norm_doi
from .net import Blocked, FetchError, fetch, fetch_json, query_url

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass
class SourceResult:
    source: str
    papers: list[Paper] = field(default_factory=list)
    warning: str | None = None

    @property
    def ok(self) -> bool:
        return self.warning is None


def clean_text(value: str | None) -> str | None:
    """Strip JATS/HTML markup and collapse whitespace."""
    if not value:
        return None
    text = html.unescape(_TAG.sub(" ", value))
    text = _WS.sub(" ", text).strip()
    return text or None


def _int(value: object) -> int | None:
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return year if 1800 < year < 2200 else None


def _query_terms(spec_facet_terms: list[str], name: str, phrases: list[str]) -> list[str]:
    ordered: list[str] = [name]
    for candidate in phrases + spec_facet_terms:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered[:4]


def _boolean_query(spec: ScopeSpec, mode: str, field: str | None) -> str:
    chunks: list[str] = []
    for facet in spec.facets:
        terms = _query_terms(facet.terms, facet.name, facet.phrases)
        parts = []
        for term in terms:
            quoted = f'"{term}"' if " " in term else term
            parts.append(f"{field}:{quoted}" if field else quoted)
        chunks.append("(" + " OR ".join(parts) + ")")
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(chunks) if chunks else (spec.intent if not field else f"{field}:{spec.intent}")


# --------------------------------------------------------------------------- #
# Europe PMC
# --------------------------------------------------------------------------- #

def search_europepmc(spec: ScopeSpec, settings: Settings, limit: int, mode: str = "and") -> SourceResult:
    query = _boolean_query(spec, mode, "TITLE_ABS")
    if spec.year_from or spec.year_to:
        query += f" AND PUB_YEAR:[{spec.year_from or 1900} TO {spec.year_to or 2100}]"
    url = query_url(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {
            "query": query,
            "format": "json",
            "pageSize": limit,
            "resultType": "core",
            # Without an explicit sort Europe PMC returns newest-first, which
            # fills the candidate pool with unread 2026 preprints. Its default
            # has no relevance ranking, so ask for the most cited matches.
            "sort": "CITED desc",
        },
    )
    try:
        data = fetch_json(url, settings, cache_ttl=3600)
    except FetchError as exc:
        return SourceResult("europepmc", warning=f"Europe PMC: {exc.message}")

    papers: list[Paper] = []
    for item in data.get("resultList", {}).get("result", []) or []:
        paper = _epmc_paper(item)
        if paper is not None:
            papers.append(paper)
    return SourceResult("europepmc", papers)


def _epmc_paper(item: dict) -> Paper | None:
    """Normalise one Europe PMC result record."""
    title = clean_text(item.get("title"))
    if not title:
        return None
    pmcid = item.get("pmcid") or None
    pub_types = item.get("pubTypeList")
    first_type = None
    if isinstance(pub_types, dict):
        values = pub_types.get("pubType") or []
        if isinstance(values, str):
            values = [values]
        first_type = values[0] if values else None
    return Paper(
        title=title,
        doi=norm_doi(item.get("doi")),
        year=_int(item.get("pubYear")),
        authors=_split_authors(item.get("authorString")),
        journal=clean_text(item.get("journalTitle")) or clean_text((item.get("bookOrReportDetails") or {}).get("publisher")),
        abstract=clean_text(item.get("abstractText")),
        url=f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id')}" if item.get("id") else None,
        pmid=item.get("pmid") or None,
        pmcid=pmcid,
        keywords=_epmc_keywords(item),
        cited_by=int(item.get("citedByCount") or 0),
        is_oa=str(item.get("isOpenAccess", "")).upper() == "Y" or bool(pmcid),
        oa_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None,
        type=first_type,
        sources=["europepmc"],
    )


def _epmc_keywords(item: dict) -> list[str]:
    out: list[str] = []
    keywords = item.get("keywordList") or {}
    if isinstance(keywords, dict):
        values = keywords.get("keyword") or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            if isinstance(value, dict):
                value = value.get("term") or value.get("keyword")
            if value:
                out.append(str(value).strip())
    for heading in (item.get("meshHeadingList") or {}).get("meshHeading", []) or []:
        if isinstance(heading, dict):
            name = heading.get("descriptorName")
            if name:
                out.append(str(name))
    return [k for k in out if k]


def _split_authors(author_string: str | None) -> list[str]:
    if not author_string:
        return []
    parts = [p.strip().rstrip(".") for p in author_string.split(",")]
    return [p for p in parts if p]


# --------------------------------------------------------------------------- #
# Crossref
# --------------------------------------------------------------------------- #

def search_crossref(spec: ScopeSpec, settings: Settings, limit: int, mode: str = "and") -> SourceResult:
    filters = ["type:journal-article"]
    if spec.year_from:
        filters.append(f"from-pub-date:{spec.year_from}-01-01")
    if spec.year_to:
        filters.append(f"until-pub-date:{spec.year_to}-12-31")
    query = spec.intent if mode == "and" else _boolean_query(spec, "or", None)
    url = query_url(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": query,
            "rows": limit,
            "filter": ",".join(filters),
            "select": "DOI,title,author,issued,container-title,abstract,is-referenced-by-count,type,subject,URL,link",
            "mailto": settings.polite_email,
        },
    )
    try:
        data = fetch_json(url, settings, cache_ttl=3600)
    except FetchError as exc:
        return SourceResult("crossref", warning=f"Crossref: {exc.message}")

    papers: list[Paper] = []
    for item in data.get("message", {}).get("items", []) or []:
        title = clean_text((item.get("title") or [""])[0])
        if not title:
            continue
        authors = []
        for author in item.get("author", []) or []:
            name = author.get("family") or author.get("name")
            given = author.get("given")
            if name:
                authors.append(f"{name} {given}".strip() if given else name)
        year = None
        for key in ("published-print", "published-online", "issued", "created"):
            parts = (item.get(key) or {}).get("date-parts") or [[]]
            if parts and parts[0] and parts[0][0]:
                year = _int(parts[0][0])
                if year:
                    break
        pdf_url = None
        for link in item.get("link", []) or []:
            if "pdf" in str(link.get("content-type", "")).lower():
                pdf_url = link.get("URL")
                break
        papers.append(
            Paper(
                title=title,
                doi=norm_doi(item.get("DOI")),
                year=year,
                authors=authors,
                journal=clean_text((item.get("container-title") or [None])[0]),
                abstract=clean_text(item.get("abstract")),
                url=item.get("URL"),
                keywords=[s for s in (item.get("subject") or []) if s],
                cited_by=int(item.get("is-referenced-by-count") or 0),
                is_oa=bool(pdf_url),
                oa_url=pdf_url,
                type=item.get("type"),
                sources=["crossref"],
            )
        )
    return SourceResult("crossref", papers)


# --------------------------------------------------------------------------- #
# arXiv
# --------------------------------------------------------------------------- #

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


def search_arxiv(spec: ScopeSpec, settings: Settings, limit: int, mode: str = "and") -> SourceResult:
    query = _boolean_query(spec, mode, "all")
    url = query_url(
        "https://export.arxiv.org/api/query",
        {"search_query": query, "start": 0, "max_results": limit, "sortBy": "relevance"},
    )
    try:
        body = fetch(url, settings, cache_ttl=3600)
        root = ET.fromstring(body)
    except FetchError as exc:
        return SourceResult("arxiv", warning=f"arXiv: {exc.message}")
    except ET.ParseError as exc:
        return SourceResult("arxiv", warning=f"arXiv: unparseable response ({exc})")

    papers: list[Paper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = clean_text(_text(entry, f"{_ATOM}title"))
        if not title:
            continue
        raw_id = _text(entry, f"{_ATOM}id") or ""
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else None
        published = _text(entry, f"{_ATOM}published") or _text(entry, f"{_ATOM}updated")
        papers.append(
            Paper(
                title=title,
                doi=norm_doi(_text(entry, f"{_ARXIV}doi")),
                year=_int(published),
                authors=[clean_text(_text(a, f"{_ATOM}name")) or "" for a in entry.findall(f"{_ATOM}author")],
                journal=clean_text(_text(entry, f"{_ARXIV}journal_ref")),
                abstract=clean_text(_text(entry, f"{_ATOM}summary")),
                url=raw_id or None,
                arxiv_id=arxiv_id,
                keywords=[c.get("term", "") for c in entry.findall(f"{_ATOM}category") if c.get("term")],
                is_oa=True,
                oa_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
                type="preprint",
                sources=["arxiv"],
            )
        )
    return SourceResult("arxiv", papers)


def _text(node: ET.Element, path: str) -> str | None:
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else None


# --------------------------------------------------------------------------- #
# Semantic Scholar
# --------------------------------------------------------------------------- #

_S2_FIELDS = (
    "title,abstract,year,authors,externalIds,venue,citationCount,isOpenAccess,"
    "openAccessPdf,fieldsOfStudy,publicationTypes,url"
)


def search_semanticscholar(spec: ScopeSpec, settings: Settings, limit: int, mode: str = "and") -> SourceResult:
    params = {"query": spec.intent, "limit": limit, "fields": _S2_FIELDS}
    if spec.year_from or spec.year_to:
        params["year"] = f"{spec.year_from or ''}-{spec.year_to or ''}"
    url = query_url("https://api.semanticscholar.org/graph/v1/paper/search", params)
    headers = {"x-api-key": settings.s2_key} if settings.s2_key else None
    try:
        data = fetch_json(url, settings, cache_ttl=3600, headers=headers)
    except FetchError as exc:
        return SourceResult("semanticscholar", warning=f"Semantic Scholar: {exc.message}")

    papers: list[Paper] = []
    for item in data.get("data", []) or []:
        title = clean_text(item.get("title"))
        if not title:
            continue
        ids = item.get("externalIds") or {}
        pdf = (item.get("openAccessPdf") or {}).get("url")
        papers.append(
            Paper(
                title=title,
                doi=norm_doi(ids.get("DOI")),
                year=_int(item.get("year")),
                authors=[a.get("name", "") for a in item.get("authors", []) or []],
                journal=clean_text(item.get("venue")) or None,
                abstract=clean_text(item.get("abstract")),
                url=item.get("url"),
                pmid=str(ids["PubMed"]) if ids.get("PubMed") else None,
                arxiv_id=ids.get("ArXiv"),
                keywords=list(item.get("fieldsOfStudy") or []),
                cited_by=int(item.get("citationCount") or 0),
                is_oa=bool(pdf) or bool(item.get("isOpenAccess")),
                oa_url=pdf,
                type=(item.get("publicationTypes") or [None])[0],
                sources=["semanticscholar"],
            )
        )
    return SourceResult("semanticscholar", papers)


# --------------------------------------------------------------------------- #
# OpenAlex
# --------------------------------------------------------------------------- #

_RATE_NOTE = (
    "OpenAlex free daily budget is exhausted for this IP; set PAPERHOUND_OPENALEX_KEY "
    "for a key (free) or drop openalex from PAPERHOUND_SOURCES"
)


def search_openalex(spec: ScopeSpec, settings: Settings, limit: int, mode: str = "and") -> SourceResult:
    filters = ["type:article"]
    if spec.year_from:
        filters.append(f"from_publication_date:{spec.year_from}-01-01")
    if spec.year_to:
        filters.append(f"to_publication_date:{spec.year_to}-12-31")
    params = {
        "search": spec.intent,
        "per-page": min(limit, 200),
        "filter": ",".join(filters),
        "mailto": settings.mailto or None,
        "api_key": settings.openalex_key or None,
    }
    url = query_url("https://api.openalex.org/works", params)
    try:
        data = fetch_json(url, settings, cache_ttl=3600)
    except FetchError as exc:
        note = _RATE_NOTE if exc.status in (429, 403) else exc.message
        return SourceResult("openalex", warning=f"OpenAlex: {note}")

    papers: list[Paper] = []
    for item in data.get("results", []) or []:
        title = clean_text(item.get("title") or item.get("display_name"))
        if not title:
            continue
        oa = item.get("best_oa_location") or item.get("primary_location") or {}
        papers.append(
            Paper(
                title=title,
                doi=norm_doi(item.get("doi")),
                year=_int(item.get("publication_year")),
                authors=[
                    (a.get("author") or {}).get("display_name", "")
                    for a in item.get("authorships", []) or []
                ],
                journal=((item.get("primary_location") or {}).get("source") or {}).get("display_name"),
                abstract=reconstruct_abstract(item.get("abstract_inverted_index")),
                url=item.get("id"),
                keywords=[t.get("display_name", "") for t in (item.get("topics") or [])[:5]],
                cited_by=int(item.get("cited_by_count") or 0),
                is_oa=bool(item.get("is_oa")),
                oa_url=oa.get("pdf_url") or oa.get("landing_page_url"),
                type=item.get("type"),
                sources=["openalex"],
            )
        )
    return SourceResult("openalex", papers)


def reconstruct_abstract(inverted: dict | None) -> str | None:
    """OpenAlex ships abstracts as word -> positions; rebuild the sentence."""
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, places in inverted.items():
        for place in places or []:
            positions.append((int(place), word))
    if not positions:
        return None
    positions.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positions)


SEARCHERS = {
    "europepmc": search_europepmc,
    "crossref": search_crossref,
    "arxiv": search_arxiv,
    "semanticscholar": search_semanticscholar,
    "openalex": search_openalex,
}


# --------------------------------------------------------------------------- #
# Single-record lookups
# --------------------------------------------------------------------------- #

def fetch_by_doi(doi: str, settings: Settings, *, deep_meta: bool = True) -> Paper | None:
    """Resolve one DOI to a Paper, preferring the records with an abstract."""
    doi = norm_doi(doi) or ""
    if not doi:
        return None
    paper: Paper | None = None

    # Europe PMC usually has the abstract and the PMC full-text pointer.
    url = query_url(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {"query": f'DOI:"{doi}"', "format": "json", "pageSize": 1, "resultType": "core"},
    )
    try:
        items = fetch_json(url, settings, cache_ttl=86400).get("resultList", {}).get("result", [])
        if items:
            paper = _epmc_paper(items[0])
    except (FetchError, KeyError, TypeError):
        paper = None

    needs_more = paper is None or not paper.abstract
    if deep_meta or needs_more:
        try:
            data = fetch_json(
                f"https://api.crossref.org/works/{urllib.parse.quote(doi)}",
                settings,
                cache_ttl=86400,
            )
            item = data.get("message", {})
            title = clean_text((item.get("title") or [""])[0])
            if title:
                cross = Paper(
                    title=title,
                    doi=doi,
                    year=_int(((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]),
                    authors=[
                        " ".join(x for x in (a.get("family"), a.get("given")) if x)
                        for a in item.get("author", []) or []
                    ],
                    journal=(item.get("container-title") or [None])[0],
                    abstract=clean_text(item.get("abstract")),
                    url=item.get("URL"),
                    keywords=[s for s in (item.get("subject") or []) if s],
                    cited_by=int(item.get("is-referenced-by-count") or 0),
                    type=item.get("type"),
                    sources=["crossref"],
                )
                if paper is None:
                    paper = cross
                else:
                    paper.merge(cross)
        except FetchError:
            pass

    if paper is None:
        try:
            params = {"fields": _S2_FIELDS}
            data = fetch_json(
                query_url(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}", params),
                settings,
                cache_ttl=86400,
                headers={"x-api-key": settings.s2_key} if settings.s2_key else None,
            )
            title = clean_text(data.get("title"))
            if title:
                paper = Paper(
                    title=title,
                    doi=doi,
                    year=_int(data.get("year")),
                    authors=[a.get("name", "") for a in data.get("authors", []) or []],
                    journal=data.get("venue"),
                    abstract=clean_text(data.get("abstract")),
                    url=data.get("url"),
                    cited_by=int(data.get("citationCount") or 0),
                    oa_url=(data.get("openAccessPdf") or {}).get("url"),
                    sources=["semanticscholar"],
                )
        except FetchError:
            pass
    return paper


def find_by_title(title: str, settings: Settings, *, rows: int = 5) -> Paper | None:
    """Best-effort title -> Paper, used when an agent only has a title."""
    url = query_url(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": title,
            "rows": rows,
            "select": "DOI,title,author,issued,container-title,abstract,is-referenced-by-count",
            "mailto": settings.polite_email,
        },
    )
    try:
        items = fetch_json(url, settings, cache_ttl=3600).get("message", {}).get("items", [])
    except FetchError:
        return None
    from difflib import SequenceMatcher

    best: Paper | None = None
    best_ratio = 0.0
    for item in items:
        candidate_title = clean_text((item.get("title") or [""])[0])
        if not candidate_title:
            continue
        ratio = SequenceMatcher(None, candidate_title.lower(), title.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = Paper(
                title=candidate_title,
                doi=norm_doi(item.get("DOI")),
                year=_int(((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]),
                authors=[
                    " ".join(x for x in (a.get("family"), a.get("given")) if x)
                    for a in item.get("author", []) or []
                ],
                journal=clean_text((item.get("container-title") or [None])[0]),
                abstract=clean_text(item.get("abstract")),
                cited_by=int(item.get("is-referenced-by-count") or 0),
                sources=["crossref"],
            )
    return best if best and best_ratio >= 0.45 else None
