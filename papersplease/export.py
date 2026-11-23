"""Citation export: BibTeX, RIS and CSL-JSON for reference managers.

An agent that has just screened a literature list usually needs to hand it to a
reference manager, not retype twenty entries. These exporters take the same
``Finding`` objects the rest of the pipeline produces and emit importable text.

Both formats are intentionally plain: no dependency, no clever escaping beyond
what breaks a parser.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable, Sequence

from .models import Paper
from .pipeline import Finding

_ASCII = re.compile(r"[^a-z0-9]+")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _ascii(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    return _ASCII.sub("", text.lower())


def _surname(author: str) -> str:
    # "Gentil Paulo" (Europe PMC) and "Paulo Gentil" (Crossref) both end up as
    # the token we pick; it only has to make the key readable and unique.
    parts = [p for p in re.split(r"[ ,]+", author.strip()) if p]
    return _ascii(parts[0]) if parts else "anon"


def _paper(item: Finding | Paper) -> Paper:
    return item.paper if isinstance(item, Finding) else item


def _key(paper: Paper, used: set[str]) -> str:
    stem = (_surname(paper.authors[0]) if paper.authors else "anon")[:18]
    year = str(paper.year or "nd")
    word = ""
    for token in re.findall(r"[A-Za-z]{4,}", paper.title or ""):
        if token.lower() not in {"with", "from", "that", "this", "effects", "effect", "study"}:
            word = _ascii(token)[:14]
            break
    base = f"{stem}{year}{word}" or "paper"
    key, suffix = base, ord("a")
    while key in used:
        key = f"{base}{chr(suffix)}"
        suffix += 1
    used.add(key)
    return key


def _escape_bib(value: str | None) -> str:
    text = (value or "").replace("\\", "\\textbackslash{}")
    for char in ("&", "%", "$", "#", "_"):
        text = text.replace(char, "\\" + char)
    return " ".join(text.split())


def to_bibtex(items: Sequence[Finding | Paper]) -> str:
    """One @article per paper, ready for \\bibliography."""
    used: set[str] = set()
    blocks: list[str] = []
    for item in items:
        paper = _paper(item)
        fields = [
            ("title", _escape_bib(paper.title)),
            ("author", " and ".join(_escape_bib(a) for a in paper.authors) or None),
            ("journal", _escape_bib(paper.journal)),
            ("year", str(paper.year) if paper.year else None),
            ("doi", paper.doi),
            ("url", paper.url),
        ]
        lines = [f"@article{{{_key(paper, used)},"]
        body = [f"  {name} = {{{value}}}," for name, value in fields if value]
        if body:
            body[-1] = body[-1].rstrip(",")
        lines.extend(body)
        lines.append("}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_ris(items: Sequence[Finding | Paper]) -> str:
    """RIS records, importable by Zotero, EndNote and Mendeley."""
    blocks: list[str] = []
    for item in items:
        paper = _paper(item)
        lines = ["TY  - JOUR", f"TI  - {paper.title}"]
        for author in paper.authors:
            lines.append(f"AU  - {author}")
        if paper.journal:
            lines.append(f"JO  - {paper.journal}")
        if paper.year:
            lines.append(f"PY  - {paper.year}")
        if paper.doi:
            lines.append(f"DO  - {paper.doi}")
        for url in _urls(item):
            lines.append(f"UR  - {url}")
        if paper.abstract:
            lines.append("AB  - " + " ".join(paper.abstract.split())[:2000])
        lines.extend(["ER  - ", ""])
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _urls(item: Finding | Paper) -> Iterable[str]:
    if isinstance(item, Finding):
        if item.access:
            if item.access.doi_url:
                yield item.access.doi_url
            if item.access.best and item.access.best.get("url"):
                yield item.access.best["url"]
        elif item.paper.url:
            yield item.paper.url
    elif item.url:
        yield item.url


def _csl_name(author: str) -> dict[str, str]:
    """Split "Surname Given" (the pipeline's convention) into CSL name parts."""
    parts = [part for part in re.split(r"\s+", author.strip()) if part]
    if len(parts) < 2:
        return {"literal": author.strip() or "anonymous"}
    return {"family": parts[0], "given": " ".join(parts[1:])}


def to_csl_json(items: Sequence[Finding | Paper]) -> str:
    """CSL-JSON, the format Zotero, Pandoc and citeproc consume directly."""
    records: list[dict] = []
    used: set[str] = set()
    for item in items:
        paper = _paper(item)
        record: dict = {
            "id": paper.doi or _key(paper, used),
            "type": "article-journal",
            "title": paper.title,
        }
        if paper.authors:
            record["author"] = [_csl_name(author) for author in paper.authors]
        if paper.year:
            record["issued"] = {"date-parts": [[paper.year]]}
        if paper.journal:
            record["container-title"] = paper.journal
        if paper.doi:
            record["DOI"] = paper.doi
        if paper.url:
            record["URL"] = paper.url
        records.append(record)
    return json.dumps(records, ensure_ascii=False, indent=2) + "\n"


FORMATS = ("text", "md", "json", "bibtex", "ris", "csljson")


def render(items: Sequence[Finding | Paper], fmt: str) -> str:
    if fmt == "bibtex":
        return to_bibtex(items)
    if fmt == "ris":
        return to_ris(items)
    if fmt == "csljson":
        return to_csl_json(items)
    raise ValueError(f"unsupported citation format: {fmt}")
