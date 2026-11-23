"""Citation export: BibTeX for LaTeX, RIS for Zotero/EndNote/Mendeley.

An agent that has just screened a literature list usually needs to hand it to a
reference manager, not retype twenty entries. These exporters take the same
``Finding`` objects the rest of the pipeline produces and emit importable text.

Both formats are intentionally plain: no dependency, no clever escaping beyond
what breaks a parser.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

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


def _key(finding: Finding, used: set[str]) -> str:
    paper = finding.paper
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


def to_bibtex(findings: Sequence[Finding]) -> str:
    """One @article per paper, ready for \\bibliography."""
    used: set[str] = set()
    blocks: list[str] = []
    for finding in findings:
        paper = finding.paper
        fields = [
            ("title", _escape_bib(paper.title)),
            ("author", " and ".join(_escape_bib(a) for a in paper.authors) or None),
            ("journal", _escape_bib(paper.journal)),
            ("year", str(paper.year) if paper.year else None),
            ("doi", paper.doi),
            ("url", paper.url),
        ]
        lines = [f"@article{{{_key(finding, used)},"]
        body = [f"  {name} = {{{value}}}," for name, value in fields if value]
        if body:
            body[-1] = body[-1].rstrip(",")
        lines.extend(body)
        lines.append("}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_ris(findings: Sequence[Finding]) -> str:
    """RIS records, importable by Zotero, EndNote and Mendeley."""
    blocks: list[str] = []
    for finding in findings:
        paper = finding.paper
        lines = ["TY  - JOUR", f"TI  - {paper.title}"]
        for author in paper.authors:
            lines.append(f"AU  - {author}")
        if paper.journal:
            lines.append(f"JO  - {paper.journal}")
        if paper.year:
            lines.append(f"PY  - {paper.year}")
        if paper.doi:
            lines.append(f"DO  - {paper.doi}")
        for url in _urls(finding):
            lines.append(f"UR  - {url}")
        if paper.abstract:
            lines.append("AB  - " + " ".join(paper.abstract.split())[:2000])
        lines.extend(["ER  - ", ""])
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _urls(finding: Finding) -> Iterable[str]:
    if finding.access:
        if finding.access.doi_url:
            yield finding.access.doi_url
        if finding.access.best and finding.access.best.get("url"):
            yield finding.access.best["url"]
    elif finding.paper.url:
        yield finding.paper.url


FORMATS = ("text", "md", "json", "bibtex", "ris")


def render(findings: Sequence[Finding], fmt: str) -> str:
    if fmt == "bibtex":
        return to_bibtex(findings)
    if fmt == "ris":
        return to_ris(findings)
    raise ValueError(f"unsupported citation format: {fmt}")
