"""Core data structures shared by every part of PaperHound."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)


def norm_doi(value: str | None) -> str | None:
    """Normalise a DOI to its bare, lowercase form, or None if unusable."""
    if not value:
        return None
    doi = _DOI_PREFIX.sub("", str(value).strip())
    doi = doi.strip().rstrip(".,;")
    if not doi.lower().startswith("10."):
        return None
    # DOIs are case-insensitive; the resolvers we query all accept lowercase.
    return doi.lower()


def title_key(title: str | None) -> str:
    """A stable key for de-duplicating papers that have no DOI."""
    if not title:
        return ""
    key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return key[:110]


@dataclass
class Paper:
    """One work, merged from every source that returned it."""

    title: str
    doi: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    abstract: str | None = None
    url: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    keywords: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    cited_by: int = 0
    is_oa: bool = False
    oa_url: str | None = None
    type: str | None = None
    sources: list[str] = field(default_factory=list)
    # Filled in by the reading step.
    excerpt: str | None = None
    fulltext_chars: int = 0
    read_note: str | None = None

    def __post_init__(self) -> None:
        # DOIs are case-insensitive, so normalise on construction: de-duplication
        # otherwise treats 10.1/A and 10.1/a as different papers.
        self.doi = norm_doi(self.doi)
        self.authors = [a for a in self.authors if a]

    def key(self) -> str:
        return self.doi or title_key(self.title)

    def author_line(self, limit: int = 3) -> str:
        if not self.authors:
            return "unknown authors"
        if len(self.authors) <= limit:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:limit]) + " et al."

    def merge(self, other: "Paper") -> None:
        """Absorb a duplicate record, keeping the richest field of each."""
        if other.doi and not self.doi:
            self.doi = other.doi
        if other.title and len(other.title) > len(self.title or ""):
            self.title = other.title
        for name in ("pmid", "pmcid", "arxiv_id", "journal", "url", "type"):
            if not getattr(self, name) and getattr(other, name):
                setattr(self, name, getattr(other, name))
        if other.abstract and len(other.abstract) > len(self.abstract or ""):
            self.abstract = other.abstract
        if other.year and (self.year is None or other.year < self.year):
            self.year = other.year
        self.cited_by = max(self.cited_by, other.cited_by)
        self.is_oa = self.is_oa or other.is_oa
        self.oa_url = self.oa_url or other.oa_url
        for name in ("authors", "keywords", "concepts", "sources"):
            bucket = getattr(self, name)
            for item in getattr(other, name):
                if item and item not in bucket:
                    bucket.append(item)

    def to_dict(self, *, abstract_chars: int | None = None) -> dict[str, Any]:
        data = asdict(self)
        if abstract_chars is not None and data.get("abstract"):
            text = data["abstract"]
            if len(text) > abstract_chars:
                data["abstract"] = text[:abstract_chars].rstrip() + " [...]"
        return data


@dataclass
class Facet:
    """One concept the user's intent demands, and the words that express it.

    ``weight`` lets generic words ("study", "effect") count for less than
    content-bearing ones, so a paper cannot pass screening on filler alone.
    """

    name: str
    terms: list[str]
    weight: float = 1.0
    phrases: list[str] = field(default_factory=list)
    # The human-readable thesaurus group, when this facet came from one. `terms`
    # also carries generated plurals for matching; `canonical` is what belongs
    # in a search query and in explanations.
    canonical: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScopeSpec:
    """A parsed research intent: what the user wants, and what disqualifies."""

    intent: str
    facets: list[Facet] = field(default_factory=list)
    # The head noun of the leading phrase ("strength" in "muscle strength in
    # older adults"). A paper must contain it to count as on-target: coverage
    # alone would happily accept a paper about muscle *mass*.
    head: Facet | None = None
    core: list[Facet] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    min_scope: float = 0.7
    extra_notes: list[str] = field(default_factory=list)

    def facet_names(self) -> list[str]:
        return [f.name for f in self.facets]

    def query(self, mode: str = "and", fields: Iterable[str] = ()) -> str:
        """Build a source query string from the facets.

        ``and`` is precision-first (every facet must appear); ``or`` is the
        recall fallback the pipeline uses when ``and`` returns too little.
        """
        joiner = " AND " if mode == "and" else " OR "
        chunks: list[str] = []
        for facet in self.facets:
            # Prefer the curated synonyms over generated plurals, and keep the
            # query to single words: an unquoted multi-word term is an AND in
            # most of these APIs and would narrow the search unpredictably.
            curated = [word for word in (facet.canonical or facet.terms) if " " not in word]
            words = (curated or facet.terms)[:6]
            chunks.append("(" + " OR ".join(words) + ")")
        # Exclusions are applied during screening, not in the source query: a
        # paper that mentions the excluded concept at all is disqualified, and
        # most APIs have no reliable NOT operator.
        query = joiner.join(chunks) or self.intent
        if fields:
            field = next(iter(fields))
            query = joiner.join(f"{field}:{c}" for c in chunks) if chunks else self.intent
        return query

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "facets": [f.to_dict() for f in self.facets],
            "head": self.head.to_dict() if self.head else None,
            "core": [f.to_dict() for f in self.core],
            "exclude": self.exclude,
            "year_from": self.year_from,
            "year_to": self.year_to,
            "min_scope": self.min_scope,
            "notes": self.extra_notes,
        }


@dataclass
class Verdict:
    """The screening decision for one paper against one ScopeSpec."""

    doi: str | None
    title: str
    score: float
    coverage: float
    title_coverage: float
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    in_scope: bool = False
    bucket: str = "out_of_scope"  # in_scope | adjacent | out_of_scope
    reason: str = ""
    excluded_reason: str | None = None
    llm: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
