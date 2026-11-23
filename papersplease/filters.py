"""Post-search filters: author, venue, citations and publication type.

Keyword APIs disagree about how to express "papers by Hinton at NeurIPS since
2020 with 100+ citations": some push part of it down, some silently ignore it,
and the merged pool mixes records no single provider filtered. Applying the
filter after the merge gives one contract that works the same for every source.

Missing metadata is kept rather than guessed: an absent venue is not evidence
of the wrong venue. Citation counts are the exception -- an unknown count
cannot satisfy a minimum, so ``min_citations`` excludes it, as the other
CLI tools that expose this filter do.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import Paper

TYPES = ("journal", "conference", "preprint", "book", "other")

_ALIASES = {
    "article": "journal",
    "journal-article": "journal",
    "journal_article": "journal",
    "proceedings": "conference",
    "proceedings-article": "conference",
    "conference-paper": "conference",
    "posted-content": "preprint",
    "book-chapter": "book",
    "monograph": "book",
}


def normalize_type(value: str | None) -> str | None:
    """Map a provider's type string onto the five words users understand."""
    if not value:
        return None
    text = value.strip().lower().replace("_", "-")
    if text in _ALIASES:
        return _ALIASES[text]
    if text in TYPES:
        return text
    if "preprint" in text:
        return "preprint"
    if "proceedings" in text or "conference" in text:
        return "conference"
    if "journal" in text or "article" in text:
        return "journal"
    if "book" in text or "monograph" in text or "chapter" in text:
        return "book"
    return "other"


def paper_type(paper: Paper) -> str:
    """Classify a paper from its provider type, falling back to identifiers."""
    kind = normalize_type(paper.type)
    if kind and kind != "other":
        return kind
    if paper.arxiv_id:
        return "preprint"
    if paper.journal:
        # Europe PMC uses labels like "review" and "clinical trial"; a record
        # in a journal is peer-reviewed even when the type word is unfamiliar.
        return "journal"
    return kind or "other"


def keep(
    paper: Paper,
    *,
    authors: Iterable[str] = (),
    venues: Iterable[str] = (),
    min_citations: int | None = None,
    types: Iterable[str] = (),
    peer_reviewed: bool = False,
    preprints_only: bool = False,
) -> bool:
    """True when a paper satisfies every requested filter."""
    wanted_authors = [a.strip().lower() for a in authors if a and a.strip()]
    if wanted_authors:
        haystack = " ".join(paper.authors).lower()
        if not any(name in haystack for name in wanted_authors):
            return False

    wanted_venues = [v.strip().lower() for v in venues if v and v.strip()]
    if wanted_venues:
        venue = (paper.journal or "").lower()
        if not any(name in venue for name in wanted_venues):
            return False

    if min_citations is not None and paper.cited_by < min_citations:
        return False

    allowed = {normalize_type(item) for item in types}
    allowed.discard(None)
    if peer_reviewed:
        allowed.update({"journal", "conference", "book"})
    if preprints_only:
        allowed.add("preprint")
    if allowed and paper_type(paper) not in allowed:
        return False
    return True


def apply(papers: Sequence[Paper], **criteria) -> list[Paper]:
    """Filter a merged pool, preserving order."""
    return [paper for paper in papers if keep(paper, **criteria)]
