"""The pipeline: search -> read -> screen -> rank -> resolve access.

This is the whole promise of the tool in one function. Given a research intent
it fans out to several bibliographic APIs, merges duplicates, reads what it can,
rejects papers that are not actually on the subject (the step the original
crawler left to a human), and only then attaches Sci-Hub links to what survived.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from . import filters, reading, resolver, screening, sources, store
from .config import Settings
from .intent import exclusion_terms, parse_intent
from .models import Paper, ScopeSpec, Verdict, title_key

SearchFilter = Callable[..., sources.SourceResult]
Searcher = Callable[..., sources.SourceResult]
ProgressCallback = Callable[[str, dict[str, Any]], None]


def _emit(on_progress: ProgressCallback | None, stage: str, **data: Any) -> None:
    if on_progress is None:
        return
    try:
        on_progress(stage, data)
    except Exception as exc:  # isolate UI callback unless debugging
        if os.environ.get("PAPERSPLEASE_DEBUG"):
            raise
        print(f"papersplease warning: progress callback failed: {exc}", file=sys.stderr)


@dataclass
class Finding:
    paper: Paper
    verdict: Verdict
    access: resolver.Access | None = None

    def to_dict(self, abstract_chars: int = 700) -> dict:
        return {
            "title": self.paper.title,
            "doi": self.paper.doi,
            "year": self.paper.year,
            "authors": self.paper.authors[:6],
            "journal": self.paper.journal,
            "cited_by": self.paper.cited_by,
            **(
                {"abstract": (self.paper.abstract or "")[:abstract_chars]}
                if abstract_chars and self.paper.abstract
                else {}
            ),
            "found_in": self.paper.sources,
            "scope": self.verdict.to_dict(),
            "access": self.access.to_dict() if self.access else None,
        }


@dataclass
class Report:
    spec: ScopeSpec
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    deep_reads: int = 0
    # How many papers passed screening, before the presentation cap.
    in_scope_total: int | None = None
    # Verdict/access cache counters, so a caller can see a repeat run was free.
    cache_stats: dict[str, Any] = field(default_factory=dict)

    def bucket(self, name: str) -> list[Finding]:
        return [f for f in self.findings if f.verdict.bucket == name]

    @property
    def in_scope(self) -> list[Finding]:
        return self.bucket("in_scope")

    @property
    def adjacent(self) -> list[Finding]:
        return self.bucket("adjacent")

    @property
    def out_of_scope(self) -> list[Finding]:
        return self.bucket("out_of_scope")

    def summary(self) -> str:
        total = self.in_scope_total if self.in_scope_total is not None else len(self.in_scope)
        shown = f" (showing {len(self.in_scope)})" if total > len(self.in_scope) else ""
        candidate_word = "candidate" if self.candidates == 1 else "candidates"
        return (
            f"{total} in scope{shown} / {len(self.adjacent)} adjacent / "
            f"{len(self.out_of_scope)} rejected from {self.candidates} {candidate_word}"
        )

    def to_dict(self, *, abstract_chars: int = 700, include_rejected: bool = False) -> dict:
        keep = self.findings if include_rejected else [f for f in self.findings if f.verdict.bucket != "out_of_scope"]
        return {
            "schema_version": 1,
            "intent": self.spec.intent,
            "scope": self.spec.to_dict(),
            "summary": self.summary(),
            "in_scope_total": self.in_scope_total if self.in_scope_total is not None else len(self.in_scope),
            "sources": self.source_counts,
            "warnings": self.warnings,
            "cache": self.cache_stats or None,
            "results": [f.to_dict(abstract_chars=abstract_chars) for f in keep],
            "rejected": [
                {"title": f.paper.title, "doi": f.paper.doi, "reason": f.verdict.reason,
                 "coverage": f.verdict.coverage}
                for f in self.out_of_scope
            ] if include_rejected else [],
        }


def explain_intent(
    intent: str,
    *,
    settings: Settings | None = None,
    min_scope: float | None = None,
    require: Iterable[str] = (),
    exclude: Iterable[str] = (),
    require_in: str = "",
    since: int | None = None,
    until: int | None = None,
) -> dict[str, Any]:
    """Show exactly what the screener will enforce for an intent.

    Costs nothing and touches no network, so an agent can call it before
    spending a search: a bad parse (a wrong head concept, an exclusion that
    swallowed the topic) becomes visible here instead of silently returning
    plausible-looking rubbish.
    """
    settings = settings or Settings.from_env()
    spec = parse_intent(
        intent,
        min_scope=min_scope if min_scope is not None else settings.min_scope,
        require=require,
        exclude=exclude,
        require_in=require_in,
    )
    if since:
        spec.year_from = since
    if until:
        spec.year_to = until
    issues: list[str] = []
    if len(spec.facets) < 2:
        issues.append("only one concept was parsed; add detail or pass require=[...]")
    listed = any("lists several concepts" in note for note in spec.extra_notes)
    if spec.head is None and not listed:
        issues.append("no head concept detected; the specificity guard is weaker")
    if len(spec.facets) > 6:
        issues.append(
            f"{len(spec.facets)} concepts parsed; source queries use the 4 strongest and "
            "screening enforces all of them"
        )
    return {
        "schema_version": 1,
        "intent": spec.intent,
        "search_text": spec.search_text,
        "facets": [
            {
                "name": facet.name,
                "weight": facet.weight,
                # Curated synonyms read better than generated plurals.
                "terms": (facet.canonical or facet.terms)[:12],
            }
            for facet in spec.facets
        ],
        "head": spec.head.name if spec.head else None,
        "exclude": spec.exclude,
        "exclude_terms": {term: exclusion_terms(term) for term in spec.exclude},
        "years": {"from": spec.year_from, "to": spec.year_to},
        "min_scope": spec.min_scope,
        "query": spec.query(),
        "required": [list(item) for item in spec.required],
        "require_field": spec.require_field,
        "anchors": spec.anchors,
        "relation": spec.relation.to_dict() if spec.relation else None,
        "sense_excludes": [term for term, _ in spec.sense_excludes],
        "notes": spec.extra_notes,
        "issues": issues,
        "how_to_read": (
            "facets are the concepts coverage is computed over; head is the concept that "
            "must be present and must co-occur with another; anchors are the key concepts of "
            "a list-style intent (two must be covered, two named in the title); exclude lists "
            "hard disqualifiers (and the spellings each one catches); issues lists parse "
            "problems worth fixing before searching."
        ),
    }


def _screen_all(papers: Sequence[Paper], spec: ScopeSpec, caches: store.Caches) -> list[Verdict]:
    """Screen papers, reusing a stored verdict when the paper and scope match."""
    verdicts: list[Verdict] = []
    for paper in papers:
        cached = caches.get_verdict(spec, paper.key())
        if cached is not None:
            verdicts.append(cached)
            continue
        verdict = screening.screen_one(paper, spec)
        caches.put_verdict(spec, paper.key(), verdict)
        verdicts.append(verdict)
    return verdicts


def _access(
    paper: Paper,
    settings: Settings,
    caches: store.Caches,
    *,
    probe: bool,
    include_oa: bool,
) -> resolver.Access:
    """Resolve links, reusing a recent answer for the same DOI and access flags."""
    mirrors = tuple(settings.mirrors)
    if paper.doi:
        raw = caches.get_access(paper.doi, include_oa=include_oa, probe=probe, mirrors=mirrors)
        if raw:
            try:
                return resolver.Access(**raw)
            except TypeError:
                pass
    access = resolver.access_for(paper, settings, probe=probe, include_oa=include_oa)
    if paper.doi:
        caches.put_access(
            paper.doi,
            access.to_dict(),
            include_oa=include_oa,
            probe=probe,
            mirrors=mirrors,
        )
    return access


def merge_papers(papers: Iterable[Paper]) -> list[Paper]:
    """Deduplicate by DOI, then by title for records missing a DOI."""
    by_key: dict[str, Paper] = {}
    for paper in papers:
        if not paper.title:
            continue
        key = paper.key()
        if key in by_key:
            by_key[key].merge(paper)
        else:
            by_key[key] = paper

    by_title: dict[str, list[Paper]] = {}
    for paper in by_key.values():
        key = title_key(paper.title)
        bucket = by_title.setdefault(key, [])
        merged = False
        for existing in bucket:
            if not existing.doi or not paper.doi:
                existing.merge(paper)
                merged = True
                break
        if not merged:
            # Same title, distinct DOIs: keep every distinct DOI record.
            bucket.append(paper)
    return [paper for bucket in by_title.values() for paper in bucket]


def _search_all(
    spec: ScopeSpec,
    settings: Settings,
    limit: int,
    source_names: Sequence[str],
    searchers: dict[str, Searcher],
    mode: str,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[Paper], dict[str, int], list[str]]:
    papers: list[Paper] = []
    counts: dict[str, int] = {}
    warnings: list[str] = []
    usable = [name for name in source_names if name in searchers]
    for name in source_names:
        if name not in searchers:
            warnings.append(f"unknown source '{name}' (try: {', '.join(sorted(searchers))})")

    _emit(on_progress, "search_start", sources=usable, mode=mode)
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(usable)))) as pool:
        futures = {
            pool.submit(searchers[name], spec, settings, limit, mode): name for name in usable
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - a broken adapter must not kill the run
                warnings.append(f"{name}: {type(exc).__name__}: {exc}")
                _emit(on_progress, "source_done", source=name, count=0, ok=False, error=str(exc))
                continue
            if result.warning:
                warnings.append(result.warning)
            counts[name] = len(result.papers)
            papers.extend(result.papers)
            _emit(
                on_progress,
                "source_done",
                source=name,
                count=len(result.papers),
                ok=not bool(result.warning),
                error=result.warning,
            )
    return papers, counts, warnings


def run_search(
    spec: ScopeSpec,
    settings: Settings,
    *,
    limit: int,
    source_names: Sequence[str],
    searchers: dict[str, Searcher],
    on_progress: ProgressCallback | None = None,
) -> tuple[list[Paper], dict[str, int], list[str]]:
    # Ask each source for roughly twice the requested pool: screening throws
    # papers away, and a pool of only 10 candidates is mostly reviews.
    per_source = min(80, max(15, limit * 2))
    papers, counts, warnings = _search_all(
        spec, settings, per_source, source_names, searchers, "and", on_progress=on_progress
    )
    if len(papers) < 5:
        # Precision-first queries can come back empty on small topics; widen.
        extra, extra_counts, extra_warnings = _search_all(
            spec, settings, per_source, source_names, searchers, "or", on_progress=on_progress
        )
        papers.extend(extra)
        for name, count in extra_counts.items():
            counts[name] = max(counts.get(name, 0), count)
        warnings.extend(w for w in extra_warnings if w not in warnings)
    return papers, counts, warnings


def find(
    intent: str,
    *,
    settings: Settings | None = None,
    limit: int = 20,
    source_names: Sequence[str] | None = None,
    deep: bool = False,
    deep_limit: int = 5,
    use_llm: bool | None = None,
    min_scope: float | None = None,
    require: Iterable[str] = (),
    exclude: Iterable[str] = (),
    since: int | None = None,
    until: int | None = None,
    authors: Iterable[str] = (),
    venues: Iterable[str] = (),
    min_citations: int | None = None,
    types: Iterable[str] = (),
    peer_reviewed: bool = False,
    preprints_only: bool = False,
    probe_mirrors: bool = True,
    include_oa: bool = True,
    access_limit: int | None = None,
    require_in: str = "",
    expand: str = "none",
    expand_limit: int = 25,
    use_cache: bool | None = None,
    searchers: dict[str, Searcher] | None = None,
    on_progress: ProgressCallback | None = None,
) -> Report:
    """Run the full pipeline for one research intent."""
    settings = settings or Settings.from_env()
    searchers = searchers or sources.SEARCHERS
    spec = parse_intent(
        intent,
        min_scope=min_scope if min_scope is not None else settings.min_scope,
        require=require,
        exclude=exclude,
        require_in=require_in,
    )
    if since:
        spec.year_from = since
    if until:
        spec.year_to = until
    if spec.year_from and spec.year_to and spec.year_from > spec.year_to:
        spec.year_from, spec.year_to = spec.year_to, spec.year_from

    _emit(
        on_progress,
        "parse",
        facets=[f.name for f in spec.facets],
        year_from=spec.year_from,
        year_to=spec.year_to,
        head=spec.head.name if spec.head else None,
    )

    names = list(source_names or settings.sources)
    papers, counts, warnings = run_search(
        spec, settings, limit=limit, source_names=names, searchers=searchers, on_progress=on_progress
    )
    merged = merge_papers(papers)
    report = Report(spec=spec, warnings=warnings, source_counts=counts, candidates=len(merged))
    before_filters = len(merged)
    merged = filters.apply(
        merged,
        authors=authors,
        venues=venues,
        min_citations=min_citations,
        types=types,
        peer_reviewed=peer_reviewed,
        preprints_only=preprints_only,
    )
    report.candidates = len(merged)
    _emit(on_progress, "search_done", candidates=len(merged), counts=counts)
    if len(merged) != before_filters:
        report.warnings.append(f"filters kept {len(merged)} of {before_filters} candidates")
    caches = store.Caches(settings, enabled=use_cache)

    # Read: fill in missing abstracts before judging anything (batched OpenAlex, then EPMC).
    if merged:
        missing_abs = [p for p in merged if not p.abstract]
        if missing_abs:
            _emit(on_progress, "read", done=0, total=len(missing_abs))
            try:
                reading.batch_fetch_abstracts(missing_abs, settings)
            except Exception as exc:  # noqa: BLE001 - defensive
                report.warnings.append(f"batch abstract lookup failed: {exc}")
            remaining = [p for p in missing_abs if not p.abstract and (p.pmid or p.pmcid or p.doi)][:12]
            done_count = len(missing_abs) - len(remaining)
            _emit(on_progress, "read", done=done_count, total=len(missing_abs))
            if remaining:
                completed = done_count
                with ThreadPoolExecutor(max_workers=min(4, len(remaining))) as pool:
                    futures = {pool.submit(reading.read, paper, settings): paper for paper in remaining}
                    for future in as_completed(futures):
                        paper = futures[future]
                        try:
                            future.result()
                        except Exception as exc:  # noqa: BLE001 - defensive
                            report.warnings.append(
                                f"read failed for {paper.doi or paper.title[:40]}: {exc}"
                            )
                        completed += 1
                        _emit(on_progress, "read", done=completed, total=len(missing_abs))
            _emit(on_progress, "read", done=len(missing_abs), total=len(missing_abs))
        else:
            _emit(on_progress, "read", done=0, total=0)
    else:
        _emit(on_progress, "read", done=0, total=0)

    verdicts = _screen_all(merged, spec, caches)
    _emit(
        on_progress,
        "screen_done",
        in_scope=sum(1 for v in verdicts if v.bucket == "in_scope"),
        adjacent=sum(1 for v in verdicts if v.bucket == "adjacent"),
        out_of_scope=sum(1 for v in verdicts if v.bucket == "out_of_scope"),
    )

    # Citation-graph expansion: one hop out from the best on-target papers
    # finds newer work (citations) and foundational work (references) that no
    # keyword query surfaced, then screens it like everything else.
    if expand in ("citations", "references", "both") and merged:
        seeds = [
            paper for paper, verdict in screening.rank(merged, verdicts)
            if verdict.bucket == "in_scope"
        ][:4]
        directions = ["citations", "references"] if expand == "both" else [expand]
        extra: list[Paper] = []
        for direction in directions:
            found, note = sources.expand_citations(
                seeds, settings, direction=direction, limit=max(5, expand_limit)
            )
            extra.extend(found)
            if note:
                report.warnings.append(note)
        known = {p.key() for p in merged}
        known_titles = {title_key(p.title) for p in merged}
        fresh = [
            p for p in merge_papers(extra)
            if p.key() not in known and title_key(p.title) not in known_titles
        ]
        if fresh:
            try:
                reading.batch_fetch_abstracts([p for p in fresh if not p.abstract], settings)
            except Exception as exc:  # noqa: BLE001 - a missing batch abstract must not end the run
                report.warnings.append(f"expansion abstract lookup failed: {exc}")
            fresh_missing = [p for p in fresh if not p.abstract][: max(12, expand_limit)]
            if fresh_missing:
                with ThreadPoolExecutor(max_workers=min(4, len(fresh_missing))) as pool:
                    futures = {pool.submit(reading.read, paper, settings): paper for paper in fresh_missing}
                    for future in as_completed(futures):
                        paper = futures[future]
                        try:
                            future.result()
                        except Exception as exc:  # noqa: BLE001 - a missing abstract must not end the run
                            report.warnings.append(
                                f"expand read failed for {paper.doi or paper.title[:40]}: {exc}"
                            )
            merged.extend(fresh)
            verdicts.extend(_screen_all(fresh, spec, caches))
            report.candidates = len(merged)
            report.warnings.append(
                f"citation expansion ({expand}) added {len(fresh)} candidates "
                f"from {len(seeds)} seed paper(s)"
            )

    llm_counts = use_llm if use_llm is not None else settings.llm_enabled
    if llm_counts and settings.llm_enabled and merged:
        shortlist = [
            (index, paper)
            for index, (paper, verdict) in enumerate(zip(merged, verdicts))
            if verdict.coverage >= 0.34
        ][:48]
        llm_map, note = screening.llm_screen([p for _, p in shortlist], spec, settings)
        if note:
            report.warnings.append(note)
        for local_index, item in llm_map.items():
            if 0 <= local_index < len(shortlist):
                screening.combine(verdicts[shortlist[local_index][0]], item)
    elif use_llm and not settings.llm_enabled:
        report.warnings.append(
            "LLM screening requested but no key set (PAPERSPLEASE_LLM_KEY / OPENAI_API_KEY); "
            "used lexical screening only"
        )

    # Deep read: re-examine the survivors against their full text.
    if deep:
        ranked = screening.rank(merged, verdicts)
        survivors = [pair for pair in ranked if pair[1].bucket == "in_scope"][:deep_limit]
        for paper, verdict in survivors:
            try:
                before = (verdict.coverage, verdict.bucket)
                reading.read(paper, settings, deep=True)
                refreshed = screening.screen_one(paper, spec)
                verdict.coverage = refreshed.coverage
                verdict.matched = refreshed.matched
                verdict.missing = refreshed.missing
                verdict.title_coverage = refreshed.title_coverage
                verdict.score = refreshed.score
                verdict.bucket = refreshed.bucket
                verdict.in_scope = refreshed.in_scope
                verdict.reason = f"{refreshed.reason} [re-checked against full text: {paper.read_note}]"
                if before[1] != verdict.bucket:
                    report.warnings.append(
                        f"full text changed the verdict for {paper.doi or paper.title[:40]}: "
                        f"{before[1]} -> {verdict.bucket}"
                    )
                report.deep_reads += 1
            except Exception as exc:  # noqa: BLE001 - defensive
                report.warnings.append(f"deep read failed for {paper.doi or paper.title[:40]}: {exc}")

    ranked = screening.rank(merged, verdicts)
    findings = [Finding(paper=paper, verdict=verdict) for paper, verdict in ranked]

    # Present at most `limit` on-target papers; the rest are counted in the
    # summary but not listed, so every listed paper can carry working links.
    in_scope_total = sum(1 for finding in findings if finding.verdict.bucket == "in_scope")
    capped: list[Finding] = []
    shown_in_scope = 0
    for finding in findings:
        if finding.verdict.bucket == "in_scope":
            if shown_in_scope >= limit:
                continue
            shown_in_scope += 1
        capped.append(finding)
    findings = capped
    report.in_scope_total = in_scope_total

    # Resolve access only for what is shown: all displayed in_scope findings
    # plus up to 6 adjacent candidates by default (not dozens of adjacent hits).
    if access_limit is None:
        in_scope_targets = [f for f in findings if f.verdict.bucket == "in_scope"]
        adjacent_targets = [f for f in findings if f.verdict.bucket == "adjacent"][:6]
        access_targets = in_scope_targets + adjacent_targets
    else:
        access_targets = [f for f in findings if f.verdict.bucket != "out_of_scope"][:access_limit]
    _emit(on_progress, "access", done=0, total=len(access_targets))
    if probe_mirrors and access_targets:
        try:
            resolver.mirror_health(settings)
        except Exception as exc:  # noqa: BLE001 - best-effort mirror warm-up
            report.warnings.append(f"mirror warm-up failed: {exc}")
    resolved = 0
    if access_targets:
        with ThreadPoolExecutor(max_workers=min(4, len(access_targets))) as pool:
            futures = {
                pool.submit(
                    _access,
                    finding.paper,
                    settings,
                    caches,
                    probe=probe_mirrors,
                    include_oa=include_oa,
                ): finding
                for finding in access_targets
            }
            for future in as_completed(futures):
                finding = futures[future]
                try:
                    finding.access = future.result()
                    if finding.access.notes:
                        report.warnings.extend(finding.access.notes)
                except Exception as exc:  # noqa: BLE001 - defensive
                    report.warnings.append(f"access lookup failed for {finding.paper.doi}: {exc}")
                resolved += 1
                _emit(on_progress, "access", done=resolved, total=len(access_targets))

    report.findings = findings
    caches.flush()
    report.cache_stats = caches.stats()
    # De-duplicate warnings while preserving order.
    seen: set[str] = set()
    report.warnings = [w for w in report.warnings if not (w in seen or seen.add(w))]
    _emit(
        on_progress,
        "done",
        in_scope=len(report.in_scope),
        adjacent=len(report.adjacent),
        out_of_scope=len(report.out_of_scope),
    )
    return report


def screen_only(
    intent: str,
    papers: Sequence[Paper],
    *,
    settings: Settings | None = None,
    use_llm: bool | None = None,
    min_scope: float | None = None,
    require: Iterable[str] = (),
    exclude: Iterable[str] = (),
    require_in: str = "",
    with_access: bool = False,
    use_cache: bool | None = None,
) -> list[Finding]:
    """Screen papers the caller already has (e.g. from a reference list)."""
    settings = settings or Settings.from_env()
    spec = parse_intent(
        intent,
        min_scope=min_scope if min_scope is not None else settings.min_scope,
        require=require,
        exclude=exclude,
        require_in=require_in,
    )
    caches = store.Caches(settings, enabled=use_cache)
    verdicts = _screen_all(papers, spec, caches)
    if (use_llm if use_llm is not None else settings.llm_enabled) and settings.llm_enabled:
        shortlist = [(i, p) for i, (p, v) in enumerate(zip(papers, verdicts)) if v.coverage >= 0.2][:48]
        llm_map, _ = screening.llm_screen([p for _, p in shortlist], spec, settings)
        for local, item in llm_map.items():
            if 0 <= local < len(shortlist):
                screening.combine(verdicts[shortlist[local][0]], item)
    ranked = screening.rank(papers, verdicts)
    findings = [Finding(paper=p, verdict=v) for p, v in ranked]
    if with_access:
        for finding in findings:
            finding.access = _access(
                finding.paper, settings, caches, probe=True, include_oa=True
            )
    caches.flush()
    return findings


def read_one(
    doi_or_title: str,
    *,
    settings: Settings | None = None,
    deep: bool = False,
) -> Paper | None:
    """Look up a single paper by DOI, arXiv id, URL or title."""
    settings = settings or Settings.from_env()
    candidate = doi_or_title.strip()
    paper = sources.resolve_identifier(candidate, settings)
    if paper is None:
        return None
    try:
        reading.read(paper, settings, deep=deep)
    except Exception as exc:  # noqa: BLE001 - preserve metadata when deep read fails
        paper.read_note = f"read failed: {type(exc).__name__}: {exc}"
    if paper.doi and not paper.oa_url:
        oa, _notes = resolver.oa_candidates(paper, settings)
        if oa:
            paper.oa_url = oa[0]["url"]
            paper.is_oa = True
    return paper
