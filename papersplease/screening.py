"""Scope screening: is this paper actually about what was asked for?

Two passes:

1. Lexical, always on, no API key. Each facet of the intent must show up
   somewhere in title / keywords / abstract, and the *fraction* of facets that
   show up is the coverage score. A paper about "muscle" alone therefore scores
   0.25 against "muscle strength in older adults", no matter how many times it
   says muscle. Title coverage and sentence proximity break remaining ties.

2. Optional LLM second opinion (``Settings.llm_enabled``). It can veto a
   lexical false positive ("this is about muscle *mass*, not *strength*") and
   its verdict carries equal weight in the final score.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Sequence

from .config import Settings
from .intent import exclusion_terms, variants
from .models import Paper, ScopeSpec, Verdict
from .net import FetchError, fetch

_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def normalize(text: str | None) -> str:
    """Lowercase, strip punctuation/hyphens, collapse whitespace."""
    if not text:
        return ""
    text = text.lower().replace("-", " ").replace("‐", " ").replace("’", "'")
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _pattern(term: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in normalize(term).split()]
    if not words:
        return re.compile(r"$^")
    body = r"\s+".join(words)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _hit(pattern: re.Pattern[str], text: str) -> bool:
    return bool(pattern.search(text)) if text else False


def _citation_norm(count: int, ceiling: int) -> float:
    if ceiling <= 0:
        return 0.0
    return math.log1p(max(count, 0)) / math.log1p(ceiling)


def screen_one(paper: Paper, spec: ScopeSpec) -> Verdict:
    """Score one paper against the parsed intent."""
    title = normalize(paper.title)
    abstract = normalize(paper.abstract)
    excerpt = normalize(paper.excerpt)
    # Deep reads add full-text prose; screen against everything we have read.
    body = f"{abstract} {excerpt}".strip()
    keywords = normalize(" ; ".join(paper.keywords + paper.concepts))
    # Split *before* normalising: normalise() strips the punctuation that marks
    # sentence boundaries, which turned every abstract into one long sentence
    # and made the co-occurrence guard pass on keyword scatter.
    raw_sentences = [paper.title or ""] + _SENTENCE_SPLIT.split(paper.abstract or "")
    if paper.excerpt:
        raw_sentences += _SENTENCE_SPLIT.split(paper.excerpt)
    sentences = [normalize(item) for item in raw_sentences if item and item.strip()]

    total_weight = sum(f.weight for f in spec.facets) or 1.0
    matched_weight = 0.0
    matched: list[str] = []
    missing: list[str] = []
    title_hits: list[str] = []
    best_sentence_weight = 0.0

    for facet in spec.facets:
        patterns = [_pattern(t) for t in facet.terms]
        in_title = any(_hit(p, title) for p in patterns)
        in_keywords = any(_hit(p, keywords) for p in patterns)
        in_body = any(_hit(p, body) for p in patterns)
        if in_title or in_keywords or in_body:
            matched.append(facet.name)
            matched_weight += facet.weight
            if in_title:
                title_hits.append(facet.name)
        else:
            missing.append(facet.name)

    # Proximity: the strongest single sentence, as a share of total facet weight.
    best_sentence_facets: list[str] = []
    if sentences and spec.facets:
        for sentence in sentences:
            covered = 0.0
            names: list[str] = []
            for facet in spec.facets:
                if any(_hit(_pattern(t), sentence) for t in facet.terms):
                    covered += facet.weight
                    names.append(facet.name)
            if covered > best_sentence_weight:
                best_sentence_weight = covered
                best_sentence_facets = names

    coverage = round(matched_weight / total_weight, 3)
    title_coverage = round(len(title_hits) / max(len(spec.facets), 1), 3)
    proximity = round(min(1.0, best_sentence_weight / total_weight), 3)

    score = 0.72 * coverage + 0.18 * title_coverage + 0.10 * proximity
    if not title_hits:
        score *= 0.78  # abstract-only matches are usually the broad ones
    score = round(min(1.0, score), 3)

    verdict = Verdict(
        doi=paper.doi,
        title=paper.title,
        score=score,
        coverage=coverage,
        title_coverage=title_coverage,
        matched=matched,
        missing=missing,
    )

    # Explicit exclusions and date filters are hard disqualifiers.
    haystack = " ".join((title, keywords, body))
    head_terms = spec.head.terms if spec.head else []
    head_present = not head_terms or any(
        _hit(_pattern(term), haystack) for term in head_terms
    )
    # Keywords sprinkled across an abstract are not a topic. For multi-facet
    # intents the head concept has to be discussed *with* another concept, in
    # one sentence or in the title.
    head_name = spec.head.name if spec.head else None
    cooccur = True
    if head_name and len(spec.facets) >= 2:
        cooccur = (
            (head_name in best_sentence_facets and len(best_sentence_facets) >= 2)
            or (head_name in title_hits and len(title_hits) >= 2)
        )
    elif len(spec.facets) >= 2:
        # A list of variables has no head noun, but the same anti-scatter rule
        # applies: two of the concepts must be discussed together somewhere,
        # not sprinkled one per paragraph.
        cooccur = len(best_sentence_facets) >= 2 or len(title_hits) >= 2
    # Anchor concepts: for list-style intents, a tangential paper (a creatine
    # trial that happened to record sets and sessions per week) can still reach
    # 70% weighted coverage by mentioning every variable once. Requiring two of
    # the top concepts *and* one of them in the title is what separates "this
    # paper studies volume" from "this paper mentions volume".
    anchor_ok = True
    anchor_hits: list[str] = []
    title_anchors: list[str] = []
    if len(spec.anchors) >= 2:
        anchor_hits = [name for name in spec.anchors if name in matched]
        title_anchors = [name for name in spec.anchors if name in title_hits]
        # The variables must be covered, and at least one must be named in the
        # title. A supplement trial measures the outcome and records sets and
        # sessions in its methods; it does not put "volume" in its title, and
        # that is the difference between studying a variable and logging it.
        needed = min(2, len(spec.anchors))
        anchor_ok = len(anchor_hits) >= needed and bool(title_anchors)
    # A facet and an exclusion may share a word ("muscle strength not about
    # muscle mass"); only an exact facet-name collision is ignored, so the
    # narrower excluded phrase still applies.
    facet_names = {f.name for f in spec.facets}
    for term in spec.exclude:
        if term in facet_names:
            continue
        for pattern in (_pattern(item) for item in exclusion_terms(term)):
            found = pattern.search(haystack) if haystack else None
            if found:
                # Name the spelling that actually matched: excluding "rodents"
                # rejecting a paper that says "mice" must not look arbitrary.
                verdict.excluded_reason = (
                    f"mentions excluded concept '{term}' (matched '{found.group(0)}')"
                )
                verdict.bucket = "out_of_scope"
                verdict.in_scope = False
                verdict.reason = verdict.excluded_reason
                return verdict
    if paper.year and spec.year_from and paper.year < spec.year_from:
        verdict.excluded_reason = f"year {paper.year} is before {spec.year_from}"
        verdict.bucket = "out_of_scope"
        verdict.reason = verdict.excluded_reason
        return verdict
    if paper.year and spec.year_to and paper.year > spec.year_to:
        verdict.excluded_reason = f"year {paper.year} is after {spec.year_to}"
        verdict.bucket = "out_of_scope"
        verdict.reason = verdict.excluded_reason
        return verdict

    # Required-in-field: the caller asked for a term by name in a given part of
    # the record ("the title must say strength"), which is stricter than the
    # thesaurus-backed facet coverage above.
    if spec.required:
        missing_required: list[str] = []
        for term, field in spec.required:
            haystacks = []
            if field in ("title", "title_or_abstract"):
                haystacks.append(title)
            if field in ("abstract", "title_or_abstract"):
                haystacks.append(body)
            if not any(
                _hit(_pattern(spelling), text)
                for spelling in variants(term)
                for text in haystacks
            ):
                missing_required.append(term)
        if missing_required:
            where = spec.require_field.replace("_", " ") or "requested field"
            verdict.excluded_reason = (
                f"required term(s) absent from the {where}: " + ", ".join(missing_required)
            )
            verdict.bucket = "out_of_scope"
            verdict.in_scope = False
            verdict.reason = verdict.excluded_reason
            return verdict

    if (
        coverage >= spec.min_scope
        and len(matched) >= min(2, len(spec.facets))
        and head_present
        and cooccur
        and anchor_ok
    ):
        verdict.bucket = "in_scope"
        verdict.in_scope = True
    elif coverage >= max(0.34, spec.min_scope * 0.55):
        verdict.bucket = "adjacent"
        verdict.in_scope = False
    else:
        verdict.bucket = "out_of_scope"
        verdict.in_scope = False

    bits = []
    if not head_present and spec.head is not None:
        bits.append(f"head concept '{spec.head.name}' absent")
    if head_present and not cooccur and head_name:
        bits.append(f"'{head_name}' never appears with the other concepts in one sentence")
    if not anchor_ok:
        missing_anchors = [name for name in spec.anchors if name not in matched]
        if missing_anchors:
            bits.append("anchor concepts missing: " + ", ".join(missing_anchors))
        if not title_anchors:
            bits.append(
                "title names none of the anchor concepts ("
                + ", ".join(spec.anchors) + "); need at least one"
            )
    if matched:
        bits.append("matched " + ", ".join(matched))
    if missing:
        bits.append("missing " + ", ".join(missing))
    if title_hits:
        bits.append("title names " + ", ".join(title_hits))
    elif title:
        bits.append("title does not name the topic (abstract-only match)")
    verdict.reason = "; ".join(bits)
    return verdict


def screen_many(papers: Sequence[Paper], spec: ScopeSpec) -> list[Verdict]:
    return [screen_one(p, spec) for p in papers]


def _llm_payload(spec: ScopeSpec, papers: Sequence[Paper]) -> list[dict[str, Any]]:
    rows = []
    for index, paper in enumerate(papers, start=1):
        rows.append(
            {
                "id": str(index),
                "title": paper.title,
                "abstract": (paper.abstract or paper.excerpt or "")[:1200],
            }
        )
    return rows


_PROMPT = """You are a strict literature screener for a researcher.

Research intent (verbatim): "{intent}"

The intent was decomposed into these concepts that a paper must actually be
about: {facets}

Judge each candidate on SUBJECT-LEVEL scope, not keyword presence. A paper on a
neighbouring topic is out of scope, e.g. for "muscle strength in older adults":
- "muscle mass and ageing" -> too_broad (mass is not strength)
- "strength training in young athletes" -> adjacent (population wrong)
- "grip strength predicts mortality in the elderly" -> in_scope
- "muscle fibre typing in rodents" -> off_topic

Return ONLY minified JSON of exactly this shape:
{{"verdicts":[{{"id":"1","verdict":"in_scope|adjacent|out_of_scope","specificity":"on_target|too_broad|too_specific|adjacent|off_topic","score":0.0,"reason":"<=25 words"}}]}}

Candidates:
{candidates}"""


def llm_screen(
    papers: Sequence[Paper],
    spec: ScopeSpec,
    settings: Settings,
    *,
    batch_size: int = 8,
    max_papers: int = 48,
) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Ask an OpenAI-compatible endpoint for a second opinion.

    Returns ({paper_index: verdict_dict}, warning). Any failure degrades to a
    warning: lexical screening already ran, so the tool stays useful offline.
    """
    if not settings.llm_enabled or not papers:
        return {}, None
    if len(papers) > max_papers:
        papers = papers[:max_papers]
    out: dict[int, dict[str, Any]] = {}
    offset = 0
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        prompt = _PROMPT.format(
            intent=spec.intent,
            facets=", ".join(f.name for f in spec.facets),
            candidates=json.dumps(_llm_payload(spec, batch), ensure_ascii=False),
        )
        body: dict[str, Any] = {
            "model": settings.llm_model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if settings.llm_json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            raw = fetch(
                f"{settings.llm_base}/chat/completions",
                settings,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.llm_key}",
                },
                retries=1,
                timeout=90,
            )
            text = json.loads(raw.decode("utf-8"))["choices"][0]["message"]["content"]
            parsed = _parse_json_block(text)
        except (FetchError, KeyError, IndexError, ValueError) as exc:
            return out, f"LLM screening unavailable, kept lexical verdicts ({exc})"
        for item in parsed.get("verdicts", []):
            try:
                local = int(str(item.get("id"))) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= local < len(batch):
                out[offset + local] = item
        offset += len(batch)
    return out, None


def _parse_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in LLM reply")
    return json.loads(text[start:end + 1])


def combine(verdict: Verdict, llm: dict[str, Any] | None) -> Verdict:
    """Fold an LLM verdict into the lexical one without losing either reason."""
    if not llm:
        return verdict
    verdict.llm = llm
    try:
        llm_score = float(llm.get("score", 0.0))
    except (TypeError, ValueError):
        llm_score = 0.0
    llm_score = max(0.0, min(1.0, llm_score))
    verdict.score = round(0.5 * verdict.score + 0.5 * llm_score, 3)
    call = str(llm.get("verdict", "")).lower()
    specificity = str(llm.get("specificity", "")).lower()
    reason = str(llm.get("reason", "")).strip()

    if call == "out_of_scope" or specificity in {"too_broad", "off_topic"}:
        verdict.bucket = "out_of_scope"
        verdict.in_scope = False
        verdict.excluded_reason = verdict.excluded_reason or f"LLM: {specificity or call}"
    elif call == "in_scope" and verdict.bucket != "out_of_scope":
        verdict.bucket = "in_scope"
        verdict.in_scope = True
    elif call == "adjacent" and verdict.bucket == "in_scope":
        verdict.bucket = "adjacent"
        verdict.in_scope = False
    if reason:
        verdict.reason = f"{verdict.reason} | reviewer: {reason}" if verdict.reason else f"reviewer: {reason}"
    return verdict


def rank(papers: Iterable[Paper], verdicts: Sequence[Verdict]) -> list[tuple[Paper, Verdict]]:
    """Order screened papers: scope first, then citations, then recency."""
    papers = list(papers)
    pairs = list(zip(papers, verdicts))
    ceiling = max((p.cited_by for p in papers), default=0)
    years = [p.year for p in papers if p.year]
    newest = max(years) if years else None
    oldest = min(years) if years else None

    def sort_key(pair: tuple[Paper, Verdict]) -> tuple[float, float]:
        paper, verdict = pair
        recency = 0.0
        if newest and oldest and newest != oldest and paper.year:
            recency = (paper.year - oldest) / (newest - oldest)
        blended = 0.82 * verdict.score + 0.13 * _citation_norm(paper.cited_by, ceiling) + 0.05 * recency
        # Sort descending, so invert for the tuple comparison.
        return (-blended, -verdict.coverage)

    return sorted(pairs, key=sort_key)
