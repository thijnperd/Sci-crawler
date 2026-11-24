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

import functools
import json
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

from .config import Settings
from .intent import exclusion_terms, variants
from .models import Paper, ScopeSpec, Verdict
from .net import FetchError, fetch

_HYPHENS = re.compile(r"[-\u2010-\u2015\u2212]+")
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def normalize(text: str | None) -> str:
    """Lowercase, strip punctuation/hyphens, collapse whitespace."""
    if not text:
        return ""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = _HYPHENS.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


@functools.lru_cache(maxsize=2048)
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


_TYPE2_TITLE_PATTERNS = (
    _pattern("type 2 diabetes"),
    _pattern("diabetes type 2"),
    _pattern("type ii diabetes"),
    _pattern("t2d"),
    _pattern("t2dm"),
    _pattern("non insulin dependent diabetes"),
    _pattern("gestational diabetes"),
)
_TYPE1_TITLE_PATTERNS = (
    _pattern("type 1 diabetes"),
    _pattern("diabetes type 1"),
    _pattern("type i diabetes"),
    _pattern("t1d"),
    _pattern("t1dm"),
    _pattern("insulin dependent diabetes"),
)

_DIET_REVERSE_ALIASES = (
    "eating habit", "eating habits", "eating behavior", "eating behaviour",
    "eating behaviors", "eating behaviours", "disordered eating",
    "eating disorder", "eating disorders", "dietary behavior",
    "dietary behaviour", "dietary habits", "food choice", "food choices",
    "dietary adherence", "diet", "diets", "dietary", "nutrition",
    "nutritional", "food intake",
)

_DIABETES_REVERSE_ALIASES = ("diabetes", "diabetes mellitus", "diabetic")

_BUCKET_ORDER = {"in_scope": 0, "adjacent": 1, "out_of_scope": 2}


def _alt_pattern(terms: Iterable[str]) -> str:
    cleaned = sorted({normalize(t) for t in terms if normalize(t)}, key=len, reverse=True)
    return "|".join(re.escape(t) for t in cleaned)


def _compile_causal_patterns(a_alt: str, b_alt: str) -> list[re.Pattern[str]]:
    if not a_alt or not b_alt:
        return []
    a = rf"(?<![a-z0-9])(?:{a_alt})(?![a-z0-9])"
    b = rf"(?<![a-z0-9])(?:{b_alt})(?![a-z0-9])"
    return [
        re.compile(
            rf"(?:influence|effect|effects|impact|impacts|role|consequence|consequences|contribution)"
            rf"\s+of\s+.{{0,45}}?{a}.{{0,65}}?\s+(?:on|upon|toward|towards)\s+.{{0,45}}?{b}"
        ),
        re.compile(
            rf"{a}.{{0,45}}?\s+(?:causes?|causing|leads?\s+to|leading\s+to|induces?|inducing|"
            rf"triggers?|triggering|increases?|increasing|reduces?|reducing|prevents?|preventing|"
            rf"improves?|improving|worsens?|worsening|promotes?|promoting|affects?|affecting|"
            rf"influences?|influencing|impacts?|impacting|predicts?|predicting|"
            rf"contributes?\s+to|modulates?|exacerbates?|ameliorates?|"
            rf"alters?|altering|changes?|changing|shapes?|shaping|disrupts?|disrupting)"
            rf"\s+.{{0,55}}?{b}"
        ),
        re.compile(
            rf"{b}.{{0,40}}?\s+(?:caused\s+by|induced\s+by|triggered\s+by|attributable\s+to|"
            rf"resulting\s+from|due\s+to)\s+.{{0,40}}?{a}"
        ),
        re.compile(
            rf"(?:risk|onset|development|incidence|progression|control|management|complications?)"
            rf"\s+of\s+.{{0,30}}?{b}.{{0,45}}?\s+(?:from|by|after|following)\s+.{{0,30}}?{a}"
        ),
    ]


def _check_direction(
    title_norm: str,
    sentence_pairs: list[tuple[str, str]],
    spec: ScopeSpec,
) -> tuple[str | None, str, str]:
    """Verify causal direction (X -> Y) against reversed (Y -> X) or descriptive framing."""
    if not spec.relation or not spec.relation.source or not spec.relation.target:
        return None, "", ""

    facet_map = {f.name: f for f in spec.facets}
    src_terms: set[str] = set()
    for name in spec.relation.source:
        if name in facet_map:
            src_terms.update(facet_map[name].terms)
            src_terms.update(facet_map[name].canonical)
        else:
            src_terms.add(name)

    dst_terms: set[str] = set()
    for name in spec.relation.target:
        if name in facet_map:
            dst_terms.update(facet_map[name].terms)
            dst_terms.update(facet_map[name].canonical)
        else:
            dst_terms.add(name)

    rev_src_terms = set(src_terms)
    if any(n in ("unhealthy diet", "diet") for n in spec.relation.source):
        rev_src_terms.update(_DIET_REVERSE_ALIASES)

    rev_dst_terms = set(dst_terms)
    if any(n in ("type 1 diabetes", "type 2 diabetes") for n in spec.relation.target):
        rev_dst_terms.update(_DIABETES_REVERSE_ALIASES)

    fwd_patterns = _compile_causal_patterns(_alt_pattern(src_terms), _alt_pattern(dst_terms))
    rev_patterns = _compile_causal_patterns(_alt_pattern(rev_dst_terms), _alt_pattern(rev_src_terms))

    src_rev_alt = _alt_pattern(rev_src_terms)
    dst_rev_alt = _alt_pattern(rev_dst_terms)
    desc_pattern = (
        re.compile(
            rf"(?<![a-z0-9])(?:{src_rev_alt})(?![a-z0-9]).{{0,45}}?\s+(?:in|among|amongst|of)\s+"
            rf".{{0,45}}?\b(?:with|having|diagnosed\s+with)\s+.{{0,25}}?"
            rf"(?<![a-z0-9])(?:{dst_rev_alt})(?![a-z0-9])"
        )
        if src_rev_alt and dst_rev_alt
        else None
    )

    def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
        return any(pat.search(text) for pat in patterns) if text else False

    title_fwd = _matches(fwd_patterns, title_norm)
    title_rev = _matches(rev_patterns, title_norm)
    title_desc = bool(desc_pattern and title_norm and desc_pattern.search(title_norm))

    src_label = ", ".join(spec.relation.source)
    dst_label = ", ".join(spec.relation.target)

    if title_rev and not title_fwd:
        raw_t = sentence_pairs[0][0] if sentence_pairs else ""
        return "reversed", f"reversed direction: studies {dst_label} -> {src_label}, asked for {src_label} -> {dst_label}", raw_t
    if title_fwd:
        raw_t = sentence_pairs[0][0] if sentence_pairs else ""
        return "aligned", f"direction aligned ({src_label} -> {dst_label})", raw_t

    fwd_sentence = ""
    rev_hits = 0
    for raw_s, norm_s in sentence_pairs[1:]:
        if _matches(fwd_patterns, norm_s) and not fwd_sentence:
            fwd_sentence = raw_s
        if _matches(rev_patterns, norm_s):
            rev_hits += 1

    if rev_hits > 0 and not fwd_sentence:
        return "reversed", f"reversed direction: studies {dst_label} -> {src_label}, asked for {src_label} -> {dst_label}", ""
    if fwd_sentence:
        return "aligned", f"direction aligned ({src_label} -> {dst_label})", fwd_sentence
    if title_desc:
        raw_t = sentence_pairs[0][0] if sentence_pairs else ""
        return "descriptive", f"describes {src_label} in {dst_label} without clear {src_label} -> {dst_label} effect", raw_t
    return "ambiguous", "", ""


def _subtype_mismatch(title_norm: str, spec: ScopeSpec) -> str | None:
    if not title_norm:
        return None
    facet_map = {f.name: f for f in spec.facets}
    if "type 1 diabetes" in facet_map and "type 2 diabetes" not in facet_map:
        t1_patterns = [_pattern(t) for t in facet_map["type 1 diabetes"].terms]
        has_t1 = any(_hit(p, title_norm) for p in t1_patterns) or bool(
            re.search(r"\btype\s+(?:1|i)\b", title_norm)
        )
        if any(_hit(p, title_norm) for p in _TYPE2_TITLE_PATTERNS) and not has_t1:
            return "title is about type 2 / gestational diabetes, not type 1 diabetes"
    elif "type 2 diabetes" in facet_map and "type 1 diabetes" not in facet_map:
        t2_patterns = [_pattern(t) for t in facet_map["type 2 diabetes"].terms]
        has_t2 = any(_hit(p, title_norm) for p in t2_patterns) or bool(
            re.search(r"\btype\s+(?:2|ii)\b", title_norm)
        )
        if any(_hit(p, title_norm) for p in _TYPE1_TITLE_PATTERNS) and not has_t2:
            return "title is about type 1 diabetes, not type 2 diabetes"
    return None


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
    sentence_pairs = [
        (" ".join(item.split()), normalize(item))
        for item in raw_sentences
        if item and item.strip()
    ]

    total_weight = sum(f.weight for f in spec.facets) or 1.0
    matched_weight = 0.0
    matched: list[str] = []
    missing: list[str] = []
    title_hits: list[str] = []
    best_sentence_weight = 0.0
    best_sentence_raw = ""

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
    if sentence_pairs and spec.facets:
        for raw_sentence, sentence in sentence_pairs:
            covered = 0.0
            names: list[str] = []
            for facet in spec.facets:
                if any(_hit(_pattern(t), sentence) for t in facet.terms):
                    covered += facet.weight
                    names.append(facet.name)
            if covered > best_sentence_weight:
                best_sentence_weight = covered
                best_sentence_facets = names
                best_sentence_raw = raw_sentence

    coverage = round(matched_weight / total_weight, 3)
    title_coverage = round(len(title_hits) / max(len(spec.facets), 1), 3)
    proximity = round(min(1.0, best_sentence_weight / total_weight), 3)

    score = 0.72 * coverage + 0.18 * title_coverage + 0.10 * proximity
    if not title_hits:
        score *= 0.78  # abstract-only matches are usually the broad ones
    if not body:
        score *= 0.90  # title-only records carry higher uncertainty than verified abstracts
    score = round(min(1.0, score), 3)

    verdict = Verdict(
        doi=paper.doi,
        title=paper.title,
        score=score,
        coverage=coverage,
        title_coverage=title_coverage,
        matched=matched,
        missing=missing,
        matched_snippet=best_sentence_raw[:320],
    )

    # Explicit exclusions and date filters are hard disqualifiers.
    haystack = f"{title} {keywords} {body}"
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

    for sense_term, label in spec.sense_excludes:
        found = _pattern(sense_term).search(haystack) if haystack else None
        if found:
            verdict.excluded_reason = (
                f"wrong domain sense ({label}: matched '{found.group(0)}')"
            )
            verdict.bucket = "out_of_scope"
            verdict.in_scope = False
            verdict.reason = verdict.excluded_reason
            return verdict

    subtype_reason = _subtype_mismatch(title, spec)
    if subtype_reason:
        verdict.excluded_reason = subtype_reason
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

    direction, dir_note, dir_snippet = _check_direction(title, sentence_pairs, spec)
    verdict.direction = direction
    if dir_snippet and (direction == "aligned" or not verdict.matched_snippet):
        verdict.matched_snippet = dir_snippet[:320]
    if direction == "reversed":
        verdict.score = round(verdict.score * 0.45, 3)
        verdict.excluded_reason = dir_note
        verdict.bucket = "out_of_scope"
        verdict.in_scope = False
        verdict.reason = dir_note
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

    if direction == "descriptive":
        if verdict.bucket == "in_scope":
            verdict.bucket = "adjacent"
            verdict.in_scope = False
        verdict.score = round(verdict.score * 0.78, 3)
    elif direction == "aligned":
        verdict.score = round(min(1.0, verdict.score + 0.08), 3)

    bits = []
    if dir_note:
        bits.append(dir_note)
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

Judge each candidate on SUBJECT-LEVEL scope, causal direction, and anatomical/domain
sense, not mere keyword presence:
- Check causal direction: if the intent asks how X influences Y, a paper about how Y
  influences X (reversed causation) or merely describing X in patients with Y is NOT
  in_scope (use "out_of_scope" for reversed, "adjacent" for descriptive).
- Check domain sense: e.g. exercise-induced cardiac/ventricular hypertrophy is NOT
  skeletal muscle hypertrophy ("off_topic"); type 2 diabetes is NOT type 1 diabetes.
- A paper on a neighbouring topic is out of scope, e.g. for "muscle strength in older adults":
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
    """Order screened papers: bucket (in_scope -> adjacent -> rejected), then score, citations, recency."""
    papers = list(papers)
    if len(papers) != len(verdicts):
        raise ValueError(
            f"rank() length mismatch: {len(papers)} papers != {len(verdicts)} verdicts"
        )
    pairs = list(zip(papers, verdicts))
    ceiling = max((p.cited_by for p in papers), default=0)
    years = [p.year for p in papers if p.year]
    newest = max(years) if years else None
    oldest = min(years) if years else None

    def sort_key(pair: tuple[Paper, Verdict]) -> tuple[int, float, float]:
        paper, verdict = pair
        recency = 0.0
        if newest and oldest and newest != oldest and paper.year:
            recency = (paper.year - oldest) / (newest - oldest)
        blended = 0.84 * verdict.score + 0.11 * _citation_norm(paper.cited_by, ceiling) + 0.05 * recency
        # Sort bucket ascending (in_scope first), then score/coverage descending.
        return (_BUCKET_ORDER.get(verdict.bucket, 2), -blended, -verdict.coverage)

    return sorted(pairs, key=sort_key)
