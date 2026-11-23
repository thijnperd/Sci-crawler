"""Turn a natural-language research intent into a machine-checkable ScopeSpec.

The user's complaint about naive literature searching is precise: ask for
"muscle strength" and you get back everything ever written about muscles. So an
intent is decomposed into *facets* -- one per content-bearing concept, each with
its synonyms -- and screening later requires coverage of most of them. A paper
that only clears the "muscle" facet fails "muscle + strength + older + adults".
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Iterable

from .models import Facet, ScopeSpec
from .normalize import normalize_intent

# Words carrying no topical information. Removed outright.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "with", "without",
    "about", "into", "from", "by", "to", "at", "as", "is", "are", "was", "were",
    "be", "been", "being", "that", "this", "these", "those", "it", "its",
    "their", "there", "which", "who", "whom", "what", "when", "where", "how",
    "why", "any", "all", "some", "more", "most", "less", "least", "than",
    "very", "can", "could", "would", "should", "do", "does", "did", "has",
    "have", "had", "i", "we", "you", "he", "she", "they", "me", "us", "my",
    "our", "your", "please", "find", "search", "look", "papers", "paper",
    "articles", "article", "literature", "evidence", "studies", "study",
    "research", "work", "works", "publications", "publication", "scihub",
    "sci", "hub", "pdf", "pdfs", "doi", "dois", "link", "links", "read",
    # Connectives left behind by a negation phrase ("but not cancer") must not
    # become facets: they would win the head-concept slot and screen everything out.
    "but", "however", "whereas", "although", "though", "yet", "rather",
    "instead", "just", "merely", "only", "also", "plus", "either",
    "neither", "regarding", "concerning", "related", "versus", "vs",
    "not", "no", "nor", "excluding", "exclude", "except", "without",
    "unrelated", "anything", "cannot", "can't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "shouldn't",
    "couldn't", "significant", "significantly",
    # Filler that carries no subject: "optimal rep count" is about reps, not
    # about optimality. Age phrases are handled by _AGE_RE, not as facets.
    "optimal", "optimum", "best", "ideal", "recommended", "recommendation",
    "recommendations", "guideline", "guidelines", "count", "number", "many",
    "much", "often", "type", "types", "kind", "kinds", "year", "years",
    "old", "aged", "hours", "hour", "minutes", "times", "per", "week",
    "weeks", "day", "days", "month", "months",
}

# Kept as facets, but worth less than a content word: these appear in most
# abstracts and should never be enough on their own to pass screening.
# Thesaurus groups whose name is a container word rather than the topic. They
# stay facets (a paper should mention exercise) but at reduced weight, so a
# verbose question does not demand that every filler noun appear.
GENERIC_GROUPS = {"exercise", "adult", "child", "program"}

# Population words bound who a paper studied, not what it is about. They keep
# their facet (the population still has to match) but never become the anchor
# concepts of a list-style intent.
POPULATION_GROUPS = {"adult", "child", "young adults", "older"}

GENERIC = {
    "effect", "effects", "affect", "role", "roles", "relationship", "relation",
    "association", "associations", "impact", "impacts", "influence",
    "comparison", "difference", "differences", "change", "changes", "outcome",
    "outcomes", "analysis", "review", "reviews", "meta", "systematic",
    "randomized", "randomised", "controlled", "trial", "trials", "human",
    "humans", "patient", "patients", "participant", "participants", "subject",
    "subjects", "people", "persons", "individual", "individuals", "mechanism",
    "mechanisms", "method", "methods", "approach", "approaches", "new",
    "novel", "review", "overview", "update", "commentary", "perspective",
}

# A small, conservative thesaurus. It exists so that "elderly" satisfies
# "older adults" and "torque"/"force" satisfy "strength" -- never so that a
# merely neighbouring topic counts as a match (power is not strength).
_THESAURUS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("muscle", "muscles", "muscular", "musculature", "myofiber", "myofibre", "myofibrillar"),
    ("strength", "force", "torque", "isometric", "maximal voluntary contraction", "1rm", "one repetition maximum", "dynamometry"),
    ("weakness", "weak", "frailty", "frail", "low strength"),
    ("older", "elderly", "aged", "senior", "seniors", "old", "ageing", "aging", "advanced age"),
    ("adult", "adults", "men", "women", "volunteers", "participants", "subjects"),
    ("resistance training", "strength training", "weight training", "resistance exercise", "resistance exercise training", "gym"),
    ("hypertrophy", "muscle hypertrophy", "muscle growth", "muscle mass", "lean mass", "cross-sectional area", "muscle thickness"),
    ("sarcopenia", "sarcopenic", "muscle loss", "muscle wasting", "muscle atrophy", "atrophy"),
    ("exercise", "exercises", "training", "physical activity", "exercise training", "physical exercise"),
    ("grip", "handgrip", "hand grip", "grip strength", "handgrip strength"),
    ("protein", "proteins", "protein intake", "dietary protein", "amino acid", "amino acids", "supplementation"),
    ("mice", "mouse", "murine", "rat", "rats", "rodent", "rodents", "animal model"),
    ("child", "children", "adolescent", "adolescents", "youth", "pediatric", "paediatric", "young"),
    ("knee", "quadriceps", "extensor", "knee extension", "leg extension"),
    ("mobility", "gait", "walking speed", "physical function", "functional capacity"),
    ("fall", "falls", "fall risk"),
    ("electromyography", "emg", "motor unit", "activation", "neural", "neural drive"),
    ("machine learning", "deep learning", "neural network", "neural networks", "transformer", "transformers", "language model", "llm", "llms"),
    ("cancer", "tumor", "tumour", "oncology", "carcinoma"),
    ("brain", "cognition", "cognitive", "memory"),
    ("sleep", "circadian"),
    ("inflammation", "inflammatory", "cytokine", "cytokines"),
    ("dose", "dosing", "dosage", "dose-response"),
    # Training variables. Grouping them keeps a verbose strength-training
    # question at a handful of facets instead of one facet per noun, which is
    # what made long intents impossible to satisfy.
    ("volume", "training volume", "weekly volume", "set volume", "number of sets", "sets"),
    ("repetition", "repetitions", "rep", "reps", "repetition range", "rep range"),
    ("frequency", "training frequency", "weekly frequency", "sessions per week",
     "times per week", "hours per week", "training hours", "training duration"),
    ("program", "training program", "training programme", "training schedule",
     "routine", "split", "periodization", "periodisation", "exercise selection",
     "exercise order"),
    ("intensity", "training intensity", "load intensity", "proximity to failure",
     "repetitions in reserve", "rir", "rpe"),
    ("rest", "rest interval", "rest period", "interset rest", "between-set rest"),
    ("young adults", "young adult", "younger adults", "younger adult",
     "young men", "young women", "younger men", "younger women",
     "college students", "university students", "18-25", "18-30"),
)

_SYNONYM_INDEX: dict[str, tuple[str, ...]] = {}
for _group in _THESAURUS_GROUPS:
    for _word in _group:
        _SYNONYM_INDEX[_word] = _group

_PHRASES = sorted(
    (p for group in _THESAURUS_GROUPS for p in group if " " in p),
    key=len,
    reverse=True,
)

# "18 years old" / "18-year-olds" is a population, not a date filter and not
# two throwaway words; collapse it before the year regexes ever see it.
_AGE_RE = re.compile(
    r"\b(1[2-9]|[2-3]\d)\s*(?:-|\s)?\s*(?:year|yr)s?[\s-]*old\b", re.I
)

_YEAR = r"(?:19|20)\d{2}"
_RANGE_RE = re.compile(rf"\b({_YEAR})\s*(?:-|–|to|until|through)\s*({_YEAR})\b")
_SINCE_RE = re.compile(rf"\b(?:since|after|from|post|newer than|published after)\s*({_YEAR})\b")
_UNTIL_RE = re.compile(rf"\b(?:before|until|pre|older than|published before)\s*({_YEAR})\b")
_LAST_RE = re.compile(r"\b(?:last|past|recent)\s+(\d{1,2})\s+years?\b", re.I)
_DECADE_RE = re.compile(r"\b(?:last|past)\s+decade\b", re.I)
_IN_PAREN_RE = re.compile(rf"\(({_YEAR})\s*[-–]\s*({_YEAR})\)")
# Negation cues. Longest-first matching is essential: "not about" must win over
# "not", and "but not" must consume the "but" instead of leaving it behind as a
# topic word. Word boundaries are equally essential -- substring matching turned
# "amino acid" into "no acid" and ate the rest of the intent.
_CUE_KINDS: dict[str, str] = {
    "not just about": "contrast",
    "not only about": "contrast",
    "not merely about": "contrast",
    "not just": "contrast",
    "not only": "contrast",
    "not merely": "contrast",
    "not related to": "hard",
    "not concerned with": "hard",
    "not about": "hard",
    "not on": "hard",
    "unrelated to": "hard",
    "other than": "hard",
    "except for": "hard",
    "anything but": "hard",
    "excluding": "hard",
    "exclude": "hard",
    "except": "hard",
    "but not": "hard",
    "and not": "hard",
    "without": "hard",
    "not": "hard",
}

_CUE_RE = re.compile(
    r"(?<![a-z0-9])(?P<cue>"
    + "|".join(re.escape(cue) for cue in sorted(_CUE_KINDS, key=len, reverse=True))
    + r")(?![a-z0-9])",
    re.I,
)

# An excluded concept is a noun phrase, not the rest of the sentence. Stopping at
# prepositions keeps "without equipment in older adults" from excluding the
# population too.
_EXCLUDE_STOP = re.compile(
    r"[,;:.!?()\[\]\"\u201c\u201d]"
    r"|\s+(?:but|however|whereas|although|though|yet)\s+"
    r"|\s+(?:in|among|amongst|for|with|without|during|after|before|between|from|"
    r"across|within|at|on|to|via|using|than|that|which|whose|when|where|"
    r"versus|vs|compared)\s+",
    re.I,
)
_TERM_SPLIT = re.compile(r"\s*(?:,|;|/|&|\+|\band\b|\bor\b)\s*", re.I)
# "no rodents" is a negation; "amino acid" and "no significant difference" are not.
_NO_RE = re.compile(r"(?<![a-z0-9])no\s+(?P<piece>[a-z0-9][^,;:.!?()]*)", re.I)
_NO_FALSE_FRIENDS = {
    "longer", "more", "less", "fewer", "significant", "effect", "effects",
    "difference", "differences", "association", "evidence", "increase",
    "decrease", "change", "changes", "doubt", "matter", "one", "two",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

# Words that attach a following phrase ("strength IN older adults") rather than
# extending the topic. "and"/"or"/"of" are deliberately absent: they join the
# head phrase together ("effect of resistance training on muscle strength").
_CORE_BOUNDARY = {
    "in", "among", "for", "with", "during", "after", "before", "between",
    "from", "across", "within", "at", "on", "to", "versus", "vs",
    "compared", "using", "via", "into", "than", "that", "which", "whose",
    "amongst", "whilst", "when", "while", "where",
}


_NON_PLURAL_ADJECTIVES = {
    "younger", "older", "lower", "higher", "greater", "smaller", "larger",
    "better", "worse", "faster", "slower", "stronger", "weaker", "heavier",
    "lighter", "shorter", "longer", "wider", "narrower", "deeper",
}


def _variants(token: str) -> list[str]:
    """Cheap, predictable morphological variants (plurals, -ing, -ed)."""
    out = {token}
    if token.endswith("ies") and len(token) > 4:
        out.add(token[:-3] + "y")
    elif token.endswith("es") and len(token) > 4:
        stem = token[:-2]
        # "boxes" -> "box", but "muscles" -> "muscle" (never "muscl").
        out.add(stem if stem.endswith(("s", "x", "z", "ch", "sh")) else token[:-1])
    elif (
        token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
        and len(token) > 3
    ):
        out.add(token[:-1])
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        out.add(base)
        if not base.endswith("e"):
            out.add(base + "e")
    if token.endswith("ed") and len(token) > 5:
        base = token[:-2]
        out.add(base)
        if not base.endswith("e"):
            out.add(base + "e")
    # Irregulars the rule below would turn into "mans" / "childs".
    for singular, plural in (("man", "men"), ("woman", "women"), ("child", "children"),
                             ("foot", "feet"), ("tooth", "teeth"), ("mouse", "mice")):
        if token == singular:
            out.add(plural)
        elif token == plural:
            out.add(singular)
    if token.endswith("y") and len(token) > 3:
        out.add(token[:-1] + "ies")
    if not token.endswith(("s", "y")) and token not in _NON_PLURAL_ADJECTIVES:
        out.add(token + "s")
    # -es is only productive after a sibilant stem ("process" -> "processes");
    # "adult" must never become "adultses".
    if token.endswith(("x", "z", "ch", "sh", "ss")):
        out.add(token + "es")
    return sorted(o for o in out if o)


def variants(term: str) -> list[str]:
    """Public spelling variants for strict, field-specific matching."""
    return _variants(" ".join((term or "").lower().split())) if term else []


def _strip_years(text: str, spec: ScopeSpec, today: _dt.date) -> str:
    """Pull explicit date filters out of the intent and record them."""
    range_match = _IN_PAREN_RE.search(text) or _RANGE_RE.search(text)
    if range_match:
        spec.year_from, spec.year_to = int(range_match.group(1)), int(range_match.group(2))
        text = text.replace(range_match.group(0), " ")
    since = _SINCE_RE.search(text)
    if since:
        spec.year_from = int(since.group(1))
        text = text.replace(since.group(0), " ")
    until = _UNTIL_RE.search(text)
    if until:
        spec.year_to = int(until.group(1))
        text = text.replace(until.group(0), " ")
    last = _LAST_RE.search(text)
    if last:
        spec.year_from = today.year - int(last.group(1))
        text = text.replace(last.group(0), " ")
    if _DECADE_RE.search(text):
        spec.year_from = today.year - 10
        text = text.replace(_DECADE_RE.search(text).group(0), " ")
    return text


def _exclusion_span(text: str, start: int) -> int:
    """End of the noun phrase that begins at ``start``.

    An exclusion is a noun phrase, not the rest of the sentence: in "not about
    rodents in older adults" only "rodents" is excluded, and the population
    stays in the positive intent.
    """
    stop = _EXCLUDE_STOP.search(text[start:])
    return start + stop.start() if stop else len(text)


def _split_terms(piece: str) -> list[str]:
    """Split an excluded list ("mice, rats and rabbits") into its concepts."""
    terms: list[str] = []
    for part in _TERM_SPLIT.split(piece):
        cleaned = part.strip(" .,;:'\"-")
        if cleaned and cleaned.lower() not in STOPWORDS:
            terms.append(cleaned)
    return terms


def _extract_negations(text: str) -> tuple[str, list[str], list[str]]:
    """Split "not about X"-style exclusions out of the positive intent.

    Returns (remaining text, excluded concepts, notes). Three rules matter:

    * cues match on word boundaries and longest-first, so "amino acid" is not
      read as "no acid", and "but not cancer" does not leave a "but" facet
      that would become the head concept;
    * an excluded concept is a noun phrase and stops at the next preposition or
      clause boundary, so the population after "in" survives;
    * contrastive phrasing ("not just about muscles but about muscle strength")
      names the broader topic, so it is dropped rather than turned into a hard
      exclusion -- otherwise every paper that says "muscle" would be rejected.
    """
    spans: list[tuple[int, int]] = []
    excludes: list[str] = []
    notes: list[str] = []

    def free(position: int) -> bool:
        return not any(start <= position < end for start, end in spans)

    for match in _NO_RE.finditer(text):
        if not free(match.start()):
            continue
        prefix = text[: match.start()].rstrip().lower()
        # "no" only negates as a determiner at the start of a clause late
        # enough to be deliberate; "amino acid" is not "no acid".
        if prefix and not re.search(r"(?:^|[,;:]|\b(?:and|but|or|also|plus))$", prefix):
            continue
        if match.group("piece").split()[0].lower() in _NO_FALSE_FRIENDS:
            continue
        end = _exclusion_span(text, match.start("piece"))
        piece = text[match.start("piece"):end].strip(" .,;:")
        terms = _split_terms(piece)
        if terms:
            excludes.extend(terms)
            spans.append((match.start(), end))

    for match in _CUE_RE.finditer(text):
        if not free(match.start()):
            continue
        cue = match.group("cue").lower()
        spacer = re.match(r"\s+", text[match.end():])
        if not spacer:
            continue
        start = match.end() + spacer.end()
        kind = _CUE_KINDS.get(cue, "hard")
        if kind == "hard":
            # "but not just about muscles" is contrastive: the leftmost cue is
            # "but not", but the phrase means "not merely".
            softened = re.match(
                r"(?:just|only|merely)\s+(?:about\s+)?", text[start:], re.I
            )
            if softened:
                kind = "contrast"
                start += softened.end()
        end = _exclusion_span(text, start)
        piece = text[start:end].strip(" .,;:")
        if not piece:
            continue
        if kind == "contrast":
            connector = re.match(
                r"\s*,?\s*(?:but|rather|instead)\s+"
                r"(?:about\s+|on\s+|of\s+|regarding\s+|concerning\s+|in\s+)?",
                text[end:],
                re.I,
            )
            if connector:
                end += connector.end()
            written = " ".join(text[match.start():end].split())
            notes.append(
                f"read {written!r} as emphasis, not an exclusion "
                "(it contrasts a broader topic with a narrower one)"
            )
            spans.append((match.start(), end))
            continue
        terms = _split_terms(piece)
        if not terms:
            continue
        excludes.extend(terms)
        spans.append((match.start(), end))

    for start, end in sorted(spans, reverse=True):
        text = text[:start] + " " + text[end:]

    # Also honour an explicit "-term" shorthand.
    kept: list[str] = []
    for token in text.split():
        if token.startswith("-") and len(token) > 2 and token[1:].isalpha():
            excludes.append(token[1:])
        else:
            kept.append(token)
    text = " ".join(kept)

    cleaned: list[str] = []
    for term in excludes:
        normalised = " ".join(term.lower().split())
        if normalised and normalised not in cleaned:
            cleaned.append(normalised)
    return text, cleaned, notes


def _phrase_terms(phrase: str) -> list[str]:
    group = _SYNONYM_INDEX[phrase]
    terms: set[str] = set()
    for word in group:
        terms.update(_variants(word))
    return sorted(terms)


def exclusion_terms(term: str) -> list[str]:
    """Spellings of an excluded concept that should disqualify a paper.

    ``--exclude rodents`` has to catch a paper that says "mice": matching only
    the literal word is how a filter quietly stops filtering. Single words get
    their thesaurus group; phrases get per-word plurals ("animal study"), so
    that "muscle mass" does not expand into the whole hypertrophy vocabulary.
    """
    cleaned = " ".join(re.findall(r"[a-z0-9']+", (term or "").lower()))
    if not cleaned:
        return []
    out: set[str] = {cleaned}
    words = cleaned.split()
    if len(words) == 1:
        group = _SYNONYM_INDEX.get(cleaned)
        if group:
            out.update(group)
        out.update(_variants(cleaned))
    else:
        for index, word in enumerate(words):
            for variant in _variants(word):
                out.add(" ".join(words[:index] + [variant] + words[index + 1:]))
    return sorted(item for item in out if item)


def _core_facets(working: str) -> tuple[list[Facet], bool]:
    """The leading noun phrase of the intent, as facets, in order.

    Used to find the *head* concept: in "muscle strength in older adults" that
    is "strength", which is what separates an on-target paper from one about
    muscle mass.
    """
    tokens = _TOKEN_RE.findall(working.lower().replace("-", " "))
    lead: list[str] = []
    # The raw leading phrase, stopwords included: multi-word concepts live in
    # it ("hours per week" is frequency), and filtering first would split them.
    raw: list[str] = []
    listed = False
    started = False
    for token in tokens:
        if token in _CORE_BOUNDARY:
            if started:
                break
            continue
        if token in ("and", "or"):
            # "volume and frequency" has no head noun: it is a list of
            # variables, and pretending the last one is the head rejects
            # papers about the others.
            listed = listed or started
            continue
        if not started and (token in STOPWORDS or token.isdigit() or len(token) < 2):
            continue
        started = True
        raw.append(token)
        if token in STOPWORDS or token.isdigit() or len(token) < 2:
            continue
        lead.append(token)
    if not lead:
        return [], listed

    joined = " ".join(raw)
    facets: list[Facet] = []
    consumed: set[str] = set()
    for phrase in _PHRASES:
        if phrase in joined:
            words = set(phrase.split())
            if words & consumed:
                continue
            facets.append(
                Facet(name=_SYNONYM_INDEX[phrase][0], terms=_phrase_terms(phrase),
                      weight=0.6 if _SYNONYM_INDEX[phrase][0] in GENERIC_GROUPS else 1.2,
                      phrases=[p for p in _SYNONYM_INDEX[phrase] if " " in p],
                      canonical=list(_SYNONYM_INDEX[phrase]))
            )
            consumed |= words
    for token in lead:
        if token in consumed:
            continue
        facet = _facet_for(token)
        if facet is not None and facet.name not in {f.name for f in facets}:
            facets.append(facet)
    return facets, listed


def _facet_for(token: str) -> Facet | None:
    group = _SYNONYM_INDEX.get(token)
    if group:
        terms: set[str] = set()
        for word in group:
            terms.update(_variants(word))
        weight = 0.6 if group[0] in GENERIC_GROUPS else 1.0
        return Facet(name=group[0], terms=sorted(terms), weight=weight, canonical=list(group))
    weight = 0.5 if token in GENERIC else 1.0
    return Facet(name=token, terms=_variants(token), weight=weight)


# Above this, one abstract can no longer plausibly cover the intent and
# coverage becomes unattainable; the first-mentioned concepts win.
MAX_FACETS = 10


def parse_intent(
    text: str,
    *,
    min_scope: float = 0.7,
    require: Iterable[str] = (),
    exclude: Iterable[str] = (),
    require_in: str = "",
    today: _dt.date | None = None,
) -> ScopeSpec:
    """Parse a free-text intent into facets, date filters and exclusions."""
    today = today or _dt.date.today()
    intent = " ".join((text or "").split())
    if not intent:
        raise ValueError("intent must not be empty")

    spec = ScopeSpec(intent=intent, min_scope=min_scope)
    # A question written in Dutch, or in gym shorthand, is translated into the
    # vocabulary the indexes actually contain before anything is parsed.
    translated, language_notes = normalize_intent(intent)
    spec.extra_notes.extend(language_notes)
    working = _strip_years(translated, spec, today)
    age = _AGE_RE.search(working)
    if age:
        working = working[: age.start()] + " young adults " + working[age.end():]
        spec.extra_notes.append(
            f"read '{age.group(0)}' as the population 'young adults' "
            f"(age {int(age.group(1))})"
        )
    working, negated, negation_notes = _extract_negations(working)
    spec.extra_notes.extend(negation_notes)

    # Multi-word thesaurus phrases are real facets in their own right; note
    # them so the query builder and the screener can use the phrase form.
    # Keep the unblanked text for head detection: see below.
    positive = working
    spec.search_text = " ".join(positive.split())
    lowered = working.lower()
    phrase_facets: list[Facet] = []
    for phrase in _PHRASES:
        if phrase in lowered:
            group = _SYNONYM_INDEX[phrase]
            terms: set[str] = set()
            for word in group:
                terms.update(_variants(word))
            phrase_facets.append(
                Facet(name=group[0], terms=sorted(terms),
                      weight=0.6 if group[0] in GENERIC_GROUPS else 1.2,
                      phrases=[p for p in group if " " in p],
                      canonical=list(group))
            )
            lowered = lowered.replace(phrase, " ")
    working = lowered

    facets: list[Facet] = list(phrase_facets)
    seen_names = {f.name for f in facets}
    seen_sets = {tuple(f.terms) for f in facets}

    for token in _TOKEN_RE.findall(working.replace("-", " ")):
        if token in STOPWORDS or len(token) < 2 or token.isdigit():
            continue
        facet = _facet_for(token)
        if facet is None:
            continue
        if facet.name in seen_names or tuple(facet.terms) in seen_sets:
            continue
        seen_names.add(facet.name)
        seen_sets.add(tuple(facet.terms))
        facets.append(facet)

    # Explicitly required concepts are always facets, whatever the wording.
    field = (require_in or "").strip().lower()
    if field not in {"", "title", "abstract", "title_or_abstract"}:
        raise ValueError("require_in must be title, abstract or title_or_abstract")
    spec.require_field = field
    required_terms = [term.strip().lower() for term in require if term.strip()]
    for term in required_terms:
        facet = _facet_for(term)
        if facet and facet.name not in seen_names:
            facet.weight = max(facet.weight, 1.2)
            facets.append(facet)
            seen_names.add(facet.name)
    if field:
        # Literal terms, not thesaurus-expanded: "--require-in title strength"
        # means the title must say strength, not that it may say torque.
        spec.required = [(term, field) for term in required_terms]
        if not required_terms:
            spec.extra_notes.append("require_in has no effect without --require")

    if not facets:
        # Single generic word ("sarcopenia" is fine, "review" is not): fall back
        # to the raw tokens so the tool still does something sensible.
        fallback = [t for t in _TOKEN_RE.findall(intent.lower()) if len(t) > 2]
        facets = [Facet(name=t, terms=_variants(t), weight=1.0) for t in fallback[:6]]

    if len(facets) > MAX_FACETS:
        def position(facet: Facet) -> int:
            for term in [facet.name] + list(facet.canonical):
                index = positive.lower().find(term.lower())
                if index >= 0:
                    return index
            return len(positive)

        keep = sorted(
            sorted(range(len(facets)), key=lambda i: (-facets[i].weight, position(facets[i]), i))[
                :MAX_FACETS
            ]
        )
        dropped = [facets[i].name for i in range(len(facets)) if i not in set(keep)]
        facets = [facets[i] for i in keep]
        spec.extra_notes.append(
            f"kept {len(facets)} of {len(facets) + len(dropped)} concepts, dropped "
            f"{', '.join(dropped)} (pass --require to force one back)"
        )

    spec.facets = facets
    # Detect the head from the text *before* multi-word phrases were blanked:
    # otherwise "strength training in older adults" loses its leading phrase and
    # the head silently becomes "adult".
    core, listed = _core_facets(positive)
    # A head concept must also be a facet, otherwise coverage can never reach
    # the threshold no matter how relevant the paper is.
    known = {f.name for f in facets}
    core = [f for f in core if f.name in known] or core[:1]
    # A generic word ("effect", "review") must never be the head concept: it is
    # in every abstract and would defeat the specificity check.
    content = [f for f in core if f.weight >= 1.0]
    spec.core = core
    if listed:
        spec.head = None
        # No head noun to anchor on, so anchor on the variables the question is
        # about -- the leading phrase, not the goal after "for". In "sets and
        # frequency for hypertrophy" the anchors are sets and frequency; the
        # outcome (hypertrophy) is what every tangential training study
        # measures, so it is exactly the wrong thing to anchor on.
        pool = [f for f in core if f.name not in POPULATION_GROUPS]
        if len(pool) < 2:
            pool = [f for f in facets if f.name not in POPULATION_GROUPS]
        top = max((f.weight for f in pool), default=0.0)
        spec.anchors = [f.name for f in pool if f.weight >= top and top > 0][:4]
        spec.extra_notes.append(
            "the intent lists several concepts rather than one head noun; "
            f"anchors: {', '.join(spec.anchors) or '(none)'}"
        )
    else:
        if content:
            spec.head = content[-1]
        else:
            non_generic = [
                f for f in facets
                if f.weight >= 1.0 and f.name not in POPULATION_GROUPS
            ]
            spec.head = non_generic[0] if non_generic else (core[-1] if core else None)
    spec.exclude = sorted({e for e in negated if e} | {x.strip().lower() for x in exclude if x.strip()})
    if spec.year_from and spec.year_to and spec.year_from > spec.year_to:
        spec.year_from, spec.year_to = spec.year_to, spec.year_from
    if spec.year_from and spec.year_from < 1800:
        spec.year_from = None
    if spec.year_to and spec.year_to < 1800:
        spec.year_to = None
    return spec
