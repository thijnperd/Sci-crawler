"""Normalise everyday wording into the English the literature is written in.

A model asking PaperHound a question rarely writes a keyword query. It writes
the user's own sentence, often in the user's own language:

    "optimale rep en set count en uren en schema en oefeningen in de gym
     voor man 18 jaar oud voor spiergroei"

Searching five bibliographic APIs for those literal tokens finds nothing, and
screening against fifteen Dutch "facets" rejects everything. So the intent is
translated word-by-word through a small technical lexicon before it is parsed:
enough for training, physiology and study-design vocabulary, and deliberately
not a general translator.

Two rules keep this from mangling an English intent:

* the Dutch map is only applied when the text is *detected* as Dutch (at least
  two unambiguous markers), because words like "of", "in" and "man" collide;
* the fitness shorthand map ("rep", "set count") is language-neutral and always
  applied, since it is jargon rather than another language.

Every change is reported as a note, so `paperhound explain` shows exactly what
was rewritten instead of silently answering a different question.
"""

from __future__ import annotations

import re

# Words that only occur in Dutch intents. Two of them are enough to justify
# translating the whole sentence. Deliberately excludes words shared with
# English ("training", "set", "gym", "rep", "in", "of").
_DUTCH_MARKERS = {
    "de", "het", "een", "van", "voor", "met", "zonder", "niet", "geen", "maar",
    "ook", "jaar", "oud", "uren", "uur", "weken", "spiergroei", "spieren",
    "kracht", "oefeningen", "schema", "sportschool", "mannen", "vrouwen",
    "tijdens", "tussen", "tot", "bij", "hoeveel", "hoelang", "vaak", "optimale",
    "optimaal", "beste", "richtlijnen", "ouderen", "kinderen", "voeding",
    "eiwit", "eiwitten", "herstel", "blessure", "blessures", "afvallen",
    "conditie", "gezondheid", "prestatie", "herhalingen", "herhaling", "rust",
    "trainingsschema", "spieropbouw", "krachttraining", "vetverlies",
}

# Dutch -> English. Longest key wins, so "set count" is not eaten by "set".
_DUTCH_MAP: dict[str, str] = {
    # training variables
    "spiergroei": "muscle hypertrophy",
    "spieropbouw": "muscle hypertrophy",
    "krachttraining": "resistance training",
    "trainingsschema": "training program",
    "oefeningen": "exercises",
    "oefening": "exercise",
    "schema": "training program",
    "herhalingen": "repetitions",
    "herhaling": "repetition",
    "set count": "number of sets",
    "rep count": "repetitions",
    "rust": "rest interval",
    "frequentie": "frequency",
    "intensiteit": "intensity",
    "hoeveel": "how many",
    "hoelang": "how long",
    "vaak": "often",
    "uren": "hours per week",
    "uur": "hour",
    "minuten": "minutes",
    "weken": "weeks",
    "week": "week",
    "per": "per",
    # populations
    "mannen": "men",
    "man": "men",
    "vrouwen": "women",
    "vrouw": "women",
    "ouderen": "older adults",
    "kinderen": "children",
    "adolescenten": "adolescents",
    "beginners": "beginners",
    "gevorderden": "trained",
    "jaar": "years",
    "oud": "old",
    "leeftijd": "age",
    # physiology
    "spieren": "muscles",
    "spier": "muscle",
    "kracht": "strength",
    "uithoudingsvermogen": "endurance",
    "conditie": "fitness",
    "gezondheid": "health",
    "prestatie": "performance",
    "herstel": "recovery",
    "blessure": "injury",
    "blessures": "injuries",
    "vetverlies": "fat loss",
    "afvallen": "weight loss",
    "voeding": "nutrition",
    "eiwitten": "protein",
    "eiwit": "protein",
    # study words
    "onderzoek": "study",
    "studie": "study",
    "meta-analyse": "meta-analysis",
    "optimale": "optimal",
    "optimaal": "optimal",
    "beste": "best",
    "ideaal": "ideal",
    "aanbevolen": "recommended",
    "richtlijnen": "guidelines",
    "effect": "effect",
    # function words
    "tijdens": "during",
    "tussen": "between",
    "zonder": "without",
    "met": "with",
    "voor": "for",
    "van": "of",
    "en": "and",
    "of": "or",
    "het": "",
    "de": "",
    "een": "",
    "in": "in",
    "op": "on",
    "bij": "in",
    "tot": "to",
    "vanaf": "since",
    "naar": "to",
    "over": "about",
    "ook": "also",
    "niet": "not",
    "geen": "no",
    "maar": "but",
    "alleen": "only",
    "zowel": "both",
}

# Jargon that is English already, or English-ish in every language.
_SHORTHAND: dict[str, str] = {
    "set count": "number of sets",
    "rep count": "repetitions",
    "rep range": "repetition range",
    "reps": "repetitions",
    "rep": "repetition",
}

_MARKER_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(sorted(_DUTCH_MARKERS, key=len, reverse=True)) + r")(?![a-z0-9])",
    re.I,
)


def _apply(text: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Replace whole-word keys in one pass, longest first."""
    if not mapping:
        return text, []
    pattern = re.compile(
        r"(?<![a-z0-9])(" + "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
        + r")(?![a-z0-9])",
        re.I,
    )
    changes: list[str] = []

    def swap(match: re.Match[str]) -> str:
        key = match.group(0).lower()
        if key not in changes:
            changes.append(key)
        return mapping[key]

    return pattern.sub(swap, text), changes


def _is_dutch(text: str) -> bool:
    return len({m.group(0).lower() for m in _MARKER_RE.finditer(text)}) >= 2


def normalize_intent(text: str) -> tuple[str, list[str]]:
    """Return (English-ish intent, notes describing what was rewritten)."""
    notes: list[str] = []
    original = " ".join((text or "").split())
    working = original
    if _is_dutch(original):
        working, changed = _apply(original, _DUTCH_MAP)
        if changed:
            notes.append(
                "translated Dutch terms before parsing: "
                + ", ".join(changed[:12])
                + (" ..." if len(changed) > 12 else "")
            )
    working, shorthand = _apply(working, _SHORTHAND)
    if shorthand:
        notes.append("expanded training shorthand: " + ", ".join(sorted(set(shorthand))))
    working = " ".join(working.split())
    if working != original:
        notes.append(f"parsed as: {working}")
    return working, notes
