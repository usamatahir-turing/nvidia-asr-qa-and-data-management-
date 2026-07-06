"""Language-specific filler word canonical forms and conservative variant detection."""

from __future__ import annotations

import re
from functools import lru_cache

# (canonical_form, [non-canonical variants])
# Lexical fillers with high false-positive risk are omitted intentionally.

FILLER_PAIRS_BY_LANG: dict[str, list[tuple[str, list[str]]]] = {
    "EN": [
        ("hm", ["mm", "hmm", "mmm", "mmh", "hmmm", "hmhmm", "hmmhmm"]),
        ("um", ["umm", "uhm", "umh", "ummh"]),
        ("uh", ["uhh", "uuh"]),
        ("ah", ["ahh", "aah", "ahhh"]),
        ("er", ["err", "erm"]),
        ("oh", ["ohh"]),
        ("eh", ["ehh"]),
        ("ooh", ["oooh", "aw"]),
        ("mn", ["mmn", "mnn"]),
        ("mm-hmm", ["mhm", "m-hm", "mmhm", "mhmm", "mm-hm", "mm hm"]),
        ("uh-huh", ["uhhuh", "uh-hum", "uhhum", "unh-hun", "uh huh"]),
        ("uh-uh", ["unh-uh", "uh uh"]),
        ("nuh-uh", ["nuhuh", "nuh uh"]),
        ("mm-mm", ["mm mm", "mhm-mm"]),
        ("hm-mm", ["hmm-mm", "hm mm"]),
    ],
    "AR": [
        ("ام", ["اممم", "امم"]),
        ("آه", ["آآه", "آاه"]),
        ("إيه", ["إيييه"]),
        ("أا", ["أأا"]),
        ("مم", ["ممم"]),
        ("ها", ["هاا"]),
    ],
    "GR": [
        ("äh", ["ääh", "äääh"]),
        ("ähm", ["ähmm", "ääähm"]),
        ("hm", ["hmm", "hmmm"]),
        ("mm", ["mmm"]),
        ("ah", ["aah", "ahh"]),
        ("ach", ["aach", "achh"]),
        ("oh", ["ohh"]),
        ("uff", ["ufff", "uuff"]),
        ("mhm", ["mm-hmm", "mhmm"]),
        ("mh-mh", ["mm-mm", "mh mh"]),
        ("aha", ["ah-ha", "ahaa"]),
        ("hä", ["hää"]),
        ("puh", ["puuh", "puhh"]),
        ("boah", ["boa", "boaaah"]),
        ("aua", ["auaa"]),
        ("autsch", ["autschh"]),
    ],
    "ES": [
        ("eh", ["eeh", "eeeh"]),
        ("em", ["emm", "emmm"]),
        ("mmm", ["mmmm"]),
        ("ajá", ["ajaa"]),
        ("ay", ["ayy"]),
    ],
    "FR": [
        ("euh", ["euuh", "euuuuh", "heu", "heuu"]),
        ("hum", ["humm", "hmmm"]),
        ("bah", ["baah"]),
        ("ah", ["aaah"]),
        ("oh", ["oooh"]),
        ("hein", ["heiiin"]),
    ],
    "IT": [
        ("eh", ["ehh", "ehhh"]),
        ("uhm", ["uhmm"]),
        ("mm", ["mmm"]),
        ("ah", ["ahh"]),
    ],
    "JA": [
        ("え", ["えー", "ええー"]),
        ("あ", ["あー"]),
        ("あの", ["あのー", "あのお"]),
        ("えっと", ["えっとー", "えーっと"]),
        ("うーん", ["うーーん"]),
        ("ふーん", ["ふーーん"]),
        ("はい", ["はいー", "はーい", "はぃ"]),
        ("へえ", ["へー", "へぇ"]),
        ("おお", ["おー"]),
    ],
    "KO": [
        ("어", ["어어", "어어어"]),
        ("음", ["으음", "음음"]),
        ("아", ["아아"]),
        ("에", ["에에"]),
        ("아하", ["아하아"]),
        ("어휴", ["어휴우"]),
        ("아이고", ["아이고오"]),
        ("흠", ["흐음"]),
    ],
    "PT": [
        ("é", ["ééé"]),
        ("eh", ["ehh"]),
        ("ah", ["ahhh"]),
        ("hum", ["humm", "hummm"]),
        ("mmm", ["mmmm"]),
        ("uhm", ["uhmm"]),
        ("aham", ["ahamm"]),
        ("uhum", ["uhumm"]),
        ("hum-hum", ["humhum", "hum hum"]),
        ("nossa", ["nossaa"]),
        ("ai", ["aii"]),
    ],
    "RU": [
        ("а", ["аа", "а-а-а"]),
        ("э", ["ээ", "э-э"]),
        ("эм", ["ээм", "эмм"]),
        ("м", ["мм"]),
        ("м-да", ["мда", "мдя"]),
        ("ах", ["ахх", "а-а-ах"]),
        ("у", ["уу", "у-у"]),
        ("ага", ["ага-а", "агаа"]),
        ("угу", ["у-г-у", "угуу"]),
        ("мгм", ["мм-гм", "мхм"]),
        ("м-м", ["м м"]),
        ("вау", ["уау"]),
        ("ей-богу", ["ейбогу", "ей богу"]),
    ],
}

_TASK_LANG_RE = re.compile(r"^NV-([A-Z]{2})-", re.IGNORECASE)
_EDGE_PUNCT_RE = re.compile(r"^[^\w\u0600-\u06FF\u3040-\u30FF\u4E00-\u9FFF-]+|[^\w\u0600-\u06FF\u3040-\u30FF\u4E00-\u9FFF-]+$")


def language_from_task_id(task_id: str) -> str | None:
    match = _TASK_LANG_RE.match(task_id)
    if not match:
        return None
    lang = match.group(1).upper()
    if lang == "SP":
        return "ES"
    if lang in FILLER_PAIRS_BY_LANG:
        return lang
    return None


def _normalize_filler_key(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text.isascii():
        text = re.sub(r"\s+", " ", text.lower())
    return text


def _strip_edge_punct(token: str) -> str:
    return _EDGE_PUNCT_RE.sub("", token)


@lru_cache(maxsize=16)
def _variant_lookup_for_lang(lang: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, variants in FILLER_PAIRS_BY_LANG.get(lang, []):
        canonical_key = _normalize_filler_key(canonical)
        for variant in variants:
            variant_key = _normalize_filler_key(variant)
            if not variant_key or variant_key == canonical_key:
                continue
            lookup[variant_key] = canonical
    return lookup


def _phrase_boundary_pattern(variant_key: str) -> re.Pattern[str]:
    """Match spaced filler variants only as standalone tokens, not across word edges."""
    flags = re.IGNORECASE if variant_key.isascii() else 0
    return re.compile(
        r"(?<!\w)" + re.escape(variant_key) + r"(?!\w)",
        flags,
    )


@lru_cache(maxsize=16)
def _phrase_patterns_for_lang(lang: str) -> list[tuple[re.Pattern[str], str]]:
    patterns: list[tuple[re.Pattern[str], str]] = []
    lookup = _variant_lookup_for_lang(lang)
    for variant_key, canonical in sorted(lookup.items(), key=lambda item: -len(item[0])):
        if " " not in variant_key:
            continue
        patterns.append((_phrase_boundary_pattern(variant_key), canonical))
    return patterns


def find_noncanonical_fillers(
    words: str,
    lang: str | None,
    *,
    is_inside_bracket_span,
) -> list[tuple[str, str]]:
    """Return (detected_variant, canonical_form) pairs found in *words*."""
    if not lang:
        return []

    lookup = _variant_lookup_for_lang(lang)
    if not lookup:
        return []

    findings: list[tuple[str, str]] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_match(start: int, end: int, detected: str, canonical: str) -> None:
        if is_inside_bracket_span(start, end):
            return
        span = (start, end)
        if span in seen_spans:
            return
        seen_spans.add(span)
        findings.append((detected, canonical))

    for pattern, canonical in _phrase_patterns_for_lang(lang):
        for match in pattern.finditer(words):
            add_match(match.start(), match.end(), match.group(), canonical)

    for match in re.finditer(r"\S+", words):
        raw = match.group()
        start, end = match.span()
        if (start, end) in seen_spans:
            continue
        core = _strip_edge_punct(raw)
        key = _normalize_filler_key(core)
        if key in lookup:
            add_match(start, end, raw, lookup[key])

    return findings
