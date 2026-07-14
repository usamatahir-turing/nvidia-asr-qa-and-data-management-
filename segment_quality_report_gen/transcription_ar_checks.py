#!/usr/bin/env python3
"""Transcription annotation review checks for seglst JSON files.

Detects:
  1. Numeric words (digit tokens and grouped/decimal forms that should be spoken-form).
  2. Unknown NSV tokens (bracket annotations not in the allowed list).
  3. Compact written symbols (@ . / : -) in URLs, emails, paths, etc.
  4. Non-canonical filler words (language-specific, from task folder code).
  5. Non-canonical abbreviations, alphanumeric compact forms, and ordinals.

Used by ``generate_report_v2.py`` to append a *Transcription Words Report* section.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from filler_word_rules import find_noncanonical_fillers, language_from_task_id

ALLOWED_NSVS = frozenset(
    {
        "breath",
        "inhale",
        "exhale",
        "sigh",
        "sniff",
        "gasp",
        "blow",
        "laugh",
        "chuckle",
        "giggle",
        "snort",
        "scoff",
        "grunt",
        "groan",
        "cry",
        "hum-tune",
        "whoop",
        "whistle",
        "tongue-click",
        "tsk",
        "lip-smack",
        "teeth-suck",
        "lip-trill",
        "shush",
        "swallow",
        "clear-throat",
        "cough",
        "sneeze",
        "yawn",
        "hiccup",
        "unintelligible",
        "other-noise",
    }
)

NUMERIC_WORD_RE = re.compile(
    r"^(?:"
    r"\d+"
    r"|"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|"
    r"\d+[.,]\d+"
    r")$"
)
_TOKEN_BODY_RE = re.compile(r"^[A-Za-z]+(?:-\s*[A-Za-z]+|\s+[A-Za-z]+)*$")
_WORD_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")

NUMBERS_RECOMMENDATION = "Listen to the audio and change to the spoken-form"
UNKNOWN_NSV_RECOMMENDATION = (
    "Non-canonical NSV found. Fix the spelling or use the common canonical form."
)
SYMBOLS_RECOMMENDATION = "Listen to the audio to ensure it's written in spoken form"
SPOKEN_FORM_RECOMMENDATION = "Listen to the audio and change to the spoken-form"
STUTTER_RECOMMENDATION = "Add space if stutter"

_DOT_ACRONYM_RE = re.compile(r"^[A-Za-z](?:\.[A-Za-z])+\.?$")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
_DIGIT_THEN_LETTERS_RE = re.compile(r"^\d+[A-Za-z]+$")
_LETTERS_THEN_DIGIT_RE = re.compile(r"^[A-Za-z]{2,}\d+$")
_LETTER_DIGIT_LETTER_RE = re.compile(r"^[A-Za-z]\d+[A-Za-z]$")
_PRO_SPAN_RE = re.compile(r"\{PRO:\s*[^}]+\}")


def filler_recommendation(canonical: str) -> str:
    return f"Use canonical filler form: {canonical}"


def abbreviation_recommendation(canonical: str | None, *, is_stutter: bool = False) -> str:
    if is_stutter:
        return STUTTER_RECOMMENDATION
    if canonical:
        return f"Listen and use the canonicalspoken-form of the word. Potential correct canonical acronym form: {canonical}"
    return SPOKEN_FORM_RECOMMENDATION

_COMPACT_SYMBOL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("url", re.compile(r"\bhttps?://\S+", re.IGNORECASE)),
    ("www", re.compile(r"\bwww\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)),
    (
        "domain_path",
        re.compile(r"\b[A-Za-z0-9][A-Za-z0-9-]*\.[A-Za-z]{2,}/\S+"),
    ),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("windows_path", re.compile(r"\b[A-Za-z]:\\(?:[^\\\s]+\\?)+")),
    (
        "unix_path",
        re.compile(r"\b(?:/[A-Za-z0-9._-]+){2,}(?:/[A-Za-z0-9._-]+)*\b"),
    ),
    (
        "file_path",
        re.compile(
            r"\b[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*){2,}\b"
        ),
    ),
    ("handle", re.compile(r"(?<!\w)@[A-Za-z_][A-Za-z0-9_]{1,}\b")),
    ("host_port", re.compile(r"\b[A-Za-z][A-Za-z0-9.-]*:\d{2,5}\b")),
    (
        "domain",
        re.compile(r"\b[A-Za-z0-9][A-Za-z0-9-]*\.[A-Za-z]{2,}\b"),
    ),
    ("slug", re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,}\b")),
)


@dataclass
class NumericFinding:
    segment_index: int
    start: float
    end: float
    detected: str
    words_preview: str


@dataclass
class UnknownNsvFinding:
    segment_index: int
    start: float
    end: float
    detected: str
    words_preview: str


@dataclass
class SymbolFinding:
    segment_index: int
    start: float
    end: float
    detected: str
    words_preview: str


@dataclass
class FillerFinding:
    segment_index: int
    start: float
    end: float
    detected: str
    canonical: str
    words_preview: str


@dataclass
class AbbreviationFinding:
    segment_index: int
    start: float
    end: float
    detected: str
    canonical: str | None
    words_preview: str
    is_stutter: bool = False


@dataclass
class SpeakerTranscriptionReport:
    speaker_id: str
    seglst_path: Path
    numeric_findings: list[NumericFinding] = field(default_factory=list)
    unknown_nsv_findings: list[UnknownNsvFinding] = field(default_factory=list)
    symbol_findings: list[SymbolFinding] = field(default_factory=list)
    filler_findings: list[FillerFinding] = field(default_factory=list)
    abbreviation_findings: list[AbbreviationFinding] = field(default_factory=list)


@dataclass
class TaskTranscriptionReport:
    task_id: str
    speakers: list[SpeakerTranscriptionReport] = field(default_factory=list)

    @property
    def numeric_count(self) -> int:
        return sum(len(s.numeric_findings) for s in self.speakers)

    @property
    def unknown_nsv_count(self) -> int:
        return sum(len(s.unknown_nsv_findings) for s in self.speakers)

    @property
    def symbol_count(self) -> int:
        return sum(len(s.symbol_findings) for s in self.speakers)

    @property
    def filler_count(self) -> int:
        return sum(len(s.filler_findings) for s in self.speakers)

    @property
    def abbreviation_count(self) -> int:
        return sum(len(s.abbreviation_findings) for s in self.speakers)


def _seglst_suffix(variant: str) -> str:
    return f"_{variant}.seglst.json"


def _pair_regex(variant: str) -> re.Pattern[str]:
    return re.compile(rf"^(.+)_{re.escape(variant)}\.seglst\.json$", re.IGNORECASE)


def _parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _escape_md_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _find_numeric_raw_token(words: str, core: str) -> str | None:
    for raw in words.split():
        if not raw:
            continue
        if _WORD_EDGE_PUNCT_RE.sub("", raw) == core:
            return raw
    return None


def _truncate_keeping_range(text: str, range_start: int, range_end: int, max_len: int) -> str:
    if len(text) <= max_len:
        return text

    highlight_len = range_end - range_start
    if highlight_len >= max_len:
        return text[range_start:range_end]

    extra = max_len - highlight_len
    before = extra // 2
    after = extra - before

    start = max(0, range_start - before)
    end = min(len(text), range_end + after)

    while end - start < max_len:
        expanded = False
        if start > 0:
            start -= 1
            expanded = True
        if end - start >= max_len:
            break
        if end < len(text):
            end += 1
            expanded = True
        if end - start >= max_len or not expanded:
            break

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _format_words_with_highlight(
    words: str,
    issue: str,
    *,
    max_len: int = 72,
) -> str:
    """Return segment text with the issue wrapped in markdown bold, kept visible."""
    if not words.strip():
        return _escape_md_table_cell(f"**{issue}**")

    highlight = issue
    start = words.find(highlight)
    if start == -1:
        raw_numeric = _find_numeric_raw_token(words, issue)
        if raw_numeric is not None:
            highlight = raw_numeric
            start = words.find(highlight)

    if start == -1:
        display = f"**{issue}** — {_words_preview(words, max_len=max(24, max_len - len(issue) - 5))}"
        return _escape_md_table_cell(display)

    end = start + len(highlight)
    highlighted = f"{words[:start]}**{highlight}**{words[end:]}"
    bold_start = start
    bold_end = start + 2 + len(highlight) + 2
    display = _truncate_keeping_range(highlighted, bold_start, bold_end, max_len)
    return _escape_md_table_cell(display)


def _words_preview(words: str, max_len: int = 48) -> str:
    text = " ".join(str(words).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def normalize_nsv_content(content: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", content).strip()
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.lower()


def is_numeric_word(token: str) -> bool:
    """True for digit tokens including grouped thousands and decimals (not ``3D``)."""
    core = _WORD_EDGE_PUNCT_RE.sub("", token)
    if not core:
        return False
    if not core.replace(",", "").replace(".", "").isdigit():
        return False
    return NUMERIC_WORD_RE.fullmatch(core) is not None


def find_numeric_tokens(words: str) -> list[str]:
    """Return distinct numeric words found in a segment transcript."""
    found: list[str] = []
    seen: set[str] = set()
    for raw in words.split():
        if not raw:
            continue
        core = _WORD_EDGE_PUNCT_RE.sub("", raw)
        if is_numeric_word(raw) and core not in seen:
            seen.add(core)
            found.append(core)
    return found


def _looks_like_token_body(text: str) -> bool:
    return bool(_TOKEN_BODY_RE.fullmatch(text.strip()))


def iter_nsv_candidates(words: str) -> Iterator[tuple[str, str]]:
    """Yield ``(raw_span, inner_content)`` for NSV-like bracket tokens."""
    i = 0
    length = len(words)

    while i < length:
        ch = words[i]
        if ch == "[":
            close = words.find("]", i + 1)
            if close != -1:
                inner = words[i + 1 : close]
                yield words[i : close + 1], inner
                i = close + 1
                continue

            j = i + 1
            while j < length and words[j] not in " \t":
                j += 1
            inner = words[i + 1 : j]
            if inner.strip() and _looks_like_token_body(inner):
                yield words[i:j], inner
            i = max(j, i + 1)
            continue

        if ch.isascii() and ch.isalpha():
            close = words.find("]", i + 1)
            if close == -1:
                i += 1
                continue
            inner = words[i:close]
            if _looks_like_token_body(inner):
                yield words[i : close + 1], inner
                i = close + 1
                continue

        i += 1


def find_unknown_nsv_tokens(words: str) -> list[str]:
    """Return distinct unknown NSV spans in a segment transcript."""
    found: list[str] = []
    seen: set[str] = set()
    for raw_span, inner in iter_nsv_candidates(words):
        normalized = normalize_nsv_content(inner)
        if not normalized:
            continue
        if normalized in ALLOWED_NSVS:
            continue
        key = raw_span.lower()
        if key not in seen:
            seen.add(key)
            found.append(raw_span)
    return found


def _bracket_spans(words: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for raw_span, _ in iter_nsv_candidates(words):
        idx = words.find(raw_span)
        if idx != -1:
            spans.append((idx, idx + len(raw_span)))
    return spans


def _is_inside_bracket_span(words: str, start: int, end: int) -> bool:
    for span_start, span_end in _bracket_spans(words):
        if start >= span_start and end <= span_end:
            return True
    return False


def _pro_spans(words: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _PRO_SPAN_RE.finditer(words)]


def _is_inside_pro_span(words: str, start: int, end: int) -> bool:
    for span_start, span_end in _pro_spans(words):
        if start >= span_start and end <= span_end:
            return True
    return False


def _is_letter_hyphen_acronym(text: str) -> bool:
    if _stutter_hyphen_spaced_form(text) is not None:
        return False
    parts = [part for part in text.split("-") if part]
    return len(parts) >= 2 and all(len(part) == 1 and part.isalpha() for part in parts)


def _is_natural_language_hyphen_compound(text: str) -> bool:
    """True for spoken compounds like ``peer-to-peer``, not technical slugs."""
    parts = [part for part in text.split("-") if part]
    if len(parts) < 3:
        return False
    return all(part.isalpha() and len(part) >= 2 for part in parts)


def _skip_slug_compact_symbol(text: str) -> bool:
    return (
        _is_letter_hyphen_acronym(text)
        or _stutter_hyphen_spaced_form(text) is not None
        or _is_natural_language_hyphen_compound(text)
    )


def _stutter_hyphen_spaced_form(token: str, *, raw: str | None = None) -> str | None:
    """Return spaced stutter form (``I-I-`` → ``I- I-``) for ASCII letter repeats."""
    parts = [part for part in token.split("-") if part]
    if len(parts) < 2:
        return None
    if not all(
        len(part) == 1 and part.isascii() and part.isalpha() for part in parts
    ):
        return None
    if len({part.lower() for part in parts}) != 1:
        return None
    spaced = "- ".join(parts[:-1]) + f"- {parts[-1]}"
    source = raw if raw is not None else token
    if source.endswith("-"):
        spaced += "-"
    return spaced


def _canonical_dot_acronym(token: str) -> str:
    return token.replace(".", "").upper()


def _is_dot_acronym(token: str) -> bool:
    if not _DOT_ACRONYM_RE.fullmatch(token):
        return False
    return len(_canonical_dot_acronym(token)) >= 2


def _is_alphanumeric_compact(token: str) -> bool:
    """Compact digit/letter forms like ``5G``, ``3D``, ``H2O`` — not ``s0``."""
    return bool(
        _DIGIT_THEN_LETTERS_RE.fullmatch(token)
        or _LETTERS_THEN_DIGIT_RE.fullmatch(token)
        or _LETTER_DIGIT_LETTER_RE.fullmatch(token)
    )


def _overlaps_compact_symbol_span(words: str, start: int, end: int) -> bool:
    for label, pattern in _COMPACT_SYMBOL_PATTERNS:
        for match in pattern.finditer(words):
            span_start, span_end = match.span()
            text = match.group()
            if label == "slug" and _skip_slug_compact_symbol(text):
                continue
            if _is_inside_bracket_span(words, span_start, span_end):
                continue
            if label == "ip" and not _valid_ip_address(text):
                continue
            if label in {"file_path", "domain_path"} and _is_common_slash_phrase(text):
                continue
            if label == "handle" and "@" in text[1:] and "." in text:
                continue
            if not (end <= span_start or start >= span_end):
                return True
    return False


def find_noncanonical_abbreviations(
    words: str,
    language: str | None = None,
) -> list[tuple[str, str | None, bool]]:
    """Return ``(detected, canonical, is_stutter)`` abbreviation / compact-form issues.

    * ``canonical`` set — non-canonical dot acronym spelling (``F.B.I.`` → ``FBI``).
    * ``canonical`` set, ``is_stutter`` — hyphen stutter (``I-I-`` → ``I- I-``); skipped for JA.
    * ``canonical`` None — alphanumeric compact (``5G``) or ordinal (``1st``).
    """
    findings: list[tuple[str, str | None, bool]] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_finding(
        start: int,
        end: int,
        detected: str,
        canonical: str | None,
        *,
        is_stutter: bool = False,
    ) -> None:
        if (start, end) in seen_spans:
            return
        if _is_inside_bracket_span(words, start, end):
            return
        if _is_inside_pro_span(words, start, end):
            return
        if _overlaps_compact_symbol_span(words, start, end):
            return
        seen_spans.add((start, end))
        findings.append((detected, canonical, is_stutter))

    for match in re.finditer(r"\S+", words):
        raw = match.group()
        start, end = match.span()
        if (start, end) in seen_spans:
            continue
        core = _WORD_EDGE_PUNCT_RE.sub("", raw)
        if not core:
            continue

        stutter_form = None
        if language != "JA":
            stutter_form = _stutter_hyphen_spaced_form(core, raw=raw)
        if stutter_form is not None:
            add_finding(start, end, raw, stutter_form, is_stutter=True)
            continue

        canonical: str | None = None
        if _is_dot_acronym(core):
            canonical = _canonical_dot_acronym(core)
        elif _ORDINAL_RE.fullmatch(core):
            canonical = None
        elif _is_alphanumeric_compact(core):
            canonical = None
        else:
            continue

        add_finding(start, end, raw, canonical)

    return findings


def _valid_ip_address(text: str) -> bool:
    parts = text.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        value = int(part)
        if value < 0 or value > 255:
            return False
    return True


def _is_common_slash_phrase(text: str) -> bool:
    if text.count("/") != 1:
        return False
    left, right = text.split("/", 1)
    common = {
        "and",
        "or",
        "he",
        "she",
        "his",
        "her",
        "yes",
        "no",
        "pro",
        "con",
    }
    return left.lower() in common and right.lower() in common


def _merge_non_overlapping_spans(
    spans: list[tuple[int, int, str]],
) -> list[str]:
    ordered = sorted(spans, key=lambda item: (-(item[1] - item[0]), item[0]))
    selected: list[tuple[int, int, str]] = []
    for start, end, text in ordered:
        if any(not (end <= sel_start or start >= sel_end) for sel_start, sel_end, _ in selected):
            continue
        selected.append((start, end, text))
    return [text for _, _, text in sorted(selected, key=lambda item: item[0])]


def find_compact_symbol_spans(words: str) -> list[str]:
    """Return compact URL/email/path-style spans that should be spoken-form."""
    matches: list[tuple[int, int, str]] = []

    for label, pattern in _COMPACT_SYMBOL_PATTERNS:
        for match in pattern.finditer(words):
            start, end = match.span()
            text = match.group()
            if _is_inside_bracket_span(words, start, end):
                continue
            if label == "ip" and not _valid_ip_address(text):
                continue
            if label in {"file_path", "domain_path"} and _is_common_slash_phrase(text):
                continue
            if label == "handle" and "@" in text[1:] and "." in text:
                continue
            if label == "slug" and _skip_slug_compact_symbol(text):
                continue
            matches.append((start, end, text))

    return _merge_non_overlapping_spans(matches)


def load_seglst(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return sorted(data, key=lambda item: _parse_time(item["start_time"]))


def discover_seglst_files(task_dir: Path, variant: str) -> list[tuple[str, Path]]:
    suffix = _seglst_suffix(variant)
    pair_re = _pair_regex(variant)
    pairs: list[tuple[str, Path]] = []
    for seglst_path in sorted(task_dir.glob(f"*{suffix}")):
        match = pair_re.match(seglst_path.name)
        if match:
            pairs.append((match.group(1), seglst_path))
    return pairs


def analyze_speaker_transcription(
    speaker_id: str,
    seglst_path: Path,
    language: str | None,
) -> SpeakerTranscriptionReport:
    report = SpeakerTranscriptionReport(
        speaker_id=speaker_id,
        seglst_path=seglst_path,
    )
    for idx, segment in enumerate(load_seglst(seglst_path)):
        words = str(segment.get("words", ""))
        start = _parse_time(segment["start_time"])
        end = _parse_time(segment["end_time"])
        for numeric in find_numeric_tokens(words):
            report.numeric_findings.append(
                NumericFinding(
                    segment_index=idx,
                    start=start,
                    end=end,
                    detected=numeric,
                    words_preview=_format_words_with_highlight(words, numeric),
                )
            )

        for unknown in find_unknown_nsv_tokens(words):
            report.unknown_nsv_findings.append(
                UnknownNsvFinding(
                    segment_index=idx,
                    start=start,
                    end=end,
                    detected=unknown,
                    words_preview=_format_words_with_highlight(words, unknown),
                )
            )

        for symbol in find_compact_symbol_spans(words):
            report.symbol_findings.append(
                SymbolFinding(
                    segment_index=idx,
                    start=start,
                    end=end,
                    detected=symbol,
                    words_preview=_format_words_with_highlight(words, symbol),
                )
            )

        for detected, canonical in find_noncanonical_fillers(
            words,
            language,
            is_inside_bracket_span=lambda s, e, w=words: _is_inside_bracket_span(w, s, e),
        ):
            report.filler_findings.append(
                FillerFinding(
                    segment_index=idx,
                    start=start,
                    end=end,
                    detected=detected,
                    canonical=canonical,
                    words_preview=_format_words_with_highlight(words, detected),
                )
            )

        for detected, canonical, is_stutter in find_noncanonical_abbreviations(
            words, language
        ):
            report.abbreviation_findings.append(
                AbbreviationFinding(
                    segment_index=idx,
                    start=start,
                    end=end,
                    detected=detected,
                    canonical=canonical,
                    words_preview=_format_words_with_highlight(words, detected),
                    is_stutter=is_stutter,
                )
            )

    return report


def analyze_task_transcription_pairs(
    task_id: str,
    speaker_seglst_pairs: list[tuple[str, Path]],
) -> TaskTranscriptionReport | None:
    if not speaker_seglst_pairs:
        return None

    language = language_from_task_id(task_id)
    task = TaskTranscriptionReport(task_id=task_id)
    for speaker_id, seglst_path in speaker_seglst_pairs:
        task.speakers.append(
            analyze_speaker_transcription(speaker_id, seglst_path, language)
        )
    return task


def analyze_task_transcription(task_dir: Path, variant: str) -> TaskTranscriptionReport | None:
    pairs = discover_seglst_files(task_dir, variant)
    return analyze_task_transcription_pairs(task_dir.name, pairs)


def _render_numbers_table(report: TaskTranscriptionReport) -> list[str]:
    lines = [
        "## Numbers",
        "",
        "| Speaker | # | start | end | detected | words | recommendation |",
        "|---------|--:|------:|----:|----------|-------|----------------|",
    ]
    rows = 0
    for speaker in report.speakers:
        for finding in speaker.numeric_findings:
            lines.append(
                f"| {speaker.speaker_id} | {finding.segment_index} | "
                f"{_format_timestamp(finding.start)} | {_format_timestamp(finding.end)} | "
                f"`{finding.detected}` | {finding.words_preview} | "
                f"{NUMBERS_RECOMMENDATION} |"
            )
            rows += 1
    if rows == 0:
        return ["## Numbers", "", "*No numeric words detected.*", ""]
    lines.append("")
    return lines


def _render_unknown_nsv_table(report: TaskTranscriptionReport) -> list[str]:
    lines = [
        "## Unknown NSV",
        "",
        "| Speaker | # | start | end | detected | words | recommendation |",
        "|---------|--:|------:|----:|----------|-------|----------------|",
    ]
    rows = 0
    for speaker in report.speakers:
        for finding in speaker.unknown_nsv_findings:
            detected = finding.detected.replace("|", "\\|")
            lines.append(
                f"| {speaker.speaker_id} | {finding.segment_index} | "
                f"{_format_timestamp(finding.start)} | {_format_timestamp(finding.end)} | "
                f"`{detected}` | {finding.words_preview} | "
                f"{UNKNOWN_NSV_RECOMMENDATION} |"
            )
            rows += 1
    if rows == 0:
        return ["## Unknown NSV", "", "*No unknown NSV tokens detected.*", ""]
    lines.append("")
    return lines


def _render_symbols_table(report: TaskTranscriptionReport) -> list[str]:
    lines = [
        "## Compact symbols",
        "",
        "| Speaker | # | start | end | detected | words | recommendation |",
        "|---------|--:|------:|----:|----------|-------|----------------|",
    ]
    rows = 0
    for speaker in report.speakers:
        for finding in speaker.symbol_findings:
            detected = finding.detected.replace("|", "\\|").replace("`", "")
            lines.append(
                f"| {speaker.speaker_id} | {finding.segment_index} | "
                f"{_format_timestamp(finding.start)} | {_format_timestamp(finding.end)} | "
                f"`{detected}` | {finding.words_preview} | "
                f"{SYMBOLS_RECOMMENDATION} |"
            )
            rows += 1
    if rows == 0:
        return ["## Compact symbols", "", "*No compact symbol forms detected.*", ""]
    lines.append("")
    return lines


def _render_fillers_table(report: TaskTranscriptionReport) -> list[str]:
    lines = [
        "## Non-canonical Fillers",
        "",
        "| Speaker | # | start | end | detected | canonical | words | recommendation |",
        "|---------|--:|------:|----:|----------|-----------|-------|----------------|",
    ]
    rows = 0
    for speaker in report.speakers:
        for finding in speaker.filler_findings:
            detected = finding.detected.replace("|", "\\|").replace("`", "")
            canonical = finding.canonical.replace("|", "\\|").replace("`", "")
            lines.append(
                f"| {speaker.speaker_id} | {finding.segment_index} | "
                f"{_format_timestamp(finding.start)} | {_format_timestamp(finding.end)} | "
                f"`{detected}` | `{canonical}` | {finding.words_preview} | "
                f"{filler_recommendation(finding.canonical)} |"
            )
            rows += 1
    if rows == 0:
        return ["## Non-canonical Fillers", "", "*No non-canonical fillers detected.*", ""]
    lines.append("")
    return lines


def _render_abbreviations_table(report: TaskTranscriptionReport) -> list[str]:
    lines = [
        "## Acronyms and Stutters",
        "",
        "| Speaker | # | start | end | detected | canonical | words | recommendation |",
        "|---------|--:|------:|----:|----------|-----------|-------|----------------|",
    ]
    rows = 0
    for speaker in report.speakers:
        for finding in speaker.abbreviation_findings:
            detected = finding.detected.replace("|", "\\|").replace("`", "")
            canonical = (
                finding.canonical.replace("|", "\\|").replace("`", "")
                if finding.canonical
                else "—"
            )
            lines.append(
                f"| {speaker.speaker_id} | {finding.segment_index} | "
                f"{_format_timestamp(finding.start)} | {_format_timestamp(finding.end)} | "
                f"`{detected}` | `{canonical}` | {finding.words_preview} | "
                f"{abbreviation_recommendation(finding.canonical, is_stutter=finding.is_stutter)} |"
            )
            rows += 1
    if rows == 0:
        return [
            "## Acronyms and Stutters",
            "",
            "*No acronym or stutter issues detected.*",
            "",
        ]
    lines.append("")
    return lines


def render_transcription_words_report(report: TaskTranscriptionReport) -> list[str]:
    """Return markdown lines for the Transcription Words Report section."""
    lines = [
        "# Transcription Words Report",
        "",
        f"- Numeric words: **{report.numeric_count}** | "
        f"Unknown NSV tokens: **{report.unknown_nsv_count}** | "
        f"Compact symbols: **{report.symbol_count}** | "
        f"Non-canonical Fillers: **{report.filler_count}** | "
        f"Acronyms and stutters: **{report.abbreviation_count}**",
        "",
    ]
    lines.extend(_render_numbers_table(report))
    lines.extend(_render_unknown_nsv_table(report))
    lines.extend(_render_symbols_table(report))
    lines.extend(_render_fillers_table(report))
    lines.extend(_render_abbreviations_table(report))
    return lines
