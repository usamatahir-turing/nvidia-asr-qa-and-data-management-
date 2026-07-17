#!/usr/bin/env python3
"""
Normalize non-speech tokens and session_id in *_approved.seglst.json files.

Default paths (relative to this script):

    --input  : ../drive_data          (read-only source from Google Drive sync)
    --output : ./output_data          (fixed files land here, mirroring the input tree)

Input tree (read-only)::

    drive_data/
        NV-KO-SS03-CONVO08/
            SPK01_approved.seglst.json
            ...

Output tree (overwritten on each run)::

    seglst_fixes_and_rttm_gen/output_data/
        NV-KO-SS03-CONVO08/
            SPK01_approved.seglst.json
            ...

Token fixes applied to each segment's ``words`` field:

- Add missing opening bracket: ``inhale]`` / ``inhale ]`` -> ``[inhale]``
- Close after known NSV when ``[ inhale`` has no ``]``: ``[ inhale speech`` -> ``[inhale] speech``
- Remove space after ``[``: ``[ exhale]`` -> ``[exhale]``
- Hyphenate compounds: ``other- noise``, ``other - noise``, ``[other noise]`` -> ``[other-noise]``
- Collapse repeated hyphens: ``clear--throat`` -> ``clear-throat``
- Collapse duplicate brackets: ``[[inhale]`` / ``[inhale]]`` -> ``[inhale]``
- Remove empty brackets: ``[]``, ``()``, ``{}`` (optional whitespace inside) -> ````
- Remove zero-width / invisible format characters (e.g. U+200B in ``[​inhale]``)
- Lowercase token text inside brackets
- Correct common token misspellings (see ``TOKEN_SPELLING_FIXES``)
- Rename deprecated NSV: ``[click]`` -> ``[other-noise]``

Timing fixes applied per segment (sorted by ``start_time``):

- Zero-duration segments (``end_time == start_time``) are expanded to 0.01 s without
  overlapping neighbors (touching boundaries are allowed).
- Negative-duration segments (``end_time < start_time``) are left unchanged for manual fix.

``session_id`` is set to the parent task folder name (e.g. ``NV-KO-SS03-CONVO08``).

``speaker`` is set to the filename stem before ``_approved`` (e.g.
``mohamed.h2@turing.com`` from ``mohamed.h2@turing.com_approved.seglst.json``).
Files containing more than one distinct speaker value are reported and corrected.

Usage::

    python fix_seglst_tokens.py
    python fix_seglst_tokens.py --dry-run
    python fix_seglst_tokens.py --input path/to/drive_data --output path/to/output_data

Only conversations listed in ``batch_conversations_list.txt`` (one folder name per line)
are processed. Update that file before each batch run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEGLST_GLOB = "*_approved.seglst.json"
SEGLST_FILENAME_RE = re.compile(r"^(.+)_(approved|fixed)\.seglst\.json$")
DEFAULT_INPUT_DIR = Path("..") / "drive_data"
DEFAULT_OUTPUT_DIR = Path("output_data")
DEFAULT_BATCH_FILE = "batch_conversations_list.txt"
MIN_ZERO_DURATION_SEC = 0.01
_TIME_EPSILON = 1e-9

# Canonical NSV token names (after ``normalize_token_content``).
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

# Latin annotation token body (single or compound).
_TOKEN_BODY_RE = re.compile(r"^[A-Za-z]+(?:-\s*[A-Za-z]+|\s+[A-Za-z]+)*$")
# Trailing ``[`` with no token (e.g. ``inhale][`` -> ``[inhale]`` after other fixes).
_TRAILING_ORPHAN_BRACKET_RE = re.compile(r"\[$")
# Repair accidental ``[i[nhale]`` corruption from an earlier regex-based pass.
_SPLIT_BRACKET_RE = re.compile(r"\[([a-z])\[([a-z-]+)\]")
_MULTI_DASH_RE = re.compile(r"-{2,}")
_DUPLICATE_OPEN_BRACKET_RE = re.compile(r"\[{2,}")
_DUPLICATE_CLOSE_BRACKET_RE = re.compile(r"\]{2,}")
_EMPTY_BRACKET_RE = re.compile(r"\[\s*\]|\{\s*\}|\(\s*\)")
# Zero-width / invisible format chars that can break NSV matching (e.g. U+200B).
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")

# Misspelling -> canonical token text (applied after bracket and hyphen normalization).
TOKEN_SPELLING_FIXES: dict[str, str] = {
    # -> unintelligible
    "inintelligible": "unintelligible",
    "ununtelligible": "unintelligible",
    "unintelliglible": "unintelligible",
    "uninteligible": "unintelligible",
    "unintelligeble": "unintelligible",
    "unintelligibble": "unintelligible",
    "unintelliggible": "unintelligible",
    "unintgelligible": "unintelligible",
    "unitelligible": "unintelligible",
    "unintillegible": "unintelligible",
    "unintelligibile": "unintelligible",
    "untintelligible": "unintelligible",
    "unintelligble": "unintelligible",
    "inuntelligible": "unintelligible",
    "unintellegible": "unintelligible",
    "uninetlligible": "unintelligible",
    "intelligible": "unintelligible",
    "unintelliglble": "unintelligible",
    "untelligible": "unintelligible",
    "unitellegible": "unintelligible",
    "unintellligible": "unintelligible",
    "unintelligivle": "unintelligible",
    "uintelligible": "unintelligible",
    "unintelligile": "unintelligible",
    "unintrlligible": "unintelligible",

    # -> inhale
    "nhale": "inhale",
    "inahel": "inhale",
    "inahle": "inhale",
    "innhale": "inhale",
    "iinhale": "inhale",
    "ihhale": "inhale",

    # -> other-noise
    "click": "other-noise",
    "other-nosei": "other-noise",
    "othe-noise": "other-noise",
    "pther-noise": "other-noise",
    "other-noiuse": "other-noise",
    "other-nosie": "other-noise",
    "noise": "other-noise",
    "othre-noise": "other-noise",
    "other-npise": "other-noise",
    "other-noies": "other-noise",
    "other-noise~": "other-noise",
    "ther-noise": "other-noise",
    "orher-noise": "other-noise",
    "othere-noise": "other-noise"
    
    # -> lip-smack
    "lop-smack": "lip-smack",
    "lips-smack": "lip-smack",
    "lip-mack": "lip-smack",
    # -> tongue-click
    "tongue-clik": "tongue-click",
    # -> breath
    "breathe": "breath",
    "braeth": "breath",
    "breaht": "breath",
    # -> exhale
    "exahle": "exhale",
    # -> chuckle
    "chucke": "chuckle",
    "chukle": "chuckle",
    "chuckel": "chuckle",
    # -> laugh
    "laughter": "laugh",
    # -> lip-smack
    "smack": "lip-smack",
    "tongue-suck": "lip-smack",
    # -> teeth-suck
    "suck-teeth": "teeth-suck",
    "teeh-suck": "teeth-suck",
    "teech-suck": "teeth-suck",

    # -> cough
    "caugh": "cough",

    # -> clear-throat
    "cleart-throat": "clear-throat",
}


def _remove_zero_width_chars(text: str) -> str:
    """Strip zero-width space and related invisible format characters."""
    return _ZERO_WIDTH_RE.sub("", text)


def normalize_token_content(content: str) -> str:
    """Normalize the interior of a bracket token."""
    text = _remove_zero_width_chars(content).strip()
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.lower()
    return TOKEN_SPELLING_FIXES.get(text, text)


def _is_repairable_nsv_body(inner: str) -> bool:
    return normalize_token_content(inner) in ALLOWED_NSVS


def _segment_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"[A-Za-z]+", text)]


def _longest_repairable_nsv_suffix(token_run: str) -> str | None:
    """Return the longest trailing NSV suffix of a Latin token run, if any."""
    token_run = token_run.strip()
    if not token_run or not _TOKEN_BODY_RE.fullmatch(token_run):
        return None

    spans = _segment_spans(token_run)
    if not spans:
        return None

    best: str | None = None
    for start, _ in spans:
        suffix = token_run[start:].strip()
        if _TOKEN_BODY_RE.fullmatch(suffix) and _is_repairable_nsv_body(suffix):
            if best is None or len(suffix) > len(best):
                best = suffix
    return best


def _shortest_repairable_nsv_prefix(words: str, start: int) -> tuple[int, str] | None:
    """Return ``(end_exclusive, raw_text)`` for the shortest known NSV starting at *start*."""
    if start >= len(words):
        return None

    for match in re.finditer(r"[A-Za-z]+", words[start:]):
        end = start + match.end()
        chunk = words[start:end]
        if not _TOKEN_BODY_RE.fullmatch(chunk):
            break
        if _is_repairable_nsv_body(chunk):
            return end, chunk
    return None


def _repair_unclosed_open_bracket(words: str, bracket_pos: int) -> tuple[str, int] | None:
    """Repair ``[ inhale speech`` when a canonical NSV follows ``[`` but ``]`` is missing."""
    pos = bracket_pos + 1
    while pos < len(words) and words[pos].isspace():
        pos += 1

    match = _shortest_repairable_nsv_prefix(words, pos)
    if match is None:
        return None

    end_pos, _raw = match
    return f"[{normalize_token_content(words[pos:end_pos])}]", end_pos


def _repair_missing_open_bracket(words: str, start: int, close: int) -> tuple[str, int] | None:
    """Repair ``inhale]``, ``inhale ]``, or ``...speech inhale ]`` without swallowing speech."""
    token_end = close
    while token_end > start and words[token_end - 1].isspace():
        token_end -= 1
    if token_end <= start:
        return None

    inner = words[start:token_end]
    if not _TOKEN_BODY_RE.fullmatch(inner):
        return None

    repair = _longest_repairable_nsv_suffix(inner)
    if repair is None:
        return None

    suffix_start = token_end - len(repair)
    if words[suffix_start:token_end] != repair:
        return None
    if "[" in words[suffix_start:token_end]:
        return None

    prefix = words[start:suffix_start]
    return prefix + f"[{normalize_token_content(repair)}]", close + 1


def _repair_split_brackets(words: str) -> str:
    """Undo ``[i[nhale]`` -> ``[inhale]`` corruption if present."""
    while True:
        repaired = _SPLIT_BRACKET_RE.sub(r"[\1\2]", words)
        if repaired == words:
            return words
        words = repaired


def _collapse_extra_dashes(words: str) -> str:
    """Replace runs of two or more hyphens with a single hyphen."""
    return _MULTI_DASH_RE.sub("-", words)


def _collapse_duplicate_brackets(words: str) -> str:
    """Replace runs of ``[[`` / ``]]`` with a single bracket."""
    return _DUPLICATE_CLOSE_BRACKET_RE.sub(
        "]",
        _DUPLICATE_OPEN_BRACKET_RE.sub("[", words),
    )


def _remove_empty_brackets(words: str) -> str:
    """Remove empty ``[]``, ``{}``, and ``()`` (whitespace-only interiors allowed)."""
    while True:
        cleaned = _EMPTY_BRACKET_RE.sub("", words)
        if cleaned == words:
            return words
        words = cleaned


def fix_words(words: str) -> str:
    """Apply all token normalization rules to a words string."""
    words = _remove_zero_width_chars(words)
    words = _remove_empty_brackets(
        _collapse_duplicate_brackets(_repair_split_brackets(words))
    )

    result: list[str] = []
    i = 0
    length = len(words)

    while i < length:
        ch = words[i]
        if ch == "[":
            close = words.find("]", i + 1)
            if close == -1:
                repaired = _repair_unclosed_open_bracket(words, i)
                if repaired is not None:
                    replacement, next_i = repaired
                    result.append(replacement)
                    i = next_i
                    continue
                result.append("[")
                i += 1
                continue
            inner = words[i + 1 : close]
            result.append(f"[{normalize_token_content(inner)}]")
            i = close + 1
            continue

        if ch.isascii() and ch.isalpha():
            close = words.find("]", i + 1)
            if close != -1:
                repaired = _repair_missing_open_bracket(words, i, close)
                if repaired is not None:
                    replacement, next_i = repaired
                    result.append(replacement)
                    i = next_i
                    continue

        result.append(ch)
        i += 1

    output = "".join(result)
    output = _TRAILING_ORPHAN_BRACKET_RE.sub("", output)
    output = _remove_empty_brackets(_collapse_duplicate_brackets(output))
    return _collapse_extra_dashes(output)


def _parse_segment_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _format_segment_time(seconds: float) -> str:
    return f"{seconds:.2f}"


def apply_zero_duration_fixes(data: list[dict[str, Any]], report: FileReport) -> None:
    """Expand zero-duration segments to 0.01 s without overlapping neighbors."""
    if not data:
        return

    order = sorted(range(len(data)), key=lambda i: _parse_segment_time(data[i]["start_time"]))

    for pos, idx in enumerate(order):
        item = data[idx]
        start = _parse_segment_time(item["start_time"])
        end = _parse_segment_time(item["end_time"])

        if end < start - _TIME_EPSILON:
            continue
        if end > start + _TIME_EPSILON:
            continue

        prev_end: float | None = None
        if pos > 0:
            prev_end = _parse_segment_time(data[order[pos - 1]]["end_time"])

        next_start: float | None = None
        if pos + 1 < len(order):
            next_start = _parse_segment_time(data[order[pos + 1]]["start_time"])

        new_start = start
        new_end = start + MIN_ZERO_DURATION_SEC

        if prev_end is not None and new_start < prev_end - _TIME_EPSILON:
            new_start = prev_end
            new_end = new_start + MIN_ZERO_DURATION_SEC

        if next_start is not None and new_end > next_start + _TIME_EPSILON:
            new_end = next_start
            new_start = new_end - MIN_ZERO_DURATION_SEC

        if new_end <= new_start + _TIME_EPSILON:
            report.duration_unfixable += 1
            print(
                f"Warning: cannot expand zero-duration segment in {report.path.name} "
                f"(index {idx}, start={start:.2f}) without overlap",
                file=sys.stderr,
            )
            continue

        old_start = item.get("start_time")
        old_end = item.get("end_time")
        item["start_time"] = _format_segment_time(new_start)
        item["end_time"] = _format_segment_time(new_end)
        if item["start_time"] != old_start or item["end_time"] != old_end:
            report.duration_fixed += 1


@dataclass
class FileReport:
    path: Path
    task_id: str
    segments_total: int = 0
    words_changed: int = 0
    session_id_changed: int = 0
    speaker_changed: int = 0
    duration_fixed: int = 0
    duration_unfixable: int = 0
    multiple_speakers: bool = False
    speakers_found: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return (
            self.words_changed > 0
            or self.session_id_changed > 0
            or self.speaker_changed > 0
            or self.duration_fixed > 0
        )


def expected_speaker_from_path(path: Path) -> str | None:
    match = SEGLST_FILENAME_RE.match(path.name)
    if not match:
        return None
    return match.group(1)


def apply_speaker_fixes(
    data: list[dict[str, Any]],
    expected_speaker: str,
    report: FileReport,
) -> None:
    speakers_found = {str(item.get("speaker", "")) for item in data}
    report.speakers_found = tuple(sorted(speakers_found))
    if len(speakers_found) > 1:
        report.multiple_speakers = True

    for item in data:
        old_speaker = item.get("speaker")
        if old_speaker != expected_speaker:
            report.speaker_changed += 1
            item["speaker"] = expected_speaker


def process_file(src_path: Path, dst_path: Path, dry_run: bool) -> FileReport:
    task_id = src_path.parent.name
    expected_speaker = expected_speaker_from_path(src_path)
    with src_path.open(encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)

    report = FileReport(path=src_path, task_id=task_id, segments_total=len(data))

    for item in data:
        old_words = str(item.get("words", ""))
        new_words = fix_words(old_words)
        if new_words != old_words:
            report.words_changed += 1
            item["words"] = new_words

        old_session = item.get("session_id")
        if old_session != task_id:
            report.session_id_changed += 1
            item["session_id"] = task_id

    apply_zero_duration_fixes(data, report)

    if expected_speaker is not None:
        apply_speaker_fixes(data, expected_speaker, report)
    else:
        print(
            f"Warning: cannot derive expected speaker from filename: {src_path.name}",
            file=sys.stderr,
        )

    if not dry_run:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with dst_path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=4)
            fh.write("\n")

    return report


def discover_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob(f"**/{SEGLST_GLOB}"))


def load_batch_conversation_ids(batch_path: Path) -> list[str]:
    """Load unique conversation folder names from a newline-separated batch file."""
    if not batch_path.is_file():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    seen: set[str] = set()
    conversation_ids: list[str] = []
    for line_number, raw_line in enumerate(batch_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            print(
                f"Warning: duplicate conversation on line {line_number}: {line!r}",
                file=sys.stderr,
            )
            continue
        seen.add(line)
        conversation_ids.append(line)

    if not conversation_ids:
        raise ValueError(f"No conversation IDs found in batch file: {batch_path}")

    return conversation_ids


def discover_files_for_batch(
    input_root: Path,
    conversation_ids: list[str],
) -> tuple[list[Path], list[str], list[str]]:
    """Return (seglst files, not_found_ids, no_seglst_ids) for the batch list."""
    files: list[Path] = []
    not_found: list[str] = []
    no_seglst: list[str] = []

    for conversation_id in conversation_ids:
        task_dir = input_root / conversation_id
        if not task_dir.is_dir():
            not_found.append(conversation_id)
            print(
                f"Warning: conversation folder not found under {input_root}: "
                f"{conversation_id!r}",
                file=sys.stderr,
            )
            continue

        conversation_files = sorted(task_dir.glob(SEGLST_GLOB))
        if not conversation_files:
            no_seglst.append(conversation_id)
            print(
                f"Warning: no {SEGLST_GLOB} files in {conversation_id!r}",
                file=sys.stderr,
            )
            continue

        files.extend(conversation_files)

    return files, not_found, no_seglst
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize bracket tokens and session_id in *_approved seglst JSON files. "
            "Reads from --input and writes fixed copies to --output."
        )
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / DEFAULT_INPUT_DIR,
        help=(
            "Read-only source directory containing task subfolders "
            f"(default: {DEFAULT_INPUT_DIR.as_posix()} relative to this script)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / DEFAULT_OUTPUT_DIR,
        help=(
            "Destination directory for fixed files (default: "
            f"{DEFAULT_OUTPUT_DIR.as_posix()} next to this script)"
        ),
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=script_dir / DEFAULT_BATCH_FILE,
        help=(
            "Newline-separated list of conversation folder names to process "
            f"(default: {DEFAULT_BATCH_FILE} next to this script)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    batch_path = args.batch_file.resolve()

    if not input_root.is_dir():
        print(f"Error: input directory not found: {input_root}", file=sys.stderr)
        return 1

    if output_root == input_root:
        print(
            "Error: --output must differ from --input to avoid overwriting source files",
            file=sys.stderr,
        )
        return 1

    try:
        conversation_ids = load_batch_conversation_ids(batch_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    files, not_found, no_seglst = discover_files_for_batch(input_root, conversation_ids)
    if not files:
        print(
            f"No {SEGLST_GLOB} files found for any conversation in {batch_path.name}",
            file=sys.stderr,
        )
        return 1

    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"Batch : {batch_path} ({len(conversation_ids)} conversation(s))")
    if args.dry_run:
        print("Mode  : DRY RUN (no files will be written)")
    print()

    total = len(files)
    reports: list[FileReport] = []
    for index, src in enumerate(files, start=1):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        print(f"[{index}/{total}] {rel}", flush=True)
        report = process_file(src, dst, dry_run=args.dry_run)
        if report.multiple_speakers:
            print(
                f"  Warning: multiple speakers in {src.name}: "
                f"{', '.join(report.speakers_found)}",
                file=sys.stderr,
            )
        if report.speaker_changed:
            print(
                f"  Warning: speaker corrected to "
                f"{expected_speaker_from_path(src)!r} in {report.speaker_changed} "
                f"segment(s)",
                file=sys.stderr,
            )
        reports.append(report)

    changed_reports = [r for r in reports if r.changed]
    total_words = sum(r.words_changed for r in reports)
    total_session = sum(r.session_id_changed for r in reports)
    total_speaker = sum(r.speaker_changed for r in reports)
    total_duration_fixed = sum(r.duration_fixed for r in reports)
    total_duration_unfixable = sum(r.duration_unfixable for r in reports)
    files_with_multiple_speakers = sum(1 for r in reports if r.multiple_speakers)
    conversations_processed = len({r.task_id for r in reports})

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print()
    print(f"[{mode}] Batch summary")
    print(f"  Conversations requested: {len(conversation_ids)}")
    print(f"  Conversations processed: {conversations_processed}")
    print(f"  Not found under input: {len(not_found)}")
    if not_found:
        for conversation_id in not_found:
            print(f"    - {conversation_id}")
    print(f"  No {SEGLST_GLOB} files: {len(no_seglst)}")
    if no_seglst:
        for conversation_id in no_seglst:
            print(f"    - {conversation_id}")
    print(f"  Seglst files processed: {total}")
    print(f"  Segments with words changes: {total_words}")
    print(f"  Segments with session_id changes: {total_session}")
    print(f"  Segments with speaker changes: {total_speaker}")
    print(f"  Segments with zero-duration fixes: {total_duration_fixed}")
    if total_duration_unfixable:
        print(f"  Zero-duration segments not fixable (overlap): {total_duration_unfixable}")
    print(f"  Files with multiple speakers: {files_with_multiple_speakers}")
    print(f"  Files with changes applied: {len(changed_reports)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
