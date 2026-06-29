#!/usr/bin/env python3
"""
Fix seglst bracket tokens, then emit per-segment token density metrics as CSV.

Phase 1 reads *_fixed.seglst.json and *_approved.seglst.json from immediate
child folders of --input, applies the same token normalization used by
``seglst_fixes_and_rttm_generation/fix_seglst_tokens.py``, and writes fixed
copies to --fixed-output (default: ./fixed_tokens_output_folder).

Phase 2 scans immediate child folders of --fixed-output. For each speaker,
prefers *_approved.seglst.json over *_fixed.seglst.json when both exist.

Default paths (relative to this script)::

    --input        : ../drive_data
    --fixed-output : ./fixed_tokens_output_folder
    --csv-output   : ./token_segment_density.csv

Usage::

    python check_token_seg_density.py
    python check_token_seg_density.py --input path/to/drive_data
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT_INPUT_DIR = Path("..") / "drive_data"
DEFAULT_FIXED_OUTPUT_DIR = Path("fixed_tokens_output_folder")
DEFAULT_CSV_OUTPUT = Path("token_segment_density.csv")

SEGLST_VARIANTS = ("approved", "fixed")
SEGLST_FILENAME_RE = re.compile(r"^(.+)_(approved|fixed)\.seglst\.json$")
SEGLST_GLOBS = tuple(f"*_{variant}.seglst.json" for variant in SEGLST_VARIANTS)

# Latin annotation token body (single or compound).
_TOKEN_BODY_RE = re.compile(r"^[A-Za-z]+(?:-\s*[A-Za-z]+|\s+[A-Za-z]+)*$")
# Trailing ``[`` with no token (e.g. ``inhale][`` -> ``[inhale]`` after other fixes).
_TRAILING_ORPHAN_BRACKET_RE = re.compile(r"\[$")
# Repair accidental ``[i[nhale]`` corruption from an earlier regex-based pass.
_SPLIT_BRACKET_RE = re.compile(r"\[([a-z])\[([a-z-]+)\]")
_TOKEN_SPAN_RE = re.compile(r"\[[^\]]*\]")

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
    # -> inhale
    "nhale": "inhale",
    "inahel": "inhale",
    "inahle": "inhale",
}

CSV_COLUMNS = (
    "folder_name",
    "file_name",
    "session_id",
    "speaker",
    "words",
    "start_time",
    "end_time",
    "duration",
    "segment_index",
    "number_of_token_words",
    "number_of_non_token_words",
)


def normalize_token_content(content: str) -> str:
    """Normalize the interior of a bracket token."""
    text = content.strip()
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.lower()
    return TOKEN_SPELLING_FIXES.get(text, text)


def _repair_split_brackets(words: str) -> str:
    """Undo ``[i[nhale]`` -> ``[inhale]`` corruption if present."""
    while True:
        repaired = _SPLIT_BRACKET_RE.sub(r"[\1\2]", words)
        if repaired == words:
            return words
        words = repaired


def fix_words(words: str) -> str:
    """Apply all token normalization rules to a words string."""
    words = _repair_split_brackets(words)

    result: list[str] = []
    i = 0
    length = len(words)

    while i < length:
        ch = words[i]
        if ch == "[":
            close = words.find("]", i + 1)
            if close == -1:
                i += 1
                continue
            inner = words[i + 1 : close]
            result.append(f"[{normalize_token_content(inner)}]")
            i = close + 1
            continue

        if ch.isascii() and ch.isalpha():
            close = words.find("]", i + 1)
            if close == -1:
                result.append(words[i:])
                break
            inner = words[i:close]
            if _TOKEN_BODY_RE.fullmatch(inner):
                result.append(f"[{normalize_token_content(inner)}]")
                i = close + 1
                continue

        result.append(ch)
        i += 1

    output = "".join(result)
    return _TRAILING_ORPHAN_BRACKET_RE.sub("", output)


@dataclass
class FileReport:
    path: Path
    task_id: str
    segments_total: int = 0
    words_changed: int = 0
    session_id_changed: int = 0
    speaker_changed: int = 0
    multiple_speakers: bool = False
    speakers_found: tuple[str, ...] = ()


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


def iter_task_dirs(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def discover_seglst_files(input_root: Path) -> list[Path]:
    """Return seglst files directly inside each immediate task folder."""
    files: list[Path] = []
    for task_dir in iter_task_dirs(input_root):
        for pattern in SEGLST_GLOBS:
            files.extend(sorted(task_dir.glob(pattern)))
    return files


def load_seglst_segments(path: Path) -> list[dict[str, Any]] | None:
    """Load a seglst JSON file, or return None if it cannot be parsed."""
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        print(f"Warning: skipping {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, list):
        print(
            f"Warning: skipping {path}: expected JSON array, got {type(data).__name__}",
            file=sys.stderr,
        )
        return None

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            print(
                f"Warning: skipping {path}: segment at index {index} "
                f"is {type(item).__name__}, expected object",
                file=sys.stderr,
            )
            return None

    return data


def process_file(src_path: Path, dst_path: Path) -> FileReport | None:
    data = load_seglst_segments(src_path)
    if data is None:
        return None

    task_id = src_path.parent.name
    expected_speaker = expected_speaker_from_path(src_path)
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

    if expected_speaker is not None:
        apply_speaker_fixes(data, expected_speaker, report)
    else:
        print(
            f"Warning: cannot derive expected speaker from filename: {src_path.name}",
            file=sys.stderr,
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)
        fh.write("\n")

    return report


def fix_seglst_files(input_root: Path, output_root: Path) -> list[FileReport]:
    if output_root == input_root:
        raise ValueError("--fixed-output must differ from --input")

    files = discover_seglst_files(input_root)
    if not files:
        raise FileNotFoundError(
            f"No *_fixed.seglst.json or *_approved.seglst.json files found "
            f"in immediate child folders of {input_root}"
        )

    print(f"Input       : {input_root}")
    print(f"Fixed output: {output_root}")
    print()

    reports: list[FileReport] = []
    skipped = 0
    total = len(files)
    for index, src in enumerate(files, start=1):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        print(f"[{index}/{total}] {rel}", flush=True)
        report = process_file(src, dst)
        if report is None:
            skipped += 1
        else:
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

    changed_reports = [
        r
        for r in reports
        if r.words_changed or r.session_id_changed or r.speaker_changed
    ]
    total_words = sum(r.words_changed for r in reports)
    total_session = sum(r.session_id_changed for r in reports)
    total_speaker = sum(r.speaker_changed for r in reports)
    files_with_multiple_speakers = sum(1 for r in reports if r.multiple_speakers)

    print()
    print(f"Fixed {len(reports)} file(s)")
    if skipped:
        print(f"Skipped {skipped} file(s) due to JSON or structure errors")
    print(f"  Segments with words changes: {total_words}")
    print(f"  Segments with session_id changes: {total_session}")
    print(f"  Segments with speaker changes: {total_speaker}")
    print(f"  Files with multiple speakers: {files_with_multiple_speakers}")
    print(f"  Files with changes applied: {len(changed_reports)}")
    print()

    if not reports:
        raise FileNotFoundError(
            f"No valid seglst files could be processed under {input_root}"
        )

    return reports


def select_seglst_files(task_dir: Path) -> list[Path]:
    """Pick one seglst file per speaker, preferring approved over fixed."""
    approved: dict[str, Path] = {}
    fixed: dict[str, Path] = {}

    for path in sorted(task_dir.iterdir()):
        if not path.is_file():
            continue
        match = SEGLST_FILENAME_RE.match(path.name)
        if not match:
            continue
        speaker_key, variant = match.group(1), match.group(2)
        if variant == "approved":
            approved[speaker_key] = path
        else:
            fixed[speaker_key] = path

    selected: list[Path] = []
    for speaker_key in sorted(set(approved) | set(fixed)):
        if speaker_key in approved:
            selected.append(approved[speaker_key])
        else:
            selected.append(fixed[speaker_key])
    return selected


def count_token_and_non_token_words(words: str) -> tuple[int, int]:
    if not words or not words.strip():
        return 0, 0

    token_count = len(_TOKEN_SPAN_RE.findall(words))
    remainder = _TOKEN_SPAN_RE.sub(" ", words)
    non_token_count = len(remainder.split())
    return token_count, non_token_count


def segment_duration(start_time: Any, end_time: Any) -> float:
    return float(end_time) - float(start_time)


def iter_density_rows(fixed_output_root: Path) -> Iterator[dict[str, Any]]:
    for task_dir in iter_task_dirs(fixed_output_root):
        folder_name = task_dir.name
        for seglst_path in select_seglst_files(task_dir):
            segments = load_seglst_segments(seglst_path)
            if segments is None:
                continue

            file_name = seglst_path.name
            for segment_index, segment in enumerate(segments):
                words = str(segment.get("words", ""))
                token_count, non_token_count = count_token_and_non_token_words(words)
                start_time = segment.get("start_time", "")
                end_time = segment.get("end_time", "")

                yield {
                    "folder_name": folder_name,
                    "file_name": file_name,
                    "session_id": segment.get("session_id", ""),
                    "speaker": segment.get("speaker", ""),
                    "words": words,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": segment_duration(start_time, end_time),
                    "segment_index": segment_index,
                    "number_of_token_words": token_count,
                    "number_of_non_token_words": non_token_count,
                }


def write_density_csv(fixed_output_root: Path, csv_path: Path) -> int:
    rows = list(iter_density_rows(fixed_output_root))
    if not rows:
        raise FileNotFoundError(
            f"No seglst files selected for analysis under {fixed_output_root}"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} segment row(s) to {csv_path}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fix seglst bracket tokens and write per-segment token density metrics."
        )
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / DEFAULT_INPUT_DIR,
        help=(
            "Source directory containing task subfolders "
            f"(default: {DEFAULT_INPUT_DIR.as_posix()} relative to this script)"
        ),
    )
    parser.add_argument(
        "--fixed-output",
        type=Path,
        default=script_dir / DEFAULT_FIXED_OUTPUT_DIR,
        help=(
            "Destination for fixed seglst files "
            f"(default: {DEFAULT_FIXED_OUTPUT_DIR.as_posix()} next to this script)"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=script_dir / DEFAULT_CSV_OUTPUT,
        help=(
            "CSV report path "
            f"(default: {DEFAULT_CSV_OUTPUT.as_posix()} next to this script)"
        ),
    )
    args = parser.parse_args()

    input_root = args.input.resolve()
    fixed_output_root = args.fixed_output.resolve()
    csv_output = args.csv_output.resolve()

    if not input_root.is_dir():
        print(f"Error: input directory not found: {input_root}", file=sys.stderr)
        return 1

    try:
        fix_seglst_files(input_root, fixed_output_root)
        write_density_csv(fixed_output_root, csv_output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
