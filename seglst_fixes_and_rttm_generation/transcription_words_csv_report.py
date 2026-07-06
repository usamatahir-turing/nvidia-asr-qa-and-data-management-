#!/usr/bin/env python3
"""Export transcription word-quality findings to CSV from output_data seglsts.

Runs the same checks as ``generate_report_v2.py`` (via ``transcription_ar_checks``)
on ``*_approved.seglst.json`` files under ``output_data`` (after ``fix_seglst_tokens.py``).

Default layout::

    output_data/
        NV-EN-SS14-CONVO36/
            speaker@turing.com_approved.seglst.json
            ...

Usage::

    python transcription_words_csv_report.py
    python transcription_words_csv_report.py --input output_data --output issues.csv
    python transcription_words_csv_report.py --batch-file batch_conversations_list.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_GEN_DIR = SCRIPT_DIR.parent / "segment_quality_report_gen"
sys.path.insert(0, str(REPORT_GEN_DIR))

from transcription_ar_checks import (  # noqa: E402
    NUMBERS_RECOMMENDATION,
    SYMBOLS_RECOMMENDATION,
    UNKNOWN_NSV_RECOMMENDATION,
    TaskTranscriptionReport,
    abbreviation_recommendation,
    analyze_task_transcription_pairs,
    discover_seglst_files,
    filler_recommendation,
)

SEGLST_GLOB = "*_approved.seglst.json"
DEFAULT_INPUT_DIR = Path("output_data")
DEFAULT_OUTPUT_CSV = "transcription_words_issues.csv"
DEFAULT_BATCH_FILE = "batch_conversations_list.txt"
VARIANT = "approved"

CSV_COLUMNS = (
    "conversation",
    "speaker",
    "category",
    "segment_index",
    "start",
    "end",
    "detected",
    "canonical",
    "words",
    "recommendation",
)

CATEGORY_NUMBERS = "Numbers"
CATEGORY_UNKNOWN_NSV = "Unknown NSV"
CATEGORY_SYMBOLS = "Compact symbols"
CATEGORY_FILLERS = "Non-canonical Fillers"
CATEGORY_ACRONYMS = "Acronyms and Stutters"

_MD_BOLD_RE = re.compile(r"\*\*")


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def _plain_words(text: str) -> str:
    return _MD_BOLD_RE.sub("", text).replace("|", "\\|")


@dataclass
class CsvRow:
    conversation: str
    speaker: str
    category: str
    segment_index: int
    start: str
    end: str
    detected: str
    canonical: str
    words: str
    recommendation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "conversation": self.conversation,
            "speaker": self.speaker,
            "category": self.category,
            "segment_index": str(self.segment_index),
            "start": self.start,
            "end": self.end,
            "detected": self.detected,
            "canonical": self.canonical,
            "words": self.words,
            "recommendation": self.recommendation,
        }


def load_batch_conversation_ids(batch_path: Path) -> list[str]:
    if not batch_path.is_file():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    seen: set[str] = set()
    conversation_ids: list[str] = []
    for line_number, raw_line in enumerate(
        batch_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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


def discover_conversation_dirs(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def resolve_conversation_dirs(
    input_root: Path,
    batch_path: Path | None,
) -> tuple[list[Path], list[str], list[str]]:
    """Return (task_dirs, not_found_ids, no_seglst_ids)."""
    if batch_path is not None:
        conversation_ids = load_batch_conversation_ids(batch_path)
        task_dirs: list[Path] = []
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
            if not discover_seglst_files(task_dir, VARIANT):
                no_seglst.append(conversation_id)
                print(
                    f"Warning: no {SEGLST_GLOB} files in {conversation_id!r}",
                    file=sys.stderr,
                )
                continue
            task_dirs.append(task_dir)

        return task_dirs, not_found, no_seglst

    task_dirs = []
    no_seglst = []
    for task_dir in discover_conversation_dirs(input_root):
        if discover_seglst_files(task_dir, VARIANT):
            task_dirs.append(task_dir)
        else:
            no_seglst.append(task_dir.name)

    return task_dirs, [], no_seglst


def rows_from_report(report: TaskTranscriptionReport) -> list[CsvRow]:
    conversation = report.task_id
    rows: list[CsvRow] = []

    for speaker_report in report.speakers:
        speaker = speaker_report.speaker_id

        for finding in speaker_report.numeric_findings:
            rows.append(
                CsvRow(
                    conversation=conversation,
                    speaker=speaker,
                    category=CATEGORY_NUMBERS,
                    segment_index=finding.segment_index,
                    start=_format_timestamp(finding.start),
                    end=_format_timestamp(finding.end),
                    detected=finding.detected,
                    canonical="",
                    words=_plain_words(finding.words_preview),
                    recommendation=NUMBERS_RECOMMENDATION,
                )
            )

        for finding in speaker_report.unknown_nsv_findings:
            rows.append(
                CsvRow(
                    conversation=conversation,
                    speaker=speaker,
                    category=CATEGORY_UNKNOWN_NSV,
                    segment_index=finding.segment_index,
                    start=_format_timestamp(finding.start),
                    end=_format_timestamp(finding.end),
                    detected=finding.detected,
                    canonical="",
                    words=_plain_words(finding.words_preview),
                    recommendation=UNKNOWN_NSV_RECOMMENDATION,
                )
            )

        for finding in speaker_report.symbol_findings:
            rows.append(
                CsvRow(
                    conversation=conversation,
                    speaker=speaker,
                    category=CATEGORY_SYMBOLS,
                    segment_index=finding.segment_index,
                    start=_format_timestamp(finding.start),
                    end=_format_timestamp(finding.end),
                    detected=finding.detected,
                    canonical="",
                    words=_plain_words(finding.words_preview),
                    recommendation=SYMBOLS_RECOMMENDATION,
                )
            )

        for finding in speaker_report.filler_findings:
            rows.append(
                CsvRow(
                    conversation=conversation,
                    speaker=speaker,
                    category=CATEGORY_FILLERS,
                    segment_index=finding.segment_index,
                    start=_format_timestamp(finding.start),
                    end=_format_timestamp(finding.end),
                    detected=finding.detected,
                    canonical=finding.canonical,
                    words=_plain_words(finding.words_preview),
                    recommendation=filler_recommendation(finding.canonical),
                )
            )

        for finding in speaker_report.abbreviation_findings:
            rows.append(
                CsvRow(
                    conversation=conversation,
                    speaker=speaker,
                    category=CATEGORY_ACRONYMS,
                    segment_index=finding.segment_index,
                    start=_format_timestamp(finding.start),
                    end=_format_timestamp(finding.end),
                    detected=finding.detected,
                    canonical=finding.canonical or "",
                    words=_plain_words(finding.words_preview),
                    recommendation=abbreviation_recommendation(
                        finding.canonical,
                        is_stutter=finding.is_stutter,
                    ),
                )
            )

    return rows


def write_csv(path: Path, rows: list[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run transcription word-quality checks on output_data *_approved "
            "seglst files and write findings to CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / DEFAULT_INPUT_DIR,
        help=f"Root folder with conversation subfolders (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=None,
        help=(
            "Optional newline-separated conversation list. "
            f"If omitted, all folders under --input with {SEGLST_GLOB} are scanned."
        ),
    )
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_path = args.output.resolve()
    batch_path = args.batch_file.resolve() if args.batch_file is not None else None

    if not input_root.is_dir():
        print(f"Error: input directory not found: {input_root}", file=sys.stderr)
        return 1

    try:
        task_dirs, not_found, no_seglst = resolve_conversation_dirs(input_root, batch_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not task_dirs:
        print(
            f"No conversations with {SEGLST_GLOB} files found under {input_root}",
            file=sys.stderr,
        )
        return 1

    all_rows: list[CsvRow] = []
    conversations_processed = 0
    speakers_processed = 0

    print(f"Input : {input_root}")
    print(f"Output: {output_path}")
    if batch_path is not None:
        print(f"Batch : {batch_path}")
    print(f"Conversations to scan: {len(task_dirs)}")
    print()

    for task_dir in task_dirs:
        pairs = discover_seglst_files(task_dir, VARIANT)
        report = analyze_task_transcription_pairs(task_dir.name, pairs)
        if report is None:
            continue
        conversations_processed += 1
        speakers_processed += len(report.speakers)
        all_rows.extend(rows_from_report(report))
        print(
            f"  {task_dir.name}: "
            f"{report.numeric_count} numeric / "
            f"{report.unknown_nsv_count} unknown NSV / "
            f"{report.symbol_count} compact symbols / "
            f"{report.filler_count} fillers / "
            f"{report.abbreviation_count} acronyms/stutters",
            flush=True,
        )

    write_csv(output_path, all_rows)

    category_counts = {
        CATEGORY_NUMBERS: sum(1 for r in all_rows if r.category == CATEGORY_NUMBERS),
        CATEGORY_UNKNOWN_NSV: sum(1 for r in all_rows if r.category == CATEGORY_UNKNOWN_NSV),
        CATEGORY_SYMBOLS: sum(1 for r in all_rows if r.category == CATEGORY_SYMBOLS),
        CATEGORY_FILLERS: sum(1 for r in all_rows if r.category == CATEGORY_FILLERS),
        CATEGORY_ACRONYMS: sum(1 for r in all_rows if r.category == CATEGORY_ACRONYMS),
    }

    print()
    print("Summary")
    print(f"  Conversations processed: {conversations_processed}")
    if batch_path is not None:
        print(f"  Not found under input: {len(not_found)}")
        for conversation_id in not_found:
            print(f"    - {conversation_id}")
    print(f"  No {SEGLST_GLOB} files: {len(no_seglst)}")
    if no_seglst:
        for conversation_id in no_seglst:
            print(f"    - {conversation_id}")
    print(f"  Speakers processed: {speakers_processed}")
    print(f"  Total issues written: {len(all_rows)}")
    print(f"    Numbers: {category_counts[CATEGORY_NUMBERS]}")
    print(f"    Unknown NSV: {category_counts[CATEGORY_UNKNOWN_NSV]}")
    print(f"    Compact symbols: {category_counts[CATEGORY_SYMBOLS]}")
    print(f"    Non-canonical Fillers: {category_counts[CATEGORY_FILLERS]}")
    print(f"    Acronyms and Stutters: {category_counts[CATEGORY_ACRONYMS]}")
    print(f"  CSV: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
