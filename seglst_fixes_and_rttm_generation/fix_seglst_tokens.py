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

- Add missing opening bracket: ``inhale]`` -> ``[inhale]``
- Remove space after ``[``: ``[ exhale]`` -> ``[exhale]``
- Hyphenate compounds: ``other- noise``, ``other - noise``, ``[other noise]`` -> ``[other-noise]``
- Lowercase token text inside brackets
- Correct common token misspellings (see ``TOKEN_SPELLING_FIXES``)

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

# Latin annotation token body (single or compound).
_TOKEN_BODY_RE = re.compile(r"^[A-Za-z]+(?:-\s*[A-Za-z]+|\s+[A-Za-z]+)*$")
# Trailing ``[`` with no token (e.g. ``inhale][`` -> ``[inhale]`` after other fixes).
_TRAILING_ORPHAN_BRACKET_RE = re.compile(r"\[$")
# Repair accidental ``[i[nhale]`` corruption from an earlier regex-based pass.
_SPLIT_BRACKET_RE = re.compile(r"\[([a-z])\[([a-z-]+)\]")

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

    @property
    def changed(self) -> bool:
        return (
            self.words_changed > 0
            or self.session_id_changed > 0
            or self.speaker_changed > 0
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
    print(f"  Files with multiple speakers: {files_with_multiple_speakers}")
    print(f"  Files with changes applied: {len(changed_reports)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
