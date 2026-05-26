#!/usr/bin/env python3
"""
Convert seglst JSON files to RTTM under each task subfolder.

Expected layout under --input (default: ``./output_data`` next to this script):

    output_data/
        NV-KO-SS03-CONVO08/
            SPK01.seglst.json
            SPK01.rttm            # created by this script
            ...

Each ``*.seglst.json`` produces a sibling ``.rttm`` with the same basename
(e.g. ``SPK01.seglst.json`` -> ``SPK01.rttm``).

RTTM lines use the speaker id as the file URI (first field) and as the speaker label::

    SPEAKER {speaker} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>

Usage::

    python seglst_to_rttm.py
    python seglst_to_rttm.py --dry-run
    python seglst_to_rttm.py --input path/to/output_data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEGLST_SUFFIX = ".seglst.json"
RTTM_CHANNEL = "1"
DEFAULT_INPUT_SUBDIR = "output_data"


def seglst_to_rttm_path(seglst_path: Path) -> Path:
    """Map ``SPK01.seglst.json`` -> ``SPK01.rttm``."""
    name = seglst_path.name
    if not name.endswith(SEGLST_SUFFIX):
        raise ValueError(f"Expected {SEGLST_SUFFIX!r} filename, got {name!r}")
    return seglst_path.with_name(name[: -len(SEGLST_SUFFIX)] + ".rttm")


def parse_time(value: str) -> float:
    """Parse seglst start/end times (seconds as string)."""
    return float(value.strip())


def format_rttm_line(speaker: str, start: float, end: float) -> str:
    duration = end - start
    return (
        f"SPEAKER {speaker} {RTTM_CHANNEL} {start:.3f} {duration:.3f} "
        f"<NA> <NA> {speaker} <NA> <NA>\n"
    )


@dataclass
class FileReport:
    seglst_path: Path
    rttm_path: Path
    task_id: str
    segments_total: int = 0
    segments_written: int = 0
    segments_skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def would_write(self) -> bool:
        return self.segments_written > 0


def convert_seglst(seglst_path: Path, dry_run: bool) -> FileReport:
    task_id = seglst_path.parent.name
    rttm_path = seglst_to_rttm_path(seglst_path)

    with seglst_path.open(encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"{seglst_path}: expected JSON array")

    report = FileReport(
        seglst_path=seglst_path,
        rttm_path=rttm_path,
        task_id=task_id,
        segments_total=len(data),
    )

    segments: list[tuple[float, float, str, int]] = []
    speakers: set[str] = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{seglst_path}: item #{index} is not an object")

        try:
            speaker = str(item["speaker"])
            start = parse_time(str(item["start_time"]))
            end = parse_time(str(item["end_time"]))
        except KeyError as exc:
            raise ValueError(
                f"{seglst_path}: item #{index} missing {exc.args[0]!r}"
            ) from exc

        speakers.add(speaker)

        if end <= start:
            report.segments_skipped += 1
            report.warnings.append(
                f"item #{index}: skipped non-positive duration "
                f"({item.get('start_time')!r} -> {item.get('end_time')!r})"
            )
            continue

        segments.append((start, end, speaker, index))

    if len(speakers) > 1:
        report.warnings.append(
            f"multiple speaker values in one seglst file: {sorted(speakers)}"
        )

    segments.sort(key=lambda row: row[0])

    lines: list[str] = []
    for start, end, speaker, _index in segments:
        lines.append(format_rttm_line(speaker, start, end))
        report.segments_written += 1

    if not dry_run:
        with rttm_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(lines)

    return report


def discover_seglst_files(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.glob(f"**/*{SEGLST_SUFFIX}")
        if path.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert seglst JSON files to RTTM in task subfolders."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_INPUT_SUBDIR,
        help=(
            "Root directory containing task subfolders "
            f"(default: ./{DEFAULT_INPUT_SUBDIR} next to this script)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report without writing RTTM files",
    )
    args = parser.parse_args()

    input_root = args.input.resolve()
    if not input_root.is_dir():
        print(f"Error: input directory not found: {input_root}", file=sys.stderr)
        return 1

    files = discover_seglst_files(input_root)
    if not files:
        print(f"No *{SEGLST_SUFFIX} files found under {input_root}", file=sys.stderr)
        return 1

    total = len(files)
    reports: list[FileReport] = []
    errors = 0
    total_warnings = 0
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(input_root)
        print(f"[{index}/{total}] {rel}", flush=True)
        try:
            report = convert_seglst(path, dry_run=args.dry_run)
            reports.append(report)
            for warning in report.warnings:
                total_warnings += 1
                print(f"  warning: {warning}", file=sys.stderr)
        except (ValueError, json.JSONDecodeError) as exc:
            errors += 1
            print(f"  error: {exc}", file=sys.stderr)

    if errors:
        return 1

    total_written = sum(r.segments_written for r in reports)
    total_skipped = sum(r.segments_skipped for r in reports)
    mode = "DRY RUN" if args.dry_run else "APPLIED"

    print()
    print(f"[{mode}] Processed {len(reports)} seglst file(s) under {input_root}")
    print(f"  RTTM lines written: {total_written}")
    print(f"  Segments skipped: {total_skipped}")
    print(f"  Warnings: {total_warnings}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
