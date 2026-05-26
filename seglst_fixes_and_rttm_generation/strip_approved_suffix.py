#!/usr/bin/env python3
"""
Remove ``_approved`` from seglst and RTTM filenames under task subfolders.

Renames::

    SPK01_approved.seglst.json  ->  SPK01.seglst.json
    SPK01_approved.rttm         ->  SPK01.rttm

Expected layout under --input (default: ``./output_data`` next to this script):

    output_data/
        NV-KO-SS03-CONVO08/
            SPK01_approved.seglst.json
            SPK01_approved.rttm
            ...

Usage::

    python strip_approved_suffix.py
    python strip_approved_suffix.py --dry-run
    python strip_approved_suffix.py --input path/to/output_data
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

SUFFIXES = (".seglst.json", ".rttm")
APPROVED_MARKER = "_approved"
DEFAULT_INPUT_SUBDIR = "output_data"


@dataclass
class RenameOp:
    source: Path
    target: Path


def target_name(filename: str) -> str | None:
    """Return new filename without ``_approved``, or None if not applicable."""
    for suffix in SUFFIXES:
        if filename.endswith(suffix) and APPROVED_MARKER in filename:
            stem = filename[: -len(suffix)]
            if not stem.endswith(APPROVED_MARKER):
                return None
            return stem[: -len(APPROVED_MARKER)] + suffix
    return None


def discover_renames(input_root: Path) -> list[RenameOp]:
    ops: list[RenameOp] = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        new_name = target_name(path.name)
        if new_name is None:
            continue
        target = path.with_name(new_name)
        ops.append(RenameOp(source=path, target=target))
    return ops


def apply_renames(ops: list[RenameOp], input_root: Path, dry_run: bool) -> int:
    total = len(ops)
    renamed = 0
    skipped = 0

    for index, op in enumerate(ops, start=1):
        rel = op.source.relative_to(input_root)
        if op.target.exists():
            print(f"[{index}/{total}] skip (target exists): {rel}", file=sys.stderr)
            skipped += 1
            continue

        print(f"[{index}/{total}] {rel} -> {op.target.name}", flush=True)
        if not dry_run:
            op.source.rename(op.target)
        renamed += 1

    mode = "DRY RUN" if dry_run else "APPLIED"
    print()
    print(
        f"[{mode}] {renamed} file(s) "
        f"{'would be ' if dry_run else ''}renamed, {skipped} skipped"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove _approved from seglst.json and .rttm filenames."
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
        help="Print planned renames without changing files",
    )
    args = parser.parse_args()

    input_root = args.input.resolve()
    if not input_root.is_dir():
        print(f"Error: input directory not found: {input_root}", file=sys.stderr)
        return 1

    ops = discover_renames(input_root)
    if not ops:
        print(f"No *_approved*.seglst.json or *_approved*.rttm files under {input_root}")
        return 0

    return apply_renames(ops, input_root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
