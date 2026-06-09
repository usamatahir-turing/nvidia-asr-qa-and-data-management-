#!/usr/bin/env python3
"""Compute statistics over delivery seglst JSON or RTTM annotations.

Expected layout under ``--root`` (default: ``./output_data``)::

    <root>/<session-id>/<speaker>.seglst.json
    <root>/<session-id>/<speaker>.rttm
    ...

Only immediate child folders of ``--root`` are scanned (nested subfolders such
as ``old/`` are ignored). Overlap statistics merge all per-speaker annotation
files within each conversation folder.

Statistics reported per conversation and combined ("all"):
  * Number of annotation files processed.
  * Number of total speakers (globally-unique IDs across the corpus).
  * Median / Average / STD / Min / Max of speaker segment length (sec).
  * Speech overlap ratio, computed using the canonical literature definition.

Speaker IDs are keyed as ``<session_id>::<speaker>``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from io import StringIO
from pathlib import Path
from typing import Iterable, TextIO

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "output_data"
DEFAULT_STATS_OUT = SCRIPT_DIR / "turing_json_stats.txt"
DEFAULT_SOURCE = "auto"

SEGLST_SUFFIX = ".seglst.json"
RTTM_SUFFIX = ".rttm"
NON_DELIVERY_SEGLST_SUFFIXES = (
    f"_approved{SEGLST_SUFFIX}",
    f"_fixed{SEGLST_SUFFIX}",
)
NON_DELIVERY_RTTM_SUFFIXES = ("_approved.rttm", "_fixed.rttm")

Segment = tuple[float, float, str, str | None, str | None]


def warn(message: str) -> None:
    print(message, file=sys.stderr)


def parse_timestamp(ts: str) -> float:
    """Parse a seglst timestamp into seconds."""
    return float(ts.strip())


def parse_seglst_json(path: Path) -> list[Segment]:
    """Return list of (start, duration, speaker, speaker_name, session_id)."""
    segments: list[Segment] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        warn(f"[warn] {path}: not valid JSON: {exc}")
        return segments
    if not isinstance(data, list):
        warn(
            f"[warn] {path}: top-level JSON is not a list, got {type(data).__name__}"
        )
        return segments

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            warn(f"[warn] {path}#{i}: segment is not a dict, skipping")
            continue
        try:
            start = parse_timestamp(entry["start_time"])
            end = parse_timestamp(entry["end_time"])
        except (KeyError, ValueError) as exc:
            warn(f"[warn] {path}#{i}: bad start/end_time ({exc}), skipping")
            continue
        dur = end - start
        if dur <= 0:
            warn(
                f"[warn] {path}#{i}: non-positive duration "
                f"({start:.3f} -> {end:.3f}), skipping"
            )
            continue
        speaker = str(entry.get("speaker", "<NA>"))
        speaker_name = entry.get("speaker_name")
        session_id = entry.get("session_id")
        segments.append((start, dur, speaker, speaker_name, session_id))
    return segments


def parse_rttm(path: Path, *, fallback_session_id: str) -> list[Segment]:
    """Return list of (start, duration, speaker, speaker_name, session_id)."""
    segments: list[Segment] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        warn(f"[warn] {path}: could not read file: {exc}")
        return segments

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            warn(f"[warn] {path}:{i}: invalid RTTM line, skipping")
            continue
        try:
            start = float(parts[3])
            duration = float(parts[4])
        except ValueError as exc:
            warn(f"[warn] {path}:{i}: bad RTTM timing ({exc}), skipping")
            continue
        if duration <= 0:
            warn(f"[warn] {path}:{i}: non-positive duration, skipping")
            continue
        speaker = parts[7]
        segments.append((start, duration, speaker, None, fallback_session_id))
    return segments


def parse_annotation_file(path: Path, *, session_id: str) -> list[Segment]:
    if path.name.endswith(SEGLST_SUFFIX):
        return parse_seglst_json(path)
    if path.name.endswith(RTTM_SUFFIX):
        return parse_rttm(path, fallback_session_id=session_id)
    warn(f"[warn] {path}: unsupported annotation file type")
    return []


def compute_overlap_ratio(segments: Iterable[Segment]) -> tuple[float, float, float, float]:
    """Compute speech / overlap durations for one conversation."""
    events: list[tuple[float, int]] = []
    for start, dur, *_ in segments:
        events.append((start, +1))
        events.append((start + dur, -1))
    if not events:
        return 0.0, 0.0, 0.0, 0.0

    events.sort()
    total_speech = 0.0
    total_overlap = 0.0
    active = 0
    prev_t = events[0][0]
    min_t = events[0][0]
    max_t = events[0][0]
    for t, delta in events:
        if t > prev_t and active > 0:
            span = t - prev_t
            total_speech += span
            if active >= 2:
                total_overlap += span
        active += delta
        prev_t = t
        if t > max_t:
            max_t = t
    file_span = max(0.0, max_t - min_t)
    ratio = (total_overlap / total_speech) if total_speech > 0 else 0.0
    return total_speech, total_overlap, file_span, ratio


class SplitStats:
    """Aggregate the per-conversation numbers we need for one session (or 'all')."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.n_files = 0
        self.all_durations: list[float] = []
        self.unique_speakers: set[str] = set()
        self.unique_speaker_names: set[str] = set()
        self.speakers_per_file: list[int] = []
        self.overlap_ratios: list[float] = []
        self.total_speech = 0.0
        self.total_overlap = 0.0
        self.total_filespan = 0.0

    def merge(self, other: "SplitStats") -> None:
        self.n_files += other.n_files
        self.all_durations.extend(other.all_durations)
        self.unique_speakers.update(other.unique_speakers)
        self.unique_speaker_names.update(other.unique_speaker_names)
        self.speakers_per_file.extend(other.speakers_per_file)
        self.overlap_ratios.extend(other.overlap_ratios)
        self.total_speech += other.total_speech
        self.total_overlap += other.total_overlap
        self.total_filespan += other.total_filespan

    def add_conversation(
        self,
        segments_by_file: list[tuple[Path, list[Segment]]],
        *,
        session_id: str,
    ) -> None:
        all_segments: list[Segment] = []
        conversation_speakers: set[str] = set()

        for _path, segs in segments_by_file:
            if not segs:
                continue
            self.n_files += 1
            all_segments.extend(segs)
            for _s, _d, spk, name, sess in segs:
                sid = sess or session_id
                global_key = f"{sid}::{spk}"
                conversation_speakers.add(global_key)
                if name:
                    self.unique_speaker_names.add(str(name))

        if not all_segments:
            return

        self.speakers_per_file.append(len(conversation_speakers))
        self.unique_speakers.update(conversation_speakers)
        self.all_durations.extend(duration for _s, duration, *_ in all_segments)

        t_speech, t_overlap, t_span, ratio = compute_overlap_ratio(all_segments)
        self.overlap_ratios.append(ratio)
        self.total_speech += t_speech
        self.total_overlap += t_overlap
        self.total_filespan += t_span

    def format_report(self) -> str:
        out = StringIO()
        self._write_report(out)
        return out.getvalue()

    def _write_report(self, out: TextIO) -> None:
        print(f"\n=== {self.label} ===", file=out)
        if self.n_files == 0:
            print("  (no annotation files)", file=out)
            return

        median_dur = statistics.median(self.all_durations)
        mean_dur = statistics.fmean(self.all_durations)
        std_dur = statistics.pstdev(self.all_durations)
        min_dur = min(self.all_durations)
        max_dur = max(self.all_durations)

        overlap_over_speech_pct = (
            100.0 * self.total_overlap / self.total_speech
            if self.total_speech > 0
            else 0.0
        )
        overlap_over_audio_pct = (
            100.0 * self.total_overlap / self.total_filespan
            if self.total_filespan > 0
            else 0.0
        )
        median_overlap_pct = 100.0 * statistics.median(self.overlap_ratios)
        mean_overlap_pct = 100.0 * statistics.fmean(self.overlap_ratios)

        print(f"Files processed               : {self.n_files}", file=out)
        print(f"Total speaker segments        : {len(self.all_durations)}", file=out)
        print(
            f"Number of total speakers      : {len(self.unique_speakers)} "
            f"(globally-unique IDs, keyed as session::speaker)",
            file=out,
        )
        if self.unique_speaker_names:
            print(
                f"Distinct speaker names        : "
                f"{len(self.unique_speaker_names)} "
                f"({', '.join(sorted(self.unique_speaker_names))})",
                file=out,
            )
        print(
            f"Avg speakers per conversation : "
            f"{statistics.fmean(self.speakers_per_file):.3f}",
            file=out,
        )
        print(f"Median speaker segment length : {median_dur:.3f} sec", file=out)
        print(f"Average speaker segment length: {mean_dur:.3f} sec", file=out)
        print(f"STD speaker segment length    : {std_dur:.3f} sec", file=out)
        print(f"Min speaker segment length    : {min_dur:.3f} sec", file=out)
        print(f"Max speaker segment length    : {max_dur:.3f} sec", file=out)
        print(
            f"Total speech duration         : {self.total_speech / 3600:.3f} h  "
            f"(union of all speaker segments)",
            file=out,
        )
        print(
            f"Total overlap duration        : {self.total_overlap / 3600:.3f} h  "
            f"(>=2 speakers active, counted once)",
            file=out,
        )
        print(
            f"Total file-span duration      : {self.total_filespan / 3600:.3f} h  "
            f"(sum of (max_end - min_start) per conversation)",
            file=out,
        )
        print(
            f"Speech overlap ratio          : {overlap_over_speech_pct:.2f} %  "
            f"(T_overlap / T_speech, canonical / literature definition)",
            file=out,
        )
        print(
            f"Overlap over total audio      : {overlap_over_audio_pct:.2f} %  "
            f"(T_overlap / T_filespan, includes silence; for reference only)",
            file=out,
        )
        print(
            f"Per-conversation mean overlap : {mean_overlap_pct:.2f} %  "
            f"(macro-avg over conversations; diagnostic)",
            file=out,
        )
        print(
            f"Per-conversation median overlap: {median_overlap_pct:.2f} %  (diagnostic)",
            file=out,
        )


def is_delivery_seglst(path: Path) -> bool:
    name = path.name
    return name.endswith(SEGLST_SUFFIX) and not any(
        name.endswith(suffix) for suffix in NON_DELIVERY_SEGLST_SUFFIXES
    )


def is_delivery_rttm(path: Path) -> bool:
    name = path.name
    return name.endswith(RTTM_SUFFIX) and not any(
        name.endswith(suffix) for suffix in NON_DELIVERY_RTTM_SUFFIXES
    )


def speaker_stem(path: Path) -> str:
    name = path.name
    if name.endswith(SEGLST_SUFFIX):
        return name[: -len(SEGLST_SUFFIX)]
    if name.endswith(RTTM_SUFFIX):
        return name[: -len(RTTM_SUFFIX)]
    raise ValueError(f"Not a seglst or RTTM file: {path}")


def discover_conversation_dirs(root: Path, conversation_names: list[str]) -> list[Path]:
    if conversation_names:
        missing = [
            name for name in conversation_names if not (root / name).is_dir()
        ]
        if missing:
            raise ValueError(
                f"Conversation folder(s) not found under {root}: {', '.join(missing)}"
            )
        return [root / name for name in conversation_names]

    return sorted(path for path in root.iterdir() if path.is_dir())


def discover_seglst_files(task_dir: Path) -> list[Path]:
    return sorted(path for path in task_dir.glob(f"*{SEGLST_SUFFIX}") if is_delivery_seglst(path))


def discover_rttm_files(task_dir: Path) -> list[Path]:
    return sorted(path for path in task_dir.glob(f"*{RTTM_SUFFIX}") if is_delivery_rttm(path))


def discover_annotation_files(task_dir: Path, source: str) -> list[Path]:
    """Return per-speaker annotation files for one conversation folder."""
    seglst_by_speaker = {speaker_stem(path): path for path in discover_seglst_files(task_dir)}
    rttm_by_speaker = {speaker_stem(path): path for path in discover_rttm_files(task_dir)}

    if source == "seglst":
        return [seglst_by_speaker[speaker] for speaker in sorted(seglst_by_speaker)]
    if source == "rttm":
        return [rttm_by_speaker[speaker] for speaker in sorted(rttm_by_speaker)]

    selected: list[Path] = []
    for speaker in sorted(set(seglst_by_speaker) | set(rttm_by_speaker)):
        if speaker in seglst_by_speaker:
            selected.append(seglst_by_speaker[speaker])
        else:
            selected.append(rttm_by_speaker[speaker])
    return selected


def process_conversation(task_dir: Path, source: str) -> SplitStats:
    stats = SplitStats(task_dir.name)
    segments_by_file: list[tuple[Path, list[Segment]]] = []
    for path in discover_annotation_files(task_dir, source):
        segments_by_file.append(
            (path, parse_annotation_file(path, session_id=task_dir.name))
        )
    stats.add_conversation(segments_by_file, session_id=task_dir.name)
    return stats


def collect_stats(root: Path, conversation_names: list[str], source: str) -> list[SplitStats]:
    per_conversation_stats: list[SplitStats] = []
    for task_dir in discover_conversation_dirs(root, conversation_names):
        stats = process_conversation(task_dir, source)
        if stats.n_files == 0:
            warn(
                f"[warn] {task_dir.name}: no delivery-ready seglst or RTTM files found"
            )
            continue
        per_conversation_stats.append(stats)
    return per_conversation_stats


def build_report(
    root: Path,
    source: str,
    per_conversation_stats: list[SplitStats],
    *,
    per_conversation: bool,
) -> str:
    out = StringIO()
    print(f"Input root : {root}", file=out)
    print(f"Source type: {source}", file=out)
    print(
        f"Conversations: {len(per_conversation_stats)} | "
        f"Annotation files: {sum(s.n_files for s in per_conversation_stats)}",
        file=out,
    )

    if per_conversation:
        for stats in per_conversation_stats:
            stats._write_report(out)

    if len(per_conversation_stats) > 1:
        combined = SplitStats(f"all ({len(per_conversation_stats)} conversations)")
        for stats in per_conversation_stats:
            combined.merge(stats)
        combined._write_report(out)
    else:
        combined = per_conversation_stats[0]
        if not per_conversation:
            combined._write_report(out)

    sorted_speakers = sorted(combined.unique_speakers)
    print(f"\nUnique speaker IDs ({len(sorted_speakers)}):", file=out)
    print(", ".join(sorted_speakers), file=out)
    print("\nUnique speaker IDs (one per line):", file=out)
    for speaker_id in sorted_speakers:
        print(speaker_id, file=out)
    return out.getvalue()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute annotation statistics from delivery seglst or RTTM files.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset root containing conversation subfolders (default: ./output_data).",
    )
    ap.add_argument(
        "--source",
        choices=("auto", "seglst", "rttm"),
        default=DEFAULT_SOURCE,
        help=(
            "Annotation source: auto prefers seglst and falls back to RTTM per speaker "
            "(default); seglst or rttm forces one format."
        ),
    )
    ap.add_argument(
        "conversations",
        nargs="*",
        metavar="CONVERSATION",
        help="Optional conversation folder name(s) under --root. Process all when omitted.",
    )
    ap.add_argument(
        "--per-conversation",
        action="store_true",
        help="Include per-conversation stats in addition to the combined report.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STATS_OUT,
        help=f"Path for the stats report (default: {DEFAULT_STATS_OUT.name}).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.root.exists():
        warn(f"Error: --root does not exist: {args.root}")
        return 1

    try:
        per_conversation_stats = collect_stats(args.root, args.conversations, args.source)
    except ValueError as exc:
        warn(f"Error: {exc}")
        return 1

    if not per_conversation_stats:
        warn(f"Error: no delivery-ready seglst or RTTM files found under {args.root}")
        return 1

    report = build_report(
        args.root,
        args.source,
        per_conversation_stats,
        per_conversation=args.per_conversation,
    )

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    except OSError as exc:
        warn(f"Error: could not write report to {args.output}: {exc}")
        return 1

    warn(
        f"Wrote stats for {len(per_conversation_stats)} conversation(s) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
