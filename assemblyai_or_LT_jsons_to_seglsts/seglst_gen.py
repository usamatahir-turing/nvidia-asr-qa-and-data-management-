#!/usr/bin/env python3
"""Build session seglst files from Label Studio (lt_jsons) or AssemblyAI JSON.

When the same session folder exists under both ``lt_jsons`` and ``assembly_ai_jsons``,
the Label Studio file is used. Word-level timings are split into utterances on gaps
greater than 200 ms (same rule as ``delivery_formatter.py``).

Output layout (one file per speaker)::

    output_seglsts/<session_id>/<speaker>.seglst.json

Example::

    python seglst_gen.py
    python seglst_gen.py --session NV_GR-SS01-CONVO01
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

SILENCE_THRESHOLD_MS = 200

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ASSEMBLY_ROOT = SCRIPT_DIR / "assembly_ai_jsons"
DEFAULT_LT_ROOT = SCRIPT_DIR / "lt_jsons"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output_seglsts"


def split_words_into_utterances(
    words: list[dict[str, Any]],
) -> Iterator[list[dict[str, Any]]]:
    if not words:
        return

    current_words = [words[0]]
    current_end = words[0]["end"]
    for word in words[1:]:
        silence_ms = word["start"] - current_end
        if silence_ms > SILENCE_THRESHOLD_MS:
            yield current_words
            current_words = [word]
            current_end = word["end"]
        else:
            current_words.append(word)
            current_end = max(current_end, word["end"])

    yield current_words


def iter_transcription_words(transcription: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for segment in transcription:
        if not isinstance(segment, dict):
            continue
        for word in segment.get("words", []):
            if (
                isinstance(word, dict)
                and "start" in word
                and "end" in word
                and word["end"] > word["start"]
                and word.get("text")
            ):
                yield {
                    "text": str(word["text"]),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                }


def transcription_has_words(transcription: Any) -> bool:
    if not isinstance(transcription, list) or not transcription:
        return False
    for _word in iter_transcription_words(transcription):
        return True
    return False


def iter_participant_words(participant: dict[str, Any]) -> Iterator[dict[str, Any]]:
    annotation = participant.get("annotation", {})
    if not isinstance(annotation, dict):
        return

    updated = annotation.get("updatedTranscription")
    if transcription_has_words(updated):
        yield from iter_transcription_words(updated)
        return

    original = annotation.get("originalTranscription", [])
    yield from iter_transcription_words(original)


def normalize_assemblyai_word(word: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(word, dict):
        return None
    text = word.get("text")
    start = word.get("start")
    end = word.get("end")
    if text is None or start is None or end is None:
        return None
    start_f, end_f = float(start), float(end)
    if end_f <= start_f:
        return None
    return {
        "text": str(text),
        "start": start_f,
        "end": end_f,
        "speaker": str(word.get("speaker", "unknown")),
    }


def parse_lt_json(
    data: dict[str, Any], session_id: str
) -> dict[str, list[dict[str, Any]]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for participant in data.get("participants", {}).values():
        if not isinstance(participant, dict):
            continue
        email = participant.get("email")
        if not email:
            continue
        speaker = str(email)
        words = sorted(iter_participant_words(participant), key=lambda w: w["start"])
        for utterance_words in split_words_into_utterances(words):
            by_speaker[speaker].append(make_segment(session_id, speaker, utterance_words))

    return dict(by_speaker)


def parse_assemblyai_json(
    data: dict[str, Any], session_id: str
) -> dict[str, list[dict[str, Any]]]:
    raw_words = data.get("words")
    if not isinstance(raw_words, list):
        raise ValueError("AssemblyAI JSON is missing a 'words' list")

    words_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_words:
        word = normalize_assemblyai_word(item)
        if word is None:
            continue
        speaker = word.pop("speaker")
        words_by_speaker[speaker].append(word)

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for speaker, words in words_by_speaker.items():
        words.sort(key=lambda w: w["start"])
        segments: list[dict[str, Any]] = []
        for utterance_words in split_words_into_utterances(words):
            segments.append(make_segment(session_id, speaker, utterance_words))
        by_speaker[speaker] = segments

    return by_speaker


def make_segment(
    session_id: str,
    speaker: str,
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    start_sec = min(word["start"] for word in words) / 1000.0
    end_sec = max(word["end"] for word in words) / 1000.0
    if end_sec <= start_sec:
        raise ValueError(
            f"Non-positive duration for {speaker!r}: "
            f"{start_sec:.2f}s -> {end_sec:.2f}s"
        )
    return {
        "session_id": session_id,
        "speaker": speaker,
        "start_time": f"{start_sec:.2f}",
        "end_time": f"{end_sec:.2f}",
        "words": " ".join(word["text"] for word in words if word.get("text")),
    }


def sort_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(segments, key=lambda seg: float(seg["start_time"]))


def seglst_filename_for_speaker(speaker: str) -> str:
    return f"{speaker}.seglst.json"


def find_json_in_session_dir(session_dir: Path) -> Path | None:
    if not session_dir.is_dir():
        return None
    json_files = sorted(session_dir.glob("*.json"))
    if not json_files:
        return None
    return json_files[0]


def discover_session_ids(assembly_root: Path, lt_root: Path) -> list[str]:
    session_ids: set[str] = set()
    for root in (assembly_root, lt_root):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                session_ids.add(child.name)
    return sorted(session_ids)


def resolve_source_json(
    session_id: str,
    *,
    assembly_root: Path,
    lt_root: Path,
) -> tuple[Path, str] | None:
    """Return (json_path, source_kind) with lt_jsons taking priority."""
    lt_dir = lt_root / session_id
    lt_json = find_json_in_session_dir(lt_dir)
    if lt_json is not None:
        return lt_json, "lt"

    assembly_dir = assembly_root / session_id
    assembly_json = find_json_in_session_dir(assembly_dir)
    if assembly_json is not None:
        return assembly_json, "assemblyai"

    return None


def build_seglists(
    json_path: Path, source_kind: str, session_id: str
) -> dict[str, list[dict[str, Any]]]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object")

    if source_kind == "lt":
        by_speaker = parse_lt_json(data, session_id)
    elif source_kind == "assemblyai":
        by_speaker = parse_assemblyai_json(data, session_id)
    else:
        raise ValueError(f"Unknown source kind: {source_kind!r}")

    if not by_speaker:
        raise ValueError(f"No utterance segments produced from {json_path}")

    return {
        speaker: sort_segments(segments)
        for speaker, segments in by_speaker.items()
        if segments
    }


def write_seglists(
    session_dir: Path, seglists: dict[str, list[dict[str, Any]]]
) -> list[Path]:
    written: list[Path] = []
    for speaker in sorted(seglists):
        output_path = session_dir / seglst_filename_for_speaker(speaker)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(seglists[speaker], fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        written.append(output_path)
    return written


def remove_legacy_combined_seglst(session_dir: Path, session_id: str) -> None:
    legacy = session_dir / f"{session_id}.seglst.json"
    if legacy.is_file():
        legacy.unlink()


def convert_session(
    session_id: str,
    *,
    assembly_root: Path,
    lt_root: Path,
    output_root: Path,
) -> list[Path]:
    resolved = resolve_source_json(
        session_id, assembly_root=assembly_root, lt_root=lt_root
    )
    if resolved is None:
        raise FileNotFoundError(
            f"No JSON found for session {session_id!r} under "
            f"{lt_root} or {assembly_root}"
        )

    json_path, source_kind = resolved
    seglists = build_seglists(json_path, source_kind, session_id)
    session_dir = output_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_combined_seglst(session_dir, session_id)
    written = write_seglists(session_dir, seglists)

    total_segments = sum(len(segments) for segments in seglists.values())
    print(
        f"{session_id}: {source_kind} ({json_path.name}) -> "
        f"{len(written)} speaker file(s), {total_segments} segment(s) "
        f"({', '.join(path.name for path in written)})"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate session seglst JSON from lt_jsons (priority) or assembly_ai_jsons."
        )
    )
    parser.add_argument(
        "--assembly-root",
        type=Path,
        default=DEFAULT_ASSEMBLY_ROOT,
        help=f"AssemblyAI session folders (default: {DEFAULT_ASSEMBLY_ROOT.name})",
    )
    parser.add_argument(
        "--lt-root",
        type=Path,
        default=DEFAULT_LT_ROOT,
        help=f"Label Studio session folders (default: {DEFAULT_LT_ROOT.name})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT.name})",
    )
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        metavar="SESSION_ID",
        help="Process only this session folder name (repeatable)",
    )
    args = parser.parse_args()

    assembly_root = args.assembly_root.resolve()
    lt_root = args.lt_root.resolve()
    output_root = args.output_root.resolve()

    session_ids = args.sessions or discover_session_ids(assembly_root, lt_root)
    if not session_ids:
        print(
            f"No session folders found under {assembly_root} or {lt_root}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    errors = 0
    sessions_ok = 0
    files_written = 0
    for session_id in session_ids:
        try:
            paths = convert_session(
                session_id,
                assembly_root=assembly_root,
                lt_root=lt_root,
                output_root=output_root,
            )
            sessions_ok += 1
            files_written += len(paths)
        except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
            print(f"{session_id}: ERROR: {exc}", file=sys.stderr)
            errors += 1

    print(
        f"\nWrote {files_written} seglst file(s) "
        f"across {sessions_ok} session(s) in {output_root}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
