#!/usr/bin/env python3
"""
Single-script segmentation QA for completed Gecko tasks.

For each speaker in each task folder, runs three checks against the same RMS
energy envelope:

  1. **Boundary tightness** — annotated start/end must be within ``--tolerance``
     seconds of the RMS-detected signal onset/offset (default 100 ms).
  2. **Interior silence** — any continuous silence inside an annotated segment
     longer than ``--max-silence`` seconds is flagged as needing a split
     (default 200 ms).
  3. **Uncovered audio** — any continuous signal **outside** all annotated
     segments longer than ``--min-missed`` seconds is flagged as a missing
     annotation (default 200 ms).

One combined Markdown report per task is written to::

    <output>/<TASK_ID>_<variant>_<YYYY-MM-DD>.md

Previous report versions are moved to ``<output>/old/``. A generation log for the
run is written as ``generation_log_<variant>_<YYYY-MM-DD>.md``, plus one
``<TASK_ID>_generation_log_<variant>_<YYYY-MM-DD>.md`` per conversation.

Default ``--input`` is ``../drive_data`` relative to this script; default
``--output`` is ``./reports_<variant>`` next to this script.

Expected layout under --input:

    NV-KO-SS03-CONVO08/
        SPK01.wav
        SPK01_fixed.seglst.json
        ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from transcription_ar_checks import (
    TaskTranscriptionReport,
    analyze_task_transcription_pairs,
    render_transcription_words_report,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VARIANT_SUFFIXES: dict[str, str] = {
    "legacy": "_legacy.seglst.json",
    "fixed": "_fixed.seglst.json",
    "approved": "_approved.seglst.json",
    "qwen3": "_qwen3.seglst.json",
    "pipeline_final": "_final.seglst.json",
    "final": ".seglst.json",
    "orig": ".seglst.json_orig",
}
# Promoted gold (*.seglst.json) must not match other pipeline / review suffixes.
FINAL_GOLD_EXCLUDE_SUFFIXES = (
    ".seglst.json_orig",
    "_legacy.seglst.json",
    "_qwen3.seglst.json",
    ".qwen3.seglst.json",
    "_final.seglst.json",
    "_fixed.seglst.json",
    "_approved.seglst.json",
)
FINAL_SOURCE_SUFFIX = "_final.seglst.json"
STRIP_SUFFIX_VARIANTS = frozenset({"orig", "final"})
SUPPORTED_VARIANTS = tuple(VARIANT_SUFFIXES)
DEFAULT_VARIANT = "fixed"
COMPARE_VARIANTS = (
    "legacy",
    "orig",
    "qwen3",
    "final",
    "pipeline_final",
    "fixed",
    "approved",
)
BENCHMARK_COMPARE_VARIANTS = (
    "orig",
    "qwen3",
    "pipeline_final",
    "fixed",
    "approved",
)
# Four-layer evolution: old initial → human approved → new qwen3 → new final.
EVOLUTION_COMPARE_VARIANTS = (
    "legacy",
    "approved",
    "qwen3",
    "pipeline_final",
)
EVOLUTION_LAYER_LABELS: dict[str, str] = {
    "legacy": "Old process initial (AssemblyAI-era seglst given to human)",
    "approved": "Human approved delivery",
    "qwen3": "New pipeline ASR (segmentation boundaries + Qwen3 text)",
    "pipeline_final": "New pipeline final (boundary-fixed target)",
}
EVOLUTION_DELTA_LABELS: dict[tuple[str, str], str] = {
    ("legacy", "approved"): "Human effort: initial → approved",
    ("approved", "qwen3"): "New pipeline vs delivered (ASR stage; boundaries unchanged vs seg)",
    ("qwen3", "pipeline_final"): "New pipeline boundary-fix gain",
    ("legacy", "pipeline_final"): "Total: old initial → new automated target",
    ("approved", "pipeline_final"): "Remaining gap: approved delivery → new pipeline final",
}


def _seglst_suffix(variant: str) -> str:
    try:
        return VARIANT_SUFFIXES[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {variant!r}") from exc


def _variant_glob_pattern(variant: str) -> str:
    return f"*{_seglst_suffix(variant)}"


def _variant_pattern_label(variant: str) -> str:
    if variant == "final":
        return "*.seglst.json (gold; excludes *_qwen3, *_final, *_fixed, *_orig)"
    return _variant_glob_pattern(variant)


def _is_mixed_seglst(name: str) -> bool:
    return "_mixed" in name.lower()


def _is_mixed_wav_stem(stem: str) -> bool:
    lowered = stem.lower()
    return "_mixed" in lowered or lowered.endswith("mixed")


def _is_final_gold_seglst(name: str) -> bool:
    lowered = name.lower()
    if not lowered.endswith(".seglst.json"):
        return False
    return not any(lowered.endswith(excluded) for excluded in FINAL_GOLD_EXCLUDE_SUFFIXES)


def _iter_seglst_paths(task_dir: Path, variant: str) -> list[Path]:
    if variant == "final":
        return sorted(
            path
            for path in task_dir.glob("*.seglst.json")
            if path.is_file() and _is_final_gold_seglst(path.name)
        )
    return sorted(task_dir.glob(_variant_glob_pattern(variant)))


def _find_wav_for_base(task_dir: Path, base: str) -> Path | None:
    exact = task_dir / f"{base}.wav"
    if exact.is_file():
        return exact
    base_lower = base.lower()
    for wav_path in sorted(task_dir.glob("*.wav")):
        if wav_path.is_file() and wav_path.stem.lower() == base_lower:
            return wav_path
    return None


def _append_wav_seglst_pair(
    task_dir: Path,
    seglst_path: Path,
    base: str,
    pairs: list[tuple[Path, Path]],
    *,
    variant: str,
    pattern_label: str,
    gen_log: GenerationLog | None,
) -> bool:
    if _is_mixed_seglst(seglst_path.name) or _is_mixed_wav_stem(base):
        return False
    wav_path = _find_wav_for_base(task_dir, base)
    if wav_path is not None:
        pairs.append((wav_path, seglst_path))
        return True

    message = (
        f"[{task_dir.name}] Missing WAV for seglist {seglst_path.name!r} "
        f"(expected {base}.wav)"
    )
    if seglst_path.name != seglst_path.name.strip():
        message += (
            "; seglist filename has leading/trailing whitespace — "
            "rename the file to match the WAV name"
        )
    if gen_log is not None:
        gen_log.add(
            task_dir.name,
            message,
            speaker=base,
            severity="error",
            code="missing_wav",
            seglst_path=str(seglst_path),
            expected_wav=f"{base}.wav",
        )
    print(f"Warning: {message}", file=sys.stderr)
    return False


def _explain_final_gold_rejection(name: str) -> str:
    lowered = name.lower()
    if not lowered.endswith(".seglst.json"):
        return "not .seglst.json"
    for excluded in FINAL_GOLD_EXCLUDE_SUFFIXES:
        if lowered.endswith(excluded):
            return f"excluded ({excluded})"
    if _is_mixed_seglst(name):
        return "excluded (_mixed)"
    return "ok"


def _log_final_discovery_diagnostics(task_dir: Path) -> None:
    all_seglst = sorted(task_dir.glob("*.seglst.json"))
    all_final_src = sorted(task_dir.glob(f"*{FINAL_SOURCE_SUFFIX}"))
    print(f"  [final] diagnostic for {task_dir.name}:", flush=True)
    if not all_seglst and not all_final_src:
        print("    no *.seglst.json or *_final.seglst.json files in folder", flush=True)
        return
    for path in all_seglst:
        reason = _explain_final_gold_rejection(path.name)
        print(f"    {path.name}: {reason}", flush=True)
    for path in all_final_src:
        print(f"    {path.name}: _final fallback candidate", flush=True)
    print(
        "    expected gold: {speaker}.seglst.json (promoted) or "
        "{speaker}_final.seglst.json before promote",
        flush=True,
    )
    print(
        "    tip: run upload_audio_qc_to_gecko.py --dry-run to promote _final locally",
        flush=True,
    )
    print(
        "    note: clean_seglst_from_gecko.py removes *.seglst.json and *_final; "
        "only *.seglst.json_orig is kept",
        flush=True,
    )


def _discover_final_pairs(
    task_dir: Path,
    gen_log: GenerationLog | None = None,
) -> list[tuple[Path, Path]]:
    """Gold promoted ``{speaker}.seglst.json``, with ``{speaker}_final`` fallback."""
    pattern_label = _variant_pattern_label("final")
    pairs_by_speaker: dict[str, tuple[Path, Path]] = {}
    gold_candidates = _iter_seglst_paths(task_dir, "final")
    fallback_candidates = sorted(task_dir.glob(f"*{FINAL_SOURCE_SUFFIX}"))

    if not gold_candidates and not fallback_candidates:
        print(
            f"Warning: [{task_dir.name}] no final gold (*.seglst.json) or "
            f"*{FINAL_SOURCE_SUFFIX} files found",
            file=sys.stderr,
        )
        _log_final_discovery_diagnostics(task_dir)
        return []

    for seglst_path in gold_candidates:
        base = _speaker_base_from_seglst(seglst_path.name, "final")
        if base is None:
            continue
        if base in pairs_by_speaker:
            continue
        paired: list[tuple[Path, Path]] = []
        if _append_wav_seglst_pair(
            task_dir,
            seglst_path,
            base,
            paired,
            variant="final",
            pattern_label=pattern_label,
            gen_log=gen_log,
        ):
            pairs_by_speaker[base] = paired[0]
            print(
                f"  [final] gold: {seglst_path.name} -> {paired[0][0].name}",
                flush=True,
            )

    for seglst_path in fallback_candidates:
        if not seglst_path.is_file():
            continue
        lowered = seglst_path.name.lower()
        if not lowered.endswith(FINAL_SOURCE_SUFFIX):
            continue
        base = seglst_path.name[: -len(FINAL_SOURCE_SUFFIX)]
        if not base or base in pairs_by_speaker:
            continue
        paired = []
        if _append_wav_seglst_pair(
            task_dir,
            seglst_path,
            base,
            paired,
            variant="final",
            pattern_label=pattern_label,
            gen_log=gen_log,
        ):
            pairs_by_speaker[base] = paired[0]
            print(
                f"  [final] fallback *_final: {seglst_path.name} -> {paired[0][0].name}",
                flush=True,
            )

    if not pairs_by_speaker:
        _log_final_discovery_diagnostics(task_dir)

    return sorted(pairs_by_speaker.values(), key=lambda pair: pair[0].stem.lower())

DEFAULT_TOLERANCE_S = 0.1
DEFAULT_MAX_SILENCE_S = 0.2
DEFAULT_MIN_MISSED_S = 0.4

DEFAULT_FRAME_MS = 20.0
DEFAULT_HOP_MS = 10.0
DEFAULT_NOISE_PERCENTILE = 15.0
DEFAULT_THRESHOLD_DB = 12.0
DEFAULT_HYSTERESIS_DB = 3.0
DEFAULT_UNCOVERED_EXTRA_DB = 2.0
DEFAULT_HANGOVER_MS = 50.0
DEFAULT_MIN_SILENCE_MS = 80.0
DEFAULT_MIN_ACTIVE_MS = 30.0


@dataclass
class EnergyParams:
    frame_ms: float = DEFAULT_FRAME_MS
    hop_ms: float = DEFAULT_HOP_MS
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE
    threshold_db: float = DEFAULT_THRESHOLD_DB
    hysteresis_db: float = DEFAULT_HYSTERESIS_DB
    hangover_ms: float = DEFAULT_HANGOVER_MS
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS
    min_active_ms: float = DEFAULT_MIN_ACTIVE_MS


@dataclass
class BoundaryFailure:
    segment_index: int
    start: float
    end: float
    onset_err_ms: float | None
    offset_err_ms: float | None
    signal_onset: float | None
    signal_offset: float | None
    issue: str
    words_preview: str = ""


@dataclass
class SilenceGap:
    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


@dataclass
class SilenceIssue:
    segment_index: int
    seg_start: float
    seg_end: float
    gaps: list[SilenceGap]
    words_preview: str = ""

    @property
    def longest_ms(self) -> float:
        return max((g.duration_ms for g in self.gaps), default=0.0)


@dataclass
class UncoveredAudio:
    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


@dataclass
class SpeakerReport:
    speaker_id: str
    wav_path: Path
    seglst_path: Path
    total_segments: int = 0
    boundary_failures: list[BoundaryFailure] = field(default_factory=list)
    no_signal_overlap: list[BoundaryFailure] = field(default_factory=list)
    silence_issues: list[SilenceIssue] = field(default_factory=list)
    uncovered_audio: list[UncoveredAudio] = field(default_factory=list)
    annotations: list[tuple[int, float, float]] = field(default_factory=list)
    sample_rate: int = 0
    noise_floor_rms: float = 0.0
    threshold_rms: float = 0.0


@dataclass
class TaskReport:
    task_id: str
    task_dir: Path
    speakers: list[SpeakerReport] = field(default_factory=list)

    @property
    def boundary_count(self) -> int:
        return sum(len(s.boundary_failures) for s in self.speakers)

    @property
    def silence_count(self) -> int:
        return sum(len(s.silence_issues) for s in self.speakers)

    @property
    def uncovered_count(self) -> int:
        return sum(len(s.uncovered_audio) for s in self.speakers)

    @property
    def no_signal_count(self) -> int:
        return sum(len(s.no_signal_overlap) for s in self.speakers)

    @property
    def segment_count(self) -> int:
        return sum(s.total_segments for s in self.speakers)

    @property
    def passed(self) -> bool:
        return (
            self.boundary_count == 0
            and self.silence_count == 0
            and self.uncovered_count == 0
            and self.no_signal_count == 0
        )


@dataclass
class GenerationIssue:
    conversation_id: str
    speaker: str | None
    severity: str
    code: str
    message: str
    seglst_path: str | None = None
    expected_wav: str | None = None


@dataclass
class GenerationLog:
    issues: list[GenerationIssue] = field(default_factory=list)

    def add(
        self,
        conversation_id: str,
        message: str,
        *,
        speaker: str | None = None,
        severity: str = "error",
        code: str = "unknown",
        seglst_path: str | None = None,
        expected_wav: str | None = None,
    ) -> None:
        self.issues.append(
            GenerationIssue(
                conversation_id=conversation_id,
                speaker=speaker,
                severity=severity,
                code=code,
                message=message,
                seglst_path=seglst_path,
                expected_wav=expected_wav,
            )
        )

    def for_conversation(self, conversation_id: str) -> list[GenerationIssue]:
        return [i for i in self.issues if i.conversation_id == conversation_id]

    def conversation_ids(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for issue in self.issues:
            if issue.conversation_id not in seen:
                seen.add(issue.conversation_id)
                ordered.append(issue.conversation_id)
        return ordered

    def render(self, variant: str, report_date: str, conversation_id: str | None = None) -> str:
        issues = (
            self.for_conversation(conversation_id)
            if conversation_id
            else self.issues
        )
        title = (
            f"# Generation log — {conversation_id} — {variant} — {report_date}"
            if conversation_id
            else f"# Generation log — {variant} — {report_date}"
        )
        lines = [title, ""]
        if not issues:
            lines.append("No issues recorded.")
            lines.append("")
            return "\n".join(lines)

        if conversation_id is None:
            by_conv: dict[str, list[GenerationIssue]] = {}
            for issue in issues:
                by_conv.setdefault(issue.conversation_id, []).append(issue)
            for conv_id in self.conversation_ids():
                lines.extend(self._render_conversation_section(conv_id, by_conv[conv_id]))
        else:
            lines.extend(self._render_conversation_section(conversation_id, issues))

        return "\n".join(lines)

    @staticmethod
    def _render_conversation_section(
        conversation_id: str, issues: list[GenerationIssue]
    ) -> list[str]:
        lines = [f"## {conversation_id}", ""]
        lines.append("| Speaker | Severity | Code | Issue |")
        lines.append("|---------|----------|------|-------|")
        for issue in issues:
            speaker = issue.speaker or "—"
            lines.append(
                f"| {speaker} | {issue.severity} | {issue.code} | {issue.message} |"
            )
        lines.append("")
        return lines


def report_filename(task_id: str, variant: str, report_date: str) -> str:
    return f"{task_id}_{variant}_{report_date}.md"


def generation_log_filename(variant: str, report_date: str) -> str:
    return f"generation_log_{variant}_{report_date}.md"


def conversation_log_filename(
    conversation_id: str, variant: str, report_date: str
) -> str:
    return f"{conversation_id}_generation_log_{variant}_{report_date}.md"


def archive_previous_conv_logs(
    output_root: Path, task_id: str, variant: str, report_date: str
) -> None:
    old_dir = output_root / "old"
    old_dir.mkdir(exist_ok=True)
    current_name = conversation_log_filename(task_id, variant, report_date)
    for path in sorted(output_root.glob(f"{task_id}_generation_log_{variant}_*.md")):
        if not path.is_file() or path.name == current_name:
            continue
        dest = old_dir / path.name
        if dest.exists():
            dest.unlink()
        path.rename(dest)


def archive_previous_reports(
    output_root: Path, task_id: str, variant: str, report_date: str
) -> None:
    """Move older report versions for this task into output_root/old/."""
    old_dir = output_root / "old"
    old_dir.mkdir(exist_ok=True)
    current_name = report_filename(task_id, variant, report_date)
    for path in sorted(output_root.glob(f"{task_id}_{variant}*.md")):
        if not path.is_file() or path.name == current_name:
            continue
        if path.name.startswith(f"{task_id}_generation_log_"):
            continue
        dest = old_dir / path.name
        if dest.exists():
            dest.unlink()
        path.rename(dest)


# ---------------------------------------------------------------------------
# RMS energy detector
# ---------------------------------------------------------------------------
def _load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav, int(sr)


def _compute_frame_rms(
    wav: np.ndarray, sr: int, frame_ms: float, hop_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if len(wav) < frame_len:
        t = np.array([len(wav) / (2.0 * sr)], dtype=np.float64)
        rms = np.array([np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-20)])
        return t, rms

    n_frames = 1 + (len(wav) - frame_len) // hop_len
    times = np.empty(n_frames, dtype=np.float64)
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        chunk = wav[start : start + frame_len].astype(np.float64)
        rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-20)
        times[i] = (start + frame_len * 0.5) / sr
    return times, rms


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _active_mask_hysteresis(
    rms: np.ndarray, thresh_on: float, thresh_off: float
) -> np.ndarray:
    active = np.zeros(len(rms), dtype=bool)
    in_region = False
    for i, level in enumerate(rms):
        if not in_region:
            if level >= thresh_on:
                in_region = True
                active[i] = True
        else:
            active[i] = True
            if level < thresh_off:
                in_region = False
    return active


def _file_noise_floor(
    wav: np.ndarray, sr: int, params: EnergyParams
) -> tuple[float, float, float]:
    _, rms = _compute_frame_rms(wav, sr, params.frame_ms, params.hop_ms)
    noise_floor = float(np.percentile(rms, params.noise_percentile))
    thresh_on = max(noise_floor * _db_to_linear(params.threshold_db), 1e-10)
    thresh_off = max(
        noise_floor * _db_to_linear(params.threshold_db - params.hysteresis_db),
        1e-10,
    )
    return noise_floor, thresh_on, thresh_off


# ---------------------------------------------------------------------------
# Per-segment analysis
# ---------------------------------------------------------------------------
def _local_active_mask(
    wav: np.ndarray,
    sr: int,
    seg_start: float,
    seg_end: float,
    thresh_on: float,
    thresh_off: float,
    params: EnergyParams,
) -> tuple[np.ndarray, int, int]:
    """Return (active_mask, i0_samples, hop_len) for the audio inside the segment."""
    i0 = int(seg_start * sr)
    i1 = int(seg_end * sr)
    hop_len = max(1, int(sr * params.hop_ms / 1000.0))
    if i1 <= i0:
        return np.zeros(0, dtype=bool), i0, hop_len

    _, rms = _compute_frame_rms(wav[i0:i1], sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return np.zeros(0, dtype=bool), i0, hop_len
    active = _active_mask_hysteresis(rms, thresh_on, thresh_off)
    return active, i0, hop_len


def _boundary_from_mask(
    active: np.ndarray,
    seg_start: float,
    seg_end: float,
    sr: int,
    i0: int,
    hop_len: int,
    frame_len: int,
    wav_len: int,
) -> tuple[float, float] | None:
    if not active.any():
        return None
    idx = np.where(active)[0]
    first, last = int(idx[0]), int(idx[-1])
    onset = (i0 + first * hop_len) / sr
    offset = min((i0 + last * hop_len + frame_len) / sr, wav_len / sr)
    return float(onset), float(offset)


def _file_active_regions(
    wav: np.ndarray,
    sr: int,
    thresh_on: float,
    thresh_off: float,
    params: EnergyParams,
) -> list[tuple[float, float]]:
    """Return active (start, end) regions across the full file, bridging short gaps."""
    _, rms = _compute_frame_rms(wav, sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return []

    active = _active_mask_hysteresis(rms, thresh_on, thresh_off)
    if not active.any():
        return []

    frame_len = max(1, int(sr * params.frame_ms / 1000.0))
    hop_len = max(1, int(sr * params.hop_ms / 1000.0))
    min_silence_frames = max(1, int(params.min_silence_ms / params.hop_ms))

    regions: list[tuple[float, float]] = []
    n = len(active)
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        start_idx = i
        end_idx = i
        i += 1
        while i < n:
            if active[i]:
                end_idx = i
                i += 1
                continue
            j = i
            while j < n and not active[j]:
                j += 1
            gap = j - i
            if gap < min_silence_frames and j < n:
                end_idx = j
                i = j
            else:
                break
        start_t = (start_idx * hop_len) / sr
        end_t = min(((end_idx * hop_len) + frame_len) / sr, len(wav) / sr)
        regions.append((start_t, end_t))
    return regions


def _uncovered_from_regions(
    regions: list[tuple[float, float]],
    annotations: list[tuple[float, float]],
    min_missed_s: float,
) -> list[UncoveredAudio]:
    """Subtract annotation intervals from active regions, keep gaps > min_missed_s."""
    if not regions:
        return []

    ann = sorted((s, e) for s, e in annotations if e > s)
    merged: list[tuple[float, float]] = []
    for s, e in ann:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    uncovered: list[UncoveredAudio] = []
    for rs, re in regions:
        cursor = rs
        for as_, ae in merged:
            if ae <= cursor:
                continue
            if as_ >= re:
                break
            if as_ > cursor:
                if (as_ - cursor) > min_missed_s:
                    uncovered.append(UncoveredAudio(cursor, as_))
            cursor = max(cursor, ae)
            if cursor >= re:
                break
        if cursor < re and (re - cursor) > min_missed_s:
            uncovered.append(UncoveredAudio(cursor, re))
    return uncovered


def _silence_gaps_from_mask(
    active: np.ndarray,
    seg_start: float,
    seg_end: float,
    sr: int,
    hop_len: int,
    max_silence_s: float,
    ignore_edges: bool,
) -> list[SilenceGap]:
    if len(active) == 0 or not active.any():
        return []

    active_idx = np.where(active)[0]
    first_active = int(active_idx[0])
    last_active = int(active_idx[-1])

    gaps: list[SilenceGap] = []
    i = first_active + 1
    while i <= last_active:
        if active[i]:
            i += 1
            continue
        j = i
        while j <= last_active and not active[j]:
            j += 1
        gap_start = seg_start + (i * hop_len) / sr
        gap_end = seg_start + (j * hop_len) / sr
        if (gap_end - gap_start) > max_silence_s:
            gaps.append(SilenceGap(gap_start, gap_end))
        i = j + 1

    if not ignore_edges:
        n = len(active)
        if first_active > 0:
            edge = SilenceGap(seg_start, seg_start + (first_active * hop_len) / sr)
            if edge.end - edge.start > max_silence_s:
                gaps.insert(0, edge)
        if last_active < n - 1:
            edge = SilenceGap(
                seg_start + ((last_active + 1) * hop_len) / sr, seg_end
            )
            if edge.end - edge.start > max_silence_s:
                gaps.append(edge)

    return gaps


# ---------------------------------------------------------------------------
# Seglst loading + helpers
# ---------------------------------------------------------------------------
def _parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def load_seglst(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return sorted(data, key=lambda item: _parse_time(item["start_time"]))


def _words_preview(words: str, max_len: int = 48) -> str:
    text = " ".join(str(words).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds * 1000:+.0f}"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS (e.g. 323.5 -> 05:23)."""
    total = max(0, int(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def _format_time_dual(seconds: float) -> str:
    """MM:SS plus decimal seconds for reviewers who use either format."""
    return f"{_format_timestamp(seconds)} ({seconds:.2f}s)"


def _format_time_range_dual(start: float, end: float) -> str:
    return (
        f"{_format_timestamp(start)}–{_format_timestamp(end)} "
        f"({start:.2f}–{end:.2f}s)"
    )


def _boundary_issue(
    onset_err: float,
    offset_err: float,
    tolerance_s: float,
    *,
    check_onset: bool = True,
    check_offset: bool = True,
) -> str:
    parts: list[str] = []
    if check_onset and abs(onset_err) > tolerance_s:
        parts.append("onset early" if onset_err < 0 else "onset late")
    if check_offset and abs(offset_err) > tolerance_s:
        parts.append("offset early" if offset_err < 0 else "offset late")
    return ", ".join(parts) if parts else "ok"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def analyze_speaker(
    wav_path: Path,
    seglst_path: Path,
    energy_params: EnergyParams,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
    uncovered_extra_db: float,
    ignore_edges: bool,
) -> SpeakerReport:
    segments_json = load_seglst(seglst_path)
    wav, sr = _load_mono_wav(wav_path)
    noise_floor, thresh_on, thresh_off = _file_noise_floor(wav, sr, energy_params)
    uncov_on = max(
        noise_floor
        * _db_to_linear(energy_params.threshold_db + uncovered_extra_db),
        1e-10,
    )
    uncov_off = max(
        noise_floor
        * _db_to_linear(
            energy_params.threshold_db + uncovered_extra_db - energy_params.hysteresis_db
        ),
        1e-10,
    )
    frame_len = max(1, int(sr * energy_params.frame_ms / 1000.0))

    report = SpeakerReport(
        speaker_id=wav_path.stem,
        wav_path=wav_path,
        seglst_path=seglst_path,
        total_segments=len(segments_json),
        sample_rate=sr,
        noise_floor_rms=noise_floor,
        threshold_rms=thresh_on,
    )

    annotation_intervals: list[tuple[float, float]] = []

    for idx, item in enumerate(segments_json):
        start = _parse_time(item["start_time"])
        end = _parse_time(item["end_time"])
        words = str(item.get("words", ""))

        if end <= start:
            report.boundary_failures.append(
                BoundaryFailure(
                    segment_index=idx,
                    start=start,
                    end=end,
                    onset_err_ms=None,
                    offset_err_ms=None,
                    signal_onset=None,
                    signal_offset=None,
                    issue="invalid duration (end <= start)",
                    words_preview=_words_preview(words),
                )
            )
            continue

        annotation_intervals.append((start, end))
        report.annotations.append((idx, start, end))

        active, i0, hop_len = _local_active_mask(
            wav, sr, start, end, thresh_on, thresh_off, energy_params
        )

        # Boundary check
        bounds = _boundary_from_mask(
            active, start, end, sr, i0, hop_len, frame_len, len(wav)
        )
        if bounds is None:
            report.no_signal_overlap.append(
                BoundaryFailure(
                    segment_index=idx,
                    start=start,
                    end=end,
                    onset_err_ms=None,
                    offset_err_ms=None,
                    signal_onset=None,
                    signal_offset=None,
                    issue="no RMS energy above threshold in segment",
                    words_preview=_words_preview(words),
                )
            )
        else:
            sig_on, sig_off = bounds
            onset_err = start - sig_on
            offset_err = end - sig_off
            onset_fail = abs(onset_err) > tolerance_s
            offset_fail = abs(offset_err) > tolerance_s
            if onset_fail or offset_fail:
                report.boundary_failures.append(
                    BoundaryFailure(
                        segment_index=idx,
                        start=start,
                        end=end,
                        onset_err_ms=onset_err * 1000,
                        offset_err_ms=offset_err * 1000,
                        signal_onset=sig_on,
                        signal_offset=sig_off,
                        issue=_boundary_issue(
                            onset_err,
                            offset_err,
                            tolerance_s,
                            check_onset=onset_fail,
                            check_offset=offset_fail,
                        ),
                        words_preview=_words_preview(words),
                    )
                )

        # Interior silence check
        gaps = _silence_gaps_from_mask(
            active, start, end, sr, hop_len, max_silence_s, ignore_edges
        )
        if gaps:
            report.silence_issues.append(
                SilenceIssue(
                    segment_index=idx,
                    seg_start=start,
                    seg_end=end,
                    gaps=gaps,
                    words_preview=_words_preview(words),
                )
            )

    # Uncovered audio check (signal not inside any annotated segment).
    # Uses a stricter threshold so transient noise/bleed doesn't get flagged.
    file_regions = _file_active_regions(wav, sr, uncov_on, uncov_off, energy_params)
    report.uncovered_audio = _uncovered_from_regions(
        file_regions, annotation_intervals, min_missed_s
    )

    return report


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _speaker_base_from_seglst(seglst_name: str, variant: str) -> str | None:
    suffix = _seglst_suffix(variant)
    if variant in STRIP_SUFFIX_VARIANTS:
        if not seglst_name.lower().endswith(suffix):
            return None
        return seglst_name[: -len(suffix)]
    pair_re = re.compile(
        rf"^(.+){re.escape(suffix)}$",
        re.IGNORECASE,
    )
    match = pair_re.match(seglst_name)
    return match.group(1) if match else None


def discover_pairs(
    task_dir: Path,
    variant: str,
    gen_log: GenerationLog | None = None,
) -> list[tuple[Path, Path]]:
    if variant == "final":
        return _discover_final_pairs(task_dir, gen_log)

    pattern_label = _variant_pattern_label(variant)
    pairs: list[tuple[Path, Path]] = []
    for seglst_path in _iter_seglst_paths(task_dir, variant):
        base = _speaker_base_from_seglst(seglst_path.name, variant)
        if base is None:
            if gen_log is not None:
                gen_log.add(
                    task_dir.name,
                    f"Seglist filename does not match expected pattern "
                    f"{pattern_label}: {seglst_path.name!r}",
                    severity="warning",
                    code="regex_mismatch",
                    seglst_path=str(seglst_path),
                )
            print(
                f"Warning: [{task_dir.name}] seglist filename does not match pattern: "
                f"{seglst_path.name}",
                file=sys.stderr,
            )
            continue
        _append_wav_seglst_pair(
            task_dir,
            seglst_path,
            base,
            pairs,
            variant=variant,
            pattern_label=pattern_label,
            gen_log=gen_log,
        )
    return pairs


def discover_tasks(
    input_root: Path,
    variant: str,
    conversation_ids: list[str] | None = None,
    gen_log: GenerationLog | None = None,
) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    tasks = [
        p
        for p in sorted(input_root.iterdir())
        if p.is_dir() and not p.name.startswith(".") and p.name != "reports"
    ]
    scorable = [t for t in tasks if discover_pairs(t, variant, gen_log)]

    if not conversation_ids:
        return scorable

    by_name = {task_dir.name: task_dir for task_dir in scorable}
    selected: list[Path] = []
    for conversation_id in conversation_ids:
        task_dir = by_name.get(conversation_id)
        if task_dir is not None:
            selected.append(task_dir)
            continue

        candidate = input_root / conversation_id
        if candidate.is_dir():
            message = (
                f"Folder exists but has no WAV + {_seglst_suffix(variant)} pairs; skipping"
            )
            if gen_log is not None:
                gen_log.add(
                    conversation_id,
                    message,
                    severity="warning",
                    code="no_scorable_pairs",
                )
            print(
                f"Warning: {conversation_id!r} exists under {input_root} but has no "
                f"WAV + {_seglst_suffix(variant)} pairs; skipping",
                file=sys.stderr,
            )
        else:
            message = f"Conversation folder not found under {input_root}"
            if gen_log is not None:
                gen_log.add(
                    conversation_id,
                    message,
                    severity="error",
                    code="conversation_not_found",
                )
            print(
                f"Warning: conversation folder not found: {conversation_id!r} "
                f"(under {input_root})",
                file=sys.stderr,
            )

    return selected


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _render_boundary_table(failures: list[BoundaryFailure]) -> list[str]:
    lines = [
        "| # | start | end | onset err | offset err | signal onset | signal offset | issue | words |",
        "|---|------:|----:|----------:|-----------:|-------------:|--------------:|-------|-------|",
    ]
    for fail in failures:
        onset_cell = (
            f"**{_format_ms(fail.onset_err_ms / 1000)} ms**"
            if fail.onset_err_ms is not None
            else "—"
        )
        offset_cell = (
            f"**{_format_ms(fail.offset_err_ms / 1000)} ms**"
            if fail.offset_err_ms is not None
            else "—"
        )
        sig_on = (
            _format_time_dual(fail.signal_onset)
            if fail.signal_onset is not None
            else "—"
        )
        sig_off = (
            _format_time_dual(fail.signal_offset)
            if fail.signal_offset is not None
            else "—"
        )
        lines.append(
            f"| {fail.segment_index} | {_format_time_dual(fail.start)} | "
            f"{_format_time_dual(fail.end)} | "
            f"{onset_cell} | {offset_cell} | {sig_on} | {sig_off} | "
            f"{fail.issue} | {fail.words_preview} |"
        )
    return lines


def _nearest_annotations(
    target_start: float,
    target_end: float,
    annotations: list[tuple[int, float, float]],
) -> str:
    before = [a for a in annotations if a[2] <= target_start]
    after = [a for a in annotations if a[1] >= target_end]
    parts: list[str] = []
    if before:
        idx, s, e = before[-1]
        parts.append(f"after #{idx} ({_format_time_range_dual(s, e)})")
    if after:
        idx, s, e = after[0]
        parts.append(f"before #{idx} ({_format_time_range_dual(s, e)})")
    if not parts:
        return "—"
    return " → ".join(parts)


def _render_uncovered_table(
    uncovered: list[UncoveredAudio],
    annotations: list[tuple[int, float, float]],
) -> list[str]:
    lines = [
        "| start | end | duration | nearest segments |",
        "|------:|----:|---------:|------------------|",
    ]
    for item in uncovered:
        nearest = _nearest_annotations(item.start, item.end, annotations)
        lines.append(
            f"| {_format_time_dual(item.start)} | {_format_time_dual(item.end)} | "
            f"**{item.duration_ms:.0f} ms** | {nearest} |"
        )
    return lines


def _render_silence_table(issues: list[SilenceIssue]) -> list[str]:
    lines = [
        "| # | start | end | longest gap | gaps (start–end, ms) | words |",
        "|---|------:|----:|------------:|----------------------|-------|",
    ]
    for issue in issues:
        gap_strs = ", ".join(
            f"{_format_time_range_dual(g.start, g.end)} ({g.duration_ms:.0f} ms)"
            for g in issue.gaps
        )
        lines.append(
            f"| {issue.segment_index} | {_format_time_dual(issue.seg_start)} | "
            f"{_format_time_dual(issue.seg_end)} | "
            f"**{issue.longest_ms:.0f} ms** | {gap_strs} | {issue.words_preview} |"
        )
    return lines


def render_speaker_section(
    spk: SpeakerReport,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {spk.speaker_id}")
    lines.append("")
    lines.append(
        f"- WAV: `{spk.wav_path.name}` | segments: **{spk.total_segments}** | "
        f"SR: **{spk.sample_rate}** Hz"
    )
    lines.append(
        f"- RMS noise floor: **{spk.noise_floor_rms:.2e}** | "
        f"threshold (on): **{spk.threshold_rms:.2e}**"
    )
    lines.append(
        f"- Boundary failures (> {tolerance_s * 1000:.0f} ms): **{len(spk.boundary_failures)}** | "
        f"Interior silence failures (> {max_silence_s * 1000:.0f} ms): **{len(spk.silence_issues)}** | "
        f"Uncovered audio (> {min_missed_s * 1000:.0f} ms): **{len(spk.uncovered_audio)}** | "
        f"No signal in segment: **{len(spk.no_signal_overlap)}**"
    )
    lines.append("")

    if spk.boundary_failures:
        lines.append(f"### Boundary failures — {spk.speaker_id}")
        lines.append("")
        lines.extend(_render_boundary_table(spk.boundary_failures))
        lines.append("")

    if spk.silence_issues:
        lines.append(f"### Interior silence — {spk.speaker_id}")
        lines.append("")
        lines.extend(_render_silence_table(spk.silence_issues))
        lines.append("")

    if spk.uncovered_audio:
        lines.append(f"### Uncovered audio (missing annotations) — {spk.speaker_id}")
        lines.append("")
        lines.extend(_render_uncovered_table(spk.uncovered_audio, spk.annotations))
        lines.append("")

    if spk.no_signal_overlap:
        lines.append(f"### No signal — {spk.speaker_id}")
        lines.append("")
        lines.append("| # | start | end | words |")
        lines.append("|---|------:|----:|-------|")
        for item in spk.no_signal_overlap:
            lines.append(
                f"| {item.segment_index} | {_format_time_dual(item.start)} | "
                f"{_format_time_dual(item.end)} | {item.words_preview} |"
            )
        lines.append("")

    if (
        not spk.boundary_failures
        and not spk.silence_issues
        and not spk.uncovered_audio
        and not spk.no_signal_overlap
    ):
        lines.append("*All checks pass.*")
        lines.append("")

    return lines


def render_task_report(
    task: TaskReport,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
    ignore_edges: bool,
    transcription: TaskTranscriptionReport | None = None,
) -> str:
    lines = [
        f"# Segmentation quality report — {task.task_id}",
        "",
        "## What this report checks",
        "",
        f"- **Boundary failures** — a segment’s start or end is off from where the "
        f"speaker actually starts/stops talking by more than "
        f"**{tolerance_s * 1000:.0f} ms**.",
        f"- **Silence failures** — there is a stretch of silence longer than "
        f"**{max_silence_s * 1000:.0f} ms** *inside* a segment. The segment should be "
        f"split there.",
        "- **Uncovered audio** — audio is present on this speaker’s channel, "
        "but no segment is annotated. A new segment should be added.",
        "- **No signal** — a segment exists in the annotations, but no audible "
        "audio was found in that time range on this channel. The segment may be on "
        "the wrong channel, at the wrong time, or shouldn’t exist.",
        "",
        "*Segment times show MM:SS and decimal seconds, e.g. 05:23 (323.45s).*",
        "",
        "## Summary",
        "",
        "| Speaker | Segments | Boundary failures | Silence failures | Uncovered audio | No-signal |",
        "|---------|---------:|------------------:|-----------------:|----------------:|----------:|",
    ]
    for spk in task.speakers:
        lines.append(
            f"| {spk.speaker_id} | {spk.total_segments} | "
            f"{len(spk.boundary_failures)} | {len(spk.silence_issues)} | "
            f"{len(spk.uncovered_audio)} | {len(spk.no_signal_overlap)} |"
        )

    status = "PASS" if task.passed else "FAIL"
    lines.append("")
    lines.append(
        f"**Task result:** {status} "
        f"({task.boundary_count} boundary, "
        f"{task.silence_count} interior silence, "
        f"{task.uncovered_count} uncovered audio, "
        f"{task.no_signal_count} no-signal failure(s))"
    )
    lines.append("")

    lines.append("## Details")
    lines.append("")
    for spk in task.speakers:
        lines.extend(
            render_speaker_section(spk, tolerance_s, max_silence_s, min_missed_s)
        )

    if task.passed:
        lines.append("")
    else:
        lines.append("---")
        lines.append("")
        lines.append(
            "*Boundary: onset err = annotated_start − signal_onset "
            "(negative = annotation starts **early**); same convention for offset.*  "
        )
        lines.append(
            "*Silence: each gap is a continuous low-energy stretch **inside** the segment; "
            "split the segment in Gecko at the gap.*  "
        )
        lines.append(
            "*Uncovered audio: a stretch of energy that no segment covers; "
            "add a new segment in Gecko (or extend a neighbour).*"
        )
        lines.append("")

    if transcription is not None:
        lines.extend(render_transcription_words_report(transcription))

    return "\n".join(lines)


def process_task_pairs(
    task_dir: Path,
    pairs: list[tuple[Path, Path]],
    energy_params: EnergyParams,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
    uncovered_extra_db: float,
    ignore_edges: bool,
    gen_log: GenerationLog | None = None,
) -> TaskReport | None:
    if not pairs:
        return None

    task = TaskReport(task_id=task_dir.name, task_dir=task_dir)
    for wav_path, seglst_path in pairs:
        print(f"  {task.task_id} / {wav_path.stem} ...", flush=True)
        try:
            speaker_report = analyze_speaker(
                wav_path,
                seglst_path,
                energy_params,
                tolerance_s,
                max_silence_s,
                min_missed_s,
                uncovered_extra_db,
                ignore_edges,
            )
        except Exception as exc:
            message = f"Failed to analyze speaker: {exc}"
            if gen_log is not None:
                gen_log.add(
                    task.task_id,
                    message,
                    speaker=wav_path.stem,
                    severity="error",
                    code="analyze_error",
                    seglst_path=str(seglst_path),
                    expected_wav=wav_path.name,
                )
            print(f"Warning: {message}", file=sys.stderr)
            continue
        task.speakers.append(speaker_report)
    if not task.speakers:
        return None
    return task


def process_task(
    task_dir: Path,
    variant: str,
    energy_params: EnergyParams,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
    uncovered_extra_db: float,
    ignore_edges: bool,
    gen_log: GenerationLog | None = None,
) -> TaskReport | None:
    pairs = discover_pairs(task_dir, variant, gen_log)
    return process_task_pairs(
        task_dir,
        pairs,
        energy_params,
        tolerance_s,
        max_silence_s,
        min_missed_s,
        uncovered_extra_db,
        ignore_edges,
        gen_log=gen_log,
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / ".." / "drive_data").resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a combined segmentation quality report per task. "
            "One <TASK>_<variant>.md per subfolder under --input."
        )
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            "Which seglst variant to score. Examples: fixed (*_fixed.seglst.json), "
            "approved (*_approved.seglst.json), orig (*.seglst.json_orig), "
            "qwen3 (*_qwen3.seglst.json), final (*.seglst.json gold). "
            f"(default: {DEFAULT_VARIANT})"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Root folder containing per-task subfolders (default: {default_input})",
    )
    parser.add_argument(
        "--conversation",
        action="append",
        dest="conversations",
        metavar="CONVERSATION_ID",
        help=(
            "Process only these task folder names under --input (repeatable). "
            "Default: all scorable tasks."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Report output directory (default: <script>/reports_<variant>). "
            "Overrides the variant-based default if provided."
        ),
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date stamp for report filenames (default: today)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help="Max allowed |boundary error| in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--max-silence",
        type=float,
        default=DEFAULT_MAX_SILENCE_S,
        help="Max allowed interior silence in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--min-missed",
        type=float,
        default=DEFAULT_MIN_MISSED_S,
        help="Min uncovered audio length to flag as missing annotation, in seconds "
        "(default: 0.4)",
    )
    parser.add_argument(
        "--include-edges",
        action="store_true",
        help="Also count silence touching segment start/end "
        "(off by default; boundary check already covers it)",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=DEFAULT_FRAME_MS,
        help="RMS analysis frame length in ms (default: 20)",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=DEFAULT_HOP_MS,
        help="RMS hop in ms (default: 10)",
    )
    parser.add_argument(
        "--noise-percentile",
        type=float,
        default=DEFAULT_NOISE_PERCENTILE,
        help="Percentile of frame RMS used as noise floor (default: 15)",
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=DEFAULT_THRESHOLD_DB,
        help="dB above noise floor to enter active region (default: 12)",
    )
    parser.add_argument(
        "--hysteresis-db",
        type=float,
        default=DEFAULT_HYSTERESIS_DB,
        help="dB below on-threshold to leave active region (default: 3)",
    )
    parser.add_argument(
        "--uncovered-extra-db",
        type=float,
        default=DEFAULT_UNCOVERED_EXTRA_DB,
        help="Extra dB added to --threshold-db ONLY for the uncovered-audio scan "
        "(default: 6). Higher = fewer noise/bleed false positives, but quieter "
        "missed speech may be ignored.",
    )
    args = parser.parse_args()

    energy_params = EnergyParams(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        noise_percentile=args.noise_percentile,
        threshold_db=args.threshold_db,
        hysteresis_db=args.hysteresis_db,
    )

    variant = args.variant
    report_date = args.date
    input_root = args.input.resolve()
    if args.output is not None:
        output_root = args.output.resolve()
    else:
        output_root = (script_dir / f"reports_{variant}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gen_log = GenerationLog()
    conversation_ids = (
        list(dict.fromkeys(args.conversations)) if args.conversations else None
    )
    task_dirs = discover_tasks(input_root, variant, conversation_ids, gen_log)
    if not task_dirs:
        scope = (
            f"for {', '.join(conversation_ids)}"
            if conversation_ids
            else f"under {input_root}"
        )
        print(
            f"No task folders with WAV + {_seglst_suffix(variant)} {scope}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ignore_edges = not args.include_edges
    if conversation_ids:
        print(
            f"Processing {len(task_dirs)} selected task(s) under {input_root} "
            f"(variant: {variant})",
            flush=True,
        )
    else:
        print(
            f"Found {len(task_dirs)} task(s) under {input_root} "
            f"(variant: {variant})",
            flush=True,
        )
    task_reports: list[TaskReport] = []

    for task_dir in task_dirs:
        print(f"\n[{task_dir.name}]", flush=True)
        report = process_task(
            task_dir,
            variant,
            energy_params,
            args.tolerance,
            args.max_silence,
            args.min_missed,
            args.uncovered_extra_db,
            ignore_edges,
            gen_log,
        )
        if report is None:
            if gen_log.for_conversation(task_dir.name):
                pass
            else:
                gen_log.add(
                    task_dir.name,
                    f"No scorable WAV + {_seglst_suffix(variant)} pairs found",
                    severity="warning",
                    code="no_scorable_pairs",
                )
            continue
        task_reports.append(report)
        pairs = discover_pairs(task_dir, variant, gen_log)
        transcription = analyze_task_transcription_pairs(
            task_dir.name,
            [(wav_path.stem, seglst_path) for wav_path, seglst_path in pairs],
        )
        archive_previous_reports(output_root, report.task_id, variant, report_date)
        out_path = output_root / report_filename(report.task_id, variant, report_date)
        out_path.write_text(
            render_task_report(
                report,
                args.tolerance,
                args.max_silence,
                args.min_missed,
                ignore_edges,
                transcription,
            ),
            encoding="utf-8",
        )
        print(
            f"  -> {out_path.name}: "
            f"{report.boundary_count} boundary / {report.silence_count} silence / "
            f"{report.uncovered_count} uncovered / {report.no_signal_count} no-signal "
            f"failure(s) / {report.segment_count} segments",
            flush=True,
        )

    run_log_path = output_root / generation_log_filename(variant, report_date)
    run_log_path.write_text(
        gen_log.render(variant, report_date),
        encoding="utf-8",
    )

    conversation_ids_with_logs = sorted(
        {report.task_id for report in task_reports} | set(gen_log.conversation_ids())
    )
    for conversation_id in conversation_ids_with_logs:
        archive_previous_conv_logs(output_root, conversation_id, variant, report_date)
        conv_log_path = output_root / conversation_log_filename(
            conversation_id, variant, report_date
        )
        conv_log_path.write_text(
            gen_log.render(variant, report_date, conversation_id),
            encoding="utf-8",
        )

    print(f"\nWrote {len(task_reports)} task report(s) to {output_root}", flush=True)
    print(f"Wrote generation log: {run_log_path.name}", flush=True)
    if conversation_ids_with_logs:
        print(
            f"Wrote {len(conversation_ids_with_logs)} per-conversation log(s)",
            flush=True,
        )

    if any(not t.passed for t in task_reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
