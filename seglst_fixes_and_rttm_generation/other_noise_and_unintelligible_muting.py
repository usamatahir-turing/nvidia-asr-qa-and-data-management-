#!/usr/bin/env python3
"""Mute and drop short ``[unintelligible]`` / lone ``[other-noise]`` segments on Drive.

For each ``*_approved.seglst.json`` and matching ``{speaker}.wav`` under the
Gecko Drive root:

- Drop segments whose ``words`` contain ``[unintelligible]`` and duration is
  ``<= 1.0`` s (other words may be present).
- Drop segments whose stripped ``words`` is exactly ``[other-noise]``.
- Silence the corresponding time ranges in the channel WAV (original rate/format
  kept when possible).
- Overwrite both files on Drive. Empty seglst lists are uploaded as ``[]``.

Example::

    cd seglst_fixes_and_rttm_generation/
    python other_noise_and_unintelligible_muting.py
    python other_noise_and_unintelligible_muting.py --dry-run
    python other_noise_and_unintelligible_muting.py NV-KO-SS03-CONVO08
    python other_noise_and_unintelligible_muting.py --batch
    python other_noise_and_unintelligible_muting.py --resume
"""

from __future__ import annotations

import argparse
import array
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

DRIVE_FOLDER_ID = "1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
FOLDER_MIME = "application/vnd.google-apps.folder"
JSON_MIME = "application/json"
WAV_MIME = "audio/wav"
APPROVED_SUFFIX = "_approved.seglst.json"
WAV_SUFFIX = ".wav"
UNINTELLIGIBLE_TOKEN = "[unintelligible]"
OTHER_NOISE_TOKEN = "[other-noise]"
MAX_UNINTELLIGIBLE_DURATION_SEC = 1.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_FILE = SCRIPT_DIR / "unintelligible_othernoise_fixes_batch.txt"
DEFAULT_PROGRESS_FILE = SCRIPT_DIR / "unintelligible_othernoise_progress.json"


def get_authenticated_drive_service():
    """Authenticate to Drive using service-account impersonation."""
    print("Authenticating...")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/jupyter/.config/gcloud/application_default_credentials.json"
    )

    base_credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    base_credentials.refresh(Request())

    url = (
        f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
        f"{TARGET_SERVICE_ACCOUNT}:generateAccessToken"
    )
    headers = {
        "Authorization": f"Bearer {base_credentials.token}",
        "Content-Type": "application/json",
    }
    payload = {
        "scope": ["https://www.googleapis.com/auth/drive"],
        "lifetime": "3600s",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Authentication Failed! API Error {response.status_code}: {response.text}"
        )

    sa_token = response.json()["accessToken"]
    creds = Oauth2Credentials(sa_token)
    return build("drive", "v3", credentials=creds)


def load_batch_folder_names(batch_path: Path) -> list[str]:
    if not batch_path.is_file():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    seen: set[str] = set()
    names: list[str] = []
    for line_number, raw_line in enumerate(
        batch_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            print(
                f"Warning: duplicate folder on line {line_number}: {line!r}",
                file=sys.stderr,
            )
            continue
        seen.add(line)
        names.append(line)

    if not names:
        raise ValueError(f"No folder names found in batch file: {batch_path}")
    return names


def load_progress(path: Path) -> dict:
    if not path.is_file():
        return {"drive_folder_id": DRIVE_FOLDER_ID, "completed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(
            f"Warning: could not parse progress file {path}; starting empty",
            file=sys.stderr,
        )
        return {"drive_folder_id": DRIVE_FOLDER_ID, "completed": {}}
    if not isinstance(data, dict):
        return {"drive_folder_id": DRIVE_FOLDER_ID, "completed": {}}
    completed = data.get("completed")
    if not isinstance(completed, dict):
        completed = {}
    return {
        "drive_folder_id": data.get("drive_folder_id", DRIVE_FOLDER_ID),
        "completed": completed,
    }


def save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


def mark_conversation_complete(
    progress: dict,
    path: Path,
    conversation: str,
    *,
    files: int,
    note: str = "",
) -> None:
    entry: dict[str, Any] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    if note:
        entry["note"] = note
    progress["completed"][conversation] = entry
    save_progress(path, progress)


def list_drive_subfolders(service, parent_id: str) -> dict[str, str]:
    subfolders: dict[str, str] = {}
    page_token = None
    query = (
        f"'{parent_id}' in parents and mimeType = '{FOLDER_MIME}' "
        "and trashed = false"
    )
    while True:
        res = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for folder in res.get("files", []):
            subfolders[folder["name"]] = folder["id"]
        page_token = res.get("nextPageToken")
        if page_token is None:
            break
    return subfolders


def list_drive_files(service, folder_id: str) -> dict[str, dict[str, str]]:
    files: dict[str, dict[str, str]] = {}
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        res = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for item in res.get("files", []):
            files[item["name"]] = {"id": item["id"], "mimeType": item["mimeType"]}
        page_token = res.get("nextPageToken")
        if page_token is None:
            break
    return files


def download_drive_file(service, file_id: str, dest_path: Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with dest_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def upload_drive_file(service, file_id: str, local_path: Path, mime_type: str) -> None:
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    service.files().update(
        fileId=file_id,
        media_body=media,
        supportsAllDrives=True,
    ).execute()


def wav_name_for_approved_seglst(seglst_name: str) -> str | None:
    if not seglst_name.endswith(APPROVED_SUFFIX):
        return None
    stem = seglst_name[: -len(APPROVED_SUFFIX)]
    if not stem:
        return None
    return f"{stem}{WAV_SUFFIX}"


def parse_segment_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def should_drop_segment(segment: dict[str, Any]) -> tuple[bool, str | None]:
    words = str(segment.get("words", ""))
    stripped = words.strip()
    if stripped == OTHER_NOISE_TOKEN:
        return True, "other-noise-only"
    if UNINTELLIGIBLE_TOKEN not in words:
        return False, None
    start = parse_segment_time(segment.get("start_time", 0))
    end = parse_segment_time(segment.get("end_time", 0))
    duration = end - start
    if duration <= MAX_UNINTELLIGIBLE_DURATION_SEC:
        return True, "unintelligible-short"
    return False, None


def filter_segments(
    data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[float, float, str]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[float, float, str]] = []
    for item in data:
        drop, reason = should_drop_segment(item)
        if not drop:
            kept.append(item)
            continue
        start = parse_segment_time(item.get("start_time", 0))
        end = parse_segment_time(item.get("end_time", 0))
        dropped.append((start, end, reason or "unknown"))
    return kept, dropped


def mute_wav_pcm(input_path: Path, output_path: Path, ranges: list[tuple[float, float]]) -> None:
    """Silence time ranges in a PCM WAV, preserving sample rate and format."""
    with wave.open(str(input_path), "rb") as reader:
        params = reader.getparams()
        nchannels = params.nchannels
        sampwidth = params.sampwidth
        framerate = params.framerate
        nframes = params.nframes
        frames = reader.readframes(nframes)

    typecode = {1: "B", 2: "h", 4: "i"}.get(sampwidth)
    if typecode is None:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    samples = array.array(typecode)
    samples.frombytes(frames)
    zero = 128 if sampwidth == 1 else 0
    total_frames = len(samples) // nchannels

    for start_sec, end_sec in ranges:
        if end_sec <= start_sec:
            continue
        start_frame = max(0, int(start_sec * framerate))
        end_frame = min(total_frames, int(end_sec * framerate + 0.999999))
        for frame_index in range(start_frame, end_frame):
            base = frame_index * nchannels
            for channel in range(nchannels):
                samples[base + channel] = zero

    with wave.open(str(output_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(samples.tobytes())


def mute_wav_ffmpeg(
    input_path: Path, output_path: Path, ranges: list[tuple[float, float]]
) -> None:
    clauses = []
    for start_sec, end_sec in ranges:
        if end_sec <= start_sec:
            continue
        clauses.append(f"between(t,{start_sec:.6f},{end_sec:.6f})")
    if not clauses:
        shutil.copy2(input_path, output_path)
        return
    enable = "+".join(clauses)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"volume=0:enable='{enable}'",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def mute_wav(
    input_path: Path, output_path: Path, ranges: list[tuple[float, float]]
) -> None:
    try:
        mute_wav_pcm(input_path, output_path, ranges)
    except (wave.Error, ValueError):
        mute_wav_ffmpeg(input_path, output_path, ranges)


def process_speaker_pair(
    service,
    conversation: str,
    seglst_name: str,
    seglst_id: str,
    wav_name: str,
    wav_id: str,
    work_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, int] | None:
    label = f"{conversation}/{seglst_name}"
    conv_dir = work_dir / conversation
    conv_dir.mkdir(parents=True, exist_ok=True)
    seglst_src = conv_dir / seglst_name
    seglst_dst = conv_dir / f"fixed_{seglst_name}"
    wav_src = conv_dir / wav_name
    wav_dst = conv_dir / f"muted_{wav_name}"

    try:
        print(f"  Downloading {seglst_name}...")
        download_drive_file(service, seglst_id, seglst_src)
        with seglst_src.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("seglst JSON must be a list")

        kept, dropped = filter_segments(data)
        if not dropped:
            print(f"  No matching segments in {seglst_name}")
            return {
                "dropped_unintelligible": 0,
                "dropped_other_noise": 0,
                "kept": len(kept),
                "empty": 0,
                "changed": 0,
            }

        counts = {
            "dropped_unintelligible": sum(
                1 for _s, _e, reason in dropped if reason == "unintelligible-short"
            ),
            "dropped_other_noise": sum(
                1 for _s, _e, reason in dropped if reason == "other-noise-only"
            ),
            "kept": len(kept),
            "empty": 1 if not kept else 0,
            "changed": 1,
        }
        if not kept:
            print(
                f"  Warning: {label} has no segments left after filtering; "
                "uploading empty []",
                file=sys.stderr,
            )
        print(
            f"  Dropping {len(dropped)} segment(s) "
            f"(unintelligible<=1s: {counts['dropped_unintelligible']}, "
            f"other-noise-only: {counts['dropped_other_noise']})"
        )

        print(f"  Downloading {wav_name}...")
        download_drive_file(service, wav_id, wav_src)
        mute_ranges = [(start, end) for start, end, _reason in dropped]
        mute_wav(wav_src, wav_dst, mute_ranges)

        seglst_dst.write_text(
            json.dumps(kept, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )

        if dry_run:
            print(f"  [dry-run] would overwrite {seglst_name} and {wav_name}")
        else:
            print(f"  Uploading {seglst_name}...")
            upload_drive_file(service, seglst_id, seglst_dst, JSON_MIME)
            print(f"  Uploading {wav_name}...")
            upload_drive_file(service, wav_id, wav_dst, WAV_MIME)
        return counts
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(f"Error: ffmpeg failed for {label}: {stderr}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"Error: failed to process {label}: {exc}", file=sys.stderr)
        return None
    finally:
        for path in (seglst_src, seglst_dst, wav_src, wav_dst):
            if path.exists():
                path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop short [unintelligible] and lone [other-noise] segments from "
            "approved seglsts on Drive and mute those regions in the matching WAVs."
        )
    )
    parser.add_argument(
        "conversations",
        nargs="*",
        metavar="CONVERSATION",
        help=(
            "Conversation folder name(s) under the Drive root. "
            "Defaults to all subfolders unless --batch is set."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Process only conversations listed in "
            f"{DEFAULT_BATCH_FILE.name} (or --batch-file)."
        ),
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=None,
        help=f"Batch file used with --batch (default: {DEFAULT_BATCH_FILE.name}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and report changes without overwriting Drive files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip conversations already recorded as complete in the progress file",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=DEFAULT_PROGRESS_FILE,
        help=f"JSON progress file (default: {DEFAULT_PROGRESS_FILE.name})",
    )
    parser.add_argument(
        "--clear-progress",
        action="store_true",
        help="Clear the progress file before starting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch_file is not None and not args.batch:
        print("Error: --batch-file requires --batch.", file=sys.stderr)
        return 1
    if args.batch and args.conversations:
        print(
            "Error: pass conversation names or --batch, not both.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Muting unintelligible/other-noise segments on Drive folder "
        f"{DRIVE_FOLDER_ID}..."
    )
    if args.dry_run:
        print("Mode: DRY RUN (Drive files will not be overwritten)")

    try:
        service = get_authenticated_drive_service()
        subfolders = list_drive_subfolders(service, DRIVE_FOLDER_ID)
        if not subfolders:
            print("Error: no conversation folders found on Drive.", file=sys.stderr)
            return 1

        if args.batch:
            batch_path = (
                args.batch_file.resolve()
                if args.batch_file is not None
                else DEFAULT_BATCH_FILE
            )
            conversations = load_batch_folder_names(batch_path)
            print(f"Batch file: {batch_path} ({len(conversations)} conversation(s))")
        elif args.conversations:
            conversations = list(dict.fromkeys(args.conversations))
        else:
            conversations = sorted(subfolders)

        print(f"Conversations requested: {len(conversations)}")

        progress_path = args.progress_file.resolve()
        if args.clear_progress and progress_path.is_file():
            progress_path.unlink()
            print(f"Cleared progress file: {progress_path}")
        progress = load_progress(progress_path)
        progress["drive_folder_id"] = DRIVE_FOLDER_ID

        skipped_resume = 0
        if args.resume:
            print(f"Mode: RESUME (progress file: {progress_path})")
            remaining: list[str] = []
            for name in conversations:
                if name in progress["completed"]:
                    print(f"Skipping {name} (already complete in progress file)")
                    skipped_resume += 1
                    continue
                remaining.append(name)
            conversations = remaining
            if not conversations:
                print("All requested conversations already complete.")
                return 0

        totals = {
            "dropped_unintelligible": 0,
            "dropped_other_noise": 0,
            "speakers_changed": 0,
            "speakers_empty": 0,
            "speakers_ok": 0,
        }
        failed = 0
        missing_folders: list[str] = []
        no_seglst: list[str] = []

        with tempfile.TemporaryDirectory(prefix="mute_unint_othernoise_") as tmp:
            work_dir = Path(tmp)
            for conversation in conversations:
                print(f"\n--- {conversation} ---")
                if conversation not in subfolders:
                    print(
                        f"Warning: conversation folder not found on Drive: "
                        f"{conversation!r}",
                        file=sys.stderr,
                    )
                    missing_folders.append(conversation)
                    continue

                files = list_drive_files(service, subfolders[conversation])
                approved = sorted(
                    name
                    for name, info in files.items()
                    if name.endswith(APPROVED_SUFFIX) and info["mimeType"] != FOLDER_MIME
                )
                if not approved:
                    print(
                        f"Warning: no *_approved.seglst.json files in {conversation!r}",
                        file=sys.stderr,
                    )
                    no_seglst.append(conversation)
                    if not args.dry_run:
                        mark_conversation_complete(
                            progress,
                            progress_path,
                            conversation,
                            files=0,
                            note="no approved seglst files",
                        )
                    continue

                conv_failed = 0
                conv_ok = 0
                for seglst_name in approved:
                    wav_name = wav_name_for_approved_seglst(seglst_name)
                    if wav_name is None or wav_name not in files:
                        print(
                            f"Warning: {conversation}/{seglst_name}: matching WAV "
                            f"{wav_name!r} not found — skipping speaker",
                            file=sys.stderr,
                        )
                        conv_failed += 1
                        failed += 1
                        continue
                    result = process_speaker_pair(
                        service,
                        conversation,
                        seglst_name,
                        files[seglst_name]["id"],
                        wav_name,
                        files[wav_name]["id"],
                        work_dir,
                        dry_run=args.dry_run,
                    )
                    if result is None:
                        conv_failed += 1
                        failed += 1
                        continue
                    conv_ok += 1
                    totals["dropped_unintelligible"] += result["dropped_unintelligible"]
                    totals["dropped_other_noise"] += result["dropped_other_noise"]
                    totals["speakers_ok"] += 1
                    totals["speakers_changed"] += result["changed"]
                    totals["speakers_empty"] += result["empty"]

                if conv_failed == 0 and not args.dry_run:
                    mark_conversation_complete(
                        progress,
                        progress_path,
                        conversation,
                        files=conv_ok,
                    )

        mode = "DRY RUN" if args.dry_run else "APPLIED"
        print()
        print(f"[{mode}] Summary")
        print(f"  Conversations requested: {len(conversations) + skipped_resume}")
        if args.resume:
            print(f"  Skipped (already complete): {skipped_resume}")
        print(f"  Not found on Drive: {len(missing_folders)}")
        for name in missing_folders:
            print(f"    - {name}")
        print(f"  No approved seglst files: {len(no_seglst)}")
        for name in no_seglst:
            print(f"    - {name}")
        print(f"  Speakers processed: {totals['speakers_ok']}")
        print(f"  Speakers with changes: {totals['speakers_changed']}")
        print(f"  Speakers uploaded as empty []: {totals['speakers_empty']}")
        print(
            f"  Segments dropped (unintelligible <= 1s): "
            f"{totals['dropped_unintelligible']}"
        )
        print(
            f"  Segments dropped (other-noise only): {totals['dropped_other_noise']}"
        )
        print(f"  Speakers failed / missing WAV: {failed}")
        return 1 if failed else 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
