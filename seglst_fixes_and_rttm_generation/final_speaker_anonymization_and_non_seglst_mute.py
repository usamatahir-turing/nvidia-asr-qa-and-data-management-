#!/usr/bin/env python3
"""Anonymize speakers and mute non-seglst WAV regions for a delivery batch.

Reads conversation IDs from ``final_conversations_to_be_delivered.txt``, finds
each as a direct child of the configured source Drive folders, maps emails to
``SPK01`` / ``SPK02`` / ..., mutes audio outside seglst intervals (with a 50 ms
collar left unmuted), and uploads into the destination Drive folder.

Source Drive is read-only. ``*_mixed.wav`` files are ignored.

Example::

    cd seglst_fixes_and_rttm_generation/
    python final_speaker_anonymization_and_non_seglst_mute.py
    python final_speaker_anonymization_and_non_seglst_mute.py --dry-run
    python final_speaker_anonymization_and_non_seglst_mute.py --resume
    python final_speaker_anonymization_and_non_seglst_mute.py --overwrite
"""

from __future__ import annotations

import argparse
import array
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONVERSATIONS_FILE = SCRIPT_DIR / "final_conversations_to_be_delivered.txt"
DEFAULT_MAPPINGS_OUT = SCRIPT_DIR / "mappings.csv"

FOLDER_IDS = [
    "1d2qUnNWrWIooRwEE0iaFS9ktmrrTwWDW",
    "14ckZ3IbOP_vG2Q9A-qbwtf5Zygo1aZrD",
]
DESTINATION_FOLDER = "1LfJK0kdu6cOU5FOSqeQ2J2KNSPlLb4qp"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"

SEGLST_SUFFIX = ".seglst.json"
RTTM_SUFFIX = ".rttm"
WAV_SUFFIX = ".wav"
FOLDER_MIME = "application/vnd.google-apps.folder"
COLLAR_SEC = 0.05
NON_DELIVERY_SEGLST_SUFFIXES = (
    f"_approved{SEGLST_SUFFIX}",
    f"_fixed{SEGLST_SUFFIX}",
)
NON_DELIVERY_RTTM_SUFFIXES = ("_approved.rttm", "_fixed.rttm")


@dataclass(frozen=True)
class SpeakerMapping:
    conversation: str
    label: str
    email: str

    @property
    def spk_name(self) -> str:
        return f"SPK{self.label}"


def warn(message: str) -> None:
    print(message, file=sys.stderr)


def get_authenticated_drive_service():
    """Authenticate to Drive using service-account impersonation."""
    print("Authenticating...", flush=True)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/jupyter/.config/gcloud/application_default_credentials.json"
    )

    base_credentials, _project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    base_credentials.refresh(Request())

    url = (
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
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
            f"Authentication failed: API error {response.status_code}: {response.text}"
        )

    creds = Oauth2Credentials(response.json()["accessToken"])
    return build("drive", "v3", credentials=creds)


def load_conversation_names(batch_path: Path) -> list[str]:
    if not batch_path.is_file():
        raise FileNotFoundError(f"Conversations file not found: {batch_path}")

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
                f"Warning: duplicate conversation on line {line_number}: {line!r}; "
                "using first occurrence",
                file=sys.stderr,
            )
            continue
        seen.add(line)
        names.append(line)

    if not names:
        raise ValueError(f"No conversation names found in: {batch_path}")
    return names


def get_drive_items(service, folder_id: str) -> dict[str, dict]:
    items: dict[str, dict] = {}
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for file_info in response.get("files", []):
            items[file_info["name"]] = file_info
        page_token = response.get("nextPageToken")
        if page_token is None:
            break
    return items


def list_conversation_folders(service, root_folder_id: str) -> dict[str, dict]:
    items = get_drive_items(service, root_folder_id)
    return {
        name: info
        for name, info in items.items()
        if info.get("mimeType") == FOLDER_MIME
    }


def index_source_conversations(
    service, folder_ids: list[str]
) -> dict[str, list[tuple[str, dict]]]:
    """Map conversation name -> list of (source_root_id, folder_info)."""
    found: dict[str, list[tuple[str, dict]]] = {}
    for root_id in folder_ids:
        folders = list_conversation_folders(service, root_id)
        print(
            f"Source folder {root_id}: {len(folders)} conversation subfolder(s)",
            flush=True,
        )
        for name, info in folders.items():
            found.setdefault(name, []).append((root_id, info))
    return found


def get_or_create_folder(
    service, name: str, parent_id: str, cache: dict[str, dict]
) -> str:
    if name in cache and cache[name].get("mimeType") == FOLDER_MIME:
        return cache[name]["id"]

    folder_id = (
        service.files()
        .create(
            body={
                "name": name,
                "mimeType": FOLDER_MIME,
                "parents": [parent_id],
            },
            fields="id",
            supportsAllDrives=True,
        )
        .execute()["id"]
    )
    cache[name] = {"id": folder_id, "name": name, "mimeType": FOLDER_MIME}
    return folder_id


def download_drive_file(service, file_id: str) -> bytes:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _status, done = downloader.next_chunk()
    return buffer.getvalue()


def upload_drive_file(
    service,
    parent_id: str,
    filename: str,
    data: bytes,
    mime_type: str,
    dest_items: dict[str, dict],
    *,
    dry_run: bool,
    overwrite: bool,
) -> str:
    """Upload or skip. Returns 'uploaded', 'updated', 'skipped', or 'dry-run'."""
    exists = filename in dest_items
    if exists and not overwrite:
        print(f"  Skipping {filename} (already on destination)", flush=True)
        return "skipped"

    if dry_run:
        action = "overwrite" if exists else "upload"
        print(f"  [dry-run] would {action} {filename} ({len(data)} bytes)")
        return "dry-run"

    media = MediaIoBaseUpload(
        io.BytesIO(data),
        mimetype=mime_type,
        resumable=True,
    )
    if exists:
        service.files().update(
            fileId=dest_items[filename]["id"],
            media_body=media,
            supportsAllDrives=True,
        ).execute()
        print(f"  Updated {filename}", flush=True)
        return "updated"

    created = (
        service.files()
        .create(
            body={"name": filename, "parents": [parent_id]},
            media_body=media,
            fields="id, name",
            supportsAllDrives=True,
        )
        .execute()
    )
    dest_items[created["name"]] = created
    print(f"  Uploaded {filename}", flush=True)
    return "uploaded"


def is_delivery_seglst(filename: str) -> bool:
    return filename.endswith(SEGLST_SUFFIX) and not any(
        filename.endswith(suffix) for suffix in NON_DELIVERY_SEGLST_SUFFIXES
    )


def is_delivery_rttm(filename: str) -> bool:
    return filename.endswith(RTTM_SUFFIX) and not any(
        filename.endswith(suffix) for suffix in NON_DELIVERY_RTTM_SUFFIXES
    )


def is_mixed_wav(filename: str) -> bool:
    return filename.endswith(WAV_SUFFIX) and "_mixed" in filename


def discover_candidate_emails(source_items: dict[str, dict]) -> set[str]:
    emails: set[str] = set()
    for filename in source_items:
        if is_mixed_wav(filename):
            continue
        if is_delivery_seglst(filename):
            emails.add(filename[: -len(SEGLST_SUFFIX)])
        elif is_delivery_rttm(filename):
            emails.add(filename[: -len(RTTM_SUFFIX)])
    return emails


def required_source_filenames(email: str) -> tuple[str, str, str]:
    return (
        f"{email}{SEGLST_SUFFIX}",
        f"{email}{RTTM_SUFFIX}",
        f"{email}{WAV_SUFFIX}",
    )


def destination_filenames(spk_name: str) -> tuple[str, str, str]:
    return (
        f"{spk_name}{SEGLST_SUFFIX}",
        f"{spk_name}{RTTM_SUFFIX}",
        f"{spk_name}{WAV_SUFFIX}",
    )


def expected_destination_filenames(
    speaker_mappings: list[SpeakerMapping],
) -> set[str]:
    expected: set[str] = set()
    for mapping in speaker_mappings:
        expected.update(destination_filenames(mapping.spk_name))
    return expected


def build_speaker_mappings(
    conversation: str,
    source_items: dict[str, dict],
) -> list[SpeakerMapping]:
    mappings: list[SpeakerMapping] = []
    spk_index = 1

    for email in sorted(discover_candidate_emails(source_items)):
        seglst_name, rttm_name, wav_name = required_source_filenames(email)
        missing = [
            name
            for name in (seglst_name, rttm_name, wav_name)
            if name not in source_items
        ]
        if missing:
            warn(
                f"Warning: {conversation}: skipping {email} — "
                f"missing source file(s): {', '.join(missing)}"
            )
            continue

        label = f"{spk_index:02d}"
        mappings.append(
            SpeakerMapping(conversation=conversation, label=label, email=email)
        )
        spk_index += 1

    return mappings


def write_mappings_csv(path: Path, mappings: list[SpeakerMapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["Conversation", "Speaker_label", "Speaker"],
        )
        writer.writeheader()
        for mapping in mappings:
            writer.writerow(
                {
                    "Conversation": mapping.conversation,
                    "Speaker_label": mapping.label,
                    "Speaker": mapping.email,
                }
            )


def transform_seglst(content: bytes, new_speaker: str) -> tuple[bytes, list[dict[str, Any]]]:
    data = json.loads(content.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("seglst JSON must be a list")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"seglst segment #{index} is not an object")
        item["speaker"] = new_speaker

    output = json.dumps(data, ensure_ascii=False, indent=4) + "\n"
    return output.encode("utf-8"), data


def transform_rttm(content: bytes, old_speaker: str, new_speaker: str) -> bytes:
    lines_out: list[str] = []
    text = content.decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            lines_out.append(line)
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            if parts[1] == old_speaker:
                parts[1] = new_speaker
            if parts[7] == old_speaker:
                parts[7] = new_speaker
            lines_out.append(" ".join(parts))
        else:
            lines_out.append(line)
    return ("\n".join(lines_out) + ("\n" if text.endswith("\n") else "")).encode("utf-8")


def parse_segment_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def keep_ranges_from_seglst(
    segments: list[dict[str, Any]], collar_sec: float = COLLAR_SEC
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for item in segments:
        start = parse_segment_time(item.get("start_time", 0)) - collar_sec
        end = parse_segment_time(item.get("end_time", 0)) + collar_sec
        start = max(0.0, start)
        if end > start:
            ranges.append((start, end))
    return merge_ranges(ranges)


def mute_ranges_from_keep(
    keep: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    mute: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in keep:
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        if start > cursor:
            mute.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        mute.append((cursor, duration))
    return mute


def wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as reader:
            rate = reader.getframerate()
            frames = reader.getnframes()
        if rate <= 0:
            raise ValueError("WAV has non-positive sample rate")
        return frames / float(rate)
    except (wave.Error, ValueError):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())


def mute_wav_pcm(input_path: Path, output_path: Path, ranges: list[tuple[float, float]]) -> None:
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
    if not ranges:
        shutil.copy2(input_path, output_path)
        return
    try:
        mute_wav_pcm(input_path, output_path, ranges)
    except (wave.Error, ValueError):
        mute_wav_ffmpeg(input_path, output_path, ranges)


def mute_wav_bytes(wav_bytes: bytes, segments: list[dict[str, Any]], work_dir: Path) -> bytes:
    src_path = work_dir / "source.wav"
    dst_path = work_dir / "muted.wav"
    src_path.write_bytes(wav_bytes)

    duration = wav_duration_seconds(src_path)
    keep = keep_ranges_from_seglst(segments)
    mute_ranges = mute_ranges_from_keep(keep, duration)
    mute_wav(src_path, dst_path, mute_ranges)
    return dst_path.read_bytes()


def conversation_complete_on_dest(
    dest_items: dict[str, dict],
    speaker_mappings: list[SpeakerMapping],
) -> bool:
    expected = expected_destination_filenames(speaker_mappings)
    return bool(expected) and all(name in dest_items for name in expected)


def process_conversation(
    service,
    conversation: str,
    speaker_mappings: list[SpeakerMapping],
    source_folder_id: str,
    dest_parent_id: str,
    *,
    dry_run: bool,
    overwrite: bool,
    resume: bool,
) -> str:
    """Return 'copied', 'skipped-resume', or 'failed'."""
    print(f"\n--- {conversation} ---", flush=True)
    source_items = get_drive_items(service, source_folder_id)

    dest_parent_items = get_drive_items(service, dest_parent_id)
    dest_folder_exists = (
        conversation in dest_parent_items
        and dest_parent_items[conversation].get("mimeType") == FOLDER_MIME
    )

    if dest_folder_exists:
        dest_folder_id = dest_parent_items[conversation]["id"]
        dest_items = get_drive_items(service, dest_folder_id)
    else:
        dest_folder_id = None
        dest_items = {}

    if resume and conversation_complete_on_dest(dest_items, speaker_mappings):
        print(
            f"Skipping {conversation} (already complete on destination)",
            flush=True,
        )
        return "skipped-resume"

    if dry_run:
        if dest_folder_id is None:
            print(f"  [dry-run] would create destination folder {conversation}")
        for mapping in speaker_mappings:
            _src_seglst, _src_rttm, _src_wav = required_source_filenames(mapping.email)
            dst_seglst, dst_rttm, dst_wav = destination_filenames(mapping.spk_name)
            print(f"  {mapping.email} -> {mapping.spk_name}", flush=True)
            for dest_name in (dst_seglst, dst_rttm, dst_wav):
                exists = dest_name in dest_items
                if exists and not overwrite:
                    print(f"  Skipping {dest_name} (already on destination)", flush=True)
                elif exists:
                    print(f"  [dry-run] would overwrite {dest_name}", flush=True)
                else:
                    print(f"  [dry-run] would upload {dest_name}", flush=True)
        print(f"Completed {conversation} ({len(speaker_mappings)} speaker(s)) [dry-run]")
        return "copied"

    if dest_folder_id is None:
        dest_folder_id = get_or_create_folder(
            service, conversation, dest_parent_id, dest_parent_items
        )
        dest_items = get_drive_items(service, dest_folder_id)

    with tempfile.TemporaryDirectory(prefix="spk_anon_mute_") as tmp:
        tmp_root = Path(tmp)
        for mapping in speaker_mappings:
            src_seglst, src_rttm, src_wav = required_source_filenames(mapping.email)
            dst_seglst, dst_rttm, dst_wav = destination_filenames(mapping.spk_name)

            print(f"Copying {mapping.email} -> {mapping.spk_name}", flush=True)

            need_seglst = overwrite or dst_seglst not in dest_items
            need_rttm = overwrite or dst_rttm not in dest_items
            need_wav = overwrite or dst_wav not in dest_items

            seglst_bytes = None
            segments: list[dict[str, Any]] = []
            if need_seglst or need_wav:
                seglst_bytes = download_drive_file(
                    service, source_items[src_seglst]["id"]
                )
                transformed_seglst, segments = transform_seglst(
                    seglst_bytes, mapping.spk_name
                )
            else:
                transformed_seglst = b""

            if need_seglst:
                upload_drive_file(
                    service,
                    dest_folder_id,
                    dst_seglst,
                    transformed_seglst,
                    "application/json",
                    dest_items,
                    dry_run=False,
                    overwrite=overwrite,
                )
            else:
                print(f"  Skipping {dst_seglst} (already on destination)", flush=True)

            if need_rttm:
                rttm_bytes = download_drive_file(service, source_items[src_rttm]["id"])
                transformed_rttm = transform_rttm(
                    rttm_bytes, mapping.email, mapping.spk_name
                )
                upload_drive_file(
                    service,
                    dest_folder_id,
                    dst_rttm,
                    transformed_rttm,
                    "text/plain",
                    dest_items,
                    dry_run=False,
                    overwrite=overwrite,
                )
            else:
                print(f"  Skipping {dst_rttm} (already on destination)", flush=True)

            if need_wav:
                wav_bytes = download_drive_file(service, source_items[src_wav]["id"])
                speaker_tmp = tmp_root / mapping.spk_name
                speaker_tmp.mkdir(parents=True, exist_ok=True)
                muted_wav = mute_wav_bytes(wav_bytes, segments, speaker_tmp)
                print(
                    f"  Muted non-seglst regions in {dst_wav} "
                    f"(collar={COLLAR_SEC * 1000:.0f} ms)",
                    flush=True,
                )
                upload_drive_file(
                    service,
                    dest_folder_id,
                    dst_wav,
                    muted_wav,
                    "audio/wav",
                    dest_items,
                    dry_run=False,
                    overwrite=overwrite,
                )
            else:
                print(f"  Skipping {dst_wav} (already on destination)", flush=True)

    print(f"Completed {conversation} ({len(speaker_mappings)} speaker(s))")
    return "copied"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map speakers to SPK labels, mute non-seglst WAV regions with a "
            "50 ms collar, and upload to the destination Drive folder."
        )
    )
    parser.add_argument(
        "--conversations-file",
        type=Path,
        default=DEFAULT_CONVERSATIONS_FILE,
        help=(
            "Newline-separated conversation IDs "
            f"(default: {DEFAULT_CONVERSATIONS_FILE.name})"
        ),
    )
    parser.add_argument(
        "--mappings-out",
        type=Path,
        default=DEFAULT_MAPPINGS_OUT,
        help=f"Path for generated mappings CSV (default: {DEFAULT_MAPPINGS_OUT.name})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write mappings.csv and report uploads without writing to destination Drive",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination files that already exist (default: skip them)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip conversations that already have all expected SPK files on destination"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        conversations = load_conversation_names(args.conversations_file)
    except (FileNotFoundError, ValueError) as exc:
        warn(f"Error: {exc}")
        return 1

    try:
        service = get_authenticated_drive_service()
    except Exception as exc:
        warn(f"Error: authentication failed: {exc}")
        return 1

    print(f"Conversations file: {args.conversations_file}")
    print(f"Source Drive folders: {', '.join(FOLDER_IDS)}")
    print(f"Destination Drive folder: {DESTINATION_FOLDER}")
    if args.dry_run:
        print("Mode: DRY RUN (destination Drive will not be modified)")
    if args.overwrite:
        print("Mode: OVERWRITE (existing destination files will be replaced)")
    else:
        print("Mode: SKIP EXISTING (files already on destination will not be replaced)")
    if args.resume:
        print("Mode: RESUME (skip conversations already complete on destination)")
    print(f"Conversations listed: {len(conversations)}", flush=True)

    source_index = index_source_conversations(service, FOLDER_IDS)

    all_mappings: list[SpeakerMapping] = []
    planned: list[tuple[str, list[SpeakerMapping], str]] = []
    failures = 0
    skipped_lookup = 0

    for conversation in conversations:
        matches = source_index.get(conversation, [])
        if not matches:
            warn(
                f"Warning: {conversation}: not found as a direct subfolder of any "
                f"source folder; skipped"
            )
            skipped_lookup += 1
            continue
        if len(matches) > 1:
            locations = ", ".join(root_id for root_id, _info in matches)
            warn(
                f"Warning: {conversation}: found in multiple source folders "
                f"({locations}); skipped"
            )
            skipped_lookup += 1
            continue

        _root_id, folder_info = matches[0]
        source_items = get_drive_items(service, folder_info["id"])
        mappings = build_speaker_mappings(conversation, source_items)
        if not mappings:
            warn(
                f"Warning: {conversation}: no speakers with a complete seglst/rttm/wav "
                f"set found; skipped"
            )
            failures += 1
            continue

        all_mappings.extend(mappings)
        planned.append((conversation, mappings, folder_info["id"]))

    if not planned:
        warn("Error: no conversations could be mapped")
        return 1

    write_mappings_csv(args.mappings_out, all_mappings)
    print(f"Wrote {len(all_mappings)} mapping row(s) to {args.mappings_out}")

    successes = 0
    skipped_resume = 0
    for conversation, mappings, source_folder_id in planned:
        try:
            result = process_conversation(
                service,
                conversation,
                mappings,
                source_folder_id,
                DESTINATION_FOLDER,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                resume=args.resume,
            )
        except Exception as exc:
            warn(f"Error: {conversation}: {exc}")
            failures += 1
            continue
        if result == "copied":
            successes += 1
        elif result == "skipped-resume":
            skipped_resume += 1
        else:
            failures += 1

    print()
    print(f"Completed: {successes} conversation(s)")
    if skipped_lookup:
        print(f"Skipped  : {skipped_lookup} conversation(s) (not found or duplicate source)")
    if args.resume:
        print(f"Skipped  : {skipped_resume} conversation(s) (already complete on destination)")
    if failures:
        print(f"Failed   : {failures} conversation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
