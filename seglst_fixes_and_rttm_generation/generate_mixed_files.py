#!/usr/bin/env python3
"""Generate ``{conversation_id}_mixed.wav`` from channel WAVs on Drive via ffmpeg.

For each conversation subfolder under the configured Drive root, channel
``.wav`` files (not ``*_mixed.wav``) are first made sample-rate consistent
using the same majority rule as ``sample_rate_consistence.py`` (44100 vs
48000 Hz; tie/neither → 44100). Mismatched channels are resampled to
``pcm_s16le``, uploaded in place, then mixed with ffmpeg (``amix``,
``duration=longest``) at that folder target rate.

Existing mixed files are skipped unless ``--overwrite`` is passed.
Conversations with fewer than two channel WAVs are skipped.

Example::

    cd seglst_fixes_and_rttm_generation/
    python generate_mixed_files.py
    python generate_mixed_files.py --overwrite
    python generate_mixed_files.py NV-EN-SS14-CONVO34 NV-KO-SS13-CONVO30
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

DRIVE_FOLDER_ID = "1OwKt8wUnR6qRm3JwHhMPAev9iEuZGJfs"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
RATE_44100 = 44_100
RATE_48000 = 48_000
WAV_PROBE_BYTES = 4096
WAV_SUFFIX = ".wav"
MIXED_WAV_SUFFIX = "_mixed.wav"
FOLDER_MIME = "application/vnd.google-apps.folder"
WAV_MIME = "audio/wav"


def get_authenticated_drive_service():
    """Handles Service Account Impersonation to bypass Vertex VM scopes."""
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


def require_ffmpeg_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required tool not found on PATH: {tool}")


def list_drive_subfolders(service, parent_id: str) -> dict[str, str]:
    """Return {folder_name: folder_id} for immediate child folders on Drive."""
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
    """Return {file_name: {id, mimeType}} for immediate children of a Drive folder."""
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


def is_mixed_wav(filename: str) -> bool:
    return filename.lower().endswith(MIXED_WAV_SUFFIX)


def is_channel_wav(filename: str, mime_type: str) -> bool:
    if mime_type == FOLDER_MIME:
        return False
    if not filename.lower().endswith(WAV_SUFFIX):
        return False
    return not is_mixed_wav(filename)


def discover_channel_wavs(
    files: dict[str, dict[str, str]],
) -> list[tuple[str, str]]:
    """Return sorted (filename, file_id) pairs for non-mixed channel WAVs."""
    return sorted(
        (name, info["id"])
        for name, info in files.items()
        if is_channel_wav(name, info["mimeType"])
    )


def mixed_wav_name(conversation_id: str) -> str:
    return f"{conversation_id}{MIXED_WAV_SUFFIX}"


def download_drive_file(service, file_id: str, dest_path: Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with dest_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def download_wav_header(service, file_id: str) -> bytes:
    """Download only the first few KB of a Drive file (WAV header probe)."""
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    request.headers["Range"] = f"bytes=0-{WAV_PROBE_BYTES - 1}"
    content = request.execute()
    if not content:
        raise ValueError("Empty response when reading WAV header")
    return content[:WAV_PROBE_BYTES]


def parse_wav_sample_rate(header: bytes) -> int:
    """Return sample rate from a WAV header buffer."""
    if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file header")

    offset = 12
    while offset + 8 <= len(header):
        chunk_id = header[offset : offset + 4]
        chunk_size = int.from_bytes(header[offset + 4 : offset + 8], "little")
        offset += 8
        if chunk_id == b"fmt ":
            if offset + 8 > len(header):
                raise ValueError("Incomplete fmt chunk in WAV header")
            return int.from_bytes(header[offset + 4 : offset + 8], "little")
        offset += chunk_size + (chunk_size % 2)

    raise ValueError("fmt chunk not found in WAV header")


def probe_remote_sample_rate(service, file_id: str) -> int | None:
    """Read sample rate from Drive via header-only download, or None if unknown."""
    header = download_wav_header(service, file_id)
    try:
        return parse_wav_sample_rate(header)
    except ValueError:
        return None


def probe_sample_rate(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rate_str = result.stdout.strip()
    if not rate_str:
        raise ValueError(f"Could not read sample rate from {path}")
    return int(float(rate_str))


def choose_folder_target_rate(rates: list[int | None]) -> int:
    """Return 44100 or 48000 from majority of known rates. Tie → 44100."""
    n_44100 = sum(1 for rate in rates if rate == RATE_44100)
    n_48000 = sum(1 for rate in rates if rate == RATE_48000)
    if n_48000 > n_44100:
        return RATE_48000
    return RATE_44100


def resample_wav(input_path: Path, output_path: Path, target_rate: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(target_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def format_rate(rate: int | None) -> str:
    return "unknown" if rate is None else str(rate)


def upload_or_create_wav(
    service,
    folder_id: str,
    filename: str,
    local_path: Path,
    existing_file_id: str | None,
) -> None:
    media = MediaFileUpload(str(local_path), mimetype=WAV_MIME, resumable=True)
    if existing_file_id:
        service.files().update(
            fileId=existing_file_id,
            media_body=media,
            supportsAllDrives=True,
        ).execute()
        return

    service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


def mix_channel_wavs(
    input_paths: list[Path], output_path: Path, sample_rate: int
) -> None:
    """Mix channel WAVs with ffmpeg amix (duration=longest) at sample_rate pcm_s16le."""
    if len(input_paths) < 2:
        raise ValueError("Need at least two channel WAV inputs to mix")

    cmd: list[str] = ["ffmpeg", "-y"]
    for path in input_paths:
        cmd.extend(["-i", str(path)])

    n = len(input_paths)
    filter_complex = f"amix=inputs={n}:duration=longest:dropout_transition=0"
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )

    subprocess.run(cmd, check=True, capture_output=True, text=True)


def process_conversation(
    service,
    conversation: str,
    folder_id: str,
    work_dir: Path,
    *,
    overwrite: bool,
) -> str:
    """Process one conversation. Returns a result label for stats."""
    print(f"\n--- {conversation} ---")
    files = list_drive_files(service, folder_id)
    out_name = mixed_wav_name(conversation)
    existing_mixed_id = (
        files[out_name]["id"]
        if out_name in files and files[out_name]["mimeType"] != FOLDER_MIME
        else None
    )

    if existing_mixed_id is not None and not overwrite:
        print(f"Skipping {conversation}: {out_name} already exists (use --overwrite)")
        return "skipped_existing"

    channels = discover_channel_wavs(files)
    if len(channels) < 2:
        print(
            f"Warning: {conversation}: need at least 2 channel WAVs, "
            f"found {len(channels)} — skipping.",
            file=sys.stderr,
        )
        return "skipped_too_few_channels"

    probed: list[tuple[str, str, int | None]] = []
    for filename, file_id in channels:
        try:
            rate = probe_remote_sample_rate(service, file_id)
        except Exception as exc:
            print(
                f"Warning: {conversation}/{filename}: header probe failed ({exc})",
                file=sys.stderr,
            )
            rate = None
        probed.append((filename, file_id, rate))

    rate_summary = ", ".join(
        f"{filename}={format_rate(rate)}" for filename, _file_id, rate in probed
    )
    print(f"Channel rates: {rate_summary}")

    rates = [rate for _filename, _file_id, rate in probed]
    target_rate = choose_folder_target_rate(rates)
    n_44100 = sum(1 for rate in rates if rate == RATE_44100)
    n_48000 = sum(1 for rate in rates if rate == RATE_48000)
    print(
        f"Target {target_rate} Hz "
        f"(already 44100: {n_44100}, 48000: {n_48000}, "
        f"other/unknown: {len(rates) - n_44100 - n_48000})"
    )

    print(f"Mixing {len(channels)} channel WAV(s) -> {out_name} at {target_rate} Hz")
    conv_dir = work_dir / conversation
    conv_dir.mkdir(parents=True, exist_ok=True)
    channel_paths: list[Path] = []

    try:
        for index, (filename, file_id, header_rate) in enumerate(probed):
            safe_name = filename.replace("/", "_").replace("\\", "_")
            local_path = conv_dir / f"{index:02d}_{safe_name}"
            print(f"  Downloading: {filename}")
            download_drive_file(service, file_id, local_path)

            old_rate = header_rate
            if old_rate is None:
                old_rate = probe_sample_rate(local_path)

            if old_rate == target_rate:
                print(f"  {filename}: {old_rate} Hz (already at target)")
                channel_paths.append(local_path)
                continue

            resampled_path = conv_dir / f"{index:02d}_resampled_{safe_name}"
            resample_wav(local_path, resampled_path, target_rate)
            new_rate = probe_sample_rate(resampled_path)
            upload_or_create_wav(service, folder_id, filename, resampled_path, file_id)
            print(
                f"  {filename}: {old_rate} Hz -> {new_rate} Hz "
                "(resampled and uploaded)"
            )
            channel_paths.append(resampled_path)

        output_path = conv_dir / out_name
        print(
            f"  Running ffmpeg amix ({len(channel_paths)} inputs, {target_rate} Hz)..."
        )
        mix_channel_wavs(channel_paths, output_path, target_rate)

        if existing_mixed_id is not None:
            print(f"  Updating Drive file: {out_name}")
        else:
            print(f"  Uploading Drive file: {out_name}")
        upload_or_create_wav(
            service,
            folder_id,
            out_name,
            output_path,
            existing_mixed_id,
        )
        return "created" if existing_mixed_id is None else "overwritten"
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(
            f"Error: ffmpeg failed for {conversation}: {stderr}",
            file=sys.stderr,
        )
        return "failed"
    except Exception as exc:
        print(f"Error: failed to process {conversation}: {exc}", file=sys.stderr)
        return "failed"
    finally:
        if conv_dir.exists():
            shutil.rmtree(conv_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make channel WAVs sample-rate consistent (44100 or 48000 Hz), then "
            "generate {conversation_id}_mixed.wav with ffmpeg amix at that rate."
        )
    )
    parser.add_argument(
        "conversations",
        nargs="*",
        metavar="CONVERSATION",
        help=(
            "Conversation folder name(s) under the Drive root. "
            "Defaults to all subfolders."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing {conversation_id}_mixed.wav on Drive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(
        f"Generating mixed WAVs under Drive folder {DRIVE_FOLDER_ID}. "
        "Channel WAVs are first aligned to 44100 or 48000 Hz "
        "(majority per conversation; tie -> 44100), then mixed at that rate."
    )
    if args.overwrite:
        print("Mode: OVERWRITE existing mixed files when present")

    try:
        require_ffmpeg_tools()
        service = get_authenticated_drive_service()
        subfolders = list_drive_subfolders(service, DRIVE_FOLDER_ID)
        if not subfolders:
            print("Error: no conversation folders found on Drive.", file=sys.stderr)
            return 1

        if args.conversations:
            missing = [name for name in args.conversations if name not in subfolders]
            if missing:
                print(
                    f"Error: conversation folder(s) not found on Drive: "
                    f"{', '.join(missing)}",
                    file=sys.stderr,
                )
                return 1
            conversations = list(args.conversations)
        else:
            conversations = sorted(subfolders)

        print(f"Conversations to process: {len(conversations)}")

        stats: dict[str, int] = {}
        with tempfile.TemporaryDirectory(prefix="generate_mixed_files_") as tmp:
            work_dir = Path(tmp)
            for conversation in conversations:
                result = process_conversation(
                    service,
                    conversation,
                    subfolders[conversation],
                    work_dir,
                    overwrite=args.overwrite,
                )
                stats[result] = stats.get(result, 0) + 1

        print(
            f"\nDone. Created: {stats.get('created', 0)} | "
            f"Overwritten: {stats.get('overwritten', 0)} | "
            f"Skipped (exists): {stats.get('skipped_existing', 0)} | "
            f"Skipped (<2 channels): {stats.get('skipped_too_few_channels', 0)} | "
            f"Failed: {stats.get('failed', 0)}"
        )
        return 1 if stats.get("failed", 0) else 0
    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
