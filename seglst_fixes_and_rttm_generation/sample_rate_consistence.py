#!/usr/bin/env python3
"""Make WAV sample rates consistent per conversation folder on Drive.

Walks each conversation subfolder under the pre-delivery Drive root. For each
folder, header-probes every ``.wav`` and chooses **44100 Hz** or **48000 Hz**
from whichever is already more common (tie or neither present → 44100 Hz).
Files already at the target are skipped without a full download; others are
resampled with ffmpeg (``pcm_s16le``) and uploaded in place.

Example::

    cd seglst_fixes_and_rttm_generation/
    python sample_rate_consistence.py
"""

from __future__ import annotations

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

PRE_DELIVERY_DRIVE_FOLDER_ID = "1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
RATE_44100 = 44_100
RATE_48000 = 48_000
WAV_PROBE_BYTES = 4096
WAV_SUFFIX = ".wav"
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

    response = requests.post(url, headers=headers, json=payload)
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
        res = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
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
        res = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
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


def upload_drive_file(service, file_id: str, local_path: Path) -> None:
    media = MediaFileUpload(str(local_path), mimetype=WAV_MIME, resumable=True)
    service.files().update(
        fileId=file_id,
        media_body=media,
        supportsAllDrives=True,
    ).execute()


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


def process_wav(
    service,
    conversation: str,
    filename: str,
    file_id: str,
    work_dir: Path,
    target_rate: int,
    old_rate: int | None,
) -> str:
    """Process one wav file. Returns 'converted', 'skipped', or 'failed'."""
    label = f"{conversation}/{filename}"
    input_path = work_dir / "input.wav"
    output_path = work_dir / "output.wav"

    try:
        if old_rate == target_rate:
            print(f"{label}: {old_rate} Hz (header check, no download)")
            return "skipped"

        download_drive_file(service, file_id, input_path)
        if old_rate is None:
            old_rate = probe_sample_rate(input_path)
            if old_rate == target_rate:
                print(f"{label}: {old_rate} Hz -> {target_rate} Hz (no change)")
                return "skipped"

        resample_wav(input_path, output_path, target_rate)
        new_rate = probe_sample_rate(output_path)
        upload_drive_file(service, file_id, output_path)
        print(f"{label}: {old_rate} Hz -> {new_rate} Hz")
        return "converted"
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(f"Error: ffmpeg/ffprobe failed for {label}: {stderr}", file=sys.stderr)
        return "failed"
    except Exception as exc:
        print(f"Error: failed to process {label}: {exc}", file=sys.stderr)
        return "failed"
    finally:
        for path in (input_path, output_path):
            if path.exists():
                path.unlink()


def main() -> int:
    print(
        f"Making WAV sample rates consistent on pre-delivery Drive "
        f"({PRE_DELIVERY_DRIVE_FOLDER_ID}); per folder 44100 or 48000 Hz..."
    )

    try:
        require_ffmpeg_tools()
        service = get_authenticated_drive_service()
        conversations = list_drive_subfolders(service, PRE_DELIVERY_DRIVE_FOLDER_ID)
        if not conversations:
            print("Error: no conversation folders found on pre-delivery Drive.", file=sys.stderr)
            return 1

        stats = {"converted": 0, "skipped": 0, "failed": 0}

        with tempfile.TemporaryDirectory(prefix="sample_rate_consistence_") as tmp:
            work_dir = Path(tmp)
            for conversation in sorted(conversations):
                folder_id = conversations[conversation]
                files = list_drive_files(service, folder_id)
                wav_files = sorted(
                    name
                    for name, info in files.items()
                    if name.lower().endswith(WAV_SUFFIX)
                    and info["mimeType"] != FOLDER_MIME
                )
                if not wav_files:
                    continue

                print(f"\n--- {conversation} ---")
                probed: list[tuple[str, int | None]] = []
                for filename in wav_files:
                    file_id = files[filename]["id"]
                    try:
                        rate = probe_remote_sample_rate(service, file_id)
                    except Exception as exc:
                        print(
                            f"Warning: {conversation}/{filename}: "
                            f"header probe failed ({exc})",
                            file=sys.stderr,
                        )
                        rate = None
                    probed.append((filename, rate))

                rates = [rate for _, rate in probed]
                target_rate = choose_folder_target_rate(rates)
                n_44100 = sum(1 for rate in rates if rate == RATE_44100)
                n_48000 = sum(1 for rate in rates if rate == RATE_48000)
                print(
                    f"Target {target_rate} Hz "
                    f"(already 44100: {n_44100}, 48000: {n_48000}, "
                    f"other/unknown: {len(rates) - n_44100 - n_48000})"
                )

                for filename, old_rate in probed:
                    result = process_wav(
                        service,
                        conversation,
                        filename,
                        files[filename]["id"],
                        work_dir,
                        target_rate,
                        old_rate,
                    )
                    stats[result] += 1

        print(
            f"\nDone. Converted: {stats['converted']} | "
            f"Already at folder target: {stats['skipped']} | "
            f"Failed: {stats['failed']}"
        )
        return 1 if stats["failed"] else 0
    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
