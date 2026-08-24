#!/usr/bin/env python3
"""Apply ``fix_seglst_tokens.py`` to ``*_approved.seglst.json`` files on Drive.

Walks conversation subfolders under the Gecko / Files-for-Gecko Drive root,
downloads each ``*_approved.seglst.json``, runs the same token / session /
speaker / zero-duration fixes as ``fix_seglst_tokens.py``, and overwrites the
same Drive file. Local copies live in a temp directory and are deleted after
each file.

Example::

    cd seglst_fixes_and_rttm_generation/
    python fix_seglst_tokens_approved_in_drive.py
    python fix_seglst_tokens_approved_in_drive.py --dry-run
    python fix_seglst_tokens_approved_in_drive.py NV-KO-SS03-CONVO08
    python fix_seglst_tokens_approved_in_drive.py NV-EN-SS14-CONVO34 NV-KO-SS13-CONVO30
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from fix_seglst_tokens import (
    SEGLST_GLOB,
    FileReport,
    expected_speaker_from_path,
    process_file,
)

DRIVE_FOLDER_ID = "1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
FOLDER_MIME = "application/vnd.google-apps.folder"
JSON_MIME = "application/json"
APPROVED_SUFFIX = "_approved.seglst.json"


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


def discover_approved_seglst_files(
    files: dict[str, dict[str, str]],
) -> list[tuple[str, str]]:
    """Return sorted (filename, file_id) for ``*_approved.seglst.json`` files."""
    return sorted(
        (name, info["id"])
        for name, info in files.items()
        if name.endswith(APPROVED_SUFFIX) and info["mimeType"] != FOLDER_MIME
    )


def download_drive_file(service, file_id: str, dest_path: Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with dest_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def upload_drive_file(service, file_id: str, local_path: Path) -> None:
    media = MediaFileUpload(str(local_path), mimetype=JSON_MIME, resumable=True)
    service.files().update(
        fileId=file_id,
        media_body=media,
        supportsAllDrives=True,
    ).execute()


def print_file_warnings(report: FileReport, src_path: Path) -> None:
    if report.multiple_speakers:
        print(
            f"  Warning: multiple speakers in {src_path.name}: "
            f"{', '.join(report.speakers_found)}",
            file=sys.stderr,
        )
    if report.speaker_changed:
        print(
            f"  Warning: speaker corrected to "
            f"{expected_speaker_from_path(src_path)!r} in {report.speaker_changed} "
            f"segment(s)",
            file=sys.stderr,
        )


def process_drive_file(
    service,
    conversation: str,
    filename: str,
    file_id: str,
    work_dir: Path,
    *,
    dry_run: bool,
) -> FileReport | None:
    """Download, fix, overwrite Drive, then delete local copies."""
    label = f"{conversation}/{filename}"
    # process_file uses the parent folder name as session_id / task_id.
    conv_dir = work_dir / conversation
    conv_dir.mkdir(parents=True, exist_ok=True)
    src_path = conv_dir / filename
    dst_path = conv_dir / f"fixed_{filename}"

    try:
        print(f"  Downloading {filename}...")
        download_drive_file(service, file_id, src_path)
        report = process_file(src_path, dst_path, dry_run=False)
        print_file_warnings(report, src_path)

        if dry_run:
            print(f"  [dry-run] would overwrite {label}")
        else:
            print(f"  Uploading {filename}...")
            upload_drive_file(service, file_id, dst_path)
        return report
    except Exception as exc:
        print(f"Error: failed to process {label}: {exc}", file=sys.stderr)
        return None
    finally:
        for path in (src_path, dst_path):
            if path.exists():
                path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply fix_seglst_tokens.py to *_approved.seglst.json files on Drive "
            "and overwrite them in place."
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
        "--dry-run",
        action="store_true",
        help="Download and report fixes without overwriting Drive files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(
        f"Fixing *_approved.seglst.json files on Drive folder {DRIVE_FOLDER_ID}..."
    )
    if args.dry_run:
        print("Mode: DRY RUN (Drive files will not be overwritten)")

    try:
        service = get_authenticated_drive_service()
        subfolders = list_drive_subfolders(service, DRIVE_FOLDER_ID)
        if not subfolders:
            print("Error: no conversation folders found on Drive.", file=sys.stderr)
            return 1

        if args.conversations:
            conversations = list(dict.fromkeys(args.conversations))
        else:
            conversations = sorted(subfolders)

        print(f"Conversations requested: {len(conversations)}")

        reports: list[FileReport] = []
        failed_files = 0
        missing_folders: list[str] = []
        no_seglst: list[str] = []

        with tempfile.TemporaryDirectory(prefix="fix_seglst_approved_drive_") as tmp:
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
                approved = discover_approved_seglst_files(files)
                if not approved:
                    print(
                        f"Warning: no {SEGLST_GLOB} files in {conversation!r}",
                        file=sys.stderr,
                    )
                    no_seglst.append(conversation)
                    continue

                print(f"Found {len(approved)} {SEGLST_GLOB} file(s)")
                for filename, file_id in approved:
                    report = process_drive_file(
                        service,
                        conversation,
                        filename,
                        file_id,
                        work_dir,
                        dry_run=args.dry_run,
                    )
                    if report is None:
                        failed_files += 1
                    else:
                        reports.append(report)

        conversations_processed = len({r.task_id for r in reports})
        mode = "DRY RUN" if args.dry_run else "APPLIED"
        print()
        print(f"[{mode}] Drive batch summary")
        print(f"  Conversations requested: {len(conversations)}")
        print(f"  Conversations processed: {conversations_processed}")
        print(f"  Not found on Drive: {len(missing_folders)}")
        for conversation_id in missing_folders:
            print(f"    - {conversation_id}")
        print(f"  No {SEGLST_GLOB} files: {len(no_seglst)}")
        for conversation_id in no_seglst:
            print(f"    - {conversation_id}")
        print(f"  Seglst files processed: {len(reports)}")
        print(f"  Seglst files failed: {failed_files}")
        print(f"  Segments with words changes: {sum(r.words_changed for r in reports)}")
        print(
            f"  Segments with session_id changes: "
            f"{sum(r.session_id_changed for r in reports)}"
        )
        print(
            f"  Segments with speaker changes: "
            f"{sum(r.speaker_changed for r in reports)}"
        )
        print(
            f"  Segments with zero-duration fixes: "
            f"{sum(r.duration_fixed for r in reports)}"
        )
        unfixable = sum(r.duration_unfixable for r in reports)
        if unfixable:
            print(f"  Zero-duration segments not fixable (overlap): {unfixable}")
        print(
            f"  Files with multiple speakers: "
            f"{sum(1 for r in reports if r.multiple_speakers)}"
        )
        print(f"  Files with changes applied: {sum(1 for r in reports if r.changed)}")
        return 1 if failed_files else 0
    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
