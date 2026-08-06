#!/usr/bin/env python3
"""Recursively mirror a Google Drive folder locally.

Default mode downloads only files modified within the last N days (``--days``,
default 30).

Batch mode (``--latest-batch``) syncs only conversation subfolders listed in
``latest_batch.txt``, with no modification-date filter.

Examples::

    python download_and_update_data.py
    python download_and_update_data.py --days 14
    python download_and_update_data.py --latest-batch
    python download_and_update_data.py --latest-batch --batch-file path/to/list.txt
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_FILE = SCRIPT_DIR / "latest_batch.txt"
FOLDER_MIME = "application/vnd.google-apps.folder"


def get_authenticated_drive_service():
    """Handles Service Account Impersonation to bypass Vertex VM scopes."""
    print("Authenticating...")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/jupyter/.config/gcloud/application_default_credentials.json"
    )

    base_credentials, _project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    base_credentials.refresh(Request())

    target_service_account = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
    url = (
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
        f"{target_service_account}:generateAccessToken"
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
    """Load unique subfolder names from a newline-separated batch file."""
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


def list_drive_children(drive_service, folder_id: str) -> list[dict]:
    """Return all non-trashed children of a Drive folder."""
    children: list[dict] = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        res = (
            drive_service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        children.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if page_token is None:
            break
    return children


def mirror_folder_sync_recursive(
    drive_service,
    root_folder_id: str,
    root_destination_dir: str,
    *,
    days: int = 30,
    batch_folders: list[str] | None = None,
):
    stats = {"downloaded": 0, "skipped": 0, "excluded_old": 0, "deleted": 0, "missing": 0}
    valid_local_paths: set[str] = set()
    apply_date_filter = batch_folders is None
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days) if apply_date_filter else None
    )

    print(f"\nStarting recursive sync for root folder ID: {root_folder_id}")
    print(f"Destination: {root_destination_dir}")
    if batch_folders is not None:
        print(
            f"Mode: latest-batch ({len(batch_folders)} folder(s), "
            "no modification-date filter)"
        )
        print(f"Folders: {', '.join(batch_folders)}\n")
    else:
        assert cutoff is not None
        print(
            f"Sync window: files modified on or after {cutoff.isoformat()} "
            f"({days} day(s))\n"
        )

    def process_folder(drive_folder_id: str, current_local_dir: str) -> None:
        os.makedirs(current_local_dir, exist_ok=True)
        valid_local_paths.add(current_local_dir)

        for f in list_drive_children(drive_service, drive_folder_id):
            file_id = f["id"]
            file_name = f["name"]
            file_path = os.path.join(current_local_dir, file_name)

            if f["mimeType"] == FOLDER_MIME:
                valid_local_paths.add(file_path)
                process_folder(file_id, file_path)
                continue

            if "application/vnd.google-apps" in f["mimeType"]:
                continue

            drive_mtime = datetime.fromisoformat(
                f["modifiedTime"].replace("Z", "+00:00")
            )

            if cutoff is not None and drive_mtime < cutoff:
                stats["excluded_old"] += 1
                continue

            valid_local_paths.add(file_path)

            needs_download = True
            if os.path.exists(file_path):
                local_mtime = datetime.fromtimestamp(
                    os.path.getmtime(file_path), tz=timezone.utc
                )
                if local_mtime >= drive_mtime:
                    needs_download = False

            if not needs_download:
                stats["skipped"] += 1
                continue

            print(f"Downloading: {file_path}...")
            request = drive_service.files().get_media(fileId=file_id)
            with io.FileIO(file_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _status, done = downloader.next_chunk()

            drive_mtime_ts = drive_mtime.timestamp()
            os.utime(file_path, (drive_mtime_ts, drive_mtime_ts))
            stats["downloaded"] += 1

    root_destination_dir = os.path.abspath(root_destination_dir)
    os.makedirs(root_destination_dir, exist_ok=True)
    valid_local_paths.add(root_destination_dir)

    if batch_folders is not None:
        # Protect sibling local folders that are outside this batch from cleanup.
        for name in os.listdir(root_destination_dir):
            path = os.path.join(root_destination_dir, name)
            if os.path.isdir(path) and name not in batch_folders:
                valid_local_paths.add(path)

        children = list_drive_children(drive_service, root_folder_id)
        folders_by_name = {
            item["name"]: item
            for item in children
            if item.get("mimeType") == FOLDER_MIME
        }

        for folder_name in batch_folders:
            if folder_name not in folders_by_name:
                stats["missing"] += 1
                print(
                    f"Warning: folder not found on Drive: {folder_name}",
                    file=sys.stderr,
                )
                continue
            drive_folder = folders_by_name[folder_name]
            local_folder = os.path.join(root_destination_dir, folder_name)
            print(f"--- Syncing {folder_name} ---")
            process_folder(drive_folder["id"], local_folder)
    else:
        process_folder(root_folder_id, root_destination_dir)

    print("\nStarting local cleanup...")
    for root, dirs, files in os.walk(root_destination_dir, topdown=False):
        for name in files:
            if name.startswith("."):
                continue
            file_path = os.path.join(root, name)
            if file_path not in valid_local_paths:
                # In batch mode, only clean inside synced batch folders.
                if batch_folders is not None:
                    rel = os.path.relpath(file_path, root_destination_dir)
                    top = rel.split(os.sep, 1)[0]
                    if top not in batch_folders:
                        continue
                os.remove(file_path)
                stats["deleted"] += 1
                print(f"Deleted orphaned file: {file_path}")

        for name in dirs:
            if name.startswith("."):
                continue
            dir_path = os.path.join(root, name)
            if dir_path not in valid_local_paths:
                if batch_folders is not None:
                    rel = os.path.relpath(dir_path, root_destination_dir)
                    top = rel.split(os.sep, 1)[0]
                    if top not in batch_folders:
                        continue
                shutil.rmtree(dir_path)
                stats["deleted"] += 1
                print(f"Deleted orphaned directory tree: {dir_path}")

    if batch_folders is not None:
        print(
            f"\nMirror Complete! Downloaded: {stats['downloaded']} | "
            f"Skipped (up to date): {stats['skipped']} | "
            f"Missing on Drive: {stats['missing']} | "
            f"Deleted: {stats['deleted']}"
        )
    else:
        print(
            f"\nMirror Complete! Downloaded: {stats['downloaded']} | "
            f"Skipped (up to date): {stats['skipped']} | "
            f"Excluded (older than {days} day(s) on Drive): {stats['excluded_old']} | "
            f"Deleted: {stats['deleted']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively mirror a Google Drive folder locally."
    )
    parser.add_argument(
        "folder_id",
        nargs="?",
        default="1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1",
        help=(
            "The ID of the root Google Drive folder to mirror "
            "(default: 1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1)."
        ),
    )
    parser.add_argument(
        "--destination",
        default="drive_data",
        help="The local directory to mirror into.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help=(
            "Only download Drive files modified within this many days "
            "(default: 30). Ignored when --latest-batch is set."
        ),
    )
    parser.add_argument(
        "--latest-batch",
        action="store_true",
        help=(
            "Sync only subfolders listed in latest_batch.txt "
            "(no modification-date filter)."
        ),
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=None,
        help=(
            "Batch file used with --latest-batch "
            f"(default: {DEFAULT_BATCH_FILE.name} next to this script)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch_file is not None and not args.latest_batch:
        print(
            "Error: --batch-file requires --latest-batch.",
            file=sys.stderr,
        )
        return 1

    if not args.latest_batch and args.days < 1:
        print("Error: --days must be at least 1.", file=sys.stderr)
        return 1

    batch_folders: list[str] | None = None
    if args.latest_batch:
        batch_path = (
            args.batch_file.resolve()
            if args.batch_file is not None
            else DEFAULT_BATCH_FILE
        )
        try:
            batch_folders = load_batch_folder_names(batch_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Loaded {len(batch_folders)} folder(s) from {batch_path}")

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    try:
        drive_svc = get_authenticated_drive_service()
        mirror_folder_sync_recursive(
            drive_svc,
            args.folder_id,
            args.destination,
            days=args.days,
            batch_folders=batch_folders,
        )
    except Exception as exc:
        print(f"\nScript failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
