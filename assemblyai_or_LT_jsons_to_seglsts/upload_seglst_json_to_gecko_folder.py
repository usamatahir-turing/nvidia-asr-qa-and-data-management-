#!/usr/bin/env python3
"""Upload generated seglst JSON files to the Files for Gecko Drive folder.

Uploads conversation folders from output_seglsts/ into the Gecko root on Drive.
If a conversation folder already exists on Drive, it is skipped with a warning.

Example::

    python upload_seglst_json_to_gecko_folder.py NV-AR-SS03-CONVO07
    python upload_seglst_json_to_gecko_folder.py NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
"""

from __future__ import annotations

import argparse
import os
import sys

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "output_seglsts")
GECKO_DRIVE_FOLDER_ID = "1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
SEGLST_SUFFIX = ".seglst.json"


def get_authenticated_drive_service():
    """Handles Service Account Impersonation to bypass Vertex VM scopes."""
    print("Authenticating...")

    adc_path = "/home/jupyter/.config/gcloud/application_default_credentials.json"
    if os.path.exists(adc_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path

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


def list_drive_subfolders(service, parent_id: str) -> dict[str, str]:
    """Return {folder_name: folder_id} for immediate child folders on Drive."""
    subfolders: dict[str, str] = {}
    page_token = None
    query = (
        f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' "
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


def create_drive_folder(service, name: str, parent_id: str) -> str:
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(
        body=file_metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return folder["id"]


def collect_seglst_files(conversation_dir: str) -> list[str]:
    """Return sorted paths to *.seglst.json files in a conversation folder."""
    return sorted(
        os.path.join(conversation_dir, name)
        for name in os.listdir(conversation_dir)
        if name.endswith(SEGLST_SUFFIX)
        and os.path.isfile(os.path.join(conversation_dir, name))
    )


def upload_file(service, local_path: str, filename: str, parent_id: str) -> None:
    print(f"Uploading: {filename}")
    media = MediaFileUpload(local_path, resumable=True)
    file_metadata = {"name": filename, "parents": [parent_id]}
    service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


def upload_conversation(
    service,
    conversation: str,
    drive_subfolders: dict[str, str],
) -> tuple[int, int, int]:
    """Upload one conversation. Returns (uploaded, skipped_folder, failed)."""
    local_dir = os.path.join(SOURCE_DIR, conversation)
    if not os.path.isdir(local_dir):
        print(
            f"Error: conversation folder not found under {SOURCE_DIR}: {conversation}",
            file=sys.stderr,
        )
        return 0, 0, 1

    seglst_files = collect_seglst_files(local_dir)
    if not seglst_files:
        print(
            f"Error: no {SEGLST_SUFFIX} files in {local_dir}",
            file=sys.stderr,
        )
        return 0, 0, 1

    if conversation in drive_subfolders:
        print(
            f"Warning: Drive folder '{conversation}' already exists — skipping.",
            file=sys.stderr,
        )
        return 0, 1, 0

    print(f"Creating folder: {conversation}")
    folder_id = create_drive_folder(service, conversation, GECKO_DRIVE_FOLDER_ID)
    drive_subfolders[conversation] = folder_id

    uploaded = 0
    for file_path in seglst_files:
        upload_file(service, file_path, os.path.basename(file_path), folder_id)
        uploaded += 1

    return uploaded, 0, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload seglst JSON files from output_seglsts to the Files for Gecko "
            "Drive folder. Skips conversations that already exist on Drive."
        )
    )
    parser.add_argument(
        "conversations",
        nargs="+",
        metavar="CONVERSATION",
        help="Conversation folder name(s) under output_seglsts (e.g. NV-AR-SS03-CONVO07).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: source directory does not exist: {SOURCE_DIR}", file=sys.stderr)
        return 1

    conversations = args.conversations
    print(
        f"Uploading {len(conversations)} conversation(s) from {SOURCE_DIR} "
        f"to Drive folder {GECKO_DRIVE_FOLDER_ID}..."
    )
    print("Conversations:", ", ".join(conversations))

    try:
        service = get_authenticated_drive_service()
        drive_subfolders = list_drive_subfolders(service, GECKO_DRIVE_FOLDER_ID)

        total_uploaded = 0
        total_skipped = 0
        total_failed = 0

        for conversation in conversations:
            print(f"\n--- {conversation} ---")
            uploaded, skipped, failed = upload_conversation(
                service, conversation, drive_subfolders
            )
            total_uploaded += uploaded
            total_skipped += skipped
            total_failed += failed

        print(
            f"\nDone. Uploaded: {total_uploaded} file(s) | "
            f"Skipped folders: {total_skipped} | Failed: {total_failed}"
        )
        return 1 if total_failed else 0
    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
