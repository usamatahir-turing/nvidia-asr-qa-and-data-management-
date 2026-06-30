#!/usr/bin/env python3
"""Copy *_mixed.wav files from AssemblyAI Drive into pre-delivery conversation folders.

For each conversation subfolder under the pre-delivery Drive root, finds the
same-named folder recursively under the AssemblyAI Drive root and copies the
single *_mixed.wav file into the pre-delivery folder.

Example::

    cd seglst_fixes_and_rttm_generation/
    python fetch_mixed.py
"""

from __future__ import annotations

import os
import sys

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build

PRE_DELIVERY_DRIVE_FOLDER_ID = "1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf"
ASSEMBLYAI_DRIVE_FOLDER_ID = "1wceeL4NRLTXg57EIgV5peQPuDCBEthzl"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
MIXED_WAV_SUFFIX = "_mixed.wav"
FOLDER_MIME = "application/vnd.google-apps.folder"


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


def is_mixed_wav(filename: str) -> bool:
    return filename.endswith(MIXED_WAV_SUFFIX)


def find_folders_by_name_recursive(
    service,
    root_id: str,
    folder_name: str,
) -> list[str]:
    """Return folder ids under root whose name equals folder_name (recursive)."""
    matches: list[str] = []

    def walk(parent_id: str) -> None:
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
                if folder["name"] == folder_name:
                    matches.append(folder["id"])
                walk(folder["id"])
            page_token = res.get("nextPageToken")
            if page_token is None:
                break

    walk(root_id)
    return matches


def discover_mixed_wav_files(
    files: dict[str, dict[str, str]],
) -> list[tuple[str, str]]:
    """Return sorted (filename, file_id) pairs for *_mixed.wav in a folder listing."""
    return sorted(
        (name, info["id"])
        for name, info in files.items()
        if is_mixed_wav(name) and info["mimeType"] != FOLDER_MIME
    )


def copy_file_to_folder(
    service,
    source_file_id: str,
    filename: str,
    dest_folder_id: str,
) -> None:
    service.files().copy(
        fileId=source_file_id,
        body={"name": filename, "parents": [dest_folder_id]},
        supportsAllDrives=True,
        fields="id",
    ).execute()


def process_conversation(
    service,
    conversation: str,
    dest_folder_id: str,
) -> str:
    """Process one conversation. Returns a result label for stats."""
    dest_files = list_drive_files(service, dest_folder_id)
    dest_mixed = discover_mixed_wav_files(dest_files)
    if dest_mixed:
        names = ", ".join(name for name, _ in dest_mixed)
        print(
            f"Note: {conversation} already has mixed wav on pre-delivery Drive "
            f"({names}) — skipping.",
        )
        return "skipped_existing"

    source_folder_ids = find_folders_by_name_recursive(
        service,
        ASSEMBLYAI_DRIVE_FOLDER_ID,
        conversation,
    )
    if not source_folder_ids:
        print(
            f"Warning: no AssemblyAI Drive folder found for {conversation} — skipping.",
            file=sys.stderr,
        )
        return "skipped_no_source_folder"
    if len(source_folder_ids) > 1:
        print(
            f"Warning: multiple AssemblyAI Drive folders named {conversation!r} "
            f"({len(source_folder_ids)} matches) — skipping.",
            file=sys.stderr,
        )
        return "skipped_ambiguous_source_folder"

    source_files = list_drive_files(service, source_folder_ids[0])
    source_mixed = discover_mixed_wav_files(source_files)
    if not source_mixed:
        print(
            f"Warning: no {MIXED_WAV_SUFFIX} file in AssemblyAI folder "
            f"{conversation} — skipping.",
            file=sys.stderr,
        )
        return "skipped_no_mixed_wav"
    if len(source_mixed) > 1:
        names = ", ".join(name for name, _ in source_mixed)
        print(
            f"Warning: multiple {MIXED_WAV_SUFFIX} files in AssemblyAI folder "
            f"{conversation} ({names}) — skipping.",
            file=sys.stderr,
        )
        return "skipped_ambiguous_mixed_wav"

    mixed_name, mixed_id = source_mixed[0]
    print(f"Copying: {conversation}/{mixed_name}")
    copy_file_to_folder(service, mixed_id, mixed_name, dest_folder_id)
    return "copied"


def main() -> int:
    print(
        f"Fetching {MIXED_WAV_SUFFIX} files from AssemblyAI Drive "
        f"({ASSEMBLYAI_DRIVE_FOLDER_ID}) into pre-delivery folders "
        f"({PRE_DELIVERY_DRIVE_FOLDER_ID})..."
    )

    try:
        service = get_authenticated_drive_service()
        dest_subfolders = list_drive_subfolders(service, PRE_DELIVERY_DRIVE_FOLDER_ID)
        if not dest_subfolders:
            print("Error: no conversation folders found on pre-delivery Drive.", file=sys.stderr)
            return 1

        stats: dict[str, int] = {}
        for conversation in sorted(dest_subfolders):
            print(f"\n--- {conversation} ---")
            result = process_conversation(
                service,
                conversation,
                dest_subfolders[conversation],
            )
            stats[result] = stats.get(result, 0) + 1

        copied = stats.get("copied", 0)
        skipped_existing = stats.get("skipped_existing", 0)
        skipped_no_source = stats.get("skipped_no_source_folder", 0)
        skipped_ambiguous_folder = stats.get("skipped_ambiguous_source_folder", 0)
        skipped_no_mixed = stats.get("skipped_no_mixed_wav", 0)
        skipped_ambiguous_mixed = stats.get("skipped_ambiguous_mixed_wav", 0)

        print(
            f"\nDone. Copied: {copied} | Already present: {skipped_existing} | "
            f"No source folder: {skipped_no_source} | "
            f"Ambiguous source folder: {skipped_ambiguous_folder} | "
            f"No mixed wav: {skipped_no_mixed} | "
            f"Ambiguous mixed wav: {skipped_ambiguous_mixed}"
        )
        return 0
    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
