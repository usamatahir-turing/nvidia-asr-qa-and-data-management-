#!/usr/bin/env python3
"""Copy channel WAV files from AssemblyAI Drive into Files for Gecko conversation folders.

For each requested conversation, reads ``*.seglst.json`` files on Gecko Drive, finds the
matching ``<speaker>.wav`` in the AssemblyAI source folder on Drive, and copies it into
the Gecko conversation folder. Existing WAVs on Gecko are replaced.

Example::

    python copy_channel_wavs_to_gecko_folder.py NV-AR-SS03-CONVO07
    python copy_channel_wavs_to_gecko_folder.py NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSEMBLYAI_DRIVE_FOLDER_ID = "1wceeL4NRLTXg57EIgV5peQPuDCBEthzl"
GECKO_DRIVE_FOLDER_ID = "1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"
SEGLST_SUFFIX = ".seglst.json"
WAV_SUFFIX = ".wav"
FOLDER_MIME = "application/vnd.google-apps.folder"


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


def speaker_stems_from_seglsts(files: dict[str, dict[str, str]]) -> list[str]:
    """Return sorted speaker stems from *.seglst.json filenames in a folder listing."""
    stems: list[str] = []
    for name in sorted(files):
        if name.endswith(SEGLST_SUFFIX) and files[name]["mimeType"] != FOLDER_MIME:
            stems.append(name[:-len(SEGLST_SUFFIX)])
    return stems


def copy_or_replace_wav(
    service,
    source_file_id: str,
    wav_name: str,
    dest_folder_id: str,
    dest_files: dict[str, dict[str, str]],
) -> bool:
    """Copy a WAV from source Drive into dest folder, replacing any existing file."""
    replaced = wav_name in dest_files and dest_files[wav_name]["mimeType"] != FOLDER_MIME
    if replaced:
        service.files().delete(
            fileId=dest_files[wav_name]["id"],
            supportsAllDrives=True,
        ).execute()
        print(f"Replacing: {wav_name}")
    else:
        print(f"Copying: {wav_name}")

    service.files().copy(
        fileId=source_file_id,
        body={"name": wav_name, "parents": [dest_folder_id]},
        supportsAllDrives=True,
        fields="id",
    ).execute()
    return replaced


def process_conversation(
    service,
    conversation: str,
    assemblyai_subfolders: dict[str, str],
    gecko_subfolders: dict[str, str],
) -> tuple[int, int, int, int]:
    """Process one conversation. Returns (copied, replaced, skipped_missing, failed)."""
    if conversation not in gecko_subfolders:
        print(
            f"Error: Gecko Drive folder not found: {conversation}",
            file=sys.stderr,
        )
        return 0, 0, 0, 1

    if conversation not in assemblyai_subfolders:
        print(
            f"Error: AssemblyAI Drive folder not found: {conversation}",
            file=sys.stderr,
        )
        return 0, 0, 0, 1

    gecko_folder_id = gecko_subfolders[conversation]
    assemblyai_folder_id = assemblyai_subfolders[conversation]

    gecko_files = list_drive_files(service, gecko_folder_id)
    assemblyai_files = list_drive_files(service, assemblyai_folder_id)
    speaker_stems = speaker_stems_from_seglsts(gecko_files)

    if not speaker_stems:
        print(
            f"Warning: no {SEGLST_SUFFIX} files in Gecko folder '{conversation}' — skipping.",
            file=sys.stderr,
        )
        return 0, 0, 0, 0

    copied = 0
    replaced = 0
    skipped_missing = 0

    for speaker in speaker_stems:
        wav_name = f"{speaker}{WAV_SUFFIX}"
        source_entry = assemblyai_files.get(wav_name)
        if source_entry is None or source_entry["mimeType"] == FOLDER_MIME:
            print(
                f"Warning: source WAV not found for {speaker}: {wav_name} — skipping.",
                file=sys.stderr,
            )
            skipped_missing += 1
            continue

        was_replaced = copy_or_replace_wav(
            service,
            source_entry["id"],
            wav_name,
            gecko_folder_id,
            gecko_files,
        )
        if was_replaced:
            replaced += 1
        else:
            copied += 1

    return copied, replaced, skipped_missing, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy channel WAV files from the AssemblyAI Drive folder into matching "
            "conversation folders on the Files for Gecko Drive folder."
        )
    )
    parser.add_argument(
        "conversations",
        nargs="+",
        metavar="CONVERSATION",
        help="Conversation folder name (e.g. NV-AR-SS03-CONVO07).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conversations = args.conversations

    print(
        f"Copying WAVs for {len(conversations)} conversation(s) from AssemblyAI Drive "
        f"({ASSEMBLYAI_DRIVE_FOLDER_ID}) to Gecko Drive ({GECKO_DRIVE_FOLDER_ID})..."
    )
    print("Conversations:", ", ".join(conversations))

    try:
        service = get_authenticated_drive_service()
        assemblyai_subfolders = list_drive_subfolders(service, ASSEMBLYAI_DRIVE_FOLDER_ID)
        gecko_subfolders = list_drive_subfolders(service, GECKO_DRIVE_FOLDER_ID)

        total_copied = 0
        total_replaced = 0
        total_skipped_missing = 0
        total_failed = 0

        for conversation in conversations:
            print(f"\n--- {conversation} ---")
            copied, replaced, skipped_missing, failed = process_conversation(
                service,
                conversation,
                assemblyai_subfolders,
                gecko_subfolders,
            )
            total_copied += copied
            total_replaced += replaced
            total_skipped_missing += skipped_missing
            total_failed += failed

        print(
            f"\nDone. Copied: {total_copied} | Replaced: {total_replaced} | "
            f"Skipped (missing source WAV): {total_skipped_missing} | Failed: {total_failed}"
        )
        return 1 if total_failed else 0
    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
