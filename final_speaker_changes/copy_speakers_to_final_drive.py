#!/usr/bin/env python3
"""
Copy conversations from the pre-delivery Drive folder to the final delivery folder,
renaming speakers from email addresses to SPK labels (e.g. SPKA, SPKB).

Reads mappings from speaker_mappings.csv. Source Drive is read-only; transformed
files are written to the destination Drive folder.

Usage::

    cd final_speaker_changes
    python copy_speakers_to_final_drive.py
    python copy_speakers_to_final_drive.py NV-AR-SS06-CONVO16 NV-KO-SS05-CONVO13
    python copy_speakers_to_final_drive.py --dry-run NV-GR-SS04-CONVO10
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import google.auth
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPINGS = SCRIPT_DIR / "speaker_mappings.csv"

SOURCE_DRIVE_FOLDER_ID = "1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf"
DEST_DRIVE_FOLDER_ID = "1WmpHwiDiatzL9OCToyLEFhmB_Eo_H6VS"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"

SEGLST_SUFFIX = ".seglst.json"
RTTM_SUFFIX = ".rttm"
WAV_SUFFIX = ".wav"
FOLDER_MIME = "application/vnd.google-apps.folder"


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


def get_or_create_folder(service, name: str, parent_id: str, cache: dict[str, dict]) -> str:
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
) -> None:
    if dry_run:
        print(f"  [dry-run] would upload {filename} ({len(data)} bytes)")
        return

    media = MediaIoBaseUpload(
        io.BytesIO(data),
        mimetype=mime_type,
        resumable=True,
    )
    if filename in dest_items:
        service.files().update(
            fileId=dest_items[filename]["id"],
            media_body=media,
            supportsAllDrives=True,
        ).execute()
        print(f"  Updated {filename}")
    else:
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
        print(f"  Uploaded {filename}")


def load_speaker_mappings(path: Path) -> dict[str, list[SpeakerMapping]]:
    mappings: dict[str, list[SpeakerMapping]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"Conversation", "Speaker_label", "Speaker"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )
        for row in reader:
            conversation = row["Conversation"].strip()
            label = row["Speaker_label"].strip()
            email = row["Speaker"].strip()
            if not conversation or not label or not email:
                continue
            mappings.setdefault(conversation, []).append(
                SpeakerMapping(conversation=conversation, label=label, email=email)
            )
    return mappings


def transform_seglst(content: bytes, old_speaker: str, new_speaker: str) -> bytes:
    data = json.loads(content.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("seglst JSON must be a list")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"seglst segment #{index} is not an object")
        item["speaker"] = new_speaker

    output = json.dumps(data, ensure_ascii=False, indent=4) + "\n"
    return output.encode("utf-8")


def transform_rttm(content: bytes, old_speaker: str, new_speaker: str) -> bytes:
    lines_out: list[str] = []
    text = content.decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            lines_out.append(line)
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            parts[1] = new_speaker
            parts[7] = new_speaker
            lines_out.append(" ".join(parts))
        else:
            lines_out.append(line)
    return ("\n".join(lines_out) + ("\n" if text.endswith("\n") else "")).encode("utf-8")


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


def process_conversation(
    service,
    conversation: str,
    speaker_mappings: list[SpeakerMapping],
    source_folder_id: str,
    dest_parent_id: str,
    *,
    dry_run: bool,
) -> bool:
    print(f"\n--- {conversation} ---", flush=True)
    source_items = get_drive_items(service, source_folder_id)
    errors: list[str] = []

    for mapping in speaker_mappings:
        seglst_name, rttm_name, wav_name = required_source_filenames(mapping.email)
        missing = [
            name
            for name in (seglst_name, rttm_name, wav_name)
            if name not in source_items
        ]
        if missing:
            errors.append(
                f"{mapping.email}: missing source file(s): {', '.join(missing)}"
            )

    if errors:
        for message in errors:
            warn(f"Warning: {conversation}: {message}")
        warn(f"Warning: {conversation}: failed; no files copied")
        return False

    dest_items_cache = get_drive_items(service, dest_parent_id)
    dest_folder_id = get_or_create_folder(
        service, conversation, dest_parent_id, dest_items_cache
    )
    dest_items = get_drive_items(service, dest_folder_id)

    for mapping in speaker_mappings:
        src_seglst, src_rttm, src_wav = required_source_filenames(mapping.email)
        dst_seglst, dst_rttm, dst_wav = destination_filenames(mapping.spk_name)

        print(
            f"Processing {mapping.email} -> {mapping.spk_name}",
            flush=True,
        )

        seglst_bytes = download_drive_file(service, source_items[src_seglst]["id"])
        rttm_bytes = download_drive_file(service, source_items[src_rttm]["id"])
        wav_bytes = download_drive_file(service, source_items[src_wav]["id"])

        transformed_seglst = transform_seglst(
            seglst_bytes, mapping.email, mapping.spk_name
        )
        transformed_rttm = transform_rttm(
            rttm_bytes, mapping.email, mapping.spk_name
        )

        upload_drive_file(
            service,
            dest_folder_id,
            dst_seglst,
            transformed_seglst,
            "application/json",
            dest_items,
            dry_run=dry_run,
        )
        upload_drive_file(
            service,
            dest_folder_id,
            dst_rttm,
            transformed_rttm,
            "text/plain",
            dest_items,
            dry_run=dry_run,
        )
        upload_drive_file(
            service,
            dest_folder_id,
            dst_wav,
            wav_bytes,
            "audio/wav",
            dest_items,
            dry_run=dry_run,
        )

    print(f"Completed {conversation}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy conversations from pre-delivery Drive to final delivery Drive, "
            "renaming speakers using speaker_mappings.csv."
        )
    )
    parser.add_argument(
        "conversations",
        nargs="*",
        metavar="CONVERSATION",
        help="Conversation folder name(s) to process. Defaults to all in the CSV.",
    )
    parser.add_argument(
        "--mappings",
        type=Path,
        default=DEFAULT_MAPPINGS,
        help=f"Path to speaker_mappings.csv (default: {DEFAULT_MAPPINGS.name})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report actions without writing to destination Drive",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.mappings.is_file():
        warn(f"Error: mappings file not found: {args.mappings}")
        return 1

    try:
        mappings_by_conversation = load_speaker_mappings(args.mappings)
    except ValueError as exc:
        warn(f"Error: {exc}")
        return 1

    if not mappings_by_conversation:
        warn(f"Error: no mappings found in {args.mappings}")
        return 1

    conversations = args.conversations or sorted(mappings_by_conversation)
    unknown_in_csv = [
        name for name in conversations if name not in mappings_by_conversation
    ]
    if unknown_in_csv:
        for name in unknown_in_csv:
            warn(f"Warning: {name}: not found in speaker_mappings.csv")
        warn("Error: one or more requested conversations are missing from the CSV")
        return 1

    try:
        service = get_authenticated_drive_service()
    except Exception as exc:
        warn(f"Error: authentication failed: {exc}")
        return 1

    source_folders = list_conversation_folders(service, SOURCE_DRIVE_FOLDER_ID)
    dest_root_items = get_drive_items(service, DEST_DRIVE_FOLDER_ID)

    for folder_name in sorted(source_folders):
        if folder_name not in mappings_by_conversation:
            warn(
                f"Warning: {folder_name}: present on source Drive but not in "
                f"speaker_mappings.csv; skipping"
            )

    print(f"Source Drive folder: {SOURCE_DRIVE_FOLDER_ID}")
    print(f"Destination Drive folder: {DEST_DRIVE_FOLDER_ID}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print(f"Conversations to process: {', '.join(conversations)}")

    successes = 0
    failures = 0

    for conversation in conversations:
        if conversation not in source_folders:
            warn(
                f"Warning: {conversation}: conversation folder not found on "
                f"source Drive; failed"
            )
            failures += 1
            continue

        ok = process_conversation(
            service,
            conversation,
            mappings_by_conversation[conversation],
            source_folders[conversation]["id"],
            DEST_DRIVE_FOLDER_ID,
            dry_run=args.dry_run,
        )
        if ok:
            successes += 1
        else:
            failures += 1

    print()
    print(f"Completed: {successes} conversation(s)")
    if failures:
        print(f"Failed   : {failures} conversation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
