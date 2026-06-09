#!/usr/bin/env python3
"""
Copy conversations from the pre-delivery Drive folder to the final delivery folder,
renaming speakers from email addresses to SPK01, SPK02, ...

For each conversation folder on source Drive, speakers with a complete set of
delivery files (seglst + rttm + wav) are assigned SPK labels in alphabetical order
by email. Mappings are written to mappings.csv, then files are copied to the
destination Drive folder with updated seglst/rttm content.

Source Drive is read-only.

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
DEFAULT_MAPPINGS_OUT = SCRIPT_DIR / "mappings.csv"

SOURCE_DRIVE_FOLDER_ID = "1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf"
DEST_DRIVE_FOLDER_ID = "1WmpHwiDiatzL9OCToyLEFhmB_Eo_H6VS"
TARGET_SERVICE_ACCOUNT = "delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com"

SEGLST_SUFFIX = ".seglst.json"
RTTM_SUFFIX = ".rttm"
WAV_SUFFIX = ".wav"
FOLDER_MIME = "application/vnd.google-apps.folder"
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


def is_delivery_seglst(filename: str) -> bool:
    return filename.endswith(SEGLST_SUFFIX) and not any(
        filename.endswith(suffix) for suffix in NON_DELIVERY_SEGLST_SUFFIXES
    )


def is_delivery_rttm(filename: str) -> bool:
    return filename.endswith(RTTM_SUFFIX) and not any(
        filename.endswith(suffix) for suffix in NON_DELIVERY_RTTM_SUFFIXES
    )


def is_delivery_wav(filename: str) -> bool:
    return filename.endswith(WAV_SUFFIX)


def discover_candidate_emails(source_items: dict[str, dict]) -> set[str]:
    emails: set[str] = set()
    for filename in source_items:
        if is_delivery_seglst(filename):
            emails.add(filename[: -len(SEGLST_SUFFIX)])
        elif is_delivery_rttm(filename):
            emails.add(filename[: -len(RTTM_SUFFIX)])
        elif is_delivery_wav(filename):
            emails.add(filename[: -len(WAV_SUFFIX)])
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


def build_speaker_mappings(
    conversation: str,
    source_items: dict[str, dict],
) -> list[SpeakerMapping]:
    """Assign SPK01, SPK02, ... to speakers with a complete delivery file set."""
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


def transform_seglst(content: bytes, new_speaker: str) -> bytes:
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
            if parts[1] == old_speaker:
                parts[1] = new_speaker
            if parts[7] == old_speaker:
                parts[7] = new_speaker
            lines_out.append(" ".join(parts))
        else:
            lines_out.append(line)
    return ("\n".join(lines_out) + ("\n" if text.endswith("\n") else "")).encode("utf-8")


def copy_conversation(
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

    dest_items_cache = get_drive_items(service, dest_parent_id)
    dest_folder_id = get_or_create_folder(
        service, conversation, dest_parent_id, dest_items_cache
    )
    dest_items = get_drive_items(service, dest_folder_id)

    for mapping in speaker_mappings:
        src_seglst, src_rttm, src_wav = required_source_filenames(mapping.email)
        dst_seglst, dst_rttm, dst_wav = destination_filenames(mapping.spk_name)

        print(f"Copying {mapping.email} -> {mapping.spk_name}", flush=True)

        seglst_bytes = download_drive_file(service, source_items[src_seglst]["id"])
        rttm_bytes = download_drive_file(service, source_items[src_rttm]["id"])
        wav_bytes = download_drive_file(service, source_items[src_wav]["id"])

        transformed_seglst = transform_seglst(seglst_bytes, mapping.spk_name)
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

    print(f"Completed {conversation} ({len(speaker_mappings)} speaker(s))")
    return True


def resolve_conversations(
    requested: list[str],
    source_folders: dict[str, dict],
) -> list[str]:
    if requested:
        return requested
    return sorted(source_folders)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-map speakers to SPK01/SPK02 labels and copy conversations from "
            "pre-delivery Drive to final delivery Drive."
        )
    )
    parser.add_argument(
        "conversations",
        nargs="*",
        metavar="CONVERSATION",
        help="Conversation folder name(s) on source Drive. Defaults to all folders.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        service = get_authenticated_drive_service()
    except Exception as exc:
        warn(f"Error: authentication failed: {exc}")
        return 1

    source_folders = list_conversation_folders(service, SOURCE_DRIVE_FOLDER_ID)
    conversations = resolve_conversations(args.conversations, source_folders)

    print(f"Source Drive folder: {SOURCE_DRIVE_FOLDER_ID}")
    print(f"Destination Drive folder: {DEST_DRIVE_FOLDER_ID}")
    if args.dry_run:
        print("Mode: DRY RUN (destination Drive will not be modified)")
    print(f"Conversations to process: {', '.join(conversations)}")

    all_mappings: list[SpeakerMapping] = []
    planned: list[tuple[str, list[SpeakerMapping]]] = []
    failures = 0

    for conversation in conversations:
        if conversation not in source_folders:
            warn(
                f"Warning: {conversation}: conversation folder not found on "
                f"source Drive; failed"
            )
            failures += 1
            continue

        source_items = get_drive_items(service, source_folders[conversation]["id"])
        mappings = build_speaker_mappings(conversation, source_items)
        if not mappings:
            warn(
                f"Warning: {conversation}: no speakers with a complete seglst/rttm/wav "
                f"set found; failed"
            )
            failures += 1
            continue

        all_mappings.extend(mappings)
        planned.append((conversation, mappings))

    if not planned:
        warn("Error: no conversations could be mapped")
        return 1

    write_mappings_csv(args.mappings_out, all_mappings)
    print(f"Wrote {len(all_mappings)} mapping row(s) to {args.mappings_out}")

    successes = 0
    for conversation, mappings in planned:
        ok = copy_conversation(
            service,
            conversation,
            mappings,
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
