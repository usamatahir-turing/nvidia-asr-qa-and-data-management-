import argparse
import os
import sys
from dataclasses import dataclass, field
import requests
from datetime import datetime, timezone
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, 'output_data')
DRIVE_DATA_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), 'drive_data')
TARGET_DRIVE_FOLDER_ID = '1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf'
TARGET_SERVICE_ACCOUNT = 'delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com'

SEGLST_SUFFIX = ".seglst.json"
RTTM_SUFFIX = ".rttm"
WAV_SUFFIX = ".wav"
NON_DELIVERY_SUFFIXES = (
    f"_approved{SEGLST_SUFFIX}",
    f"_fixed{SEGLST_SUFFIX}",
    "_approved.rttm",
    "_fixed.rttm",
)


@dataclass
class TaskSpeakerAudit:
    task_name: str
    seglst_speakers: set[str] = field(default_factory=set)
    rttm_speakers: set[str] = field(default_factory=set)
    wav_speakers: set[str] = field(default_factory=set)
    non_delivery_files: list[str] = field(default_factory=list)
    duplicate_seglst: dict[str, list[str]] = field(default_factory=dict)
    duplicate_rttm: dict[str, list[str]] = field(default_factory=dict)
    duplicate_wav: dict[str, list[str]] = field(default_factory=dict)

    @property
    def complete_speakers(self) -> set[str]:
        return self.seglst_speakers & self.rttm_speakers & self.wav_speakers

    @property
    def all_speakers(self) -> set[str]:
        return self.seglst_speakers | self.rttm_speakers | self.wav_speakers


def _is_non_delivery_filename(filename: str) -> bool:
    return any(filename.endswith(suffix) for suffix in NON_DELIVERY_SUFFIXES)


def _collect_delivery_speakers(
    directory: str,
    suffix: str,
) -> tuple[set[str], dict[str, list[str]]]:
    """Return speaker ids and duplicate delivery-ready files with the given suffix."""
    speakers: set[str] = set()
    files_by_speaker: dict[str, list[str]] = {}

    if not os.path.isdir(directory):
        return speakers, {}

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        if not filename.endswith(suffix):
            continue
        if _is_non_delivery_filename(filename):
            continue

        speaker = filename[: -len(suffix)]
        files_by_speaker.setdefault(speaker, []).append(filename)
        speakers.add(speaker)

    duplicates = {
        speaker: filenames
        for speaker, filenames in files_by_speaker.items()
        if len(filenames) > 1
    }
    return speakers, duplicates


def audit_task_speakers(task_name: str) -> TaskSpeakerAudit:
    """Audit delivery-ready speaker files for a task folder."""
    output_task_dir = os.path.join(SOURCE_DIR, task_name)
    wav_task_dir = os.path.join(DRIVE_DATA_ROOT, task_name)
    audit = TaskSpeakerAudit(task_name=task_name)

    if os.path.isdir(output_task_dir):
        for filename in sorted(os.listdir(output_task_dir)):
            path = os.path.join(output_task_dir, filename)
            if os.path.isfile(path) and _is_non_delivery_filename(filename):
                audit.non_delivery_files.append(filename)

    audit.seglst_speakers, audit.duplicate_seglst = _collect_delivery_speakers(
        output_task_dir, SEGLST_SUFFIX
    )
    audit.rttm_speakers, audit.duplicate_rttm = _collect_delivery_speakers(
        output_task_dir, RTTM_SUFFIX
    )
    audit.wav_speakers, audit.duplicate_wav = _collect_delivery_speakers(
        wav_task_dir, WAV_SUFFIX
    )

    return audit


def print_task_speaker_report(task_name: str) -> None:
    """Print speaker counts and delivery issues for a task folder."""
    audit = audit_task_speakers(task_name)

    print(
        f"Speakers (complete set): {len(audit.complete_speakers)} | "
        f"seglst.json: {len(audit.seglst_speakers)} | "
        f"rttm: {len(audit.rttm_speakers)} | "
        f"wav: {len(audit.wav_speakers)}"
    )

    for filename in audit.non_delivery_files:
        print(
            f"Warning: non-delivery file present (run strip_approved_suffix.py): {filename}",
            file=sys.stderr,
        )

    for label, duplicates in (
        ("seglst.json", audit.duplicate_seglst),
        ("rttm", audit.duplicate_rttm),
        ("wav", audit.duplicate_wav),
    ):
        for speaker, filenames in sorted(duplicates.items()):
            print(
                f"Warning: duplicate {label} for {speaker}: {', '.join(filenames)}",
                file=sys.stderr,
            )

    for speaker in sorted(audit.all_speakers - audit.complete_speakers):
        missing = []
        if speaker not in audit.seglst_speakers:
            missing.append("seglst.json")
        if speaker not in audit.rttm_speakers:
            missing.append("rttm")
        if speaker not in audit.wav_speakers:
            missing.append("wav")
        print(
            f"Warning: {speaker} missing {', '.join(missing)}",
            file=sys.stderr,
        )


def get_authenticated_drive_service():
    """Handles Service Account Impersonation to bypass Vertex VM scopes."""
    print("Authenticating...")
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/jupyter/.config/gcloud/application_default_credentials.json"

    base_credentials, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    base_credentials.refresh(Request())

    url = f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{TARGET_SERVICE_ACCOUNT}:generateAccessToken"
    headers = {
        "Authorization": f"Bearer {base_credentials.token}",
        "Content-Type": "application/json"
    }
    payload = {
        "scope": ["https://www.googleapis.com/auth/drive"],
        "lifetime": "3600s"
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Authentication Failed! API Error {response.status_code}: {response.text}")
        
    sa_token = response.json()['accessToken']
    creds = Oauth2Credentials(sa_token)
    
    return build('drive', 'v3', credentials=creds)

def create_drive_folder(service, name, parent_id):
    """Creates a folder in Google Drive."""
    file_metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(
        body=file_metadata, 
        fields='id',
        supportsAllDrives=True
    ).execute()
    return folder.get('id')

def get_drive_items(service, folder_id):
    """Retrieves all files and folders in a specific Drive folder."""
    items = {}
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and trashed = false"
        res = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType, modifiedTime)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        for f in res.get('files', []):
            items[f['name']] = f
        page_token = res.get('nextPageToken', None)
        if page_token is None:
            break
    return items

def upload_files_recursive(service, local_path, drive_parent_id, current_rel_path=""):
    """Recursively syncs local files and folders to Google Drive."""
    # 1. Get current items on Drive
    drive_items = get_drive_items(service, drive_parent_id)
    active_local_names = set()

    for item in os.listdir(local_path):
        item_path = os.path.join(local_path, item)
        active_local_names.add(item)
        
        if os.path.isdir(item_path):
            # Handle Folder
            if item in drive_items and drive_items[item]['mimeType'] == 'application/vnd.google-apps.folder':
                folder_id = drive_items[item]['id']
                print(f"Folder exists: {item}")
            else:
                print(f"Creating folder: {item}")
                folder_id = create_drive_folder(service, item, drive_parent_id)
            
            # Recurse, passing along the relative path to keep drive_data in sync
            upload_files_recursive(service, item_path, folder_id, os.path.join(current_rel_path, item))
        
        else:
            # Handle File Sync Logic
            def sync_file(file_to_upload_path, filename):
                active_local_names.add(filename)
                file_metadata = {'name': filename, 'parents': [drive_parent_id]}
                local_mtime_ts = os.path.getmtime(file_to_upload_path)
                local_mtime = datetime.fromtimestamp(local_mtime_ts, tz=timezone.utc)

                needs_upload = True
                if filename in drive_items:
                    drive_file = drive_items[filename]
                    drive_mtime = datetime.fromisoformat(drive_file['modifiedTime'].replace('Z', '+00:00'))
                    if drive_mtime >= local_mtime:
                        needs_upload = False

                if needs_upload:
                    print(f"Uploading: {filename}...")
                    media = MediaFileUpload(file_to_upload_path, resumable=True)
                    if filename in drive_items:
                        service.files().update(
                            fileId=drive_items[filename]['id'],
                            media_body=media,
                            supportsAllDrives=True
                        ).execute()
                    else:
                        service.files().create(
                            body=file_metadata,
                            media_body=media,
                            fields='id',
                            supportsAllDrives=True
                        ).execute()

            # Sync the actual file from output_data
            sync_file(item_path, item)

            # TASK: If it's an .rttm, find and sync the corresponding .wav from drive_data
            if item.endswith('.rttm'):
                wav_filename = item.replace('.rttm', '.wav')
                wav_local_path = os.path.join(DRIVE_DATA_ROOT, current_rel_path, wav_filename)
                
                if os.path.exists(wav_local_path):
                    sync_file(wav_local_path, wav_filename)
                else:
                    print(f"Warning: Corresponding WAV not found: {wav_local_path}")

    # 2. Cleanup: Remove files on Drive that no longer exist locally (including synced WAVs).
    # Keep mixed wavs — they are not uploaded from output_data and must not be deleted.
    for drive_item_name, drive_item_info in drive_items.items():
        if drive_item_name not in active_local_names:
            if drive_item_name.endswith(WAV_SUFFIX) and "_mixed" in drive_item_name:
                print(f"Skipping cleanup of mixed wav: {drive_item_name}")
                continue
            print(f"Deleting orphaned Drive item: {drive_item_name}")
            service.files().delete(
                fileId=drive_item_info['id'],
                supportsAllDrives=True
            ).execute()

    if current_rel_path:
        print_task_speaker_report(current_rel_path)


def resolve_task_dirs(source_dir: str, task_names: list[str]) -> list[str]:
    """Return task folder names to upload, validating paths under source_dir."""
    if not task_names:
        tasks = sorted(
            name
            for name in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, name))
        )
        if not tasks:
            raise ValueError(f"No task folders found in {source_dir}")
        return tasks

    missing = [
        name for name in task_names if not os.path.isdir(os.path.join(source_dir, name))
    ]
    if missing:
        raise ValueError(
            f"Task folder(s) not found under {source_dir}: {', '.join(missing)}"
        )
    return task_names


def get_or_create_drive_folder(
    service,
    name: str,
    parent_id: str,
    drive_items: dict,
) -> str:
    """Return the Drive folder id for name, creating it when missing."""
    if name in drive_items and drive_items[name]["mimeType"] == "application/vnd.google-apps.folder":
        print(f"Folder exists: {name}")
        return drive_items[name]["id"]

    print(f"Creating folder: {name}")
    return create_drive_folder(service, name, parent_id)


def upload_task_folders(service, source_dir: str, drive_parent_id: str, task_names: list[str]):
    """Upload only the selected task folders into the target Drive folder."""
    drive_items = get_drive_items(service, drive_parent_id)

    for task_name in task_names:
        local_path = os.path.join(source_dir, task_name)
        print(f"\n--- {task_name} ---")
        folder_id = get_or_create_drive_folder(service, task_name, drive_parent_id, drive_items)
        upload_files_recursive(service, local_path, folder_id, task_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload seglst/rttm files from output_data to the pre-delivery Drive folder, "
            "including matching channel wav files from drive_data."
        )
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        metavar="TASK",
        help=(
            "Task folder name(s) under output_data (e.g. NV-AR-SS03-CONVO07). "
            "Upload all task folders when omitted."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.", file=sys.stderr)
        return 1

    try:
        task_names = resolve_task_dirs(SOURCE_DIR, args.tasks)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        drive_service = get_authenticated_drive_service()
        if args.tasks:
            print(
                f"Starting upload of {len(task_names)} task folder(s) from {SOURCE_DIR} "
                f"to Drive folder {TARGET_DRIVE_FOLDER_ID}..."
            )
        else:
            print(
                f"Starting upload of all task folders from {SOURCE_DIR} "
                f"to Drive folder {TARGET_DRIVE_FOLDER_ID}..."
            )
        print("Tasks:", ", ".join(task_names))

        if args.tasks:
            upload_task_folders(drive_service, SOURCE_DIR, TARGET_DRIVE_FOLDER_ID, task_names)
        else:
            upload_files_recursive(drive_service, SOURCE_DIR, TARGET_DRIVE_FOLDER_ID, "")

        print("Upload complete!")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
