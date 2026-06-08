import argparse
import os
import sys
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

    # 2. Cleanup: Remove files on Drive that no longer exist locally (including synced WAVs)
    for drive_item_name, drive_item_info in drive_items.items():
        if drive_item_name not in active_local_names:
            print(f"Deleting orphaned Drive item: {drive_item_name}")
            service.files().delete(
                fileId=drive_item_info['id'],
                supportsAllDrives=True
            ).execute()


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
