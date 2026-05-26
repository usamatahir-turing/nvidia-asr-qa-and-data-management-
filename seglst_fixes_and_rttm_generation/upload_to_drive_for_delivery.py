import os
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

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
    else:
        try:
            drive_service = get_authenticated_drive_service()
            print(f"Starting upload from {SOURCE_DIR} to Drive folder {TARGET_DRIVE_FOLDER_ID}...")
            upload_files_recursive(drive_service, SOURCE_DIR, TARGET_DRIVE_FOLDER_ID, "")
            print("Upload complete!")
        except Exception as e:
            print(f"An error occurred: {e}")
