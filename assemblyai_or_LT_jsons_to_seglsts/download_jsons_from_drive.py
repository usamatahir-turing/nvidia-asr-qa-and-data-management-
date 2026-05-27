import os
import io
import requests
import shutil
from datetime import datetime, timezone

import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Configuration
DRIVE_ROOT_FOLDER_ID = "1wceeL4NRLTXg57EIgV5peQPuDCBEthzl"
TARGET_SERVICE_ACCOUNT = 'delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com'
LOCAL_DESTINATION_ROOT = os.path.join(os.path.dirname(__file__), "assembly_ai_jsons")

def get_authenticated_drive_service():
    """Handles Service Account Impersonation to bypass Vertex VM scopes."""
    print("Authenticating...")
    
    # Ensure ADC is pointed to the correct file if it exists, otherwise rely on default
    if os.path.exists("/home/jupyter/.config/gcloud/application_default_credentials.json"):
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

def sync_jsons_only_flattened(drive_service, root_folder_id, dest_root):
    """
    Syncs only .json files from Drive, replicating only the immediate parent folder.
    Maintains 'updates only' logic and cleans up orphaned local files.
    """
    stats = {'downloaded': 0, 'skipped': 0, 'deleted': 0}
    valid_local_paths = set()
    
    print(f"\nStarting JSON sync from Drive Folder ID: {root_folder_id}")
    print(f"Destination: {dest_root}\n")

    def walk_drive(drive_folder_id, parent_folder_name=None):
        page_token = None
        while True:
            # Query for files and folders. We need parent ID to get parent name for top-level jsons if needed,
            # but usually we want the name of the folder we are currently in.
            query = f"'{drive_folder_id}' in parents and trashed = false"
            res = drive_service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, modifiedTime)',
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            files = res.get('files', [])
            
            for f in files:
                file_id = f['id']
                file_name = f['name']
                
                if f['mimeType'] == 'application/vnd.google-apps.folder':
                    # Recurse, passing the current folder's name as the new parent_folder_name
                    walk_drive(file_id, parent_folder_name=file_name)
                    continue
                
                # We only care about .json files
                if not file_name.lower().endswith('.json'):
                    continue
                
                # Determine local path: dest_root / parent_folder_name / file_name
                # If the json is directly in the root folder provided, parent_folder_name might be None
                # though usually it resides in a subfolder per requirements.
                effective_parent = parent_folder_name if parent_folder_name else ""
                local_parent_dir = os.path.join(dest_root, effective_parent)
                file_path = os.path.join(local_parent_dir, file_name)
                
                # Track for cleanup
                valid_local_paths.add(local_parent_dir)
                valid_local_paths.add(file_path)

                # Process file
                os.makedirs(local_parent_dir, exist_ok=True)
                
                drive_time_str = f['modifiedTime']
                drive_mtime = datetime.fromisoformat(drive_time_str.replace('Z', '+00:00'))

                needs_download = True
                if os.path.exists(file_path):
                    local_mtime_ts = os.path.getmtime(file_path)
                    local_mtime = datetime.fromtimestamp(local_mtime_ts, tz=timezone.utc)
                    
                    if local_mtime >= drive_mtime:
                        needs_download = False

                if not needs_download:
                    stats['skipped'] += 1
                    continue

                print(f"Downloading: {effective_parent}/{file_name}...")
                request = drive_service.files().get_media(fileId=file_id)
                with io.FileIO(file_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        
                # Update local timestamp to match Drive
                drive_mtime_ts = drive_mtime.timestamp()
                os.utime(file_path, (drive_mtime_ts, drive_mtime_ts))
                stats['downloaded'] += 1

            page_token = res.get('nextPageToken', None)
            if page_token is None:
                break

    # Start recursion
    walk_drive(root_folder_id)

    # Cleanup local orphans
    print("\nStarting local cleanup...")
    for root, dirs, files in os.walk(dest_root, topdown=False):
        # Skip hidden files
        for name in files:
            if name.startswith('.'): continue
            
            file_path = os.path.join(root, name)
            if file_path not in valid_local_paths:
                os.remove(file_path)
                stats['deleted'] += 1
                print(f"Deleted orphaned file: {file_path}")
                
        for name in dirs:
            if name.startswith('.'): continue
            
            dir_path = os.path.join(root, name)
            if dir_path not in valid_local_paths:
                shutil.rmtree(dir_path)
                stats['deleted'] += 1
                print(f"Deleted orphaned directory tree: {dir_path}")

    print(f"\nSync Complete! Downloaded: {stats['downloaded']} | Skipped: {stats['skipped']} | Deleted: {stats['deleted']}")

if __name__ == "__main__":
    if not os.path.exists(LOCAL_DESTINATION_ROOT):
        os.makedirs(LOCAL_DESTINATION_ROOT)
        
    try:
        drive_svc = get_authenticated_drive_service()
        sync_jsons_only_flattened(drive_svc, DRIVE_ROOT_FOLDER_ID, LOCAL_DESTINATION_ROOT)
    except Exception as e:
        print(f"\nScript failed: {e}")
