import os
import io
import argparse
import requests
import shutil
from datetime import datetime, timezone

import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def get_authenticated_drive_service():
    """Handles Service Account Impersonation to bypass Vertex VM scopes."""
    print("Authenticating...")
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/jupyter/.config/gcloud/application_default_credentials.json"

    base_credentials, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    base_credentials.refresh(Request())

    TARGET_SERVICE_ACCOUNT = 'delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com'
    
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


def mirror_folder_sync_recursive(drive_service, root_folder_id, root_destination_dir):
    # State tracking across recursive calls
    stats = {'downloaded': 0, 'skipped': 0, 'deleted': 0}
    valid_local_paths = set()
    
    print(f"\nStarting recursive sync for root folder ID: {root_folder_id}")
    print(f"Destination: {root_destination_dir}\n")

    def process_folder(drive_folder_id, current_local_dir):
        # 1. Create the local directory if it doesn't exist
        os.makedirs(current_local_dir, exist_ok=True)
        valid_local_paths.add(current_local_dir)
        
        page_token = None
        while True:
            query = f"'{drive_folder_id}' in parents and trashed = false"
            res = drive_service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, modifiedTime)',
                pageToken=page_token
            ).execute()

            files = res.get('files', [])
            
            for f in files:
                file_id = f['id']
                file_name = f['name']
                file_path = os.path.join(current_local_dir, file_name)
                
                # Add this exact path to our tracker so it doesn't get deleted later
                valid_local_paths.add(file_path)

                # 2. If it's a folder, RECURSE into it
                if f['mimeType'] == 'application/vnd.google-apps.folder':
                    process_folder(file_id, file_path)
                    continue
                    
                # Skip Google Workspace documents (Docs, Sheets)
                if 'application/vnd.google-apps' in f['mimeType']:
                    continue

                # 3. File Processing & Timestamps
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

                # 4. Download file
                print(f"Downloading: {file_path}...")
                request = drive_service.files().get_media(fileId=file_id)
                with io.FileIO(file_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        
                # Update timestamp
                drive_mtime_ts = drive_mtime.timestamp()
                os.utime(file_path, (drive_mtime_ts, drive_mtime_ts))
                
                stats['downloaded'] += 1

            page_token = res.get('nextPageToken', None)
            if page_token is None:
                break

    # Kick off the recursion from the root
    process_folder(root_folder_id, root_destination_dir)

    # 5. The Recursive Cleanup Phase
    print("\nStarting local cleanup...")
    # os.walk bottom-up ensures we delete files before trying to delete the folder holding them
    for root, dirs, files in os.walk(root_destination_dir, topdown=False):
        
        # Check files
        for name in files:
            if name.startswith('.'): continue # Skip hidden Jupyter files
            
            file_path = os.path.join(root, name)
            if file_path not in valid_local_paths:
                os.remove(file_path)
                stats['deleted'] += 1
                print(f"Deleted orphaned file: {file_path}")
                
        # Check directories
        for name in dirs:
            if name.startswith('.'): continue # Skip hidden Jupyter directories
            
            dir_path = os.path.join(root, name)
            if dir_path not in valid_local_paths:
                shutil.rmtree(dir_path)
                stats['deleted'] += 1
                print(f"Deleted orphaned directory tree: {dir_path}")

    print(f"\nMirror Complete! Downloaded: {stats['downloaded']} | Skipped: {stats['skipped']} | Deleted: {stats['deleted']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively mirror a Google Drive folder locally.")
    parser.add_argument("folder_id", nargs='?', default="1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1", help="The ID of the root Google Drive folder to mirror (default: 1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1).")
    parser.add_argument("--destination", default="drive_data", help="The local directory to mirror into.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
        
    try:
        drive_svc = get_authenticated_drive_service()
        mirror_folder_sync_recursive(drive_svc, args.folder_id, args.destination)
    except Exception as e:
        print(f"\nScript failed: {e}")