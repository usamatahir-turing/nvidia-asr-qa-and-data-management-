import os
import requests
import argparse
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as Oauth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configuration
PARENT_FOLDER_ID = '1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1'
TARGET_SERVICE_ACCOUNT = 'delivery-nvidia@delivery-nvidia.iam.gserviceaccount.com'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def get_subfolder_id(service, parent_id, folder_name):
    """Finds a subfolder ID by name within a parent folder."""
    query = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(
        q=query, 
        spaces='drive', 
        fields='files(id, name)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    files = results.get('files', [])
    if not files:
        return None
    return files[0]['id']

def get_file_id(service, parent_id, file_name):
    """Finds a file ID by name within a parent folder."""
    query = f"'{parent_id}' in parents and name = '{file_name}' and trashed = false"
    results = service.files().list(
        q=query, 
        spaces='drive', 
        fields='files(id, name)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    files = results.get('files', [])
    if not files:
        return None
    return files[0]['id']

def push_reports(variant):
    source_dir = os.path.join(SCRIPT_DIR, f'reports_{variant}')
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    service = get_authenticated_drive_service()
    
    # Pre-cache subfolders in the parent directory to avoid redundant API calls
    print(f"Fetching subfolders from Drive folder {PARENT_FOLDER_ID}...")
    query = f"'{PARENT_FOLDER_ID}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(
        q=query, 
        spaces='drive', 
        fields='files(id, name)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    drive_subfolders = {f['name']: f['id'] for f in results.get('files', [])}

    for filename in os.listdir(source_dir):
        if not filename.endswith(f'_{variant}.md'):
            continue
            
        file_path = os.path.join(source_dir, filename)
        # Extract subfolder name: NV-AR-SS03-CONVO07_approved.md -> NV-AR-SS03-CONVO07
        subfolder_name = filename.rsplit('_', 1)[0]
        
        target_folder_id = drive_subfolders.get(subfolder_name)
        
        if not target_folder_id:
            print(f"Warning: Subfolder '{subfolder_name}' not found on Drive. Skipping {filename}.")
            continue
            
        print(f"Processing {filename} -> Drive Folder: {subfolder_name}")
        
        existing_file_id = get_file_id(service, target_folder_id, filename)
        media = MediaFileUpload(file_path, mimetype='text/markdown', resumable=True)
        
        if existing_file_id:
            print(f"Updating existing file: {filename}")
            service.files().update(
                fileId=existing_file_id,
                media_body=media,
                supportsAllDrives=True
            ).execute()
        else:
            print(f"Uploading new file: {filename}")
            file_metadata = {
                'name': filename,
                'parents': [target_folder_id]
            }
            service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push quality reports to Google Drive.")
    parser.add_argument("--variant", choices=['fixed', 'approved'], required=True, help="The variant of reports to upload (fixed or approved).")
    
    args = parser.parse_args()
    
    try:
        push_reports(args.variant)
        print("\nAll tasks completed.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
