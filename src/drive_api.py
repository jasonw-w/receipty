import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io
import json
# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/drive.file"
]
# Handle potential missing file for CLIENT_SECRET_FILE
try:
    ptr_data = json.load(open("client_secret.json"))
    CLIENT_SECRET_FILE = ptr_data["CLIENT_SECRET_FILE"]
except Exception:
    CLIENT_SECRET_FILE = None

import streamlit as st

class DriveAPI:
    def __init__(self):
        self.creds = None
        self.service = None
        self._try_load_existing_creds()

    def _try_load_existing_creds(self):
        """Attempts to load existing credentials from token.json or Streamlit secrets."""
        # 1. Check local token file
        if os.path.exists("token.json"):
            try:
                self.creds = Credentials.from_authorized_user_file("token.json", SCOPES)
            except Exception:
                self.creds = None
        
        # 2. Check Streamlit Secrets for a pre-generated token (preferred for Cloud)
        if (not self.creds or not self.creds.valid) and "gcp_token" in st.secrets:
             try:
                token_info = dict(st.secrets["gcp_token"])
                self.creds = Credentials.from_authorized_user_info(token_info, SCOPES)
             except Exception:
                 pass

        # If we have valid creds, build the service immediately
        if self.creds and self.creds.valid:
            self.service = build("drive", "v3", credentials=self.creds)
        elif self.creds and self.creds.expired and self.creds.refresh_token:
            try:
                self.creds.refresh(Request())
                self.service = build("drive", "v3", credentials=self.creds)
                # Save refreshed token
                with open("token.json", "w") as token:
                    token.write(self.creds.to_json())
            except Exception:
                self.creds = None
                self.service = None

    def get_auth_flow(self):
        """Creates and returns the OAuth flow object."""
        if "gcp_oauth" in st.secrets:
            config = dict(st.secrets["gcp_oauth"])
            # Ensure it is wrapped in 'installed' or 'web'
            if "installed" not in config and "web" not in config:
                config = {"installed": config}
            return InstalledAppFlow.from_client_config(config, SCOPES)
        elif CLIENT_SECRET_FILE and os.path.exists(CLIENT_SECRET_FILE):
             return InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        else:
             raise FileNotFoundError("No Google Drive secrets found.")

    def get_auth_url(self, redirect_uri):
        """Generates the authorization URL for the user to visit."""
        flow = self.get_auth_flow()
        flow.redirect_uri = redirect_uri
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        return auth_url

    def authenticate_with_code(self, code, redirect_uri):
        """Exchanges the auth code for a token and initializes the service."""
        flow = self.get_auth_flow()
        flow.redirect_uri = redirect_uri
        flow.fetch_token(code=code)
        self.creds = flow.credentials
        self.service = build("drive", "v3", credentials=self.creds)
        
        # Save locally for caching (optional in Cloud, but good for session)
        with open("token.json", "w") as token:
            token.write(self.creds.to_json())
        return True

    def list_files(self, page_size=10):
        """Lists files from Google Drive."""
        try:
            results = (
                self.service.files()
                .list(pageSize=page_size, fields="nextPageToken, files(id, name)")
                .execute()
            )
            items = results.get("files", [])
            return items
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []

    def download_file(self, file_id, save_path):
        """Downloads a file from Google Drive."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                # print(f"Download {int(status.progress() * 100)}%.")
            
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            print(f"Downloaded to {save_path}")
            return True
        except HttpError as error:
            print(f"An error occurred: {error}")
            return False
    
    
    def get_file_id(self, filename):
        """Finds a file ID by its name."""
        try:
            query = f"name = '{filename}' and trashed = false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            if files:
                return files[0]['id']
            return None
        except HttpError as error:
             print(f"An error occurred searching for file: {error}")
             return None

    def create_csv_if_not_exists(self, filename="receipts.csv", initial_content="Date,Store,Item,Amount\n"):
        # Check if exists
        file_id = self.get_file_id(filename)
        
        if file_id:
            print(f"File '{filename}' already exists (ID: {file_id}).")
            return file_id
            
        print(f"File '{filename}' not found. Creating...")
        
        # Determine media upload
        if os.path.exists(initial_content): # If argument is a path
             media = MediaFileUpload(initial_content, mimetype='text/csv')
        else:
             # Create from string using MediaIoBaseUpload (NOT MediaFileUpload)
             fh = io.BytesIO(initial_content.encode('utf-8'))
             from googleapiclient.http import MediaIoBaseUpload
             media = MediaIoBaseUpload(fh, mimetype='text/csv')

        try:
            file = self.service.files().create(
                body={
                    'name': filename,
                    'mimeType': 'text/csv' # Generic CSV
                    # 'mimeType': 'application/vnd.google-apps.spreadsheet' # Unwrap to Sheet
                },
                media_body=media, 
                fields='id'
            ).execute()
            print(f"Created file '{filename}' with ID: {file.get('id')}")
            return file.get('id')
        except HttpError as error:
             print(f"An error occurred: {error}")
             return None
    
    def append_to_csv(self, file_id, new_content_str):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        current_text = fh.getvalue().decode('utf-8')
        from googleapiclient.http import MediaIoBaseUpload
        if not current_text.endswith('\n'):
            current_text += '\n'    
        final_content = current_text + new_content_str
        self.service.files().update(
            fileId=file_id,
            media_body=MediaIoBaseUpload(io.BytesIO(final_content.encode('utf-8')), mimetype='text/csv'),
            fields='id'
        ).execute()
    
    def read_csv(self, file_id):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8')
        
            
        
            
if __name__ == "__main__":
    drive = DriveAPI()
    
    # Test CSV creation logic
    print("Testing 'receipts_log.csv' creation/check...")
    csv_id = drive.create_csv_if_not_exists("receipts_log.csv", "Date,Store,Item,Price\n")


