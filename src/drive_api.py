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
# Update this with the specific client secret file name in your root directory
CLIENT_SECRET_FILE = json.load(open("client_secret.json"))["CLIENT_SECRET_FILE"]
class DriveAPI:
    def __init__(self):
        self.creds = self._authenticate()
        self.service = build("drive", "v3", credentials=self.creds)

    def _authenticate(self):
        """Authenticates the user and returns credentials."""
        creds = None
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRET_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return creds

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


