import io
import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google.cloud import storage

SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/devstorage.read_only'
]

def get_user_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

class GoogleCloudHelper:
    def __init__(self):
        self.credentials = get_user_credentials()
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.slides_service = build('slides', 'v1', credentials=self.credentials)
        self.storage_client = storage.Client(credentials=self.credentials)

    def download_gcs_file_as_stringio(self, bucket_name, source_blob_name):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        content = blob.download_as_string().decode('utf-8')
        return io.StringIO(content)

    def create_drive_folder(self, folder_name, parent_folder_id=None):
        parents = [parent_folder_id] if parent_folder_id else []
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': parents
        }
        folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

    def upload_buffer_to_drive(self, buffer, folder_id, file_name, mimetype):
        buffer.seek(0)
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaIoBaseUpload(buffer, mimetype=mimetype, resumable=True)
        file = self.drive_service.files().create(
            body=file_metadata, media_body=media, fields='id, webContentLink'
        ).execute()
        print(f"Uploaded '{file_name}' to Drive. ID: {file.get('id')}")
        return file

    def replace_image_in_slides(self, presentation_id, image_url, placeholder_text):
        requests = [{'replaceAllShapesWithImage': {
            'imageUrl': image_url, 'replaceMethod': 'CENTER_INSIDE',
            'containsText': {'text': f'{{{{{placeholder_text}}}}}', 'matchCase': False}
        }}]
        self.slides_service.presentations().batchUpdate(
            presentationId=presentation_id, body={'requests': requests}
        ).execute()
