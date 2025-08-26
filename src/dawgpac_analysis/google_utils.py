import io
import os
# import pickle
import time
import random
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google.cloud import storage

# Define the scopes needed for the APIs that will use the user OAuth flow.
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/presentations'
]


def get_user_drive_slides_credentials():
    """
    Handles the OAuth 2.0 flow using credentials from the `secrets/` directory.
    This is used to authenticate the Personal Account for Drive and Slides.
    """
    # Define paths to your secret files within the 'secrets' directory.
    token_path = 'secrets/token.json'
    credentials_path = 'secrets/credentials.json'

    creds = None
    # The token.json file stores the user's access and refresh tokens, and is
    # created automatically in the secrets/ folder on the first successful login.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # This uses the `credentials.json` from your personal GCP project.
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run in the secrets/ folder.
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return creds


class GoogleCloudHelper:
    """
    A helper class to interact with Google services using a dual-authentication strategy.
    """
    def __init__(self, gcs_quota_project_id=None):
        """
        Initializes Google Cloud clients with separate authentication:
        - GCS uses Application Default Credentials (from your STANFORD account via `gcloud`).
        - Drive and Slides use user-provided OAuth credentials (from your PERSONAL account).
        - gcs_quota_project_id: The ID of a GCP project to use for GCS API quotas.
        """
        print("Initializing Google Cloud Helper with dual authentication...")

        # Pass the provided project ID to the GCS client.
        # This satisfies the client's need for a quota project, while still using
        # the Org account's Application Default Credentials for authentication.
        self.storage_client = storage.Client(project=gcs_quota_project_id)
        print(f"  - GCS Client initialized using project '{gcs_quota_project_id}' for quota and Org Account for auth.")

        # Initialize Drive and Slides clients using Personal Account credentials.
        # This calls our dedicated function to get credentials from the OAuth flow.
        drive_slides_creds = get_user_drive_slides_credentials()
        self.drive_service = build('drive', 'v3', credentials=drive_slides_creds)
        self.slides_service = build('slides', 'v1', credentials=drive_slides_creds)
        print("  - Drive and Slides services initialized using user OAuth flow (Personal Account).")

    def download_gcs_file_as_stringio(self, bucket_name, source_blob_name):
        """Downloads a file from GCS and returns it as an in-memory text buffer."""
        # This correctly uses self.storage_client (STANFORD Account)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        try:
            content = blob.download_as_string().decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not download GCS file '{source_blob_name}'. Error: {e}")
            content = ""
        return io.StringIO(content)

    def create_drive_folder(self, folder_name, parent_folder_id=None):
        """Creates a new folder in Google Drive."""
        # This correctly uses self.drive_service (Personal Account)
        parents = [parent_folder_id] if parent_folder_id else []
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': parents
        }
        folder = self.drive_service.files().create(
            body=file_metadata, fields='id').execute()
        return folder.get('id')

    def upload_buffer_to_drive(self, buffer, folder_id, file_name, mimetype):
        """Uploads an in-memory buffer to a specific folder in Google Drive."""
        # This correctly uses self.drive_service (Personal Account)
        buffer.seek(0)
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaIoBaseUpload(buffer, mimetype=mimetype, resumable=True)
        request = self.drive_service.files().create(
            body=file_metadata, media_body=media, fields='id, webContentLink'
        )
        # Implement retry logic with exponential backoff for timeouts
        max_retries = 3
        for i in range(max_retries):
            try:
                file = request.execute()
                print(f"Uploaded '{file_name}' to Drive. ID: {file.get('id')}")
                return file
            except TimeoutError:
                if i < max_retries - 1:
                    wait_time = (2 ** i) + random.random()
                    print(f"--> TimeoutError on '{file_name}'. Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"--> TimeoutError on '{file_name}'. Max retries exceeded.")
                    raise

    def replace_image_in_slides(self, presentation_id, image_url, placeholder_text):
        """Finds a shape with placeholder text and replaces it with an image."""
        # This correctly uses self.slides_service (Personal Account)
        requests = [{
            'replaceAllShapesWithImage': {
                'imageUrl': image_url,
                'replaceMethod': 'CENTER_INSIDE',
                'containsText': {
                    'text': f'{{{{{placeholder_text}}}}}',
                    'matchCase': False
                }
            }
        }]
        body = {'requests': requests}
        self.slides_service.presentations().batchUpdate(
            presentationId=presentation_id, body=body
        ).execute()
