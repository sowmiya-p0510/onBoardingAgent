import datetime
import os
from google.cloud import storage
from typing import Optional

class GCSManager:
    def __init__(self):
        """Initialize the GCS Manager."""
        self.bucket_name: Optional[str] = os.environ.get("GCS_BUCKET_NAME")
        self.client = storage.Client()
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME is not set in environment variables")
        self.bucket = self.client.bucket(self.bucket_name)

    def get_signed_url(self, path: Optional[str], expiration=3600) -> str:
        """
        Generate a signed URL for a GCS object with authentication.

        Args:
            path: Path in GCS, just the folder and object path without bucket name
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Signed URL to the object or empty string if file doesn't exist
        """
        if not path:
            return ""

        # Remove any bucket name if it was accidentally included
        if self.bucket_name and path.startswith(self.bucket_name):
            path = path.replace(f"{self.bucket_name}/", "", 1)

        blob = self.bucket.blob(path)
        if not blob.exists():
            print(f"Warning: File does not exist at path: {path}")
            return ""

        try:
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(seconds=expiration),
                method="GET"
            )
            return url
        except Exception as e:
            print(f"Error generating signed URL: {e}")
            return ""

    def list_files_in_folder(self, folder_path: Optional[str]) -> list[str]:
        """
        List all files in a GCS folder.

        Args:
            folder_path: Path to the folder in GCS

        Returns:
            List of file paths that actually exist
        """
        if not folder_path:
            return []

        if not folder_path.endswith('/'):
            folder_path += '/'

        try:
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=folder_path))
            return [blob.name for blob in blobs if not blob.name.endswith('/')]
        except Exception as e:
            print(f"Error listing files in folder {folder_path}: {e}")
            return []

    def check_file_exists(self, path: Optional[str]) -> bool:
        """
        Check if a file exists in GCS.

        Args:
            path: Path in GCS

        Returns:
            True if file exists, False otherwise
        """
        if not path:
            return False

        if self.bucket_name and path.startswith(self.bucket_name):
            path = path.replace(f"{self.bucket_name}/", "", 1)

        blob = self.bucket.blob(path)
        return blob.exists()
