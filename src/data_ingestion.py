import os
import requests
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataIngestion:
    """
    Downloads a file from a Google Drive share link and saves it locally.
    """

    def __init__(self, share_url: str, save_dir: str = SAVE_DIR, save_name: str = SAVE_NAME):
        self.share_url = share_url
        self.save_dir = save_dir
        self.save_name = save_name
        self.file_id = self._extract_file_id()

        logger.info("Data ingestion Initialized.")

    def _extract_file_id(self) -> str:
        try:
            """
            Extracts file ID from the Google Drive share link.
            """
            if "id=" in self.share_url:
                return self.share_url.split("id=")[-1]
            elif "/d/" in self.share_url:
                return self.share_url.split("/d/")[1].split("/")[0]
                
            logger.info("File id extracted successfyully.")
        
        except Exception as e:
            logger.error("Failed to extract file id.")
            raise CustomException("Error while extracting file id", e)

    def _build_download_url(self) -> str:
        try:
            """
            Builds the direct download URL for Google Drive.
            """
            logger.info("Download url builded successfully.")

            return f"https://drive.google.com/uc?export=download&id={self.file_id}"  

        except Exception as e:
            logger.error("Failed to build download url.")
            raise CustomException("Error while building download url.", e)

    def download_file(self):
        try:
            """
            Downloads the file from Google Drive and saves it to the specified directory.
            """
            os.makedirs(self.save_dir, exist_ok=True)
            download_url = self._build_download_url()

            print(f" Downloading file from: {download_url}")

            response = requests.get(download_url, stream=True)
            if response.status_code != 200:
                raise Exception(f" Download failed (HTTP {response.status_code})")

            file_path = os.path.join(self.save_dir, self.save_name)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f" File saved successfully at: {file_path}")

            logger.info("File downloaded successfully.")
            return file_path

        except Exception as e:
            logger.error("Error while downloading file.")
            raise CustomException("Failed to download file.", e)

