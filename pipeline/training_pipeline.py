from src.data_ingestion import DataIngestion
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class TrainingPipeline:

    def __init__(self, drive_link : str):
        
        self.drive_link = drive_link

    def run_pipeline(self):

        try:
            logger.info("Training pipeline initialized successfully.")

            # Data Ingestion
            logger.info("Data Ingestion Started.")

            ingest = DataIngestion(share_url = self.drive_link)
            ingest.download_file()

            logger.info("Data ingestion completed.")

            # Data Preprocessing
            logger.info("Data Processing started.")
            logger.info("Data processing completed.")

            logger.info("Training pipeline completed successfully.")

        except Exception as e:

            logger.error("Failed to run training pipeline.")
            raise CustomException("Error while running trainig pipeline.", e)
          

if __name__ == "__main__":

    trainer = TrainingPipeline(GDRIVE_LINK)
    trainer.run_pipeline()