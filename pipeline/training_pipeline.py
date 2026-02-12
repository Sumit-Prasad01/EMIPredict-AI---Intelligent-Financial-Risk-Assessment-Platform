from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training_regression import TrainRegressionModel
from src.model_training_classification import TrainClassificationModel
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class TrainingPipeline:

    def run_pipeline(self):

        try:
            logger.info("Training pipeline initialized successfully.")

            ingest = DataIngestion(GDRIVE_LINK)
            ingest.download_file()

            data_processor = DataProcessor(RAW_DATA_PATH, PROCESSED_DATA_PATH_CL, PROCESSED_DATA_PATH_REG)
            data_processor.run()

            trainer = TrainClassificationModel(SAVE_CL_MODEL_PATH)
            trainer.run()

            trainer = TrainRegressionModel(SAVE_REG_MODEL_PATH)
            trainer.run()


            logger.info("Training pipeline completed successfully.")

        except Exception as e:

            logger.error("Failed to run training pipeline.")
            raise CustomException("Error while running trainig pipeline.", e)
          

if __name__ == "__main__":

    trainer = TrainingPipeline()
    trainer.run_pipeline()