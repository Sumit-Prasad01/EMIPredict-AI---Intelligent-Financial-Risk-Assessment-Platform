from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *
import pandas as pd
import joblib
import numpy as np

logger = get_logger(__name__)


class Loader:
    """Utility class for loading raw and processed data."""

    @staticmethod
    def load_data(file_path: str):
        """Load raw data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise CustomException("Error while loading and reading data", e)

    @staticmethod
    def load_processed_data(X_train_path : str , X_test_path : str , y_train_path : str , y_test_path : str ):
        """Load processed train-test data from pickle files."""
        try:
            logger.info("Loading processed data...")

            X_train = joblib.load(X_train_path)
            X_test = joblib.load(X_test_path)
            y_train = joblib.load(y_train_path)
            y_test = joblib.load(y_test_path)

            logger.info("Processed data loaded successfully.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise CustomException("Error while loading processed data", e)
        

    @staticmethod
    def load_model(model_path : str):
        """Load model from atrifacts dir."""
        try:
    
            model = joblib.load(model_path)
            logger.info("Model loaded successfully.")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model : {e}")
            raise CustomException("Error while loading model", e)
    

            