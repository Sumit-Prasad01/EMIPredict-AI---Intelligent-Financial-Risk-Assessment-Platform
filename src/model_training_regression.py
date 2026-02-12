import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.Loader import Loader
from scipy.stats import randint
import mlflow
import mlflow.sklearn


logger = get_logger(__name__)


class TrainRegressionModel:

    def __init__(self, model_output_path : str):
        
        self.lgb_sample_weights = None
        self.model = None
        self.model_output_path = model_output_path
        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS_REGRESSION


    def load_data_and_train_model(self):
        try:
            
            self.X_train, self.X_test, self.y_train, self.y_test = Loader.load_processed_regression_data(X_TRAIN_PATH_REG, X_TEST_PATH_REG, y_TRAIN_PATH_REG, y_TEST_PATH_REG)

            logger.info("Initializing Our Model.")

            lgbm_model =  lgb.LGBMRegressor(random_state = self.random_search_params["random_state"])

            logger.info("Starting our HyperParameter Tuning")

            random_search = RandomizedSearchCV(
                    estimator = lgbm_model,
                    param_distributions = self.params_dist,
                    n_iter = self.random_search_params["n_iter"],
                    cv = self.random_search_params["cv"],
                    n_jobs = self.random_search_params["n_jobs"],
                    verbose = self.random_search_params["verbose"],
                    random_state = self.random_search_params["random_state"],
                    scoring = self.random_search_params["scoring"],
            )

            logger.info("Starting our HyperParameter Tunning")

            random_search.fit(self.X_train, self.y_train, sample_weight = self.lgb_sample_weights)

            logger.info("HyperParameter Tunning Completed")

            best_params = random_search.best_params_
            self.model = random_search.best_estimator_

            logger.info(f"Best Parameters are {best_params}")


        except Exception as e:
            logger.error("Error while loading data and training model.")
            raise CustomException("Failed to load data and train model.", e)
    
    def evaluate_model(self):
        try:
            logger.info("Evaluating Our Regression Model")

            y_pred = self.model.predict(self.X_test)

            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"MAE  : {mae}")
            logger.info(f"MSE  : {mse}")
            logger.info(f"RMSE : {rmse}")
            logger.info(f"R2 Score : {r2}")

            return {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2_score": r2
            }

        except Exception as e:
            logger.error(f"Error During Evaluating model: {e}")
            raise CustomException("Failed to Evaluate Regression Model", e)
    
    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("Saving the model.")

            joblib.dump(self.model, self.model_output_path)
            logger.info(f"Model Saved to {self.model_output_path}")
        
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to Save model",e)
        


    def run(self):
        try:
            with mlflow.start_run():

                logger.info("Starting Our MLflow Experimentation")

                logger.info("Starting our model training pipeline")

                logger.info("Logging that training and testing dataset to MLFlow")
                # mlflow.log_artifact(self.X_train, artifact_path = 'datasets')
                # mlflow.log_artifact(self.X_test, artifact_path = 'datasets')

                self.load_data_and_train_model()

                metrics = self.evaluate_model()
                self.save_model()

                logger.info("Logging the model into mlflow")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Params and metrics to Mlflow")
                mlflow.log_params(self.model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model Training Successfully Completed")
            
        except Exception as e:
            logger.error(f"Error while running model training pipeline {e}")
            raise CustomException("Failed to run train pipeline",e)
        

if __name__ == "__main__":
    trainer = TrainRegressionModel(SAVE_REG_MODEL_PATH)
    trainer.run()