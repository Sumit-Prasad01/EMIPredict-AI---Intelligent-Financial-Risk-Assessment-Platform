import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.Loader import Loader
from scipy.stats import randint
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)


class TrainClassificationModel:

    def __init__(self, model_output_path):
        
        self.lgb_sample_weights = None
        self.model = None
        self.model_output_path = model_output_path
        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_data_and_train_model(self):
        try:
            
            self.X_train, self.X_test, self.y_train, self.y_test, class_weights = Loader.load_processed_classification_data(X_TRAIN_LOAD_PATH_CL, X_TEST_LOAD_PATH_CL, y_TRAIN_LOAD_PATH_CL, y_TEST_LOAD_PATH_CL, CLASS_WEIGHTS_PATH_CL)

            logger.info("Initializing Our Model.")

            lgbm_model =  lgb.LGBMClassifier(random_state = self.random_search_params["random_state"])

            self.lgb_sample_weights = np.array([class_weights[c] for c in self.y_train])

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
    
    def evaluate_mode(self):
        try:
            logger.info("Evaluating Our Model")

            y_pred = self.model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average = "weighted")
            recall = recall_score(self.y_test, y_pred, average = "weighted")
            f1score = f1_score(self.y_test, y_pred, average = "weighted")

            logger.info(f"Accuracy Score :{accuracy}")
            logger.info(f"Precision Score :{precision}")
            logger.info(f"Recall Score :{recall}")
            logger.info(f"F1 Score Score :{f1score}")


            return {
                "accuracy" : accuracy,
                "precision" : precision,
                "recall" : recall,
                "f1-score" : f1score
            }
        
        except Exception as e:
            logger.error(f"Error During Evaluating model {e}")
            raise CustomException("Failed to Evaluate model",e)
    
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

                metrics = self.evaluate_mode()
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
    trainer = TrainClassificationModel(SAVE_CL_MODEL_PATH)
    trainer.run()