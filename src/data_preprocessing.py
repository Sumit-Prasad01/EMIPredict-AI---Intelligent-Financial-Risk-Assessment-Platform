import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

from src.custom_exception import CustomException
from src.logger import get_logger
from utils.Loader import Loader
from config.paths_config import *

logger = get_logger(__name__)


class DataProcessor:

    def __init__(self, input_path, output_path):

        self.input_path : str = input_path
        self.output_path : str = output_path
        self.df = None
        self.label_encoder = None
        self.X = None
        self.y = None
        self.y_encoded = None
        self.X_train = None 
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_p = None
        self.X_test_p = None
        self.preprocessor = None

        os.makedirs(self.output_path, exist_ok = True)

        logger.info("Data processing initialized.")


    def fix_dtypes(self):
        try:

            self.df = Loader.load_data(self.input_path)

            categorical_cols = [
            "gender", "marital_status", "education", "employment_type", 
            "company_type", "house_type", "existing_loans", 
            "emi_scenario", "emi_eligibility"
            ]
            numerical_cols = [
                "age", "monthly_salary", "years_of_employment", "monthly_rent", 
                "family_size", "dependents", "school_fees", "college_fees", 
                "travel_expenses", "groceries_utilities", "other_monthly_expenses", 
                "current_emi_amount", "credit_score", "bank_balance", 
                "emergency_fund", "requested_amount", "requested_tenure", 
                "max_monthly_emi"
            ]

            for col in self.df.columns:
                if col in categorical_cols:
                    self.df[col] = self.df[col].astype('category')
                elif col in numerical_cols:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            logger.info("Dtypes fixed successfully.")
        
        except Exception as e:
            logger.error("Error while changing dtypes.")
            raise CustomException("Failed to change dtypes.". e)
        
    
    def feature_engineering(self):
        try:

            categorical_cols = [
                "gender", "marital_status", "education",
                "employment_type", "company_type",
                "house_type", "existing_loans",
                "emi_scenario"
            ]

            numerical_cols = [
                col for col in self.df.columns
                if col not in categorical_cols + ["emi_eligibility", "max_monthly_emi"]
            ]

            EPS = 1e-6

            self.df["total_monthly_expenses"] = (
                self.df["monthly_rent"] +
                self.df["school_fees"] +
                self.df["college_fees"] +
                self.df["travel_expenses"] +
                self.df["groceries_utilities"] +
                self.df["other_monthly_expenses"] +
                self.df["current_emi_amount"]
            )

            self.df["disposable_income"] = self.df["monthly_salary"] - self.df["total_monthly_expenses"]

            self.df["emi_burden_ratio"] = self.df["current_emi_amount"] / (self.df["monthly_salary"] + EPS)

            self.df["expense_income_ratio"] = self.df["total_monthly_expenses"] / (self.df["monthly_salary"] + EPS)

            self.df["emergency_fund_ratio"] = self.df["emergency_fund"] / (self.df["total_monthly_expenses"] + EPS)

            self.df["savings_ratio"] = self.df["bank_balance"] / (self.df["monthly_salary"] * 6 + EPS)

            self.df["existing_loans"] = self.df["existing_loans"].map({"Yes": 1, "No": 0})

            logger.info("Feature engineering completed successfully.")

        except Exception as e:
            logger.error("Error while creating new features.")
            raise CustomException("Failed to create new features.", e)
        
    
    def split_data(self):
        try:

            self.X = self.df.drop(columns=["emi_eligibility", "max_monthly_emi"])
            self.y = self.df["emi_eligibility"]

        except Exception as e:
            logger.error("Error while splitting data.")
            raise CustomException("Failed to split data.", e)
        
    
    def encode_and_handel_imbalance_data(self):
        try:

            self.label_encoder = LabelEncoder()
            self.y_encoded = self.label_encoder.fit_transform(self.y)

            label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
            logger.info(f"Label mappings : {label_mapping}")
            
            class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.y_encoded),
            y=self.y_encoded
            )

            class_weights = dict(zip(np.unique(self.y_encoded), class_weights_arr))
            logger.info(f"Class weights : {class_weights}")
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=self.y_encoded
            )

            logger.info("Data encoding and class imbalancing handeled successfully.")

        except Exception as e:
            logger.error("Error while encoding data and handling class imbalance.")
            raise CustomException("Failed to encode data and handel class imbalance.", e)
        
    
    def preprocess(self):
        try:

            categorical_features = [
                "gender", "marital_status", "education",
                "employment_type", "company_type",
                "house_type", "emi_scenario"
            ]

            numerical_features = [
                col for col in self.X.columns if col not in categorical_features
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            self.preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            self.X_train_p = self.preprocessor.fit_transform(self.X_train)
            self.X_test_p = self.preprocessor.transform(self.X_test)

            logger.info(f"Null Values after processing data : {np.isnan(self.X_train_p).sum()}")

            logger.info("Data processing completed.")

        except Exception as e:
            logger.error("Error while processing data.")
            raise CustomException("Failed to process data.", e)
        
    
    def save_artifacts(self):
        try:

            joblib.dump(self.X_train_p, X_TRAIN_PATH)
            joblib.dump(self.X_test_p, X_TEST_PATH)
            joblib.dump(self.y_train,y_TRAIN_PATH)
            joblib.dump(self.y_test, y_TEST_PATH)
            joblib.dump(self.label_encoder, ENCODER_PATH)
            joblib.dump(self.preprocessor, PROCESSOR_PATH)

            logger.info("Processed artifacts saved successfully.")

        except Exception as e:
            logger.error("Error while saving processed artifacts.")
            raise CustomException("Failed to save processed artifacts.", e)
        
    
    def run(self):
        try:

            logger.info("Starting data processing pipeline.")

            self.fix_dtypes()
            self.feature_engineering()
            self.split_data()
            self.encode_and_handel_imbalance_data()
            self.preprocess()
            self.save_artifacts()

            logger.info("Data processing pipeline ran successfully.")
        
        except Exception as e:
            logger.error("Error while running data processing pipeline.")
            raise CustomException("Failed to run data processing pipeline.", e)
        


if __name__ == "__main__":

    data_processor = DataProcessor(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    data_processor.run()