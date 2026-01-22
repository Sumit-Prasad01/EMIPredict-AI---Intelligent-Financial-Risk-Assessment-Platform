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

    def __init__(self, input_path, output_path_cl, output_path_reg):

        self.input_path : str = input_path
        self.output_path_cl = output_path_cl
        self.output_path_reg = output_path_reg
        self.df = None
        self.label_encoder = None
        self.X = None
        self.y = None
        # Classification
        self.X_train_cl = None 
        self.X_test_cl = None
        self.y_train_cl = None
        self.y_test_cl = None
        self.X_train_p_cl = None
        self.X_test_p_cl = None
        self.preprocessor_cl = None
        # Regression
        self.X_reg = None
        self.y_reg = None
        self.X_train_reg = None
        self.X_test_reg = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.preprocessor_reg = None
        self.X_train_p_reg = None
        self.X_test_p_reg = None

        os.makedirs(self.output_path_cl, exist_ok = True)
        os.makedirs(self.output_path_reg, exist_ok = True)

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
            y_encoded = self.label_encoder.fit_transform(self.y)

            label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
            logger.info(f"Label mappings : {label_mapping}")
            
            class_weights_arr = compute_class_weight(
            class_weight = "balanced",
            classes = np.unique(y_encoded),
            y = y_encoded
            )

            class_weights = dict(zip(np.unique(y_encoded), class_weights_arr))
            logger.info(f"Class weights : {class_weights}")
            
            self.X_train_cl, self.X_test_cl, self.y_train_cl, self.y_test_cl = train_test_split(
                self.X,
                y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded
            )

            logger.info("Data encoding and class imbalancing handeled successfully.")

        except Exception as e:
            logger.error("Error while encoding data and handling class imbalance.")
            raise CustomException("Failed to encode data and handel class imbalance.", e)
        
    
    def preprocess_classification(self):
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

            self.X_train_p_cl = self.preprocessor.fit_transform(self.X_train_cl)
            self.X_test_p_cl = self.preprocessor.transform(self.X_test_cl)

            logger.info(f"Null Values after processing classification data : {np.isnan(self.X_train_p_cl).sum()}")

            logger.info("Data processing for classification completed.")

        except Exception as e:
            logger.error("Error while processing classification data.")
            raise CustomException("Failed to process classification  data.", e)
        
    

    def preprocess_regression(self):
        try:

            self.X_reg = self.df.drop(columns=["emi_eligibility", "max_monthly_emi"])
            self.y_reg = self.df["max_monthly_emi"]

            self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(
                self.X_reg,
                self.y_reg,
                test_size=0.2,
                random_state=42
            )

            categorical_features = [
                "gender",
                "marital_status",
                "education",
                "employment_type",
                "company_type",
                "house_type",
                "emi_scenario"
            ]

            numerical_features = list(
                set(self.X_reg.columns) - set(categorical_features)
            )

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            self.preprocessor_reg = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features),
                    ("cat", cat_pipeline, categorical_features)
                ]
            )

            self.X_train_p_reg = self.preprocessor_reg.fit_transform(self.X_train_reg)
            self.X_test_p_reg = self.preprocessor_reg.transform(self.X_test_reg)

           
            logger.info(f"Null Values after processing regression data : {np.isnan(self.X_train_p_reg).sum()}")
        
        except Exception as e:
            logger.error("Error while processing regression data.")
            raise CustomException("Failed to process regression data.", e)

        
    
    def save_artifacts(self):
        try:

            joblib.dump(self.X_train_p_cl, X_TRAIN_PATH_CL)
            joblib.dump(self.X_test_p_cl, X_TEST_PATH_CL)
            joblib.dump(self.y_train_cl,y_TRAIN_PATH_CL)
            joblib.dump(self.y_test_cl, y_TEST_PATH_CL)
            joblib.dump(self.label_encoder, ENCODER_PATH_CL)
            joblib.dump(self.preprocessor_cl, PROCESSOR_PATH_CL)

            logger.info("Processed classification artifacts saved successfully.")

            joblib.dump(self.X_train_p_reg, X_TRAIN_PATH_REG)
            joblib.dump(self.X_test_p_reg, X_TEST_PATH_REG)
            joblib.dump(self.y_train_reg,y_TRAIN_PATH_REG)
            joblib.dump(self.y_test_reg, y_TEST_PATH_REG)
            joblib.dump(self.label_encoder, ENCODER_PATH_REG)
            joblib.dump(self.preprocessor_reg, PROCESSOR_PATH_REG)

            logger.info("Processed regression artifacts saved successfully.")


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
            self.preprocess_classification()
            self.preprocess_regression()
            self.save_artifacts()

            logger.info("Data processing pipeline ran successfully.")
        
        except Exception as e:
            logger.error("Error while running data processing pipeline.")
            raise CustomException("Failed to run data processing pipeline.", e)
        


if __name__ == "__main__":

    data_processor = DataProcessor(RAW_DATA_PATH, PROCESSED_DATA_PATH_CL, PROCESSED_DATA_PATH_REG)
    data_processor.run()