from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
from config.paths_config import *


# Load Artifacts
preprocessor_cls = joblib.load(CL_PREPROCESSOR)
label_encoder = joblib.load(ENCODER)
preprocessor_reg = joblib.load(REG_PREPROCESSOR)

model_cls = joblib.load(SAVED_CL_MODEL_PATH)
model_reg = joblib.load(SAVED_REG_MODEL_PATH)


app = FastAPI(title="EMIPredict AI API", version="1.0")



# Input Schema
class CustomerInput(BaseModel):
    age: int
    gender: str
    marital_status: str
    education: str
    monthly_salary: float
    employment_type: str
    years_of_employment: float
    company_type: str
    house_type: str
    monthly_rent: float
    family_size: int
    dependents: int
    school_fees: float
    college_fees: float
    travel_expenses: float
    groceries_utilities: float
    other_monthly_expenses: float
    existing_loans: str  # Yes/No
    current_emi_amount: float
    credit_score: float
    bank_balance: float
    emergency_fund: float
    emi_scenario: str
    requested_amount: float
    requested_tenure: int



# Feature Engineering Function
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    EPS = 1e-6

    # Convert Yes/No to numeric (must match training)
    df["existing_loans"] = df["existing_loans"].map({"Yes": 1, "No": 0})

    # Total monthly expenses
    df["total_monthly_expenses"] = (
        df["monthly_rent"] +
        df["school_fees"] +
        df["college_fees"] +
        df["travel_expenses"] +
        df["groceries_utilities"] +
        df["other_monthly_expenses"] +
        df["current_emi_amount"]
    )

    # Disposable income
    df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"]

    # Financial ratios
    df["emi_burden_ratio"] = df["current_emi_amount"] / (df["monthly_salary"] + EPS)
    df["expense_income_ratio"] = df["total_monthly_expenses"] / (df["monthly_salary"] + EPS)
    df["emergency_fund_ratio"] = df["emergency_fund"] / (df["total_monthly_expenses"] + EPS)
    df["savings_ratio"] = df["bank_balance"] / ((df["monthly_salary"] * 6) + EPS)

    return df



# Root Endpoint
@app.get("/")
def home():
    return {"message": "EMIPredict AI API is running"}



# Predict Both (Classification + Regression)
@app.post("/predict")
def predict(data: CustomerInput):

    # Convert input to dataframe
    input_df = pd.DataFrame([data.model_dump()])

    # Apply feature engineering
    input_df = feature_engineering(input_df)


    # Classification Prediction
    X_cls = preprocessor_cls.transform(input_df)
    cls_pred_encoded = model_cls.predict(X_cls)[0]
    cls_pred_label = label_encoder.inverse_transform([cls_pred_encoded])[0]

    cls_prob = model_cls.predict_proba(X_cls)[0]
    confidence = float(np.max(cls_prob))

    # Regression Prediction
    X_reg = preprocessor_reg.transform(input_df)
    reg_pred = model_reg.predict(X_reg)[0]

    # Business guardrail (optional but recommended)
    max_allowed = 0.5 * float(input_df["monthly_salary"].iloc[0])
    reg_pred = float(min(reg_pred, max_allowed))

    return {
        "emi_eligibility_prediction": cls_pred_label,
        "eligibility_confidence": confidence,
        "predicted_max_monthly_emi": reg_pred
    }


# Classification Only Endpoint
@app.post("/predict/classification")
def predict_classification(data: CustomerInput):

    input_df = pd.DataFrame([data.model_dump()])
    input_df = feature_engineering(input_df)

    X_cls = preprocessor_cls.transform(input_df)
    cls_pred_encoded = model_cls.predict(X_cls)[0]
    cls_pred_label = label_encoder.inverse_transform([cls_pred_encoded])[0]

    cls_prob = model_cls.predict_proba(X_cls)[0]

    return {
        "emi_eligibility_prediction": cls_pred_label,
        "probabilities": {
            label_encoder.classes_[0]: float(cls_prob[0]),
            label_encoder.classes_[1]: float(cls_prob[1]),
            label_encoder.classes_[2]: float(cls_prob[2]),
        }
    }



# Regression Only Endpoint
@app.post("/predict/regression")
def predict_regression(data: CustomerInput):

    input_df = pd.DataFrame([data.model_dump()])
    input_df = feature_engineering(input_df)

    X_reg = preprocessor_reg.transform(input_df)
    reg_pred = model_reg.predict(X_reg)[0]

    max_allowed = 0.5 * float(input_df["monthly_salary"].iloc[0])
    reg_pred = float(min(reg_pred, max_allowed))

    return {
        "predicted_max_monthly_emi": reg_pred
    }


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
    #  uvicorn.run(app, host="localhost", port=5050)