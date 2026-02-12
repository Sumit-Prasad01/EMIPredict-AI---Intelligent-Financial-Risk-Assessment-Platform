# 💳 FinEMI AI - Intelligent Loan EMI Risk & Affordability Platform

FinEMI AI is an end-to-end **Financial Risk Assessment & EMI Prediction System** that helps evaluate a customer's loan repayment capability using Machine Learning.  
The system predicts:

✅ **EMI Eligibility Classification** (`Eligible`, `High_Risk`, `Not_Eligible`)  
✅ **Maximum Safe Monthly EMI (Regression)** (`max_monthly_emi`)

This project is built with an industry-grade pipeline using **Scikit-learn**, **LightGBM**, **MLflow**, **FastAPI**, **Streamlit**, and is fully containerized using **Docker**.

---

## 🚀 Key Features

- 🔍 **Customer Risk Classification** (Eligible / High Risk / Not Eligible)
- 📈 **Maximum EMI Prediction** using regression modeling
- ⚙️ **Feature Engineering** based on real-world financial indicators
- 🧠 **LightGBM Models** for high performance and scalability
- 🧪 **MLflow Tracking** for experiment management
- 🌐 **FastAPI Backend API** for production-ready serving
- 🎨 **Streamlit Frontend UI** for interactive predictions
- 🐳 **Docker + Docker Compose** for deployment

---

## 🏗️ Project Architecture

```
FinEMI AI
│
├── api/                         # FastAPI backend
│   └── app.py
│
├── frontend/                    # Streamlit UI (optional folder)
│
├── artifacts/                   # Model artifacts and processed files
│   ├── models/
│   ├── processed/
│   └── raw/
│
├── pipeline/                    # ML pipeline scripts
│
├── src/                         # Core ML modules
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training_classification.py
│   ├── model_training_regression.py
│   ├── logger.py
│   └── custom_exception.py
│
├── logs/                        # Application logs
├── notebooks/                   # EDA + experimentation notebooks
├── mlruns/                      # MLflow tracking directory
│
├── Dockerfile.api               # FastAPI Dockerfile
├── Dockerfile.streamlit         # Streamlit Dockerfile
├── docker-compose.yaml          # Docker Compose file
│
├── requirements.txt
├── main.py
└── README.md
```

---

## 📌 Problem Statement

Financial institutions need an automated way to evaluate customer loan applications and determine:

1. Whether the applicant is eligible for EMI-based loan approval.
2. What maximum EMI amount the applicant can safely afford.

FinEMI AI solves this by using customer demographic, income, expense, and credit information to predict both **risk eligibility** and **affordable EMI capacity**.

---

## 🧠 Machine Learning Tasks

### ✅ 1. Classification Model (EMI Eligibility)
Predicts the customer’s EMI eligibility category:

- `Eligible`
- `High_Risk`
- `Not_Eligible`

### ✅ 2. Regression Model (Max EMI Prediction)
Predicts:

- `max_monthly_emi` (continuous target)

---

## 🧾 Input Features Used

The system uses the following input features:

- age  
- gender  
- marital_status  
- education  
- monthly_salary  
- employment_type  
- years_of_employment  
- company_type  
- house_type  
- monthly_rent  
- family_size  
- dependents  
- school_fees  
- college_fees  
- travel_expenses  
- groceries_utilities  
- other_monthly_expenses  
- existing_loans  
- current_emi_amount  
- credit_score  
- bank_balance  
- emergency_fund  
- emi_scenario  
- requested_amount  
- requested_tenure  

---

## ⚙️ Feature Engineering

The project performs important domain-driven feature engineering such as:

- **Total Monthly Expenses**
- **Disposable Income**
- **EMI Burden Ratio**
- **Expense-to-Income Ratio**
- **Emergency Fund Ratio**
- **Savings Ratio**

These engineered features help improve both classification and regression performance.

---

## 🛠️ Tech Stack

| Component | Tools |
|----------|------|
| ML Modeling | Scikit-learn, LightGBM |
| Experiment Tracking | MLflow |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Containerization | Docker, Docker Compose |
| Data Processing | Pandas, NumPy |

---

## 📊 Model Performance (Example)

### Classification (LightGBM)
- Strong performance with class imbalance handling using class weights

### Regression (LightGBM Regressor)
- **RMSE:** ~645  
- **MAE:** ~234  
- **R²:** ~0.99  

---

## 🚀 How to Run Locally

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run FastAPI Backend
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8080
```

FastAPI Docs:
```
http://127.0.0.1:8080/docs
```

### 4️⃣ Run Streamlit Frontend
```bash
streamlit run streamlit_app.py
```

Streamlit UI:
```
http://localhost:8501
```

---

## 🐳 Run Using Docker

### Build & Run with Docker Compose
```bash
docker-compose up --build
```

- FastAPI will run on: `http://localhost:8080`
- Streamlit will run on: `http://localhost:8501`

---

## 📦 API Endpoints

### Root
`GET /`  
Checks if API is running.

### Predict (Classification + Regression)
`POST /predict`

Returns:
- EMI eligibility prediction
- Confidence score
- Predicted max EMI

### Classification Only
`POST /predict/classification`

### Regression Only
`POST /predict/regression`

---

## 📁 Artifacts Saved

The project saves key production artifacts:

### Classification
- Preprocessor (`ColumnTransformer`)
- Label Encoder
- Trained LightGBM Model

### Regression
- Preprocessor (`ColumnTransformer`)
- Trained LightGBM Regressor

These ensure consistent preprocessing during inference.

---

## 🧪 MLflow Tracking

MLflow is used for:
- Tracking experiments
- Logging metrics
- Comparing models
- Storing model versions

To start MLflow UI:

```bash
mlflow ui
```

Then open:
```
http://127.0.0.1:5000
```

---

## 🔮 Future Improvements

- Add SHAP explainability for financial risk decisions
- Add threshold tuning for classification decisions
- Deploy on cloud (AWS/GCP/Azure)
- Add CI/CD pipeline for automated deployment
- Add monitoring with Prometheus + Grafana

---


