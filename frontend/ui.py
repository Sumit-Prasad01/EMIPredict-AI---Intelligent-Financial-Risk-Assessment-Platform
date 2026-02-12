import streamlit as st
import requests
import json


# Streamlit Page Config
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide"
)

st.title("💳 EMIPredict AI - Financial Risk Assessment Platform")
st.markdown("### AI-powered EMI Eligibility & Max EMI Prediction")


# FastAPI Backend URL
API_URL = "http://127.0.0.1:5050/predict"



# Sidebar Inputs
st.sidebar.header("📌 Customer Details")

age = st.sidebar.slider("Age", 18, 65, 30)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])

education = st.sidebar.selectbox(
    "Education",
    ["High School", "Graduate", "Post Graduate", "Professional"]
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    ["Private", "Government", "Self-employed"]
)

years_of_employment = st.sidebar.slider("Years of Employment", 0.0, 40.0, 2.0)

company_type = st.sidebar.selectbox(
    "Company Type",
    ["Startup", "Mid-size", "MNC"]
)

house_type = st.sidebar.selectbox("House Type", ["Rented", "Own", "Family"])

family_size = st.sidebar.slider("Family Size", 1, 10, 3)

dependents = st.sidebar.slider("Dependents", 0, 6, 1)

existing_loans = st.sidebar.selectbox("Existing Loans", ["Yes", "No"])

emi_scenario = st.sidebar.selectbox(
    "EMI Scenario",
    [
        "E-commerce Shopping EMI",
        "Home Appliances EMI",
        "Vehicle EMI",
        "Personal Loan EMI",
        "Education EMI"
    ]
)



# Main Inputs
st.subheader("📌 Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0.0, value=50000.0)
    bank_balance = st.number_input("Bank Balance (₹)", min_value=0.0, value=100000.0)
    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0.0, value=50000.0)

with col2:
    monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0.0, value=10000.0)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0.0, value=8000.0)
    travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0.0, value=3000.0)

with col3:
    school_fees = st.number_input("School Fees (₹)", min_value=0.0, value=0.0)
    college_fees = st.number_input("College Fees (₹)", min_value=0.0, value=0.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0.0, value=4000.0)


st.subheader("📌 Loan Request Details")

col4, col5, col6 = st.columns(3)

with col4:
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=0.0, value=200000.0)

with col5:
    requested_tenure = st.number_input("Requested Tenure (Months)", min_value=1, value=12)

with col6:
    current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0.0, value=0.0)

credit_score = st.slider("Credit Score", 300, 850, 650)


# Prediction Button
if st.button("🔍 Predict EMI Eligibility & Max EMI"):
    payload = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success("✅ Prediction Completed Successfully")

            st.subheader("📌 Results")

            colA, colB, colC = st.columns(3)

            with colA:
                st.metric("Eligibility Status", result["emi_eligibility_prediction"])

            with colB:
                st.metric("Confidence", f"{result['eligibility_confidence']*100:.2f}%")

            with colC:
                st.metric("Predicted Max EMI (₹)", f"{result['predicted_max_monthly_emi']:.2f}")

        else:
            st.error(f"❌ API Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("⚠️ Could not connect to FastAPI server.")
        st.write("Make sure FastAPI is running at:", API_URL)
        st.write("Error:", str(e))



# Footer
st.markdown("---")
st.caption("🚀 EMIPredict AI | Built using FastAPI + Streamlit + LightGBM")
