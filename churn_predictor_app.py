import streamlit as st
import pandas as pd
import joblib

st.title("📉 Telecom Churn Risk Predictor")

# Load saved model
model = joblib.load("xgb_model.pkl")

# Define user inputs (same as training features)
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0)
TotalCharges = st.number_input("Total Charges")

# Convert categorical to numeric as per training
def encode_binary(val): return 1 if val == "Yes" or val == "Male" else 0

input_data = pd.DataFrame([{
    'gender': encode_binary(gender),
    'SeniorCitizen': SeniorCitizen,
    'Partner': encode_binary(Partner),
    'Dependents': encode_binary(Dependents),
    'tenure': tenure,
    'PhoneService': encode_binary(PhoneService),
    'MultipleLines': 0 if MultipleLines == "No" else (1 if MultipleLines == "Yes" else 2),
    'InternetService': {"DSL": 0, "Fiber optic": 1, "No": 2}[InternetService],
    'OnlineSecurity': 0 if OnlineSecurity == "No" else (1 if OnlineSecurity == "Yes" else 2),
    'OnlineBackup': 0 if OnlineBackup == "No" else (1 if OnlineBackup == "Yes" else 2),
    'DeviceProtection': 0 if DeviceProtection == "No" else (1 if DeviceProtection == "Yes" else 2),
    'TechSupport': 0 if TechSupport == "No" else (1 if TechSupport == "Yes" else 2),
    'StreamingTV': 0 if StreamingTV == "No" else (1 if StreamingTV == "Yes" else 2),
    'StreamingMovies': 0 if StreamingMovies == "No" else (1 if StreamingMovies == "Yes" else 2),
    'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[Contract],
    'PaperlessBilling': encode_binary(PaperlessBilling),
    'PaymentMethod': {
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    }[PaymentMethod],
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}])

# Predict churn probability
churn_prob = model.predict_proba(input_data)[0][1]
st.metric("Predicted Churn Risk", f"{int(churn_prob * 100)}%")