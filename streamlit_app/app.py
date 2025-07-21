# app.py

import os
from dotenv import load_dotenv

import streamlit as st
import pandas as pd

from model_loader import load_model

# ── Load env vars ────────────────────────────────────────
# Make sure you have a .env in your repo root with:
# S3_BUCKET_NAME, FINAL_MODEL_PATH, AWS credentials, etc.
load_dotenv()

# ── Streamlit page config ───────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔍")

st.title("🔎 Customer Churn Predictor")
st.write(
    "Fill in the customer details below and click **Predict Churn** to see "
    "the probability that the customer will churn."
)

# ── Load & cache the model ───────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ── Sidebar Inputs ───────────────────────────────────────
st.sidebar.header("Customer Features")
age                = st.sidebar.slider("Age", 18, 90, 35)
tenure             = st.sidebar.slider("Tenure (months)", 0, 72, 24)
usage_frequency    = st.sidebar.slider("Usage Frequency (times/month)", 0, 50, 10)
support_calls      = st.sidebar.slider("Support Calls (past 6 months)", 0, 20, 2)
payment_delay      = st.sidebar.slider("Avg. Payment Delay (days)", 0, 30, 5)
total_spend        = st.sidebar.number_input("Total Spend ($)", min_value=0.0, value=300.0, step=10.0)
last_interaction   = st.sidebar.slider("Days Since Last Interaction", 0, 365, 30)

gender             = st.sidebar.selectbox("Gender", ["Male", "Female"])
subscription_type  = st.sidebar.selectbox(
    "Subscription Type", ["Basic", "Standard", "Premium"]
)
contract_length    = st.sidebar.selectbox(
    "Contract Length", ["Monthly", "Quarterly", "Annual"]
)

# ── Encode categorical ──────────────────────────────────
gender_index        = 0 if gender == "Male" else 1
subscription_index  = {"Basic": 0, "Standard": 1, "Premium": 2}[subscription_type]
contract_index      = {"Monthly": 0, "Quarterly": 1, "Annual": 2}[contract_length]

# ── Run prediction ──────────────────────────────────────
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "age":                    age,
        "tenure":                 tenure,
        "usage_frequency":        usage_frequency,
        "support_calls":          support_calls,
        "payment_delay":          payment_delay,
        "total_spend":            total_spend,
        "gender_index":           gender_index,
        "last_interaction":       last_interaction,
        "subscription_type_index": subscription_index,
        "contract_length_index":   contract_index,
    }])

    pred        = model.predict(input_df)[0]
    proba       = model.predict_proba(input_df)[0][1]
    proba_pct   = f"{proba:.2%}"

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {proba_pct}")

    if pred == 1:
        st.error("⚠️ The customer is likely to **churn**.")
    else:
        st.success("✅ The customer is likely to **stay**.")
