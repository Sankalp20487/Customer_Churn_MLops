import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from streamlit_app.model_loader import predict_churn, get_best_model_info

load_dotenv()

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔍")
st.title("🔎 Customer Churn Predictor")

# Show model information
try:
    model_info = get_best_model_info()
    st.info(f"Using {model_info['model_name']} (Accuracy: {model_info['accuracy']:.3f}, Recall: {model_info['recall']:.3f})")
except Exception as e:
    st.error(f"Cannot load model info: {e}")
    st.stop()

st.write("Fill in the customer details below and click **Predict Churn**")

# Input fields (same as before)
st.sidebar.header("Customer Features")
age = st.sidebar.slider("Age", 18, 90, 35)
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 24)
usage_frequency = st.sidebar.slider("Usage Frequency (times/month)", 0, 50, 10)
support_calls = st.sidebar.slider("Support Calls (past 6 months)", 0, 20, 2)
payment_delay = st.sidebar.slider("Avg. Payment Delay (days)", 0, 30, 5)
total_spend = st.sidebar.number_input("Total Spend ($)", min_value=0.0, value=300.0, step=10.0)
last_interaction = st.sidebar.slider("Days Since Last Interaction", 0, 365, 30)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
subscription_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

# Encode categorical variables
gender_index = 0 if gender == "Male" else 1
subscription_index = {"Basic": 0, "Standard": 1, "Premium": 2}[subscription_type]
contract_index = {"Monthly": 0, "Quarterly": 1, "Annual": 2}[contract_length]

# Prediction
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "age": age,
        "tenure": tenure,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "payment_delay": payment_delay,
        "total_spend": total_spend,
        "gender_index": gender_index,
        "last_interaction": last_interaction,
        "subscription_type_index": subscription_index,
        "contract_length_index": contract_index,
    }])

    with st.spinner("Making prediction..."):
        result = predict_churn(input_df)
    
    st.subheader("Prediction Result")
    
    if result["success"]:
        pred = result["prediction"]
        # Note: REST API might return different format than sklearn
        if isinstance(pred, list):
            churn_class = pred[0]
        else:
            churn_class = pred
            
        if churn_class == 1:
            st.error("⚠️ The customer is likely to **churn**.")
        else:
            st.success("✅ The customer is likely to **stay**.")
    else:
        st.error(f"Prediction failed: {result['error']}")
        st.info("Make sure MLflow model server is running:\n`mlflow models serve -m <model_uri> -p 5000`")