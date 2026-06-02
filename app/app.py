import streamlit as st
import numpy as np
import joblib
import os

# Fix path - works on all OS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgb_fraud_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud probability")

st.sidebar.header("Transaction Features")
amount = st.sidebar.number_input("Transaction Amount ($)", 0.0, 10000.0, 100.0)
v14 = st.sidebar.slider("V14", -20.0, 20.0, 0.0)
v17 = st.sidebar.slider("V17", -20.0, 20.0, 0.0)
v12 = st.sidebar.slider("V12", -20.0, 20.0, 0.0)
v10 = st.sidebar.slider("V10", -20.0, 20.0, 0.0)

if st.button("Predict"):
    input_data = np.zeros((1, 30))
    input_data[0][0] = amount
    input_data[0][9]  = v10
    input_data[0][11] = v12
    input_data[0][13] = v14
    input_data[0][16] = v17

    prob = model.predict_proba(input_data)[0][1]
    prediction = "🚨 FRAUD DETECTED" if prob > 0.5 else "✅ LEGITIMATE"

    st.subheader(f"Prediction: {prediction}")
    st.metric("Fraud Probability", f"{prob:.2%}")

    if prob > 0.5:
        st.error("This transaction shows high fraud indicators. Flag for review.")
    else:
        st.success("Transaction appears legitimate.")