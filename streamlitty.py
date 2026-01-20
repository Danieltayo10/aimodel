import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(layout="wide")
st.title("Universal AI Business Intelligence")

# ---- Upload CSV ----
st.header("1. Upload CSV")
uploaded_file = st.file_uploader("Upload your business data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # ---- Select Target ----
    st.header("2. Select Target Column")
    target = st.selectbox("What do you want to predict?", df.columns)

    if st.button("Train Model"):
        response = requests.post(
            f"{BACKEND_URL}/train",
            files={"csv_file": uploaded_file},
            params={"target_column": target}
        )
        st.success(response.json())

    # ---- Prediction ----
    st.header("3. Make a Prediction")
    input_data = {}

    for col in df.drop(columns=[target]).columns:
        if df[col].dtype == "object":
            input_data[col] = st.text_input(col, "UNKNOWN")
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Predict"):
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json=input_data
        ).json()

        st.subheader("Prediction")
        st.metric("Predicted Value", response["prediction"])

        st.subheader("AI Explanation")
        st.write(response.get("ai_summary", "No AI summary available"))

        st.subheader("Feature Impact")
        st.json(response["explanation"])
