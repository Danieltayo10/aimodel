import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "https://aimodel-nnmo.onrender.com"

st.set_page_config(layout="wide")
st.title("Universal AI Business Intelligence Engine")

# --- Upload CSV ---
st.header("1. Upload CSV")
uploaded_file = st.file_uploader("Upload your business data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # --- Select Target ---
    st.header("2. Select Target Column")
    target = st.selectbox("What do you want to predict?", df.columns)

    # --- Select Problem Type ---
    st.header("3. Select Problem Type")
    problem_type = st.selectbox("Problem Type", ["Auto Detect", "Regression", "Classification"])

    if st.button("Train Model"):
        uploaded_file.seek(0)
        response = requests.post(
            f"{BACKEND_URL}/train",
            files={"csv_file": uploaded_file},
            params={"target_column": target, "problem_type": problem_type}
        )
        st.success(response.json())

    # --- Prediction Inputs ---
    st.header("4. Make a Prediction")
    input_data = {}
    for col in df.drop(columns=[target]).columns:
        if df[col].dtype == "object":
            input_data[col] = st.text_input(col, "UNKNOWN")
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Predict"):
        response = requests.post(f"{BACKEND_URL}/predict", json=input_data).json()
        st.subheader("Prediction")
        st.metric("Predicted Value", response["prediction"])
        if response.get("probability") is not None:
            st.metric("Prediction Confidence", response["probability"])
        st.subheader("AI Explanation")
        st.write(response.get("ai_summary", "No AI summary available"))
        st.subheader("Feature Impact")
        st.json(response["explanation"])

# --- AI Text Tools ---
st.header("5. AI Text Tools")
ai_option = st.selectbox("Select AI Tool", ["Sentiment Analysis", "Topic Modeling", "Query Assistant"])

if ai_option == "Sentiment Analysis":
    text_input = st.text_area("Enter text to analyze sentiment")
    if st.button("Analyze Sentiment"):
        response = requests.post(f"{BACKEND_URL}/sentiment", json={"text": text_input}).json()
        st.json(response)

elif ai_option == "Topic Modeling":
    texts_input = st.text_area("Enter multiple texts, one per line")
    texts_list = [t.strip() for t in texts_input.split("\n") if t.strip()]
    if st.button("Extract Topics & Keywords"):
        response = requests.post(f"{BACKEND_URL}/topics", json=texts_list).json()
        st.json(response)

elif ai_option == "Query Assistant":
    query_input = st.text_input("Ask a business question")
    if st.button("Ask AI"):
        response = requests.post(f"{BACKEND_URL}/query", json={"query": query_input}).json()
        st.subheader("AI Answer")
        st.write(response.get("answer", "No response"))
