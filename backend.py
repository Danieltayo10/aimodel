from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
from openai import OpenAI
from train_utils import train_model, load_model, predict, explain, analyze_sentiment, extract_topics, extract_keywords

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Universal AI Decision Engine")

MODEL_READY = False

# =======================
# EXISTING ENDPOINTS
# =======================
@app.post("/train")
async def train(csv_file: UploadFile = File(...), target_column: str = "", problem_type: str = "Auto Detect"):
    global MODEL_READY
    df = pd.read_csv(csv_file.file)
    if target_column not in df.columns:
        return {"error": "Target column not found in CSV"}
    result = train_model(df, target_column, problem_type)
    MODEL_READY = True
    return {"status": "model trained", "metrics": result}

@app.post("/predict")
def run_prediction(input_data: dict):
    if not MODEL_READY:
        return {"error": "Model not trained yet"}
    model, feature_columns, problem_type = load_model()
    pred, proba = predict(model, feature_columns, input_data)
    explanation = explain(model, feature_columns, input_data)
    ai_summary = None
    if client.api_key:
        prompt = f"""
Prediction: {pred}
Probability (if classification): {proba}
Feature impacts: {explanation}

Explain this result in clear business language and suggest actionable steps.
"""
        try:
            response = client.responses.create(
                model="mistralai/mixtral-8x7b-instruct",
                input=prompt,
                max_output_tokens=200
            )
            ai_summary = response.output_text
        except:
            ai_summary = None
    return {"prediction": pred, "probability": proba, "explanation": explanation, "ai_summary": ai_summary}

# =======================
# NEW TEXT / AI ENDPOINTS
# =======================

@app.post("/sentiment")
def sentiment_analysis(text: str):
    """Analyze sentiment of a text string"""
    return analyze_sentiment(text)

@app.post("/topics")
def topic_modeling(texts: list[str]):
    """Perform topic modeling and keyword extraction on a list of text documents"""
    topics = extract_topics(texts)
    keywords = extract_keywords(texts)
    return {"topics": topics, "keywords": keywords}

@app.post("/query")
def ai_query(query: str):
    """AI chatbot / query assistant: answer business questions"""
    if not client.api_key:
        return {"error": "OpenAI API key not set"}
    prompt = f"""
You are a business intelligence assistant.
Answer the user question concisely and clearly.

Question: {query}
"""
    try:
        response = client.responses.create(
            model="mistralai/mixtral-8x7b-instruct",
            input=prompt,
            max_output_tokens=200
        )
        answer = response.output_text
    except Exception as e:
        answer = f"Failed to get AI response: {e}"
    return {"answer": answer}
