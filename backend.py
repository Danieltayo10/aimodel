from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
import openai
from train_utils import train_model, load_model, predict, explain

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Universal AI Decision Engine")

MODEL_READY = False


@app.post("/train")
async def train(csv_file: UploadFile = File(...), target_column: str = ""):
    global MODEL_READY

    df = pd.read_csv(csv_file.file)

    if target_column not in df.columns:
        return {"error": "Target column not found in CSV"}

    result = train_model(df, target_column)
    MODEL_READY = True

    return {
        "status": "model trained",
        "metrics": result
    }


@app.post("/predict")
def run_prediction(input_data: dict):
    if not MODEL_READY:
        return {"error": "Model not trained yet"}

    model, feature_columns = load_model()

    pred = predict(model, feature_columns, input_data)
    explanation = explain(model, feature_columns, input_data)

    ai_summary = None
    if openai.api_key:
        prompt = f"""
Prediction value: {pred}
Feature impacts: {explanation}

Explain this result in business-friendly language and suggest actions.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        ai_summary = response.choices[0].message.content

    return {
        "prediction": pred,
        "explanation": explanation,
        "ai_summary": ai_summary
    }
