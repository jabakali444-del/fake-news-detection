from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

from utils import clean_text


app = FastAPI(
    title="Fake News Detection API",
    description="API for predicting whether a news article is FAKE or REAL.",
    version="1.0.0"
)

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


class NewsInput(BaseModel):
    title: str = ""
    text: str = ""


@app.get("/")
def root():
    return {"message": "Fake News Detection API is running"}


@app.post("/predict")
def predict_news(data: NewsInput):
    content = f"{data.title} {data.text}".strip()

    if not content:
        raise HTTPException(status_code=400, detail="Please provide a title or text.")

    cleaned = clean_text(content)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = float(max(probabilities) * 100)

    probs = {
        label: round(float(prob * 100), 2)
        for label, prob in zip(model.classes_, probabilities)
    }

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": probs
    }