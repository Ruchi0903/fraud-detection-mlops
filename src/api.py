import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------- Load trained model ----------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "fraud_model.joblib")

model = joblib.load(MODEL_PATH)

# Preserve the exact feature order used during training.
FEATURE_NAMES = list(model.feature_names_in_)


# ---------- Create API ----------

app = FastAPI(
    title="Fraud Detection API",
    description="Predict whether a transaction may be fraudulent.",
    version="1.0.0",
)


# ---------- Request schema ----------

class Transaction(BaseModel):
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# ---------- Endpoints ----------

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction):
    transaction_data = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Match the exact training feature order.
    transaction_data = transaction_data[FEATURE_NAMES]

    prediction = int(model.predict(transaction_data)[0])
    fraud_probability = float(
        model.predict_proba(transaction_data)[0][1]
    )

    return {
        "prediction": prediction,
        "label": "fraud" if prediction == 1 else "legitimate",
        "fraud_probability": round(fraud_probability, 4),
    }
