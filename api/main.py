"""FastAPI application for serving Olist Delivery Delay Predictions."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global container for model
ml_model = None

# Get the project root directory dynamically from api/main.py
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the ML model into memory upon server startup."""
    global ml_model
    try:
        ml_model = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
        ml_model = None  # Ensure it is explicitly None if loading fails
    yield
    # Clean up model on shutdown if necessary
    ml_model = None


app = FastAPI(
    title="Olist Delivery Delay Prediction API",
    version="1.0.0",
    description="Production-style API for predicting E-commerce delivery delays.",
    lifespan=lifespan,
)


class OrderPredictionRequest(BaseModel):
    price: float = Field(..., gt=0, description="Item price in BRL")
    freight_value: float = Field(..., ge=0, description="Freight shipping cost")
    product_weight_g: float = Field(..., gt=0, description="Product weight in grams")
    product_length_cm: float = Field(..., gt=0, description="Product length in cm")
    product_height_cm: float = Field(..., gt=0, description="Product height in cm")
    product_width_cm: float = Field(..., gt=0, description="Product width in cm")
    seller_customer_distance_km: float = Field(..., ge=0, description="Haversine distance in km")
    customer_state: str = Field(..., min_length=2, max_length=2, description="2-letter customer state code e.g. SP")
    seller_state: str = Field(..., min_length=2, max_length=2, description="2-letter seller state code e.g. RJ")
    order_purchase_timestamp: str = Field(..., description="ISO format datetime string e.g. 2017-10-02 10:00:00")


class PredictionResponse(BaseModel):
    is_delayed_prediction: int
    delay_probability: float
    risk_level: str


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint to verify server status."""
    return {"status": "healthy", "model_loaded": ml_model is not None}


@app.get("/model-info", tags=["System"])
def model_info():
    """Returns metadata about the active deployed model."""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(ml_model.named_steps["classifier"]).__name__,
        "preprocessor": type(ml_model.named_steps["preprocessor"]).__name__,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(payload: OrderPredictionRequest):
    """Predicts delivery delay probability for a single order."""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request body to DataFrame
        data_dict = payload.model_dump()
        df = pd.DataFrame([data_dict])

        # Engineer features expected by pipeline
        dt = pd.to_datetime(df["order_purchase_timestamp"])
        df["purchase_hour"] = dt.dt.hour
        df["purchase_dayofweek"] = dt.dt.dayofweek
        df["purchase_month"] = dt.dt.month
        df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
        df["freight_ratio"] = df["freight_value"] / (df["price"] + 1e-5)
        df["product_volume_cm3"] = (
            df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
        )

        # Predict
        prob = float(ml_model.predict_proba(df)[:, 1][0])
        pred = int(prob >= 0.5)
        risk = "High Risk" if prob >= 0.5 else "Low Risk"

        return PredictionResponse(
            is_delayed_prediction=pred,
            delay_probability=round(prob, 4),
            risk_level=risk,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
def predict_batch(payloads: List[OrderPredictionRequest]):
    """Predicts delivery delay probability for a batch of orders."""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        data_dicts = [p.model_dump() for p in payloads]
        df = pd.DataFrame(data_dicts)

        dt = pd.to_datetime(df["order_purchase_timestamp"])
        df["purchase_hour"] = dt.dt.hour
        df["purchase_dayofweek"] = dt.dt.dayofweek
        df["purchase_month"] = dt.dt.month
        df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
        df["freight_ratio"] = df["freight_value"] / (df["price"] + 1e-5)
        df["product_volume_cm3"] = (
            df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
        )

        probs = ml_model.predict_proba(df)[:, 1]
        results = []
        for prob in probs:
            p_val = float(prob)
            results.append(
                PredictionResponse(
                    is_delayed_prediction=int(p_val >= 0.5),
                    delay_probability=round(p_val, 4),
                    risk_level="High Risk" if p_val >= 0.5 else "Low Risk",
                )
            )
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))