"""Streamlit Dashboard for Olist Delivery Delay Prediction & SHAP Explainability."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import shap
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Olist Delivery Delay Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
API_URL = os.getenv("API_URL", "http://localhost:8000")

BRAZIL_STATES = [
    "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "ES",
    "PE", "CE", "PA", "MT", "MA", "MS", "PB", "PI", "RN", "AL",
    "SE", "TO", "RO", "AM", "AC", "AP", "RR",
]


@st.cache_resource
def load_local_pipeline():
    """Loads the serialized model pipeline once and caches it for SHAP & fallback."""
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please run Phase 4 training first.")
        st.stop()
    return joblib.load(MODEL_PATH)


def engineer_single_order_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """Applies the exact runtime feature transformations required by the model preprocessor."""
    df = pd.DataFrame([payload])
    dt = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_hour"] = dt.dt.hour
    df["purchase_dayofweek"] = dt.dt.dayofweek
    df["purchase_month"] = dt.dt.month
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
    df["freight_ratio"] = df["freight_value"] / (df["price"] + 1e-5)
    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )
    return df


def get_prediction(payload: Dict[str, Any], pipeline) -> Tuple[int, float, str, str]:
    """Attempts to call the FastAPI /predict endpoint; falls back to local model if offline."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            return (
                data["is_delayed_prediction"],
                data["delay_probability"],
                data["risk_level"],
                "FastAPI Server (Online)",
            )
    except requests.exceptions.RequestException:
        pass

    # Fallback to local pipeline inference (useful for standalone Cloud deployment)
    df_feat = engineer_single_order_df(payload)
    prob = float(pipeline.predict_proba(df_feat)[:, 1][0])
    pred = int(prob >= 0.5)
    risk = "High Risk" if prob >= 0.5 else "Low Risk"
    return pred, round(prob, 4), risk, "Local Model Artifact (Standalone Mode)"


def compute_shap_plot(payload: Dict[str, Any], pipeline) -> plt.Figure:
    """Computes SHAP values for a single order and returns a Matplotlib waterfall figure."""
    df_feat = engineer_single_order_df(payload)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform row and clean feature names
    X_transformed = preprocessor.transform(df_feat)
    raw_feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.split("__")[-1] for name in raw_feature_names]

    explainer = shap.TreeExplainer(classifier)
    shap_explanation = explainer(X_transformed)
    shap_explanation.feature_names = clean_names

    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap.plots.waterfall(shap_explanation[0], max_display=8, show=False)
    plt.tight_layout()
    return fig


# --- UI LAYOUT ---
st.title("📦 Olist E-Commerce Delivery Delay Prediction Platform")
st.markdown(
    "Predict whether a Brazilian e-commerce order will miss its estimated delivery date "
    "at the exact moment of checkout, and inspect feature contributions via **SHAP**."
)

pipeline = load_local_pipeline()

tab_predict, tab_analytics = st.tabs(["🔮 Real-Time Order Predictor", "📊 Historical Delay Analytics"])

with tab_predict:
    st.subheader("Order Checkout Simulation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💰 Financials & Route**")
        price = st.number_input("Product Price (BRL)", min_value=1.0, max_value=10000.0, value=120.0, step=10.0)
        freight_value = st.number_input("Freight Cost (BRL)", min_value=0.0, max_value=1000.0, value=22.5, step=2.5)
        distance_km = st.slider("Seller-to-Customer Distance (km)", min_value=1.0, max_value=3500.0, value=432.0, step=10.0)

    with col2:
        st.markdown("**📐 Package Dimensions**")
        weight_g = st.number_input("Product Weight (grams)", min_value=50.0, max_value=30000.0, value=1200.0, step=100.0)
        length_cm = st.number_input("Package Length (cm)", min_value=5.0, max_value=150.0, value=30.0, step=5.0)
        height_cm = st.number_input("Package Height (cm)", min_value=2.0, max_value=150.0, value=15.0, step=5.0)
        width_cm = st.number_input("Package Width (cm)", min_value=5.0, max_value=150.0, value=20.0, step=5.0)

    with col3:
        st.markdown("**🌍 Location & Timing**")
        seller_state = st.selectbox("Seller State", options=BRAZIL_STATES, index=0)     # Default SP
        customer_state = st.selectbox("Customer State", options=BRAZIL_STATES, index=1) # Default RJ
        purchase_date = st.date_input("Purchase Date", value=datetime(2018, 5, 15))
        purchase_time = st.time_input("Purchase Time", value=datetime(2018, 5, 15, 14, 30).time())

    purchase_iso = f"{purchase_date} {purchase_time}"

    order_payload = {
        "price": float(price),
        "freight_value": float(freight_value),
        "product_weight_g": float(weight_g),
        "product_length_cm": float(length_cm),
        "product_height_cm": float(height_cm),
        "product_width_cm": float(width_cm),
        "seller_customer_distance_km": float(distance_km),
        "customer_state": customer_state,
        "seller_state": seller_state,
        "order_purchase_timestamp": purchase_iso,
    }

    if st.button("🚀 Predict Delivery Risk", type="primary", use_container_width=True):
        pred, prob, risk_level, source = get_prediction(order_payload, pipeline)

        st.divider()
        res_col1, res_col2 = st.columns([1, 1.6])

        with res_col1:
            st.markdown("### Prediction Verdict")
            if pred == 1:
                st.error(f"⚠️ **{risk_level}: Likely Delayed**")
            else:
                st.success(f"✅ **{risk_level}: Expected On-Time**")

            st.metric(label="Estimated Delay Probability", value=f"{prob * 100:.1f}%")
            st.caption(f"Inference Source: `{source}`")

            st.markdown("**Derived Checkout Features:**")
            st.json(
                {
                    "freight_ratio": round(freight_value / price, 3),
                    "product_volume_cm3": round(length_cm * height_cm * width_cm, 1),
                    "route": f"{seller_state} ➔ {customer_state} ({distance_km} km)",
                }
            )

        with res_col2:
            st.markdown("### 🔍 Local Feature Explanation (SHAP Waterfall)")
            st.caption(
                "Red bars push the order toward a higher delay risk; blue bars push it toward on-time delivery."
            )
            with st.spinner("Generating SHAP explanation..."):
                shap_fig = compute_shap_plot(order_payload, pipeline)
                st.pyplot(shap_fig, use_container_width=True)

with tab_analytics:
    st.subheader("Olist Historical Logistics Benchmark (110,189 Delivered Orders)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Delivered Items", "110,189")
    kpi2.metric("Baseline Delay Rate", "7.91%")
    kpi3.metric("Median Route Distance", "431.9 km")
    kpi4.metric("Best Model ROC-AUC", "0.7712")

    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # State-level historical benchmarks from our Phase 1/2 SQL analysis
        state_df = pd.DataFrame(
            {
                "Customer State": ["AL", "MA", "PI", "CE", "SE", "BA", "RJ", "MG", "PR", "SP"],
                "Delay Rate (%)": [23.9, 19.6, 15.9, 15.3, 15.2, 14.0, 13.5, 5.6, 5.0, 5.9],
            }
        )
        fig_state = px.bar(
            state_df,
            x="Customer State",
            y="Delay Rate (%)",
            color="Delay Rate (%)",
            color_continuous_scale="Reds",
            title="Historical Delay Rate by Customer State (%)",
        )
        st.plotly_chart(fig_state, use_container_width=True)

    with chart_col2:
        model_df = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Random Forest", "XGBoost (Optuna)"],
                "ROC-AUC": [0.6468, 0.7530, 0.7712],
                "PR-AUC": [0.1332, 0.2445, 0.2652],
                "F1-Score": [0.2087, 0.2941, 0.3050],
            }
        )
        fig_models = px.bar(
            model_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Phase 4 Holdout Test Performance across Models",
        )
        st.plotly_chart(fig_models, use_container_width=True)
        