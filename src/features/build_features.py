"""Reusable feature engineering pipeline for Delivery Delay Prediction."""

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HaversineDistanceTransformer(BaseEstimator, TransformerMixin):
    """Calculates the great-circle distance between seller and customer ZIP centroids."""

    def __init__(self, lat1_col: str, lon1_col: str, lat2_col: str, lon2_col: str):
        self.lat1_col = lat1_col
        self.lon1_col = lon1_col
        self.lat2_col = lat2_col
        self.lon2_col = lon2_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ = X.copy()
        # Convert degrees to radians
        lat1 = np.radians(X_[self.lat1_col])
        lon1 = np.radians(X_[self.lon1_col])
        lat2 = np.radians(X_[self.lat2_col])
        lon2 = np.radians(X_[self.lon2_col])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371.0  # Earth radius in kilometers

        X_["seller_customer_distance_km"] = r * c
        return X_


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates temporal, financial, and product features without data leakage."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ = X.copy()
        
        # Ensure datetime conversion
        purchase_dt = pd.to_datetime(X_["order_purchase_timestamp"])
        
        # Temporal features
        X_["purchase_hour"] = purchase_dt.dt.hour
        X_["purchase_dayofweek"] = purchase_dt.dt.dayofweek
        X_["purchase_month"] = purchase_dt.dt.month
        X_["is_weekend"] = (X_["purchase_dayofweek"] >= 5).astype(int)
        
        # Financial features
        X_["freight_ratio"] = X_["freight_value"] / (X_["price"] + 1e-5)
        
        # Product volume
        X_["product_volume_cm3"] = (
            X_["product_length_cm"] 
            * X_["product_height_cm"] 
            * X_["product_width_cm"]
        )
        
        return X_


def get_training_data_from_db(engine) -> pd.DataFrame:
    """Extracts raw base tables joined with geolocation centroids."""
    query = """
    SELECT 
        o.order_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date,
        o.order_delivered_customer_date,
        c.customer_state,
        c.customer_zip_code_prefix,
        s.seller_state,
        s.seller_zip_code_prefix,
        oi.price,
        oi.freight_value,
        p.product_weight_g,
        p.product_length_cm,
        p.product_height_cm,
        p.product_width_cm,
        geo_c.geolocation_lat AS customer_lat,
        geo_c.geolocation_lng AS customer_lng,
        geo_s.geolocation_lat AS seller_lat,
        geo_s.geolocation_lng AS seller_lng
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    -- Average geolocation per zip code to avoid row duplication joins
    LEFT JOIN (
        SELECT geolocation_zip_code_prefix, AVG(geolocation_lat) AS geolocation_lat, AVG(geolocation_lng) AS geolocation_lng
        FROM geolocation GROUP BY geolocation_zip_code_prefix
    ) geo_c ON c.customer_zip_code_prefix = geo_c.geolocation_zip_code_prefix
    LEFT JOIN (
        SELECT geolocation_zip_code_prefix, AVG(geolocation_lat) AS geolocation_lat, AVG(geolocation_lng) AS geolocation_lng
        FROM geolocation GROUP BY geolocation_zip_code_prefix
    ) geo_s ON s.seller_zip_code_prefix = geo_s.geolocation_zip_code_prefix
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL;
    """
    logger.info("Executing base extraction query...")
    df = pd.read_sql_query(query, con=engine)
    logger.info("Extracted %d rows.", len(df))
    return df


def build_preprocessor() -> ColumnTransformer:
    """Constructs the scikit-learn ColumnTransformer for numerical and categorical features."""
    numeric_features = [
        "price",
        "freight_value",
        "product_weight_g",
        "product_volume_cm3",
        "seller_customer_distance_km",
        "freight_ratio",
        "purchase_hour",
        "purchase_dayofweek",
        "purchase_month",
        "is_weekend",
    ]
    categorical_features = ["customer_state", "seller_state"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor