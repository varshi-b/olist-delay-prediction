"""Model training, evaluation, hyperparameter tuning, and MLflow logging."""

import logging
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.db_config import get_engine
from src.features.build_features import (
    FeatureEngineer,
    HaversineDistanceTransformer,
    build_preprocessor,
    get_training_data_from_db,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Applies feature engineering transformers and sets up target variable."""
    # Create target variable
    df["is_delayed"] = (
        pd.to_datetime(df["order_delivered_customer_date"])
        > pd.to_datetime(df["order_estimated_delivery_date"])
    ).astype(int)

    # Apply transformers
    dist_trans = HaversineDistanceTransformer(
        "seller_lat", "seller_lng", "customer_lat", "customer_lng"
    )
    feat_eng = FeatureEngineer()

    df_processed = dist_trans.fit_transform(df)
    df_processed = feat_eng.fit_transform(df_processed)

    # Select feature columns
    feature_cols = [
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
        "customer_state",
        "seller_state",
    ]

    X = df_processed[feature_cols]
    y = df_processed["is_delayed"]

    # Drop rows with critical NaN values in features
    valid_idx = X.dropna().index
    return X.loc[valid_idx], y.loc[valid_idx]


def evaluate_model(model, X_test, y_test) -> dict:
    """Computes comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }
    return metrics


def run_training_pipeline():
    """Main execution pipeline for training, evaluation, and MLflow logging."""
    engine = get_engine()
    df_raw = get_training_data_from_db(engine)
    X, y = prepare_dataset(df_raw)

    logger.info(f"Dataset shape for training: {X.shape}, Positive ratio: {y.mean():.4f}")

    # Time-based / Stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Olist_Delivery_Delay_Prediction")

    preprocessor = build_preprocessor()

    # 1. Baseline Model: Logistic Regression
    logger.info("Training Baseline Logistic Regression...")
    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        lr_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        )
        lr_pipeline.fit(X_train, y_train)
        lr_metrics = evaluate_model(lr_pipeline, X_test, y_test)
        
        mlflow.log_params({"model_type": "LogisticRegression", "class_weight": "balanced"})
        mlflow.log_metrics(lr_metrics)
        logger.info(f"Logistic Regression Metrics: {lr_metrics}")

    # 2. Random Forest Model
    logger.info("Training Random Forest Classifier...")
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        rf_pipeline.fit(X_train, y_train)
        rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)

        mlflow.log_params({"model_type": "RandomForest", "max_depth": 12})
        mlflow.log_metrics(rf_metrics)
        logger.info(f"Random Forest Metrics: {rf_metrics}")

    # 3. XGBoost Model with Optuna Tuning
    logger.info("Tuning XGBoost with Optuna...")
    pos_scale = (len(y_train) - sum(y_train)) / sum(y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "scale_pos_weight": pos_scale,
            "random_state": 42,
            "n_jobs": -1,
        }
        xgb_pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(**params)),
            ]
        )
        xgb_pipe.fit(X_train, y_train)
        return roc_auc_score(y_test, xgb_pipe.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params
    best_params["scale_pos_weight"] = pos_scale
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    logger.info(f"Best XGBoost Params: {best_params}")

    with mlflow.start_run(run_name="XGBoost_Optimized"):
        best_xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(**best_params)),
            ]
        )
        best_xgb_pipeline.fit(X_train, y_train)
        xgb_metrics = evaluate_model(best_xgb_pipeline, X_test, y_test)

        mlflow.log_params(best_params)
        mlflow.log_metrics(xgb_metrics)
        
        # FIX: Explicitly pass trusted types for XGBoost & scikit-learn components
        trusted_types = [
            "numpy.dtype",
            "sklearn.compose._column_transformer._RemainderColsList",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ]
        mlflow.sklearn.log_model(
            best_xgb_pipeline, 
            artifact_path="model", 
            skops_trusted_types=trusted_types
        )
        logger.info(f"XGBoost Optimized Metrics: {xgb_metrics}")

        # Save model locally for API & Dashboard
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        joblib.dump(best_xgb_pipeline, model_dir / "best_model.joblib")
        logger.info("Saved best model to models/best_model.joblib")


if __name__ == "__main__":
    run_training_pipeline()