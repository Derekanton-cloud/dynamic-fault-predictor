import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.utils.preprocess import Preprocessor
from src.utils.feature_validation import validate_numeric_features


class ProductionPredictor:

    def __init__(self):
        self.cnn_model_path = "models/production/cnn_model.keras"
        self.xgb_model_path = "models/production/xgb_model.pkl"
        self.scaler_path = "artifacts/scalers/scaler.pkl"

        # Prefer frozen showcase metadata when available
        showcase_path = "artifacts/metadata/hybrid_showcase.json"
        legacy_path = "artifacts/metadata/production_metadata.json"
        self.metadata_path = showcase_path if os.path.exists(showcase_path) else legacy_path

        self._validate_files()

        self.cnn_model = load_model(self.cnn_model_path)
        self.xgb_model = joblib.load(self.xgb_model_path)
        self.scaler = joblib.load(self.scaler_path)

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.threshold = self.metadata.get("threshold", 0.32)
        self.cnn_weight = self.metadata.get("cnn_weight", 0.3)
        self.xgb_weight = self.metadata.get("xgb_weight", 0.7)

    def _validate_files(self):
        if not os.path.exists(self.cnn_model_path):
            raise FileNotFoundError("CNN model not found.")

        if not os.path.exists(self.xgb_model_path):
            raise FileNotFoundError("XGBoost model not found.")

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Scaler not found.")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Metadata not found.")

    def predict_from_csv(self, csv_path, target_column="defects"):

        print("🔎 Running Production Prediction...")

        df = pd.read_csv(csv_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = np.where(df[target_column] > 0, 1, 0)

        validate_numeric_features(X)

        # Scale
        X_scaled = self.scaler.transform(X)

        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        X_flat = X_scaled

        # Predict probabilities
        cnn_prob = self.cnn_model.predict(X_cnn).ravel()
        xgb_prob = self.xgb_model.predict_proba(X_flat)[:, 1]

        y_prob = self.cnn_weight * cnn_prob + self.xgb_weight * xgb_prob

        # Apply optimized threshold
        y_pred = (y_prob >= self.threshold).astype(int)

        df["prediction"] = y_pred
        df["fault_probability"] = y_prob

        return {
            "predictions_dataframe": df,
            "threshold_used": self.threshold,
            "model_version": self.metadata.get("model_version", "unknown"),
            "training_date": self.metadata.get("training_date", "unknown")
        }
