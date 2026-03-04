"""
Verify Showcase CNN Model
==========================
Loads the frozen showcase CNN and evaluates it on the JM1 test split
using the threshold stored in metadata (0.32).

This script does NOT retrain anything — it only verifies saved artefacts.

Usage:
    python -m src.evaluation.verify_showcase_model
"""

import json
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model

from src.utils.preprocess import DataPreprocessor

# ── Paths ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "production", "production_cnn_showcase.keras")
SCALER_PATH = os.path.join("artifacts", "scalers", "production_scaler_showcase.pkl")
METADATA_PATH = os.path.join("artifacts", "metadata", "production_cnn_showcase.json")
DATASET_PATH = "data/jm1.csv"


def verify():
    # ── 1. Load metadata ─────────────────────────────────────────
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    threshold = metadata.get("threshold", 0.32)

    # ── 2. Load model & scaler ───────────────────────────────────
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ── 3. Preprocess (same splits as training) ──────────────────
    processor = DataPreprocessor()
    processor.scaler = scaler

    import pandas as pd

    df = pd.read_csv(DATASET_PATH)
    target_column = processor.detect_target_column(df)
    processor.feature_columns = [c for c in df.columns if c != target_column]

    X_scaled, y = processor.prepare_features(df, target_column=target_column, fit=False)
    X_scaled = processor.reshape_for_cnn(X_scaled)

    from sklearn.model_selection import train_test_split

    X_temp, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 4. Predict ───────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # ── 5. Metrics ───────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── 6. Print ─────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  === Showcase Verification ===")
    print("=" * 40)
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  F1             : {f1:.4f}")
    print(f"  Threshold used : {threshold}")
    print()
    print("  Confusion Matrix")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    verify()
