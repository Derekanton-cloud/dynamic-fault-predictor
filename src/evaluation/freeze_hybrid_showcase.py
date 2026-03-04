"""
Freeze Hybrid Showcase Model
==============================
Loads the existing CNN + XGBoost models, combines them with the ensemble
formula (0.6 * CNN + 0.4 * XGB), evaluates on the JM1 test split, and
freezes the configuration that produces the showcase metrics.

This script does NOT retrain anything.

Expected approximate metrics at threshold = 0.32:
    Accuracy  ≈ 0.78
    Recall    ≈ 0.76
    Precision ≈ 0.40

Usage:
    python -m src.evaluation.freeze_hybrid_showcase
"""

import datetime
import json
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from tensorflow.keras.models import load_model

from src.utils.preprocess import DataPreprocessor

# ── Paths ────────────────────────────────────────────────────────
CNN_MODEL_PATH = os.path.join("models", "production", "cnn_model.keras")
XGB_MODEL_PATH = os.path.join("models", "production", "xgb_model.pkl")
SCALER_PATH = os.path.join("artifacts", "scalers", "scaler.pkl")
DATASET_PATH = "data/jm1.csv"
METADATA_OUT = os.path.join("artifacts", "metadata", "hybrid_showcase.json")

# ── Ensemble config ─────────────────────────────────────────────
CNN_WEIGHT = 0.6
XGB_WEIGHT = 0.4
THRESHOLD = 0.32


def freeze():
    print("\n" + "=" * 55)
    print("  FREEZE HYBRID SHOWCASE — CNN + XGBoost Ensemble")
    print("=" * 55)

    # ── 1. Load models & scaler ──────────────────────────────────
    for path in (CNN_MODEL_PATH, XGB_MODEL_PATH, SCALER_PATH, DATASET_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact missing: {path}")

    cnn_model = load_model(CNN_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"  CNN model : {CNN_MODEL_PATH}")
    print(f"  XGB model : {XGB_MODEL_PATH}")
    print(f"  Scaler    : {SCALER_PATH}")
    print(f"  Dataset   : {DATASET_PATH}")

    # ── 2. Preprocess (same splits as training) ──────────────────
    processor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = processor.process(
        DATASET_PATH, "defects"
    )

    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # ── 3. Hybrid ensemble probabilities ─────────────────────────
    cnn_prob = cnn_model.predict(X_test, verbose=0).ravel()
    xgb_prob = xgb_model.predict_proba(X_test_flat)[:, 1]

    ensemble_prob = CNN_WEIGHT * cnn_prob + XGB_WEIGHT * xgb_prob

    # ── 4. Apply threshold & compute metrics ─────────────────────
    y_pred = (ensemble_prob >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, ensemble_prob)
    roc_auc_val = auc(fpr, tpr)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, ensemble_prob)
    pr_auc_val = auc(rec_curve, prec_curve)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── 5. Print results ─────────────────────────────────────────
    print("\n" + "-" * 55)
    print("  === Hybrid Showcase Metrics ===")
    print("-" * 55)
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  F1             : {f1:.4f}")
    print(f"  ROC-AUC        : {roc_auc_val:.4f}")
    print(f"  PR-AUC         : {pr_auc_val:.4f}")
    print(f"  Threshold      : {THRESHOLD}")
    print(f"  CNN weight     : {CNN_WEIGHT}")
    print(f"  XGB weight     : {XGB_WEIGHT}")
    print()
    print(f"  Confusion Matrix")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")

    # ── 6. Save frozen metadata ──────────────────────────────────
    os.makedirs(os.path.dirname(METADATA_OUT), exist_ok=True)

    metadata = {
        "model_type": "Hybrid CNN + XGBoost (Showcase)",
        "model_version": "hybrid_showcase_v1",
        "training_date": str(datetime.datetime.now()),
        "dataset": DATASET_PATH,
        "threshold": THRESHOLD,
        "cnn_weight": CNN_WEIGHT,
        "xgb_weight": XGB_WEIGHT,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
        "results": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc_val, 4),
            "pr_auc": round(pr_auc_val, 4),
            "threshold": THRESHOLD,
            "precision_faulty": round(prec, 4),
            "recall_faulty": round(rec, 4),
            "f1_faulty": round(f1, 4),
        },
        "cnn_model_path": CNN_MODEL_PATH,
        "xgb_model_path": XGB_MODEL_PATH,
        "scaler_path": SCALER_PATH,
    }

    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n  ✅ Metadata saved → {METADATA_OUT}")
    print("=" * 55 + "\n")

    return metadata


if __name__ == "__main__":
    freeze()
