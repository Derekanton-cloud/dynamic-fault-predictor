"""
Standalone Production CNN Training Pipeline
============================================
Trains a pure CNN model (no XGBoost) for software fault prediction.
Fully independent from the hybrid and dynamic pipelines.

Usage:
    python -m src.training.train_production_cnn
    python -m src.training.train_production_cnn --data data/jm1.csv
"""

import argparse
import datetime
import json
import os

import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.utils.imbalance import ImbalanceHandler
from src.utils.preprocess import DataPreprocessor

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# ── Output paths (fully separate from hybrid / dynamic) ──────────
MODEL_PATH = os.path.join("models", "production", "production_cnn.keras")
SCALER_PATH = os.path.join("artifacts", "scalers", "production_scaler.pkl")
METADATA_PATH = os.path.join("artifacts", "metadata", "production_cnn_metadata.json")
REPORT_DIR = "reports"


# ── CNN architecture (mirrors ProductionTrainer.build_cnn) ───────
def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ── Threshold sweep: max precision with recall >= 0.75 ───────────
def select_threshold(y_true, y_prob, min_recall=0.75):
    thresholds = np.arange(0.01, 1.00, 0.01)
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        rows.append({
            "threshold": round(float(t), 2),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(
        os.path.join(REPORT_DIR, "production_cnn_threshold_sweep.csv"),
        index=False,
    )

    eligible = sweep_df[sweep_df["recall"] >= min_recall]
    if eligible.empty:
        print("⚠  No threshold satisfies recall >= 0.75; falling back to best F1.")
        best = sweep_df.sort_values("f1", ascending=False).iloc[0]
    else:
        best = eligible.sort_values("precision", ascending=False).iloc[0]

    return float(best["threshold"]), sweep_df


# ── Evaluation artefacts ─────────────────────────────────────────
def save_evaluation_artifacts(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Production CNN – Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "production_cnn_confusion_matrix.png"))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Production CNN – ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "production_cnn_roc_curve.png"))
    plt.close()

    # Precision-Recall curve
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_curve, prec_curve)
    fig, ax = plt.subplots()
    ax.plot(rec_curve, prec_curve, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_title("Production CNN – Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "production_cnn_pr_curve.png"))
    plt.close()

    return roc_auc, pr_auc


# ── Main training function ───────────────────────────────────────
def train_production_cnn(dataset_path="data/jm1.csv", target_column="defects"):

    print("\n🚀 Starting Standalone Production CNN Training...")
    print(f"   Dataset : {dataset_path}")
    print(f"   Target  : {target_column}\n")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── 1. Preprocess ────────────────────────────────────────────
    processor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = processor.process(
        dataset_path,
        target_column,
    )

    # ── 2. Handle class imbalance (flatten → SMOTE → reshape) ───
    balancer = ImbalanceHandler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_bal_flat, y_train_bal = balancer.handle(X_train_flat, y_train)
    X_train_bal = X_train_bal_flat.reshape(
        X_train_bal_flat.shape[0], X_train.shape[1], 1
    )

    # ── 3. Build CNN ─────────────────────────────────────────────
    model = build_cnn((X_train.shape[1], 1))
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # ── 4. Train ─────────────────────────────────────────────────
    model.fit(
        X_train_bal,
        y_train_bal,
        epochs=80,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, lr_scheduler],
        verbose=1,
    )

    # ── 5. Threshold sweep on VALIDATION set ────────────────────
    y_val_prob = model.predict(X_val, verbose=0).ravel()
    best_threshold, sweep_df = select_threshold(y_val, y_val_prob, min_recall=0.75)
    print(f"\n🎯 Selected Threshold (from validation): {best_threshold:.2f}")

    # ── 6. Final evaluation on TEST set ──────────────────────────
    y_test_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_test_prob >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc_val, pr_auc_val = save_evaluation_artifacts(y_test, y_test_prob, best_threshold)

    print("\n=== Production CNN – Final Test Metrics ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc_val:.4f}")
    print(f"  PR-AUC    : {pr_auc_val:.4f}")

    # ── 8. Save model, scaler, metadata ──────────────────────────
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_type": "Production CNN (standalone)",
        "model_version": "production_cnn_v1",
        "training_date": str(datetime.datetime.now()),
        "dataset": dataset_path,
        "validation_threshold": best_threshold,
        "validation_policy": "max precision with recall >= 0.75",
        "threshold": best_threshold,
        "results": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc_val, 4),
            "pr_auc": round(pr_auc_val, 4),
            "threshold": round(best_threshold, 4),
            "precision_faulty": round(prec, 4),
            "recall_faulty": round(rec, 4),
            "f1_faulty": round(f1, 4),
        },
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Production CNN saved to         : {MODEL_PATH}")
    print(f"   Scaler saved to                 : {SCALER_PATH}")
    print(f"   Metadata saved to               : {METADATA_PATH}")
    print(f"   Threshold sweep saved to        : {REPORT_DIR}/production_cnn_threshold_sweep.csv")
    print(f"   Confusion matrix saved to       : {REPORT_DIR}/production_cnn_confusion_matrix.png")
    print(f"   ROC curve saved to              : {REPORT_DIR}/production_cnn_roc_curve.png")
    print(f"   PR curve saved to               : {REPORT_DIR}/production_cnn_pr_curve.png")

    return metadata


# ── CLI entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a standalone Production CNN for fault prediction."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/jm1.csv",
        help="Path to dataset CSV (default: data/jm1.csv)",
    )
    args = parser.parse_args()
    train_production_cnn(args.data)
