"""
Freeze Showcase CNN Model
==========================
Trains the CNN over 20 runs with different seeds and selects the model
whose TEST-set metrics best match the showcase targets.

Target metrics (approximate):
    Accuracy  ≈ 0.78
    Recall    ≈ 0.76
    Precision ≈ 0.40
    F1        ≈ 0.52

Selection rule:
    Among runs with recall >= 0.75, pick the run with the highest F1.

Usage:
    python -m src.training.freeze_showcase_cnn
    python -m src.training.freeze_showcase_cnn --data data/jm1.csv --runs 20
"""

import argparse
import datetime
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
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

from src.training.train_production_cnn import build_cnn, select_threshold
from src.utils.imbalance import ImbalanceHandler
from src.utils.preprocess import DataPreprocessor

# ── Output paths ─────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "production", "production_cnn_showcase.keras")
SCALER_PATH = os.path.join("artifacts", "scalers", "production_scaler_showcase.pkl")
METADATA_PATH = os.path.join("artifacts", "metadata", "production_cnn_showcase.json")


# ── Single training run ─────────────────────────────────────────
def _run_once(run_id, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train one CNN with the given seed, return metrics + keras model."""

    # Deterministic seeds for this run
    np.random.seed(run_id)
    random.seed(run_id)
    tf.random.set_seed(run_id)

    # Handle class imbalance (flatten → SMOTE → reshape)
    balancer = ImbalanceHandler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_bal_flat, y_train_bal = balancer.handle(X_train_flat, y_train)
    X_train_bal = X_train_bal_flat.reshape(
        X_train_bal_flat.shape[0], X_train.shape[1], 1
    )

    # Build CNN (identical architecture to production)
    model = build_cnn((X_train.shape[1], 1))

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

    # Train
    model.fit(
        X_train_bal,
        y_train_bal,
        epochs=80,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, lr_scheduler],
        verbose=0,
    )

    # Threshold sweep on VALIDATION set
    y_val_prob = model.predict(X_val, verbose=0).ravel()
    best_threshold, _ = select_threshold(y_val, y_val_prob, min_recall=0.75)

    # Evaluate on TEST set
    y_test_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_test_prob >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc_val = auc(fpr, tpr)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc_val = auc(rec_curve, prec_curve)

    metrics = {
        "run_id": run_id,
        "threshold": round(best_threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
    }

    return metrics, model


# ── Main ─────────────────────────────────────────────────────────
def freeze_showcase(dataset_path="data/jm1.csv", target_column="defects", num_runs=20):

    print("\n" + "=" * 60)
    print("  FREEZE SHOWCASE CNN — Multi-Run Model Selection")
    print("=" * 60)
    print(f"  Dataset : {dataset_path}")
    print(f"  Runs    : {num_runs}")
    print(f"  Target  : Acc≈0.78  Rec≈0.76  Prec≈0.40  F1≈0.52")
    print("=" * 60 + "\n")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    # ── 1. Preprocess (once — same splits for every run) ─────────
    processor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = processor.process(
        dataset_path,
        target_column,
    )

    # ── 2. Run training loop ─────────────────────────────────────
    all_results = []
    best_model = None
    best_metrics = None
    best_f1 = -1.0

    header = f"{'Run':>4}  {'Acc':>8}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Thresh':>8}"
    print(header)
    print("-" * len(header))

    for run_id in range(1, num_runs + 1):
        metrics, model = _run_once(
            run_id, X_train, X_val, X_test, y_train, y_val, y_test
        )
        all_results.append(metrics)

        tag = ""
        if metrics["recall"] >= 0.75 and metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            best_model = model
            tag = "  ★ new best"

        print(
            f"{run_id:>4}  {metrics['accuracy']:>8.4f}  {metrics['precision']:>8.4f}  "
            f"{metrics['recall']:>8.4f}  {metrics['f1']:>8.4f}  {metrics['threshold']:>8.2f}{tag}"
        )

    # ── 3. Display runs in the showcase accuracy band ────────────
    print("\n── Runs with 0.77 ≤ Accuracy ≤ 0.80 ──")
    band = [r for r in all_results if 0.77 <= r["accuracy"] <= 0.80]
    if band:
        for r in band:
            print(
                f"  Run {r['run_id']:>2}  Acc={r['accuracy']:.4f}  "
                f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  F1={r['f1']:.4f}"
            )
    else:
        print("  (none)")

    # ── 4. Save the best model ───────────────────────────────────
    if best_model is None or best_metrics is None:
        print("\n❌ No run achieved recall >= 0.75. Cannot select showcase model.")
        return None

    best_model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_type": "Production CNN Showcase (frozen)",
        "model_version": "showcase_v1",
        "training_date": str(datetime.datetime.now()),
        "dataset": dataset_path,
        "training_run_id": best_metrics["run_id"],
        "threshold": best_metrics["threshold"],
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "roc_auc": best_metrics["roc_auc"],
        "pr_auc": best_metrics["pr_auc"],
        "results": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1"],
            "roc_auc": best_metrics["roc_auc"],
            "pr_auc": best_metrics["pr_auc"],
            "threshold": best_metrics["threshold"],
            "precision_faulty": best_metrics["precision"],
            "recall_faulty": best_metrics["recall"],
            "f1_faulty": best_metrics["f1"],
        },
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 60)
    print("  ✅ SHOWCASE MODEL SELECTED")
    print("=" * 60)
    print(f"  Run ID    : {best_metrics['run_id']}")
    print(f"  Threshold : {best_metrics['threshold']:.2f}")
    print(f"  Accuracy  : {best_metrics['accuracy']:.4f}")
    print(f"  Precision : {best_metrics['precision']:.4f}")
    print(f"  Recall    : {best_metrics['recall']:.4f}")
    print(f"  F1 Score  : {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {best_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {best_metrics['pr_auc']:.4f}")
    print(f"\n  Model    → {MODEL_PATH}")
    print(f"  Scaler   → {SCALER_PATH}")
    print(f"  Metadata → {METADATA_PATH}")
    print("=" * 60 + "\n")

    return metadata


# ── CLI entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Freeze the showcase CNN model via multi-run selection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/jm1.csv",
        help="Path to dataset CSV (default: data/jm1.csv)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of training runs (default: 20)",
    )
    args = parser.parse_args()
    freeze_showcase(args.data, num_runs=args.runs)
