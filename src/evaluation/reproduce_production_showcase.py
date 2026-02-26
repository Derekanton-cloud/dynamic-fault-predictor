import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from src.utils.preprocess import DataPreprocessor


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def reproduce_production_showcase(
    dataset_path="data/jm1.csv",
    model_path="models/production/cnn_model.keras",
    scaler_path="artifacts/scalers/scaler.pkl",
):
    processor = DataPreprocessor()

    df = pd.read_csv(dataset_path)
    target_column = processor.detect_target_column(df)

    loaded_scaler = joblib.load(scaler_path)
    processor.scaler = loaded_scaler
    processor.feature_columns = [col for col in df.columns if col != target_column]

    X_scaled, y = processor.prepare_features(
        df,
        target_column=target_column,
        fit=False,
    )
    X_scaled = processor.reshape_for_cnn(X_scaled)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    val_ratio = 0.2 / (1 - 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=42,
    )

    _ = (X_train, X_val, y_train, y_val)

    model = load_model(model_path)
    y_prob = model.predict(X_test, verbose=0).ravel()

    thresholds = np.arange(0.01, 1.00, 0.01)
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "false_positives": int(fp),
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        )

    sweep_df = pd.DataFrame(rows)
    filtered = sweep_df[sweep_df["recall"] >= 0.75]

    if filtered.empty:
        raise RuntimeError("No threshold satisfies recall >= 0.75.")

    best_row = filtered.sort_values("precision", ascending=False).iloc[0]
    threshold = float(best_row["threshold"])

    y_pred_final = (y_prob >= threshold).astype(int)
    final_acc = accuracy_score(y_test, y_pred_final)
    final_prec = precision_score(y_test, y_pred_final, zero_division=0)
    final_rec = recall_score(y_test, y_pred_final, zero_division=0)
    final_f1 = f1_score(y_test, y_pred_final, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()

    print("Reproduced Production Showcase Metrics (CNN)")
    print(f"Threshold Selected: {threshold:.2f}")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall: {final_rec:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")

    target = {
        "accuracy": 0.7813,
        "precision": 0.4015,
        "recall": 0.7608,
    }

    tolerance = 0.01
    mismatch = []

    if abs(final_acc - target["accuracy"]) > tolerance:
        mismatch.append(
            f"accuracy mismatch: expected {target['accuracy']:.4f}, got {final_acc:.4f}"
        )
    if abs(final_prec - target["precision"]) > tolerance:
        mismatch.append(
            f"precision mismatch: expected {target['precision']:.4f}, got {final_prec:.4f}"
        )
    if abs(final_rec - target["recall"]) > tolerance:
        mismatch.append(
            f"recall mismatch: expected {target['recall']:.4f}, got {final_rec:.4f}"
        )

    if mismatch:
        print("⚠ WARNING: Metrics are outside tolerance (±0.01).")
        for line in mismatch:
            print(f"- {line}")
    else:
        print("✅ Metrics reproduced within tolerance (±0.01).")

    return {
        "threshold": threshold,
        "accuracy": final_acc,
        "precision": final_prec,
        "recall": final_rec,
        "f1_score": final_f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sweep": sweep_df,
        "best_row": best_row.to_dict(),
    }


if __name__ == "__main__":
    reproduce_production_showcase()
