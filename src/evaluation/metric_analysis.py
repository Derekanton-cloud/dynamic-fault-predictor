import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def threshold_sweep_analysis(y_true, y_prob, model_name="model"):

    os.makedirs("reports", exist_ok=True)

    thresholds = np.linspace(0.01, 0.99, 99)

    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results.append([t, acc, prec, rec, f1])

    df = pd.DataFrame(
        results,
        columns=["threshold", "accuracy", "precision", "recall", "f1"]
    )

    df.to_csv(f"reports/{model_name}_threshold_analysis.csv", index=False)

    # -----------------------------------------
    # Plot 1: Threshold vs Metrics
    # -----------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(df["threshold"], df["accuracy"], label="Accuracy")
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title(f"{model_name} - Metrics vs Threshold")
    plt.legend()
    plt.grid()
    plt.savefig(f"reports/{model_name}_threshold_vs_metrics.png")
    plt.close()

    # -----------------------------------------
    # Plot 2: Precision vs Recall
    # -----------------------------------------
    plt.figure(figsize=(6,6))
    plt.plot(df["recall"], df["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision vs Recall")
    plt.grid()
    plt.savefig(f"reports/{model_name}_precision_vs_recall.png")
    plt.close()

    # -----------------------------------------
    # Plot 3: Accuracy vs Recall
    # -----------------------------------------
    plt.figure(figsize=(6,6))
    plt.plot(df["recall"], df["accuracy"])
    plt.xlabel("Recall")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy vs Recall")
    plt.grid()
    plt.savefig(f"reports/{model_name}_accuracy_vs_recall.png")
    plt.close()

    # -----------------------------------------
    # Plot 4: Accuracy vs Precision
    # -----------------------------------------
    plt.figure(figsize=(6,6))
    plt.plot(df["precision"], df["accuracy"])
    plt.xlabel("Precision")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy vs Precision")
    plt.grid()
    plt.savefig(f"reports/{model_name}_accuracy_vs_precision.png")
    plt.close()

    print("✅ Threshold analysis completed and graphs saved in /reports")

    return df