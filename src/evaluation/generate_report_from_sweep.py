import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# Load Threshold Sweep File
# -------------------------------------------------------
SWEEP_PATH = "reports/production_showcase_20260222_110239_threshold_sweep.csv"
OUTPUT_DIR = "reports/documentation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(SWEEP_PATH)

# -------------------------------------------------------
# Tradeoff Graphs
# -------------------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(df["recall"], df["accuracy"])
plt.xlabel("Recall")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Recall")
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/accuracy_vs_recall.png")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(df["recall"], df["precision"])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall")
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/precision_vs_recall.png")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(df["precision"], df["accuracy"])
plt.xlabel("Precision")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Precision")
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/accuracy_vs_precision.png")
plt.close()

# -------------------------------------------------------
# Threshold Behavior Graphs
# -------------------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(df["threshold"], df["accuracy"], label="Accuracy")
plt.plot(df["threshold"], df["recall"], label="Recall")
plt.plot(df["threshold"], df["precision"], label="Precision")
plt.plot(df["threshold"], df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Metrics vs Threshold")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/metrics_vs_threshold.png")
plt.close()

# -------------------------------------------------------
# Correlation Matrix
# -------------------------------------------------------

corr = df[["accuracy", "precision", "recall", "f1"]].corr()

corr.to_csv(f"{OUTPUT_DIR}/metric_correlation.csv")

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Metrics")
plt.savefig(f"{OUTPUT_DIR}/metric_correlation_heatmap.png")
plt.close()

print("✅ All documentation graphs generated successfully.")
print(f"📂 Saved in {OUTPUT_DIR}")