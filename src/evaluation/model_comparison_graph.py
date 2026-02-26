import os
import matplotlib.pyplot as plt
import numpy as np

def generate_model_comparison():

    models = [
        "Random Forest",
        "SVM",
        "Logistic Regression",
        "AdaBoost",
        "Gaussian NB",
        "1D CNN"
    ]

    # Classical model results you computed
    accuracy = [
        0.825445,
        0.705036,
        0.706929,
        0.703143,
        0.826959,
        0.7813  # CNN
    ]

    recall = [
        0.351544,
        0.615202,
        0.572447,
        0.629454,
        0.232779,
        0.7608  # CNN
    ]

    x = np.arange(len(models))
    width = 0.35

    os.makedirs("reports/documentation", exist_ok=True)

    plt.figure(figsize=(14, 7))

    bars1 = plt.bar(x - width/2, accuracy, width, label="Accuracy")
    bars2 = plt.bar(x + width/2, recall, width, label="Recall")

    # Highlight your CNN bars
    bars1[-1].set_color("darkred")
    bars2[-1].set_color("darkblue")

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Accuracy vs Recall Comparison on JM1 Dataset")

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig("reports/documentation/jm1_accuracy_vs_recall_comparison.png", dpi=400)
    plt.close()

    print("✅ Accuracy vs Recall comparison graph saved successfully.")

if __name__ == "__main__":
    generate_model_comparison()