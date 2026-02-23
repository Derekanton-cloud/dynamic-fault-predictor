import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluator:

    def __init__(self, report_dir="reports", metadata_dir="artifacts/metadata"):
        self.report_dir = report_dir
        self.metadata_dir = metadata_dir

        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def evaluate(self, y_true, y_prob, threshold, model_name="model"):

        y_pred = (y_prob >= threshold).astype(int)

        # ----- Metrics -----
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # ----- ROC -----
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # ----- PR Curve -----
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)

        # ----- Confusion Matrix -----
        cm = confusion_matrix(y_true, y_pred)

        # Save confusion matrix
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        cm_path = os.path.join(self.report_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Save ROC curve
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_title("ROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        roc_path = os.path.join(self.report_dir, f"{model_name}_roc.png")
        plt.savefig(roc_path)
        plt.close()

        # Save PR curve
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.4f}")
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()
        pr_path = os.path.join(self.report_dir, f"{model_name}_pr.png")
        plt.savefig(pr_path)
        plt.close()

        # Structured result
        results = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "threshold": round(threshold, 4),
            # Backward-compatible aliases for fault-positive metrics
            "precision_faulty": round(prec, 4),
            "recall_faulty": round(rec, 4),
            "f1_faulty": round(f1, 4)
        }

        # Save metadata JSON
        metadata_path = os.path.join(self.metadata_dir, f"{model_name}_results.json")
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=4)

        return results
