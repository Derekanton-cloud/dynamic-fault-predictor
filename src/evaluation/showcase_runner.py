import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PRED_PATH = "reports/production_showcase_20260222_110239_predictions.csv"
THRESHOLD = 0.32

def run_showcase():

    df = pd.read_csv(PRED_PATH)

    # determine true label column
    if "true_label" in df.columns:
        y_true = df["true_label"].astype(int).values
        true_col = "true_label"
    elif "defects" in df.columns:
        y_true = (df["defects"].astype(float) > 0).astype(int).values
        true_col = "defects > 0"
    elif "label" in df.columns:
        y_true = df["label"].astype(int).values
        true_col = "label"
    else:
        raise ValueError(
            "No suitable true-label column found in predictions CSV. "
            "Expected one of: true_label, defects, label"
        )

    # determine probability column
    if "predicted_probability" in df.columns:
        y_prob = df["predicted_probability"].astype(float).values
        prob_col = "predicted_probability"
    elif "fault_probability" in df.columns:
        y_prob = df["fault_probability"].astype(float).values
        prob_col = "fault_probability"
    elif "fault_prob" in df.columns:
        y_prob = df["fault_prob"].astype(float).values
        prob_col = "fault_prob"
    else:
        raise ValueError(
            "No suitable probability column found in predictions CSV. "
            "Expected one of: predicted_probability, fault_probability, fault_prob"
        )

    y_pred = (y_prob >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== PRODUCTION SHOWCASE MODEL ===")
    print(f"Using true labels from: {true_col}")
    print(f"Using probabilities from: {prob_col}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")