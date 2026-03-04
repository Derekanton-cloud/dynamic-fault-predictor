import argparse
import os
import sys

from src.evaluation.showcase_runner import run_showcase


def _venv_python_path(project_root: str) -> str | None:
    candidates = [
        os.path.join(project_root, "venv", "Scripts", "python.exe"),
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        os.path.join(project_root, "venv", "bin", "python"),
        os.path.join(project_root, ".venv", "bin", "python"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _running_inside_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _ensure_venv_python():
    if os.environ.get("DFP_BOOTSTRAPPED") == "1":
        return

    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = _venv_python_path(project_root)

    if venv_python is None:
        return

    if _running_inside_venv():
        return

    try:
        import tensorflow  # noqa: F401
        return
    except Exception:
        pass

    os.environ["DFP_BOOTSTRAPPED"] = "1"
    os.execv(venv_python, [venv_python, *sys.argv])


_ensure_venv_python()


# -----------------------------
# TRAIN PRODUCTION MODEL
# -----------------------------
def train_mode(dataset_path):
    from src.training.train_production import train_production_model
    train_production_model(dataset_path)


# -----------------------------
# PREDICT USING PRODUCTION MODEL
# -----------------------------
def predict_mode(dataset_path):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from src.inference.predict import ProductionPredictor
    predictor = ProductionPredictor()

    results = predictor.predict_from_csv(dataset_path)

    os.makedirs("reports", exist_ok=True)
    output_path = "reports/predictions_output.csv"

    df = results["predictions_dataframe"]
    df.to_csv(output_path, index=False)

    # Compute and display classification metrics
    if "defects" in df.columns:
        y_true = np.where(df["defects"] > 0, 1, 0)
        y_pred = df["prediction"].values
        y_prob = df["fault_probability"].values

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, y_prob)

        print("\n" + "=" * 50)
        print("  PRODUCTION HYBRID MODEL — PREDICTION RESULTS")
        print("=" * 50)
        print(f"  CNN Weight:    {predictor.cnn_weight}")
        print(f"  XGB Weight:    {predictor.xgb_weight}")
        print(f"  Threshold:     {results['threshold_used']}")
        print("-" * 50)
        print(f"  Accuracy:      {acc:.4f}")
        print(f"  Precision:     {prec:.4f}")
        print(f"  Recall:        {rec:.4f}")
        print(f"  F1 Score:      {f1:.4f}")
        print(f"  ROC-AUC:       {roc:.4f}")
        print("-" * 50)
        print(f"  Dataset Size:  {len(df)}")
        print(f"  Predicted Faults: {int(y_pred.sum())} / {len(df)}")
        print("=" * 50)

    print(f"\n📦 Model Version: {results['model_version']}")
    print(f"📅 Training Date: {results['training_date']}")
    print(f"📁 Predictions saved to: {output_path}")


# -----------------------------
# DYNAMIC RETRAIN MODE
# -----------------------------
def retrain_mode(dataset_path):
    from src.training.retrain_dynamic import retrain_dynamic_model
    results = retrain_dynamic_model(dataset_path)

    print("\n🧠 Dynamic Retraining Complete")
    print(f"📦 Model Saved At: {results['model_path']}")
    print(f"📊 Threshold Selected: {results['threshold']}")


# -----------------------------
# CLI ENTRY
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="train | predict | retrain | showcase"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default=None,
        help="Path to dataset CSV (required for train, predict, retrain)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        if not args.data:
            parser.error("--data is required for mode 'train'.")
        train_mode(args.data)

    elif args.mode == "predict":
        if not args.data:
            parser.error("--data is required for mode 'predict'.")
        predict_mode(args.data)

    elif args.mode == "retrain":
        if not args.data:
            parser.error("--data is required for mode 'retrain'.")
        retrain_mode(args.data)

    elif args.mode == "showcase":
        run_showcase()

    else:
        print("❌ Invalid mode.")
        print("Use --mode train | predict | retrain | showcase")
