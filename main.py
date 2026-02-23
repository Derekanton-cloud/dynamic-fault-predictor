import argparse
import os

from src.training.train_production import train_production_model
from src.training.retrain_dynamic import retrain_dynamic_model
from src.inference.predict import ProductionPredictor


# -----------------------------
# TRAIN PRODUCTION MODEL
# -----------------------------
def train_mode(dataset_path):
    train_production_model(dataset_path)


# -----------------------------
# PREDICT USING PRODUCTION MODEL
# -----------------------------
def predict_mode(dataset_path):
    predictor = ProductionPredictor()

    results = predictor.predict_from_csv(dataset_path)

    os.makedirs("reports", exist_ok=True)
    output_path = "reports/predictions_output.csv"

    results["predictions_dataframe"].to_csv(output_path, index=False)

    print("\n🚀 Prediction Complete")
    print(f"📊 Threshold Used: {results['threshold_used']}")
    print(f"📦 Model Version: {results['model_version']}")
    print(f"📅 Training Date: {results['training_date']}")
    print(f"📁 Predictions saved to: {output_path}")


# -----------------------------
# DYNAMIC RETRAIN MODE
# -----------------------------
def retrain_mode(dataset_path):

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
        help="train | predict | retrain"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset CSV"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args.data)

    elif args.mode == "predict":
        predict_mode(args.data)

    elif args.mode == "retrain":
        retrain_mode(args.data)

    else:
        print("❌ Invalid mode.")
        print("Use --mode train | predict | retrain")
