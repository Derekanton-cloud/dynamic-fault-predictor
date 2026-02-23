# Dynamic Fault Predictor

A production-ready machine learning system for software fault prediction with three modes of operation: a stable production model, a dataset-adaptive dynamic model, and a hybrid ensemble that combines CNN and XGBoost. The project includes a Streamlit dashboard for interactive analysis, drift monitoring, and model comparison, plus a CLI for training and batch inference.

## Why this project exists
Software quality teams need early warning signals to catch risky modules before release. This project focuses on fault prediction at the code-metrics level, providing:
- A hybrid production model optimized for recall while balancing precision.
- Dynamic retraining for new datasets with automatic threshold selection.
- Transparent evaluation artifacts (metrics, ROC/PR, confusion matrices).
- An interactive dashboard that makes model decisions explainable.

## Key features
- Hybrid CNN + XGBoost ensemble for strong baseline performance.
- Dynamic retraining pipeline with imbalance handling via SMOTE.
- Threshold optimization with minimum-recall constraints.
- Streamlit UI for model selection, metric visualization, and drift checks.
- Structured metadata artifacts for reproducible deployments.

## Project structure (high level)
- app/: Streamlit dashboard UI.
- src/: training, inference, evaluation, and preprocessing modules.
- models/: saved CNN and XGBoost models.
- artifacts/: scalers and metadata for reproducibility.
- data/: sample dataset (JM1).
- reports/: generated plots and prediction exports (ignored in git).

## Quick start
1) Create and activate a virtual environment.
2) Install dependencies from requirements.txt.
3) Run the Streamlit dashboard or use the CLI.

## Running the dashboard
From the repo root:
```
streamlit run app/streamlit_app.py
```

## CLI usage
Train the hybrid production model:
```
python main.py --mode train --data data/jm1.csv
```

Run batch predictions using the production ensemble:
```
python main.py --mode predict --data data/jm1.csv
```

Retrain a dynamic model on a new dataset:
```
python main.py --mode retrain --data data/jm1.csv
```

## Data format
- Input must be a CSV file.
- The target column must exist and is expected to be named defects.
- All feature columns must be numeric; non-numeric columns should be encoded before use.

## Model artifacts
The dashboard and CLI expect the following artifacts to exist in the repo:
- models/production/cnn_model.keras
- models/production/xgb_model.pkl
- artifacts/scalers/scaler.pkl
- artifacts/metadata/production_metadata.json

Dynamic model artifacts are stored in models/experiments and artifacts/metadata with timestamps.

## Metrics snapshot (included artifacts)
Example results from the latest production showcase:
- Accuracy: 0.7813
- Recall: 0.7608
- Precision: 0.4015
- F1: 0.5256

## Notes
- If you retrain, new metadata and model files are created automatically.
- For collaborators and step-by-step setup, see README_FRIENDS.md.
