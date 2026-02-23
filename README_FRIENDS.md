## Get the project onto your computer
1) Open Visual Studio Code.
2) Open the terminal inside VS Code:
   - Click Terminal > New Terminal.
3) Clone the repo (replace the URL with the real GitHub URL):
```
git clone https://github.com/Derekanton-cloud/dynamic-fault-predictor.git
```
4) Move into the project folder:
```
cd dynamic-fault-predictor
```

## Create and activate a virtual environment
In the VS Code terminal, run:
```
python -m venv .venv
```

Activate it (Windows PowerShell):
```
.\.venv\Scripts\Activate.ps1
```

If that fails, try Command Prompt:
```
.\.venv\Scripts\activate.bat
```

## Install the project dependencies
With the virtual environment active, run:
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit dashboard (recommended first)
From the project root:
```
streamlit run app/streamlit_app.py
```

A browser page will open. Upload a CSV with a defects column to see predictions.

## Run from the command line
Train the hybrid production model:
```
python main.py --mode train --data data/jm1.csv
```

Run predictions using the production ensemble:
```
python main.py --mode predict --data data/jm1.csv
```

Retrain a dynamic model on a new dataset:
```
python main.py --mode retrain --data data/jm1.csv
```

## Simple project explanation
This project predicts which software modules are likely to be faulty based on numeric code metrics.

- main.py
  - The command line entry point.
  - It lets you train, predict, or retrain.

- app/streamlit_app.py
  - The web dashboard where you can upload a dataset and see results.

- src/training/
  - train_production.py trains the hybrid CNN + XGBoost model.
  - retrain_dynamic.py trains a dynamic CNN for a new dataset.
  - threshold_optimizer.py picks the best probability cutoff.

- src/inference/predict.py
  - Loads saved models and makes predictions.

- src/evaluation/evaluate.py
  - Calculates accuracy, recall, precision, F1, ROC, and PR curves.

- src/utils/
  - preprocess.py cleans data, handles missing values, scales features, and reshapes for CNNs.
  - imbalance.py balances data when there are too few faulty examples.
  - feature_validation.py checks the dataset before training or prediction.

- data/
  - Sample dataset (jm1.csv) you can test with.

- models/
  - Saved trained models (CNN and XGBoost).

- artifacts/
  - Saved scalers and metadata used to reproduce results.

- reports/
  - Generated plots and prediction exports after you run training or prediction.
  - This folder will be created automatically.

## Dataset rules (important)
- Your CSV must include a column named defects.
- All other columns should be numeric.
- If you have text columns, encode them before using this project.

## Troubleshooting
- If streamlit is not found, make sure the virtual environment is active and run pip install -r requirements.txt again.
- If you see missing model files, run the train command first to regenerate them.
- If Python commands fail, restart VS Code and re-activate the virtual environment.
