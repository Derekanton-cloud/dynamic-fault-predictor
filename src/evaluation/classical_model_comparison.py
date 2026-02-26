import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.utils.preprocess import DataPreprocessor
from src.utils.imbalance import ImbalanceHandler

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob)
    }

    return results


def run_comparison(csv_path="data/jm1.csv"):

    processor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = processor.process(csv_path)

    balancer = ImbalanceHandler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_train_bal, y_train_bal = balancer.handle(X_train_flat, y_train)

    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ("SVM", SVC(probability=True, random_state=SEED)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
        ("AdaBoost", AdaBoostClassifier(random_state=SEED)),
        ("Gaussian NB", GaussianNB())
    ]

    results = []

    for name, model in models:
        print(f"Training {name}...")
        res = evaluate_model(name, model, X_train_bal, y_train_bal, X_test_flat, y_test)
        results.append(res)

    df = pd.DataFrame(results)

    os.makedirs("reports/documentation", exist_ok=True)
    df.to_csv("reports/documentation/classical_model_metrics.csv", index=False)

    print("✅ Classical model comparison completed.")
    print(df)

    return df


if __name__ == "__main__":
    run_comparison()