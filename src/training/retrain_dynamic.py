import os
import json
from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.utils.preprocess import DataPreprocessor
from src.utils.imbalance import ImbalanceHandler
from src.training.threshold_optimizer import ThresholdOptimizer
from src.evaluation.evaluate import ModelEvaluator


def build_cnn_model(input_shape):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv1D(64, 3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(128, 3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def retrain_dynamic_model(dataset_path):

    print("\n🔄 Starting Dynamic Retraining...")

    # ------------------------------------------------
    # 1️⃣ Preprocessing
    # ------------------------------------------------
    df_preprocessor = DataPreprocessor()
    import pandas as pd
    df = pd.read_csv(dataset_path)

    X_scaled, y = df_preprocessor.prepare_features(df, fit=True)
    X_scaled = df_preprocessor.reshape_for_cnn(X_scaled)

    scaler = df_preprocessor.scaler

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ------------------------------------------------
    # 2️⃣ Handle Imbalance
    # ------------------------------------------------
    imbalance_handler = ImbalanceHandler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    X_train_bal, y_train_bal = imbalance_handler.handle(
        X_train_flat,
        y_train
    )

    # Reshape back to CNN format
    X_train_bal = X_train_bal.reshape(
        X_train_bal.shape[0],
        X_train.shape[1],
        1
    )

    # ------------------------------------------------
    # 3️⃣ Build + Train Model
    # ------------------------------------------------
    model = build_cnn_model((X_train.shape[1], 1))

    model.fit(
        X_train_bal,
        y_train_bal,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # ------------------------------------------------
    # 4️⃣ Threshold Optimization
    # ------------------------------------------------
    y_test_prob = model.predict(X_test).ravel()

    best_threshold, best_f1 = ThresholdOptimizer.optimize(
        y_test,
        y_test_prob,
        min_recall=0.75
    )

    print(f"\n🎯 Best Threshold: {best_threshold:.4f}")
    print(f"📈 F1 at Best Threshold: {best_f1:.4f}")

    # ------------------------------------------------
    # 5️⃣ Final Evaluation
    # ------------------------------------------------
    evaluator = ModelEvaluator()

    results = evaluator.evaluate(
        y_test,
        y_test_prob,
        best_threshold,
        model_name="dynamic_model"
    )

    # ------------------------------------------------
    # 6️⃣ Save Model + Artifacts
    # ------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = "models/experiments"
    scaler_dir = "artifacts/scalers"
    metadata_dir = "artifacts/metadata"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f"dynamic_model_{timestamp}.keras"
    )

    scaler_path = os.path.join(
        scaler_dir,
        f"dynamic_scaler_{timestamp}.pkl"
    )

    metadata_path = os.path.join(
        metadata_dir,
        f"dynamic_metadata_{timestamp}.json"
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": "Dynamic CNN",
        "timestamp": timestamp,
        "threshold": best_threshold,
        "results": results,
        "validation_f1": best_f1
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✅ Dynamic Model Saved Successfully")
    print(f"📦 Model: {model_path}")
    print(f"📊 Metadata: {metadata_path}")

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metadata_path": metadata_path,
        "threshold": best_threshold
    }
