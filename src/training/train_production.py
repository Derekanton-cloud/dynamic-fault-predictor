import os
import json
import datetime
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import save_model
from xgboost import XGBClassifier

from src.utils.preprocess import DataPreprocessor
from src.utils.imbalance import ImbalanceHandler
from src.utils.feature_validation import FeatureValidator
from src.training.threshold_optimizer import ThresholdOptimizer
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.metric_analysis import threshold_sweep_analysis


class ProductionTrainer:

    def __init__(self):
        self.model_dir = "models/production"
        self.scaler_dir = "artifacts/scalers"
        self.metadata_dir = "artifacts/metadata"

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def build_cnn(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, csv_path, target_column="defects"):

        print("🚀 Starting Hybrid Production Training...")

        # 1️⃣ Preprocess
        processor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = processor.process(
            csv_path,
            target_column
        )

        FeatureValidator(X_train.reshape(X_train.shape[0], -1))

        # 2️⃣ Handle imbalance on flattened data for both models
        balancer = ImbalanceHandler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_train_bal_flat, y_train_bal = balancer.handle(X_train_flat, y_train)

        # reshape balanced data back for CNN
        X_train_bal_cnn = X_train_bal_flat.reshape(
            X_train_bal_flat.shape[0],
            X_train.shape[1],
            1
        )

        # 3️⃣ Build CNN
        cnn_model = self.build_cnn((X_train.shape[1], 1))

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )

        cnn_model.fit(
            X_train_bal_cnn,
            y_train_bal,
            epochs=80,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, lr_scheduler],
            verbose=1
        )

        # 4️⃣ Train XGBoost on flattened data
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        xgb_model.fit(
            X_train_bal_flat,
            y_train_bal
        )

        # 5️⃣ Validation probabilities
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        cnn_val_prob = cnn_model.predict(X_val).ravel()
        xgb_val_prob = xgb_model.predict_proba(X_val_flat)[:, 1]

        # 6️⃣ Ensemble
        cnn_weight = 0.6
        xgb_weight = 0.4
        ensemble_val_prob = cnn_weight * cnn_val_prob + xgb_weight * xgb_val_prob

        best_threshold, best_f1 = ThresholdOptimizer.optimize(
            y_val,
            ensemble_val_prob,
            min_recall=0.75
        )

        print(f"Best Threshold (Ensemble): {best_threshold}")
        print(f"Validation F1 (Ensemble): {best_f1}")

        # 7️⃣ Test probabilities
        cnn_test_prob = cnn_model.predict(X_test).ravel()
        xgb_test_prob = xgb_model.predict_proba(X_test_flat)[:, 1]
        ensemble_test_prob = cnn_weight * cnn_test_prob + xgb_weight * xgb_test_prob

        # 8️⃣ Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(
            y_test,
            ensemble_test_prob,
            best_threshold,
            model_name="hybrid_production_model"
        )

        # ---------------------------------------------------
        # 8️⃣ Full Threshold Sensitivity Analysis (Professional)
        # ---------------------------------------------------
        threshold_sweep_analysis(
            y_true=y_test,
            y_prob=ensemble_test_prob,
            model_name="hybrid_production_model"
        )

        print("\n=== Final Test Metrics ===")
        print(f"Accuracy: {results['accuracy']}")
        print(f"Recall (Faulty): {results['recall_faulty']}"
              )
        print(f"Precision (Faulty): {results['precision_faulty']}")
        print(f"F1 (Faulty): {results['f1_faulty']}")

        # 9️⃣ Save Models & Scaler
        cnn_path = os.path.join(self.model_dir, "cnn_model.keras")
        xgb_path = os.path.join(self.model_dir, "xgb_model.pkl")
        cnn_model.save(cnn_path)
        joblib.dump(xgb_model, xgb_path)

        scaler_path = os.path.join(self.scaler_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        # 🔟 Save Metadata
        metadata = {
            "model_version": "hybrid_v1",
            "training_date": str(datetime.datetime.now()),
            "threshold": best_threshold,
            "results": results,
            "validation_f1": best_f1,
            "cnn_weight": cnn_weight,
            "xgb_weight": xgb_weight,
            "cnn_model_path": cnn_path,
            "xgb_model_path": xgb_path,
            "scaler_path": scaler_path
        }

        metadata_path = os.path.join(self.metadata_dir, "production_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print("✅ Hybrid Model Trained & Saved Successfully!")

        return results

def train_production_model(dataset_path):
    trainer = ProductionTrainer()
    return trainer.train(dataset_path)