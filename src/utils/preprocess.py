import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Enterprise-safe preprocessing pipeline.
    Handles:
    - Target detection
    - Binary encoding
    - Missing values
    - Feature scaling
    - CNN reshaping
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def detect_target_column(self, df):
        """
        Automatically detect target column.
        Looks for common fault labels.
        """
        possible_targets = ["defects", "bug", "bugs", "fault", "label", "target"]

        for col in df.columns:
            if col.lower() in possible_targets:
                return col

        raise ValueError(
            "No valid target column found. "
            "Dataset must contain one of: defects, bug, bugs, fault, label, target"
        )

    def prepare_features(self, df, target_column=None, fit=True):
        """
        Full preprocessing pipeline.
        """

        if target_column is None:
            target_column = self.detect_target_column(df)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert target to binary (0/1)
        y = np.where(y > 0, 1, 0)

        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))

        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()
        else:
            if self.feature_columns is None:
                raise ValueError("Feature columns not stored from training phase.")
            X = X[self.feature_columns]
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def process(
        self,
        csv_path,
        target_column="defects",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    ):
        """Load data, scale, reshape, and return train/val/test splits with scaler."""

        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1.")

        if val_size <= 0 or val_size >= 1:
            raise ValueError("val_size must be between 0 and 1.")

        df = pd.read_csv(csv_path)

        X_scaled, y = self.prepare_features(
            df,
            target_column=target_column,
            fit=True,
        )

        X_scaled = self.reshape_for_cnn(X_scaled)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        val_ratio = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler

    def reshape_for_cnn(self, X):
        """
        Reshape for 1D CNN
        (samples, features, 1)
        """
        return X.reshape(X.shape[0], X.shape[1], 1)


# Backwards compatibility for modules importing Preprocessor
Preprocessor = DataPreprocessor
