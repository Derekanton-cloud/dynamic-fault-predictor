import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter


class ImbalanceHandler:
    """
    Enterprise-safe imbalance handling.
    Automatically detects imbalance
    and applies SMOTE only if needed.
    """

    def __init__(self, imbalance_threshold=0.20):
        """
        imbalance_threshold:
        If minority class ratio < threshold,
        SMOTE will be applied.
        """
        self.imbalance_threshold = imbalance_threshold

    def check_imbalance(self, y):
        counter = Counter(y)
        total = len(y)

        ratios = {
            cls: count / total for cls, count in counter.items()
        }

        print("Class distribution:", counter)
        print("Class ratios:", ratios)

        # Detect minority class ratio
        minority_ratio = min(ratios.values())

        is_imbalanced = minority_ratio < self.imbalance_threshold

        return is_imbalanced

    def apply_smote(self, X, y):
        """
        Apply SMOTE to balance dataset.
        """
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        print("After SMOTE:", Counter(y_res))

        return X_res, y_res

    def handle(self, X, y):
        """
        Auto-handle imbalance.
        """
        if self.check_imbalance(y):
            print("⚠️ Imbalance detected. Applying SMOTE...")
            return self.apply_smote(X, y)
        else:
            print("✅ Dataset is balanced. No SMOTE applied.")
            return X, y

    def apply_class_weights(self, X, y):
        """Compatibility wrapper used by production trainer."""
        original_shape = None

        if X.ndim > 2:
            original_shape = X.shape[1:]
            X = X.reshape(X.shape[0], -1)

        X_bal, y_bal = self.handle(X, y)

        if original_shape is not None:
            X_bal = X_bal.reshape(X_bal.shape[0], *original_shape)

        return X_bal, y_bal
