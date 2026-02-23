import pandas as pd


class FeatureValidator:
    """
    Ensures dataset compatibility.
    Makes the system work with ANY software dataset.
    """

    def __init__(self, target_column="defects"):
        self.target_column = target_column

    def validate(self, df: pd.DataFrame):
        """
        Ensures:
        - Target column exists
        - Data contains numeric features
        - No empty dataframe
        """
        if df.empty:
            raise ValueError("Uploaded dataset is empty.")

        if self.target_column not in df.columns:
            raise ValueError(
                f"Dataset must contain target column '{self.target_column}'."
            )

        # Drop non-numeric columns safely
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] == 0:
            raise ValueError(
                "Dataset must contain numeric feature columns."
            )

        return numeric_df

    def separate_features_target(self, df: pd.DataFrame):
        """
        Splits dataset into X and y.
        Converts defect counts into binary classification.
        """

        X = df.drop(columns=[self.target_column])

        y = df[self.target_column]

        # Convert to binary:
        # 0 defects → 0 (clean)
        # >0 defects → 1 (faulty)
        y = (y > 0).astype(int)

        return X, y


def validate_numeric_features(df: pd.DataFrame):
    """Ensure downstream steps only see numeric features."""
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError("Dataset must contain at least one numeric feature column.")

    if numeric_df.shape[1] != df.shape[1]:
        raise ValueError("Some feature columns are non-numeric. Please encode them before inference.")

    return numeric_df
