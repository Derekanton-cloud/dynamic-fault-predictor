import numpy as np
from sklearn.metrics import recall_score, f1_score


class ThresholdOptimizer:
    """Select threshold ensuring minimum recall while maximizing F1."""

    @staticmethod
    def optimize(y_true, y_prob, min_recall=0.75):

        thresholds = np.arange(0.05, 0.9, 0.01)

        best_threshold = 0.5
        best_f1 = 0

        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            recall = recall_score(y_true, preds)

            if recall >= min_recall:
                f1 = f1_score(y_true, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

        return best_threshold, best_f1
