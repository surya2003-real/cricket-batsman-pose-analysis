import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Any, Dict, Tuple
import random

class RandomForestRangeClassifier:
    """
    RandomForest classifier for range-based features.

    This model wraps scikit-learn's RandomForestClassifier to provide
    a consistent interface with methods like `fit()`, `predict()`, and `evaluate()`.
    """
    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        """
        Initializes the RandomForestClassifier.

        Args:
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            **kwargs: Additional keyword arguments passed to RandomForestClassifier.
        """
        if random_state==-1:
            random_state = random.randint(0, 1000)
        self.model = RandomForestClassifier(random_state=random_state, **kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit the RandomForest model on the provided training data.

        Args:
            X (pd.DataFrame): Training features.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels and probabilities for the given features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Predicted class labels.
                - Predicted probabilities for the positive class.
        """
        predictions = self.model.predict(X)
        prob_predictions = self.model.predict_proba(X)[:, 1]
        return predictions, prob_predictions

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, verbose: bool = True) -> Tuple[float, float, float]:
        """
        Evaluate the model on the provided test data.

        Args:
            X (pd.DataFrame): Test features.
            y (np.ndarray): True test labels.
            verbose (bool, optional): If True, prints the evaluation metrics.

        Returns:
            Tuple[float, float, float]: Accuracy, AUC-ROC, and F1 score.
        """
        predictions, prob_predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        auc = roc_auc_score(y, prob_predictions)
        f1 = f1_score(y, predictions)
        f1_weighted = f1_score(y, predictions, average='weighted')
        if verbose:
            print(f"Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f}, F1: {f1:.4f}, F1 (weighted): {f1_weighted:.4f}")
        return accuracy, auc, f1, f1_weighted
