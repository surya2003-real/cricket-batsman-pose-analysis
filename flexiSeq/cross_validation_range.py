import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datasets.range_dataset import RangeDataset

class RangeCrossValidator:
    def __init__(self,
                 dataset_paths: Dict[str, str],
                 selected_features: List[str],
                 model_class: Any,
                 model_params: Dict[str, Any] = None,
                 verbose: bool = False):
        """
        Initializes the RangeCrossValidator.

        Args:
            dataset_paths (dict): A dictionary mapping dataset names to folder paths.
            selected_features (List[str]): List of feature names to extract ranges for.
            model_class (Any): Model class that implements fit(), predict(), and evaluate().
            model_params (dict, optional): Parameters for initializing the model. Defaults to {}.
            verbose (bool, optional): If True, prints detailed logs. Defaults to False.
        """
        self.dataset_names = list(dataset_paths.keys())
        self.dataset_paths = dataset_paths
        self.selected_features = selected_features
        self.model_class = model_class
        self.model_params = model_params if model_params is not None else {}
        self.verbose = verbose
        

    def leave_one_out(self) -> List[Tuple[float, float, float]]:
        """
        Perform leave-one-dataset-out cross validation.

        Returns:
            List[Tuple[float, float, float]]: A list containing (accuracy, auc, f1) for each fold.
        """
        results = []
        for i, test_name in enumerate(self.dataset_names):
            if self.verbose:
                print(f"\nLeave-One-Out Fold {i+1}: Testing on {test_name}")
            train_names = [name for name in self.dataset_names if name != test_name]

            rd= RangeDataset(self.dataset_paths[train_names[0]], self.selected_features)
            X_train= rd.features
            y_train= rd.labels
            for name in train_names[1:]:
                rd= RangeDataset(self.dataset_paths[name], self.selected_features)
                X_train.extend(rd.features)
                y_train.extend(rd.labels)
    
            rd= RangeDataset(self.dataset_paths[test_name], self.selected_features)
            X_test= rd.features
            y_test= rd.labels

            # Convert arrays to DataFrames (scikit-learn expects DataFrame input)
            X_train_df = pd.DataFrame(X_train, columns=self.selected_features)
            X_test_df = pd.DataFrame(X_test, columns=self.selected_features)

            # Initialize and train the model
            model = self.model_class(**self.model_params)
            model.fit(X_train_df, y_train)

            # Evaluate the model on the test set
            acc, auc, f1, f1_weighted = model.evaluate(X_test_df, y_test, verbose=self.verbose)
            if self.verbose:
                print(f"Fold {i+1}: Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, F1 (weighted) = {f1_weighted:.4f}")
            results.append((acc, auc, f1, f1_weighted))
        return results

    def cross_validate(self, folds: int = 1) -> np.ndarray:
        """
        Optionally run multiple rounds of leave-one-out cross validation.

        Args:
            folds (int, optional): Number of rounds to run. Defaults to 1.

        Returns:
            np.ndarray: Averaged results for each dataset fold in the form (accuracy, auc, f1).
        """
        all_results = []
        for run in range(folds):
            if self.verbose:
                print(f"\nCross Validation Run {run+1}")
            results = self.leave_one_out()
            all_results.append(results)
        all_results = np.array(all_results)  # shape: (folds, num_datasets, 3)
        avg_results = np.mean(all_results, axis=0)
        return avg_results
