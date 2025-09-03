import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Dict, List, Tuple, Any

# Import dataset loaders for each model type:
from datasets.variable_length_dataset import VariableLengthDataset, collate_fn
from datasets.range_dataset import RangeDataset


class EnsembleCrossValidator:
    def __init__(self,
                 dataset_paths: Dict[str, str],
                 feature_names: List[str],
                 lstm_model_class: Any,
                 lstm_model_params: Dict[str, Any],
                 rf_model_class: Any,
                 rf_model_params: Dict[str, Any],
                 batch_size: int = 256,
                 verbose: bool = False,
                 val_split: float = 0.2) -> None:
        """
        Ensemble cross validator that trains both an LSTM classifier and a RandomForest classifier,
        then ensembles their predicted probabilities.

        Args:
            dataset_paths (dict): Dictionary mapping dataset names to folder paths.
            feature_names (List[str]): List of features to use.
            lstm_model_class (Any): Class for the LSTM classifier.
            lstm_model_params (dict): Parameters for the LSTM classifier.
            rf_model_class (Any): Class for the RandomForest classifier.
            rf_model_params (dict): Parameters for the RandomForest classifier.
            batch_size (int, optional): Batch size for LSTM DataLoader. Defaults to 256.
            verbose (bool, optional): If True, prints detailed logs. Defaults to False.
            val_split (float, optional): Fraction of training data to use as validation (for LSTM). Defaults to 0.2.
        """
        self.dataset_names = list(dataset_paths.keys())
        self.dataset_paths = dataset_paths
        self.feature_names = feature_names
        self.lstm_model_class = lstm_model_class
        self.lstm_model_params = lstm_model_params
        self.rf_model_class = rf_model_class
        self.rf_model_params = rf_model_params if rf_model_params is not None else {}
        self.batch_size = batch_size
        self.verbose = verbose
        self.val_split = val_split

    def leave_one_out(self) -> List[Tuple[float, float, float]]:
        """
        Perform leave-one-dataset-out cross validation, training both models and ensembling their probabilities.

        Returns:
            List[Tuple[float, float, float]]: A list with (accuracy, auc, f1) for each fold.
        """
        results = []
        for i, test_name in enumerate(self.dataset_names):
            if self.verbose:
                print(f"\nLeave-One-Out Fold {i+1}: Testing on {test_name}")
            train_names = [name for name in self.dataset_names if name != test_name]
            # ----- Prepare training data for LSTM model (Variable-Length Sequences) -----
            # Load training datasets for all datasets except the test dataset.
            # Combine training sequences and labels.
            lstm_train_dataset= VariableLengthDataset(self.dataset_paths[train_names[0]], self.feature_names)
            for name in train_names[1:]:
                ds = VariableLengthDataset(self.dataset_paths[name], self.feature_names)
                lstm_train_dataset.sequences.extend(ds.sequences)
                lstm_train_dataset.labels.extend(ds.labels)
            
            # Split the concatenated training dataset into training and validation subsets.
            if self.val_split == 0:
                train_loader = DataLoader(lstm_train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = None
            else:
                total_size = len(lstm_train_dataset)
                val_size = int(self.val_split * total_size)
                train_size = total_size - val_size
                train_subset, val_subset = random_split(lstm_train_dataset, [train_size, val_size])
                train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

            # ----- Prepare training data for RF model (Range-Based Features) -----
            rd = RangeDataset(self.dataset_paths[train_names[0]], self.feature_names)
            X_train_rf, y_train_rf = rd.features, rd.labels
            for name in train_names[1:]:
                rd = RangeDataset(self.dataset_paths[name], self.feature_names)
                X_train_rf.extend(rd.features)
                y_train_rf.extend(rd.labels)
            X_train_rf_df = pd.DataFrame(X_train_rf, columns=self.feature_names)

            # ----- Prepare test data for both models -----
            # For LSTM:
            lstm_test_dataset = VariableLengthDataset(self.dataset_paths[test_name], self.feature_names)
            test_loader = DataLoader(lstm_test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
            # For RF:
            rd = RangeDataset(self.dataset_paths[test_name], self.feature_names)
            X_test_rf = rd.features
            y_test_rf = rd.labels
            X_test_rf_df = pd.DataFrame(X_test_rf, columns=self.feature_names)

            # ----- Train both models -----
            lstm_model = self.lstm_model_class(input_size=len(self.feature_names), **self.lstm_model_params)
            rf_model = self.rf_model_class(**self.rf_model_params)
            lstm_model.fit(train_loader, val_loader=val_loader, verbose=self.verbose)
            rf_model.fit(X_train_rf_df, y_train_rf)

            # ----- Get predictions -----
            # LSTM: predict returns (true_labels, probabilities)
            lstm_true, lstm_probs = lstm_model.predict(test_loader)
            # RF: predict returns (predicted_labels, probabilities)
            rf_pred, rf_probs = rf_model.predict(X_test_rf_df)

            # ----- Ensemble Predictions -----
            # Here we assume both models' test order aligns (same underlying samples).
            ensemble_probs = (np.array(lstm_probs) + np.array(rf_probs)) / 2.0
            ensemble_pred = [1 if prob > 0.5 else 0 for prob in ensemble_probs]
            # Ground truth labels (we assume they are the same for both loaders)
            y_true = lstm_true

            # ----- Compute Metrics -----
            acc = accuracy_score(y_true, ensemble_pred)
            auc = roc_auc_score(y_true, ensemble_probs)
            f1 = f1_score(y_true, ensemble_pred)
            f1_weighted = f1_score(y_true, ensemble_pred, average='weighted')
            if self.verbose:
                print(f"Fold {i+1}: Ensemble Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, F1 (weighted) = {f1_weighted:.4f}")
            results.append((acc, auc, f1, f1_weighted))
        return results

    def cross_validate(self, folds: int = 1) -> np.ndarray:
        """
        Optionally run multiple rounds of leave-one-out cross validation.

        Args:
            folds (int, optional): Number of rounds to run. Defaults to 1.

        Returns:
            np.ndarray: Averaged ensemble metrics for each fold (accuracy, auc, f1).
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

