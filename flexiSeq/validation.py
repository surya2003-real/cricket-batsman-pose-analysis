import numpy as np
from torch.utils.data import random_split, DataLoader
from typing import List, Tuple
from datasets.variable_length_dataset import VariableLengthDataset, collate_fn

class CrossValidator:
    def __init__(self, dataset_paths, feature_names, model_class, model_params,
                 batch_size=256, verbose=False, val_split=0.2, dataset_generator=None, 
                 smoothing_type=None):
        """
        dataset_paths: dict mapping dataset names to folder paths.
        feature_names: list of features.
        model_class: a model class that implements the BaseClassifier interface.
        model_params: parameters for model initialization.
        batch_size: training/evaluation batch size.
        verbose: whether to print detailed training/evaluation logs.
        val_split: fraction of the training data to use as a validation set.
        dataset_generator: dataset class to use for loading data.
        smoothing_type: type of smoothing to apply to the data.
        """
        self.dataset_names = list(dataset_paths.keys())
        self.dataset_paths = dataset_paths
        self.feature_names = feature_names
        self.model_class = model_class
        self.model_params = model_params
        self.batch_size = batch_size
        self.verbose = verbose
        self.val_split = val_split
        self.dataset_generator = dataset_generator if dataset_generator is not None else VariableLengthDataset
        self.smoothing_type = smoothing_type

    def validate_first_as_test(self) -> Tuple[float, float, float]:
        """
        Use the first folder as the test set and the remaining folders for training.
        Optionally, split the concatenated training set into train and validation subsets.
        
        Returns:
            Tuple[float, float, float]: (accuracy, auc, f1) on the test set.
        """
        # Use the first dataset as the test set.
        test_name = self.dataset_names[0]
        print(f"Validation: Using '{test_name}' as the test set.")

        # Use the remaining datasets for training.
        train_names = self.dataset_names[1:]
        if not train_names:
            raise ValueError("At least two datasets are required: one for testing and at least one for training.")

        # Concatenate training data from all remaining datasets.
        train_dataset = self.dataset_generator(self.dataset_paths[train_names[0]], self.feature_names, smoothing_type=self.smoothing_type)
        for name in train_names[1:]:
            ds = self.dataset_generator(self.dataset_paths[name], self.feature_names, smoothing_type=self.smoothing_type)
            train_dataset.sequences.extend(ds.sequences)
            train_dataset.labels.extend(ds.labels)
        
        # Optionally create a validation split on the training data.
        if self.val_split == 0:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = None
        else:
            total_size = len(train_dataset)
            val_size = int(self.val_split * total_size)
            train_size = total_size - val_size
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        # Create the test loader.
        test_dataset = self.dataset_generator(self.dataset_paths[test_name], self.feature_names, smoothing_type=self.smoothing_type)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        # Initialize and train the model with early stopping using the validation loader.
        model = self.model_class(input_size=len(self.feature_names), **self.model_params)
        model.fit(train_loader, val_loader=val_loader, verbose=self.verbose)

        # Evaluate the model on the test set.
        acc, auc, f1, f1_weighted = model.evaluate(test_loader, verbose=self.verbose)
        print(f"Results on Test Set '{test_name}': Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, F1 (weighted) = {f1_weighted:.4f}")
        return acc, auc, f1, f1_weighted

    def validate(self, folds=5):
        """
        Optionally run multiple rounds of the above validation scheme.
        """
        all_results = []
        for i in range(folds):
            print(f"\nValidation Run {i+1}")
            result = self.validate_first_as_test()
            all_results.append(result)
        all_results = np.array(all_results)
        avg_results = np.mean(all_results, axis=0)
        return avg_results
