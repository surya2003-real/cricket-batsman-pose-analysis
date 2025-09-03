import os
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Tuple

class RangeDataset:
    """
    Dataset loader for range-based features extracted from CSV files.

    This loader traverses directories, loads CSV files, extracts the selected features,
    and computes the range (max - min) for each feature in each file.
    """
    def __init__(self, folder_path: str, selected_features: List[str]) -> None:
        """
        Initializes the RangeDataset.

        Args:
            folder_path (str): Path to the directory containing the CSV files.
            selected_features (List[str]): List of feature names to compute range for.
        """
        self.selected_features = selected_features
        self.features, self.labels = self._load_data(folder_path)

    def _load_data(self, folder_path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Loads CSV files from the provided directories and computes the range for selected features.
        
        Args:
            folder_path (str): Path to the directory containing the CSV files.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]:
                - A list of NumPy arrays, each containing feature ranges for one dataset.
                - A list of NumPy arrays, each containing labels (1 for "High" and 0 for "Low") for one dataset.
        """
        features = []
        labels = []
        for label_folder in ["High", "Low"]:
            label_path = os.path.join(folder_path, label_folder)
            label= 1 if label_folder == "High" else 0
            # Use glob to find all CSV files
            for file_path in glob(os.path.join(label_path, "*.csv")):
                # Load CSV as DataFrame
                df = pd.read_csv(file_path).fillna(0)
                df = df[df['frame'] >= 10]
                df = df[df['frame'] <= 60]
                # Use only selected features
                df_selected = df[self.selected_features]
                if df_selected.shape[0] == 0:
                    # print(f"Warning: No data for selected features in file {file_path}")
                    continue
                # Compute range: max - min for each feature
                feature_ranges = df_selected.max() - df_selected.min()
                features.append(feature_ranges.values)  # as NumPy array
                labels.append(label)
        return features, labels

