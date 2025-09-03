import os
from glob import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class VariableLengthDataset(Dataset):
    """
        Dataset loader for variable-length sequences.

        This loader reads CSV files from a folder structure (with subfolders "High" and "Low"),
        extracts the specified features, subtracts reference values, and returns a list of PyTorch tensors
        along with their corresponding labels.
    """
    def __init__(self, folder_path, feature_names, **garbage_params):
        
        """
        Args:
            folder_path (str): Path to the dataset folder.
            feature_names (list): List of feature names to extract.
        """
        self.sequences, self.labels = self.load_data(folder_path, feature_names)

    def load_data(self, folder_path, feature_names):
        sequences = []
        labels = []
        for label_folder in ['High', 'Low']:
            label_path = os.path.join(folder_path, label_folder)
            label = 1 if label_folder == 'High' else 0
            for file in glob(os.path.join(label_path, '*.csv')):
                df = pd.read_csv(file).fillna(0)
                df = df[df['frame'] >= 10]
                df = df[df['frame'] <= 60]
                # Ensure required reference columns exist
                reference_columns = {'x_23': 0, 'y_23': 0, 'z_23': 0}
                missing_references = [col for col in reference_columns if col not in df.columns]
                if missing_references:
                    print(f"Error: Missing reference columns {missing_references} in file {file}")
                    continue

                for feature in feature_names:
                    prefix = feature[0]
                    if prefix in ['x', 'y', 'z']:
                        ref_col = f"{prefix}_23"
                        if ref_col in df.columns:
                            df[feature] -= df[ref_col]
                        else:
                            print(f"Warning: Missing reference column {ref_col} for feature {feature} in file {file}")

                try:
                    feature_data = df[feature_names].values
                except KeyError as e:
                    print(f"Error: Missing columns {e.args} in file {file}")
                    continue

                if feature_data.shape[0] == 0:
                    # print(f"Empty sequence found in {file}, skipping...")
                    continue

                feature_data = np.nan_to_num(feature_data, nan=0.0).astype(np.float32)
                sequences.append(torch.tensor(feature_data, dtype=torch.float32))
                labels.append(label)
        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """
    Custom collate function to pad sequences and return their original lengths.
    """
    from torch.nn.utils.rnn import pad_sequence
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels
