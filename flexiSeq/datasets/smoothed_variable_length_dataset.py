import os
from glob import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

def kalman_smooth(data: np.ndarray, process_variance: float = 1e-5, measurement_variance: float = 0.1) -> np.ndarray:
    """
    A simple 1D Kalman filter implementation for smoothing.
    
    Args:
        data (np.ndarray): 1D array of data points.
        process_variance (float): Process variance.
        measurement_variance (float): Measurement variance.
    
    Returns:
        np.ndarray: Smoothed data.
    """
    n = len(data)
    xhat = np.zeros(n)      # posterior estimate
    P = np.zeros(n)         # posterior error estimate
    xhat[0] = data[0]
    P[0] = 1.0
    for k in range(1, n):
        # Time update
        xhatminus = xhat[k-1]
        Pminus = P[k-1] + process_variance
        # Measurement update
        K = Pminus / (Pminus + measurement_variance)
        xhat[k] = xhatminus + K * (data[k] - xhatminus)
        P[k] = (1 - K) * Pminus
    return xhat

class SmoothedVariableLengthDataset(Dataset):
    """
    Dataset loader for variable-length sequences with optional smoothing.

    This loader reads CSV files from a folder structure (with subfolders "High" and "Low"),
    extracts the specified features, subtracts reference values, optionally applies a smoothing filter,
    and returns a list of PyTorch tensors along with their corresponding labels.
    
    Available smoothing options:
      - "moving_average": Simple moving average filter.
      - "exponential": Exponential moving average filter.
      - "savitzky_golay": Savitzky–Golay filter.
      - "gaussian": Gaussian smoothing filter.
      - "kalman": Kalman smoothing filter.
      - "butterworth": Butterworth low-pass filter.
      - "none": No smoothing.
    """
    def __init__(self, folder_path: str, feature_names: list, 
                 smooth: bool = True, smoothing_type: str = "moving_average", 
                 window_size: int = 3, alpha: float = 0.3,
                 butterworth_order: int = 4, butterworth_cutoff: float = 0.1,
                 **kalman_params):
        """
        Args:
            folder_path (str): Path to the dataset folder.
            feature_names (list): List of feature names to extract.
            smooth (bool, optional): Whether to apply smoothing. Defaults to True.
            smoothing_type (str, optional): Smoothing method to use.
                Options: "moving_average", "exponential", "savitzky_golay", "gaussian", "kalman", "butterworth", "none". Defaults to "moving_average".
            window_size (int, optional): Window size for moving average, Savitzky–Golay, or Gaussian filter. Defaults to 3.
            alpha (float, optional): Smoothing factor for exponential moving average. Defaults to 0.3.
            butterworth_order (int, optional): Order of the Butterworth filter. Defaults to 4.
            butterworth_cutoff (float, optional): Normalized cutoff frequency for the Butterworth filter (0 < cutoff < 1). Defaults to 0.1.
            **kalman_params: Additional parameters for Kalman smoothing (e.g., process_variance, measurement_variance).
        """
        self.folder_path = folder_path
        self.feature_names = feature_names
        self.smooth = smooth
        self.smoothing_type = smoothing_type
        self.window_size = window_size
        self.alpha = alpha
        self.butterworth_order = butterworth_order
        self.butterworth_cutoff = butterworth_cutoff
        self.kalman_params = kalman_params
        self.sequences, self.labels = self.load_data(folder_path, feature_names)

    def smooth_series_moving_average(self, series: pd.Series) -> pd.Series:
        return series.rolling(window=self.window_size, min_periods=1, center=True).mean()

    def smooth_series_exponential(self, series: pd.Series) -> pd.Series:
        return series.ewm(alpha=self.alpha, adjust=False).mean()

    def smooth_series_savgol(self, series: pd.Series) -> pd.Series:
        # Ensure window length is odd.
        window = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
        return pd.Series(savgol_filter(series.values, window_length=window, polyorder=2))

    def smooth_series_gaussian(self, series: pd.Series) -> pd.Series:
        return pd.Series(gaussian_filter1d(series.values, sigma=self.window_size))

    def smooth_series_kalman(self, series: pd.Series) -> pd.Series:
        smoothed = kalman_smooth(series.values, 
                                 process_variance=self.kalman_params.get("process_variance", 1e-5),
                                 measurement_variance=self.kalman_params.get("measurement_variance", 0.1))
        return pd.Series(smoothed)

    def smooth_series_butterworth(self, series: pd.Series) -> pd.Series:
        """
        Apply a low-pass Butterworth filter to the data.
        
        Args:
            series (pd.Series): The data series to smooth.
            
        Returns:
            pd.Series: Smoothed data.
        """
        b, a = butter(N=self.butterworth_order, Wn=self.butterworth_cutoff, btype='low', analog=False)
        filtered = filtfilt(b, a, series.values)
        return pd.Series(filtered)

    def load_data(self, folder_path: str, feature_names: list):
        sequences = []
        labels = []
        for label_folder in ['High', 'Low']:
            label_path = os.path.join(folder_path, label_folder)
            label = 1 if label_folder == 'High' else 0
            for file in glob(os.path.join(label_path, '*.csv')):
                df = pd.read_csv(file).fillna(0)
                df = df[df['frame'] >= 10]
                df = df[df['frame'] <= 60]
                # Ensure required reference columns exist.
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
                            df[feature] = df[feature] - df[ref_col]
                        else:
                            print(f"Warning: Missing reference column {ref_col} for feature {feature} in file {file}")

                try:
                    feature_data = df[feature_names].values
                except KeyError as e:
                    print(f"Error: Missing columns {e.args} in file {file}")
                    continue

                if feature_data.shape[0] == 0:
                    continue

                if self.smooth and self.smoothing_type != "none":
                    df_smooth = df.copy()
                    for feature in feature_names:
                        if self.smoothing_type == "moving_average":
                            df_smooth[feature] = self.smooth_series_moving_average(df_smooth[feature])
                        elif self.smoothing_type == "exponential":
                            df_smooth[feature] = self.smooth_series_exponential(df_smooth[feature])
                        elif self.smoothing_type == "savitzky_golay":
                            df_smooth[feature] = self.smooth_series_savgol(df_smooth[feature])
                        elif self.smoothing_type == "gaussian":
                            df_smooth[feature] = self.smooth_series_gaussian(df_smooth[feature])
                        elif self.smoothing_type == "kalman":
                            df_smooth[feature] = self.smooth_series_kalman(df_smooth[feature])
                        elif self.smoothing_type == "butterworth":
                            df_smooth[feature] = self.smooth_series_butterworth(df_smooth[feature])
                        else:
                            # Unknown smoothing type; leave data unchanged.
                            pass
                    feature_data = df_smooth[feature_names].values

                feature_data = np.nan_to_num(feature_data, nan=0.0).astype(np.float32)
                sequences.append(torch.tensor(feature_data, dtype=torch.float32))
                labels.append(label)
        return sequences, labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
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