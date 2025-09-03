import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
from typing import Optional, Tuple
from models.sequence_models.base_model import BaseClassifier

class Chomp1d(nn.Module):
    """
    Removes the last 'chomp_size' elements along the temporal dimension.
    This is used to ensure that the output of a convolution matches the
    original sequence length.
    """
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class TemporalBlock(nn.Module):
    """
    A single Temporal Block used in the TCN.
    It consists of two causal, dilated convolution layers with weight normalization,
    followed by ReLU activations, dropout, chomp (to remove extra padding),
    and a residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dilation: int, dropout: float = 0.2) -> None:
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation  # Extra padding for causal convolution

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu1(out + res)

class TemporalConvNet(nn.Module):
    """
    A stack of TemporalBlocks forming the TCN.
    """
    def __init__(self, num_inputs: int, num_channels: list,
                 kernel_size: int = 2, dropout: float = 0.2) -> None:
        """
        Args:
            num_inputs (int): Number of input channels (features).
            num_channels (list): List of output channel sizes for each TemporalBlock.
            kernel_size (int, optional): Convolution kernel size. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TCNClassifier(BaseClassifier):
    """
    TCN-based classifier for variable-length sequences.

    The model uses a Temporal Convolutional Network to process the input sequence.
    An adaptive max pooling layer converts the variable-length sequence into a fixed-size
    representation, which is then fed to a fully connected layer for classification.
    """
    def __init__(self,
                 input_size: int,
                 num_channels: list = [128, 128],
                 kernel_size: int = 2,
                 dropout: float = 0.2,
                 num_classes: int = 2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 early_stopping: bool = False,
                 early_stopping_patience: int = 10,
                 device: Optional[torch.device] = None) -> None:
        """
        Args:
            input_size (int): Number of input features.
            num_channels (list, optional): Filters for each TCN layer. Defaults to [128, 128].
            kernel_size (int, optional): Convolution kernel size. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            num_classes (int, optional): Number of output classes. Defaults to 2.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            epochs (int, optional): Maximum training epochs. Defaults to 100.
            early_stopping (bool, optional): Enable early stopping. Defaults to False.
            early_stopping_patience (int, optional): Epochs with no improvement before stopping. Defaults to 10.
            device (torch.device, optional): Device for training. Defaults to CUDA if available.
        """
        super(TCNClassifier, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TCN expects input as (batch, channels, seq_length)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Reduces sequence length to 1
        self.fc = nn.Linear(num_channels[-1], num_classes)
        self.model = nn.Sequential(self.tcn, self.global_pool, nn.Flatten(), self.fc).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            verbose: bool = False) -> None:
        """
        Train the TCN classifier.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader, optional): DataLoader for validation data. If not provided, training data is used.
            verbose (bool, optional): If True, displays training progress with metrics.
        """
        self.model.train()
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        epoch_iterator = tqdm(range(self.epochs), desc="Training", leave=True) if verbose else range(self.epochs)
        for epoch in epoch_iterator:
            total_train_loss = 0.0
            for sequences, lengths, labels in train_loader:
                # sequences shape: (batch, seq_length, input_size)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                # Transpose to (batch, input_size, seq_length)
                sequences = sequences.transpose(1, 2)
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Evaluate on validation set (or training set if none provided)
            current_val_loader = val_loader if val_loader is not None else train_loader
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for sequences, lengths, labels in current_val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    sequences = sequences.transpose(1, 2)
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(current_val_loader)
            
            if self.early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    if verbose:
                        epoch_iterator.set_description(f"Early stopping at epoch {epoch+1}")
                    break

            if verbose and ((epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs):
                acc, auc, f1, f1_weighted = self.evaluate(current_val_loader, verbose=False)
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{self.epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} "
                    f"Acc: {acc:.4f} AUC: {auc:.4f} F1: {f1:.4f} F1 (weighted): {f1_weighted:.4f}"
                )
            self.model.train()

    def predict(self, test_loader: torch.utils.data.DataLoader) -> Tuple[list, list]:
        """
        Generate predictions for the test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            Tuple[list, list]: A tuple of (true labels, predicted probabilities for the positive class).
        """
        self.model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for sequences, lengths, labels in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                sequences = sequences.transpose(1, 2)
                outputs = self.model(sequences)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return all_labels, all_probs

    def evaluate(self,
                 loader: torch.utils.data.DataLoader,
                 verbose: bool = False) -> Tuple[float, float, float]:
        """
        Evaluate model performance on the given data.

        Args:
            loader (DataLoader): DataLoader for evaluation.
            verbose (bool, optional): If True, prints evaluation metrics.

        Returns:
            Tuple[float, float, float]: (accuracy, AUC-ROC, F1 score)
        """
        y_true, y_probs = self.predict(loader)
        y_pred = [1 if p > 0.5 else 0 for p in y_probs]
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        if verbose:
            print(f"[TCNClassifier] Evaluation -- Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f} F1 (weighted): {f1_weighted:.4f}")
        return accuracy, auc, f1, f1_weighted
