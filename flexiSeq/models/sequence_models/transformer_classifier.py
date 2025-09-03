import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
from typing import Optional, Tuple
from models.sequence_models.base_model import BaseClassifier

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor: Output with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(BaseClassifier):
    """
    Transformer-based classifier for variable-length sequences.

    This classifier first projects the input features to an embedding dimension (d_model),
    adds sinusoidal positional encodings, processes the sequence using a transformer encoder
    (with batch_first=True), then applies mean pooling and a final linear layer to obtain class logits.
    """
    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_classes: int = 2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 early_stopping: bool = False,
                 early_stopping_patience: int = 10,
                 device: Optional[torch.device] = None) -> None:
        """
        Args:
            input_size (int): Number of input features.
            d_model (int, optional): Embedding dimension. Defaults to 128.
            nhead (int, optional): Number of attention heads. Defaults to 8.
            num_layers (int, optional): Number of transformer encoder layers. Defaults to 2.
            dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 512.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            num_classes (int, optional): Number of output classes. Defaults to 2.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            epochs (int, optional): Maximum training epochs. Defaults to 100.
            early_stopping (bool, optional): Enable early stopping if validation loss stops improving. Defaults to False.
            early_stopping_patience (int, optional): Number of epochs with no improvement before stopping. Defaults to 10.
            device (torch.device, optional): Device to run the model on. Defaults to CUDA if available.
        """
        super(TransformerClassifier, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Project input features to embedding dimension.
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Create TransformerEncoderLayer with batch_first=True.
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final classification layer.
        self.fc = nn.Linear(d_model, num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor: Logits of shape (batch, num_classes).
        """
        # Project and add positional encoding.
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_encoder(x)          # (batch, seq_len, d_model)
        # TransformerEncoder with batch_first=True accepts input as (batch, seq_len, d_model).
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # Mean pooling over time dimension.
        x = x.mean(dim=1)                # (batch, d_model)
        x = self.fc(x)                   # (batch, num_classes)
        return x

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            verbose: bool = False) -> None:
        """
        Train the transformer classifier.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader, optional): DataLoader for validation data. If None, training data is used.
            verbose (bool, optional): If True, displays training progress.
        """
        self.train()
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        epoch_iterator = tqdm(range(self.epochs), desc="Training", leave=True) if verbose else range(self.epochs)
        for epoch in epoch_iterator:
            total_train_loss = 0.0
            for sequences, lengths, labels in train_loader:
                # sequences: (batch, seq_len, input_size)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Evaluate on validation set (or training set if none provided)
            current_val_loader = val_loader if val_loader is not None else train_loader
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for sequences, lengths, labels in current_val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.forward(sequences)
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
                        epoch_iterator.set_description(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            if verbose and ((epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs):
                acc, auc, f1, f1_weighted = self.evaluate(current_val_loader, verbose=False)
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{self.epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} " +
                    f"Acc: {acc:.4f} AUC: {auc:.4f} F1: {f1:.4f} F1 (weighted): {f1_weighted:.4f}"
                )
            self.train()

    def predict(self, test_loader: torch.utils.data.DataLoader) -> Tuple[list, list]:
        """
        Generate predictions for the test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            Tuple[list, list]: A tuple of (true labels, predicted probabilities for the positive class).
        """
        self.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for sequences, lengths, labels in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(sequences)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return all_labels, all_probs

    def evaluate(self, loader: torch.utils.data.DataLoader, verbose: bool = False) -> Tuple[float, float, float]:
        """
        Evaluate model performance.

        Args:
            loader (DataLoader): DataLoader for evaluation.
            verbose (bool, optional): If True, prints evaluation metrics.

        Returns:
            Tuple[float, float, float]: (accuracy, AUC-ROC, F1 score)
        """
        y_true, y_probs = self.predict(loader)
        y_pred = [1 if p > 0.5 else 0 for p in y_probs]
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        if verbose:
            print(f"[TransformerClassifier] Evaluation -- Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f} F1 (weighted): {f1_weighted:.4f}")
        return acc, auc, f1, f1_weighted
