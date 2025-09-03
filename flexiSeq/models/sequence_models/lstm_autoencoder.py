import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from typing import Optional, Tuple
from models.sequence_models.base_model import BaseClassifier

class LSTMAutoencoder(BaseClassifier):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_classes: int = 2,
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 early_stopping: bool = False,
                 early_stopping_patience: int = 10,
                 device: Optional[torch.device] = None) -> None:
        """
        LSTM Autoencoder with a classification head.

        Args:
            input_size (int): Number of input features.
            hidden_size (int, optional): Number of hidden units in the encoder LSTM. Defaults to 256.
            num_classes (int, optional): Number of output classes. Defaults to 2.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            epochs (int, optional): Maximum number of training epochs. Defaults to 1000.
            early_stopping (bool, optional): If True, stops training if validation loss does not improve. Defaults to False.
            early_stopping_patience (int, optional): Number of epochs to wait with no improvement before stopping. Defaults to 10.
            device (torch.device, optional): Device to run the model on. Defaults to CUDA if available, else CPU.
        """
        super(LSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion_classification = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, seq_len, input_size).
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Classification output.
        """
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed_x)
        class_out = self.classifier(h_n[-1])
        return class_out

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            verbose: bool = False) -> None:
        """
        Train the LSTM Autoencoder with classification.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader, optional): DataLoader for the validation data. If None, training data is used.
            verbose (bool, optional): If True, displays a tqdm progress bar with training/validation metrics.
        """
        self.train()
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        epoch_iterator = tqdm(range(self.epochs), desc="Training", leave=True) if verbose else range(self.epochs)

        for epoch in epoch_iterator:
            total_train_loss = 0.0
            for sequences, lengths, labels in train_loader:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(sequences, lengths)
                loss = self.criterion_classification(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            current_val_loader = val_loader if val_loader is not None else train_loader
            self.eval()
            total_val_loss = 0.0
            for sequences, lengths, labels in current_val_loader:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(sequences, lengths)
                loss = self.criterion_classification(outputs, labels)
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
                    f"Epoch {epoch+1}/{self.epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} "
                    f"Acc: {acc:.4f} AUC: {auc:.4f} F1: {f1:.4f} F1 (weighted): {f1_weighted:.4f}"
                )
            self.train()

    def predict(self, test_loader: torch.utils.data.DataLoader) -> Tuple[list, list]:
        """
        Generate predictions for the test data.

        Args:
            test_loader (DataLoader): DataLoader for the test data.

        Returns:
            Tuple[list, list]: A tuple containing the true labels and predicted probabilities.
        """
        self.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for sequences, lengths, labels in test_loader:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(sequences, lengths)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return all_labels, all_probs

    def evaluate(self,
                 loader: torch.utils.data.DataLoader,
                 verbose: bool = False) -> Tuple[float, float, float]:
        """
        Evaluate the model performance.

        Args:
            loader (DataLoader): DataLoader for evaluation.
            verbose (bool, optional): If True, prints the evaluation metrics.

        Returns:
            Tuple[float, float, float]: Accuracy, AUC-ROC, and F1 score.
        """
        y_true, y_probs = self.predict(loader)
        y_pred = [1 if p > 0.5 else 0 for p in y_probs]
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        auc = roc_auc_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        if verbose:
            print(f"[LSTMAutoencoder] Evaluation -- Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        return accuracy, auc, f1, f1_weighted
