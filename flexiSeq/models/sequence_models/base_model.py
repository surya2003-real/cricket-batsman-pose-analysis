import torch
import torch.nn as nn

class BaseClassifier(nn.Module):
    """
    Abstract base class for classifiers implementing a scikit-learn-like interface.
    Subclasses should implement fit, predict, and evaluate methods.
    
    The methods can optionally print progress when verbose=True.
    """
    def __init__(self):
        super(BaseClassifier, self).__init__()

    def fit(self, train_loader, verbose=False):
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(self, test_loader):
        """
        Should return a tuple (y_true, y_probs) where y_probs are the raw probabilities.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

    def evaluate(self, test_loader, verbose=False):
        """
        Evaluates the model on test data using metrics such as accuracy, AUC, and F1.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")
