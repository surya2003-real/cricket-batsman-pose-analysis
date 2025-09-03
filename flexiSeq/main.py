from cross_validation import CrossValidator
from models.sequence_models.lstm_classifier import LSTMClassifier
from models.sequence_models.lstm_autoencoder import LSTMAutoencoder
from models.sequence_models.cnn1d import CNN1DClassifier 
from models.sequence_models.tcn_classifier import TCNClassifier
from models.sequence_models.transformer_classifier import TransformerClassifier
from datasets.smoothed_variable_length_dataset import SmoothedVariableLengthDataset
if __name__ == '__main__':
    # Define dataset paths.
    dataset_paths = {
        'Train': "../Train",
        'Test_divyansh': "../Test_divyansh",
        'Test_krrish': "../Test_krrish",
    }

    feature_names = ['y_18', 'y_20', 'y_22', 'y_16', 'y_17', 'y_19', 'y_27', 'y_15', 'y_13', 'y_31']
    model_paramsCNN1D = {
        'num_filters': 64,
        'kernel_size': 3,
        'num_classes': 2,
        'learning_rate': 0.001,
        'epochs': 1000,
        'early_stopping': False,
        'early_stopping_patience': 100  # Number of epochs to wait for improvement
    }
    # tcn_model_params = {
    #     'num_channels': [128, 128],           # Number of filters in each TCN layer
    #     'kernel_size': 2,                     # Convolution kernel size
    #     'dropout': 0.2,                       # Dropout rate
    #     'num_classes': 2,                     # Number of classes for classification
    #     'learning_rate': 0.001,               # Learning rate for the optimizer
    #     'epochs': 1000,                        # Maximum training epochs
    #     'early_stopping': False,               # Enable early stopping
    #     'early_stopping_patience': 5          # Number of epochs with no improvement before stopping
    # }
    # model_params = {
    #     'hidden_size': 256,
    #     'num_classes': 2,
    #     'learning_rate': 0.001,
    #     'epochs': 1000,
    #     'early_stopping': False,
    #     'early_stopping_patience': 100  # Number of epochs to wait for improvement
    # }
    # transformer_model_params = {
    #     'd_model': 128,                    # Embedding dimension.
    #     'nhead': 8,                        # Number of attention heads.
    #     'num_layers': 2,                   # Number of transformer encoder layers.
    #     'dim_feedforward': 512,            # Dimension of feedforward network.
    #     'dropout': 0.1,                    # Dropout rate.
    #     'num_classes': 2,                  # Number of output classes.
    #     'learning_rate': 0.001,            # Learning rate.
    #     'epochs': 1000,                     # Maximum epochs.
    #     'early_stopping': False,            # Enable early stopping.
    #     'early_stopping_patience': 100       # Early stopping patience.
    # }
    # Initialize CrossValidator with verbose output enabled.
    cv = CrossValidator(dataset_paths, feature_names, 
                        CNN1DClassifier, 
                        model_paramsCNN1D, 
                        batch_size=256, verbose=True, val_split=0,
                        dataset_generator=SmoothedVariableLengthDataset,
                        smoothing_type='kalman')
    results = cv.cross_validate(folds=5)

    print("\nFinal Results:")
    for i, (acc, auc, f1, f1_weighted) in enumerate(results):
        print(f"Fold {i+1}: Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, F1 (weighted) = {f1_weighted:.4f}")
