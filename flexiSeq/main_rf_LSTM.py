# Import model classes
from models.sequence_models.lstm_classifier import LSTMClassifier
from models.range_models.random_forest_range_classifier import RandomForestRangeClassifier
from cross_validation_rf_LSTM import EnsembleCrossValidator
# Example usage:
if __name__ == '__main__':
    dataset_paths = {
        "Train": "../Train",
        "Test_divyansh": "../Test_divyansh",
        "Test_krrish": "../Test_krrish",
        "Test_suryansh_2": "../Test_suryansh_2"
    }
    feature_names = ['x_18', 'x_20', 'x_22', 'x_16', 'x_17', 'x_19', 'x_27', 'x_15', 'x_13', 'x_31']
    lstm_model_params = {
        'hidden_size': 256,
        'num_classes': 2,
        'learning_rate': 0.001,
        'epochs': 1000,  # For demonstration, using fewer epochs
        'early_stopping': False,
        'early_stopping_patience': 5
    }
    rf_model_params = {
        'random_state': -1
    }
    ensemble_cv = EnsembleCrossValidator(dataset_paths, feature_names,
                                           lstm_model_class=LSTMClassifier,
                                           lstm_model_params=lstm_model_params,
                                           rf_model_class=RandomForestRangeClassifier,
                                           rf_model_params=rf_model_params,
                                           batch_size=256,
                                           verbose=True,
                                           val_split=0)
    results = ensemble_cv.cross_validate(folds=5)
    print("\nFinal Average Ensemble Metrics per fold:")
    for i, (acc, auc, f1, f1_weighted) in enumerate(results):
        print(f"Fold {i+1}: Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, F1 (weighted) = {f1_weighted:.4f}")