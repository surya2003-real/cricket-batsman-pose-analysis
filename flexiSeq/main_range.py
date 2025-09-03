from models.range_models.random_forest_range_classifier import RandomForestRangeClassifier
from cross_validation_range import RangeCrossValidator
# Example usage:
if __name__ == '__main__':
    # Define a dictionary mapping dataset names to folder paths.
    dataset_paths = {
        "Train": "../Train",
        "Test_divyansh": "../Test_divyansh",
        "Test_krrish": "../Test_krrish",
        "Test_suryansh_2": "../Test_suryansh_2"
    }
    selected_features = ['y_18', 'y_20', 'y_22', 'y_16', 'y_17', 'y_19', 'y_27', 'y_15', 'y_13', 'y_31']
    model_params = {'random_state': -1}  # Additional RandomForest parameters can be added here

    cv = RangeCrossValidator(dataset_paths, selected_features, RandomForestRangeClassifier,
                             model_params=model_params, verbose=True)
    results = cv.cross_validate(folds=5)
    print("\nFinal Average Metrics per fold:")
    for i, (acc, auc, f1, f1_weighted) in enumerate(results):
        print(f"Fold {i+1}: Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, F1 (weighted) = {f1_weighted:.4f}")