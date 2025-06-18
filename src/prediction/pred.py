import sys
import os
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
sys.path.append(data_dir)
sys.path.append(models_dir)

from common import CLASS_NAMES
from feature_builder import FeatureBuilder
from forest_functions import predict_forest


def predict_all(filename):
    """
    Make predictions for the data in filename.
    Args:
        filename: Path to the CSV file containing the test data.
    Returns:
        predictions: List of predictions for each row in the test data.
    """
    # Load and preprocess the test data using DataPreprocessor class
    processed_data = FeatureBuilder(filename, normalize=False, exclude_columns=[]).build_features()

    # NOTE: When the data frame will not have the label column, but for testing we can pass it in
    # Extract true labels and drop unnecessary columns
    # labels = processed_data['label']
    
    processed_data.drop(['id', 'label'], inplace=True, axis=1, errors='ignore')  # Drop 'id' and 'label' columns
    probabilities = processed_data.apply(lambda row: predict_forest(**row.to_dict()), axis=1)
    predictions = [CLASS_NAMES[np.argmax(prob)] for prob in probabilities]

    # # Calculate accuracy
    # correct_predictions = np.sum(predictions == labels.apply(lambda x : x.capitalize()))  # Compare encoded labels
    # total_predictions = len(labels)
    # accuracy = correct_predictions / total_predictions
    # print(f"Manual Prediction Accuracy: {accuracy * 100:.2f}%")
    return predictions

# predictions = predict_all("val_data.csv")
# predictions = predict_all("test_data.csv")
# predictions = predict_all("transformed_dataset.csv")
# print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    args = parser.parse_args()

    predictions = predict_all(args.input)
    print(predictions)

