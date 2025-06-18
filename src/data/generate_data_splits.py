import os
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Generate train, validation, and test splits from the raw data.

    Parameters:
        train_ratio (float): Proportion of data to allocate to training set.
        val_ratio (float): Proportion of data to allocate to validation set.
        test_ratio (float): Proportion of data to allocate to test set.
    """

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(project_root, "data", "raw", "student_survey_responses.csv")
    data = pd.read_csv(input_file)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, "data", "raw_splits")
    os.makedirs(output_dir, exist_ok=True)

    ids = data['id'].unique()  # Get unique IDs

    # Split the IDs into training, validation, and test sets
    train_ids, temp_ids = train_test_split(ids, train_size=train_ratio, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_ratio/(val_ratio + test_ratio), random_state=seed)

    # Split the data based on the selected IDs
    train_data = data[data['id'].isin(train_ids)]
    val_data = data[data['id'].isin(val_ids)]
    test_data = data[data['id'].isin(test_ids)]

    # Save the subsets to new CSV files in the processed directory
    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    return len(train_data), len(val_data), len(test_data)


if __name__ == "__main__":
    generate_splits()