# Script used to train and predict a fixed forest model on multiple different data splits.

import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_builder import FeatureBuilder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up logging to file
log_file = f"seed_testing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log(message, print_to_console=False):
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    if print_to_console:
        print(message)

# Define the seed values to try
SEEDS = list(range(10, 101))

# Define the split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Initialize results storage
results = {
    'seeds': [],
    'valid_accuracies': [],
    'test_accuracies': [],
    'train_sizes': [],
    'val_sizes': [],
    'test_sizes': []
}

# Read the CSV file
data = pd.read_csv("./ML Challenge New/cleaned_data_combined.csv")
log(f"Loaded dataset with {len(data)} rows")

# Function to generate splits and save them to CSV files
def generate_splits(seed):
    ids = data['id'].unique()  # Get unique IDs

    # Split the IDs into training, validation, and test sets
    train_ids, temp_ids = train_test_split(ids, train_size=train_ratio, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_ratio/(val_ratio + test_ratio), random_state=seed)

    # Split the data based on the selected IDs
    train_data = data[data['id'].isin(train_ids)]
    val_data = data[data['id'].isin(val_ids)]
    test_data = data[data['id'].isin(test_ids)]

    # Save the subsets to new CSV files
    train_data.to_csv("train_data.csv", index=False)
    val_data.to_csv("val_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)

    return len(train_data), len(val_data), len(test_data)

# Function to generate train_stats_dict
def generate_train_stats_dict():
    """Generate and save training statistics dictionary"""
    train_df = FeatureBuilder("train_data.csv", normalize=False).build_features2()

    # Compute statistics for all columns (except 'id' and 'label')
    d = {}
    for col in train_df.columns:
        if col not in ['id', 'label']:
            d[col] = (train_df[col].mean(), train_df[col].std(), train_df[col].median())    
    
    # Write the dictionary to a Python file
    with open("train_stats_dict.py", "w") as f:
        f.write("TRAIN_STATS_DICT = {\n")
        for col, (mean, std, median) in d.items():
            f.write(f"    '{col}': ({mean}, {std}, {median}),\n")
        f.write("}\n")
    
    return d

# Function to extract accuracy from the output of pred.py
def extract_accuracy(output):
    """Extract accuracy percentage from pred.py output"""
    try:
        return float(output.split(":")[1].strip().replace("%", ""))
    except (IndexError, ValueError):
        log(f"Warning: Could not parse accuracy from output: {output}")
        return 0.0

# Function to run the entire process for a given seed
def run_process(seed):
    log(f"\n=== Running process with SEED = {seed} ===")
    
    # Generate splits and log sizes
    train_size, val_size, test_size = generate_splits(seed)
    log(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Generate lookup tables
    log("Generating lookup tables...")
    subprocess.run(["python3", "create_bayes_params.py"])
    
    # Generate train_stats_dict
    log("Generating training statistics...")
    generate_train_stats_dict()
    
    # Train the model
    log("Training the model...")
    subprocess.run(["python3", "extract_forest.py"])
    
    # Evaluate the model and capture output
    log("Evaluating the model...")
    result = subprocess.run(["python3", "pred.py"], capture_output=True, text=True)
    
    # Log full output for debugging
    log("\nModel output:")
    log(result.stdout)
    
    # Extract accuracies
    output_lines = result.stdout.splitlines()
    valid_accuracy = extract_accuracy(output_lines[-2])
    test_accuracy = extract_accuracy(output_lines[-1])
    
    # Store results
    results['seeds'].append(seed)
    results['valid_accuracies'].append(valid_accuracy)
    results['test_accuracies'].append(test_accuracy)
    results['train_sizes'].append(train_size)
    results['val_sizes'].append(val_size)
    results['test_sizes'].append(test_size)
    
    log(f"Validation Accuracy: {valid_accuracy}%")
    log(f"Test Accuracy: {test_accuracy}%")
    
    return valid_accuracy, test_accuracy

# Main execution
log("Starting seed testing experiment")
log(f"Testing {len(SEEDS)} seeds from {SEEDS[0]} to {SEEDS[-1]}")
log(f"Train/Val/Test ratios: {train_ratio}/{val_ratio}/{test_ratio}")

# Run the process for each seed
for seed in SEEDS:
    run_process(seed)

# Compute and log statistics
def compute_stats(accuracies, name):
    if not accuracies:
        return np.nan, np.nan, np.nan, 0
    
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    avg_acc = np.mean(accuracies)
    count_90_plus = sum(1 for acc in accuracies if acc >= 90)
    
    log(f"\n{name} Accuracy Statistics:")
    log(f"  Lowest: {min_acc:.2f}%")
    log(f"  Highest: {max_acc:.2f}%")
    log(f"  Average: {avg_acc:.2f}%")
    log(f"  90%+ Count: {count_90_plus}/{len(SEEDS)}")
    
    return min_acc, max_acc, avg_acc, count_90_plus

# Calculate statistics
val_stats = compute_stats(results['valid_accuracies'], "Validation")
test_stats = compute_stats(results['test_accuracies'], "Test")

# Save results to CSV for further analysis
results_df = pd.DataFrame({
    'seed': results['seeds'],
    'valid_accuracy': results['valid_accuracies'],
    'test_accuracy': results['test_accuracies'],
    'train_size': results['train_sizes'],
    'val_size': results['val_sizes'],
    'test_size': results['test_sizes']
})
results_csv = f"seed_testing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(results_csv, index=False)
log(f"\nFull results saved to {results_csv}")

# Final summary
log("\n=== FINAL SUMMARY ===")
log(f"Best Validation Accuracy: {val_stats[1]:.2f}%")
log(f"Best Test Accuracy: {test_stats[1]:.2f}%")
log(f"Average Validation Accuracy: {val_stats[2]:.2f}%")
log(f"Average Test Accuracy: {test_stats[2]:.2f}%")
log(f"Seeds with 90%+ Validation: {val_stats[3]}")
log(f"Seeds with 90%+ Test: {test_stats[3]}")
log("\nExperiment completed successfully!")
