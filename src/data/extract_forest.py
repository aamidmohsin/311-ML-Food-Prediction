import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import _tree
from feature_builder import FeatureBuilder
from common import CLASS_NAMES

# Code inspired from: https://mljar.com/blog/extract-rules-decision-tree/

def tree_to_code(tree, feature_names, tree_index, class_names):
    """
    Converts a single Decision Tree into Python code.
    Args:
        tree: The Decision Tree model.
        feature_names: List of feature names.
        tree_index: Index of the tree in the forest.
        class_names: List of class names.
    Returns:
        code: Python code for the tree's predict function.
    """
    tree_ = tree.tree_

    # Shorten function name: t0, t1, t2, ...
    code = f"def t{tree_index}({','.join(f'f{i+1}' for i in range(len(feature_names)))}):\n"

    def recurse(node, depth):
        nonlocal code
        indent = " " * (depth * 2)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = f"f{tree_.feature[node] + 1}"
            threshold = np.round(tree_.threshold[node], 2)
            code += f"{indent}if {name}<={threshold}:\n"
            recurse(tree_.children_left[node], depth + 1)
            code += f"{indent}else:\n"
            recurse(tree_.children_right[node], depth + 1)
        else:
            class_probs = tree_.value[node][0]
            class_probs_normalized = (class_probs / np.sum(class_probs)).tolist()
            code += f"{indent}return {class_probs_normalized}\n"

    recurse(0, 1)
    return code

def save_forest_functions(forest, feature_names, class_names, output_file):
    """
    Saves a Random Forest as executable Python functions.
    Args:
        forest: Trained Forest model.
        feature_names: List of feature names.
        class_names: List of class names.
        output_file: Output Python file path.
    """
    with open(output_file, "w") as f:
        f.write("import numpy as np\n\n")
        for i, tree in enumerate(forest.estimators_):
            f.write(tree_to_code(tree, feature_names, i, class_names) + '\n')

        f.write("def predict_forest(**kw):\n")
        f.write("  fmap = {" + ",".join(f"'{name}':'f{i+1}'" for i, name in enumerate(feature_names)) + "}\n")
        f.write("  kw = {fmap[k]: v for k, v in kw.items() if k in fmap}\n")
        f.write("  ts = [" + ",".join(f"t{i}" for i in range(len(forest.estimators_))) + "]\n")
        f.write(f"  probs = np.zeros({len(class_names)})\n")
        f.write("  for t in ts: probs += np.array(t(**kw))\n")
        f.write("  return (probs/len(ts)).tolist()\n")

def tree_to_code2(tree, feature_names, tree_index, class_names):
    """
    Converts a single Decision Tree into Python code.
    Args:
        tree: The Decision Tree model.
        feature_names: List of feature names.
        tree_index: Index of the tree in the forest.
        class_names: List of class names.
    Returns:
        code: Python code for the tree's predict function.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    code = f"def predict_tree_{tree_index}({', '.join(feature_names)}):\n"

    def recurse(node, depth):
        nonlocal code
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            code += f"{indent}if {name} <= {np.round(threshold, 2)}:\n"
            recurse(tree_.children_left[node], depth + 1)
            # code += f"{indent}else:  # if {name} > {np.round(threshold, 2)}\n"
            code += f"{indent}else:\n"
            recurse(tree_.children_right[node], depth + 1)
        else:
            # Return the class with the highest probability
            # class_probs = tree_.value[node][0]
            # predicted_class = class_names[np.argmax(class_probs)]
            # code += f"{indent}return '{predicted_class}'\n"  # Return class label

            class_probs = tree_.value[node][0]
            total_samples = np.sum(class_probs)
            class_probs_normalized = class_probs / total_samples
            code += f"{indent}return {class_probs_normalized.tolist()}\n"

    recurse(0, 1)
    return code

def save_forest_functions2(rf, feature_names, class_names, output_file):
    """
    Saves the Forest's trees as executable functions in a Python file.
    Args:
        rf: The trained Forest model.
        feature_names: List of feature names.
        class_names: List of class names.
        output_file: Path to the output Python file.
    """
    n_estimators = len(rf.estimators_)  # Number of trees in the forest

    with open(output_file, "w") as f:
        f.write("import numpy as np\n\n")
        for i, tree in enumerate(rf.estimators_):
            f.write(tree_to_code2(tree, feature_names, i, class_names) + "\n\n")
        
        # Add a function to combine predictions from all trees (majority voting)
        f.write("def predict_forest(**kwargs):\n")
        f.write("    # List of tree prediction functions\n")
        f.write("    tree_predictors = [\n")
        for i in range(n_estimators):
            f.write(f"        predict_tree_{i},\n")
        f.write("    ]\n")
        f.write("    predictions = []\n")

        f.write("    # Initialize a list to store probabilities from all trees\n")
        f.write(f"    all_probabilities = np.zeros({len(class_names)})\n")
        f.write("    for tree in tree_predictors:\n")
        f.write("        # Get probabilities from the current tree\n")
        f.write("        probabilities = tree(**kwargs)\n")
        f.write("        all_probabilities += np.array(probabilities)\n")
        f.write("    # Average probabilities across all trees\n")
        f.write("    avg_probabilities = all_probabilities / len(tree_predictors)\n")
        f.write("    return avg_probabilities.tolist()\n")

        # f.write("    for tree in tree_predictors:\n")
        # f.write("        predictions.append(tree(**kwargs))\n")
        # f.write("    # Perform majority voting manually\n")
        # f.write("    unique_classes, counts = np.unique(predictions, return_counts=True)\n")
        # f.write("    return unique_classes[np.argmax(counts)]\n")

# Load the training data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_data_path = os.path.join(project_root, "data", "raw_splits", "train_data.csv")

train_data = FeatureBuilder(train_data_path, normalize=False, exclude_columns=[]).build_features()

# Prepare features and labels
y_train = train_data['label']
X_train = train_data.drop(columns=['id', 'label'])

# Train Forest
# best_params = {'bootstrap': True, 'max_depth': None, 'max_features': 1, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 180, 'random_state': 42}
# forest = RandomForestClassifier(**best_params)

best_params = {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150, 'random_state': 42}
forest = ExtraTreesClassifier(**best_params)

forest.fit(X_train, y_train)

# Save the Forest as executable functions
output_dir = os.path.join(project_root, "src", "models")
save_forest_functions(forest, X_train.columns.tolist(), CLASS_NAMES, os.path.join(output_dir, "forest_functions.py"))
print("Forest functions saved to 'forest_functions.py'")
