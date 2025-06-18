# Script to generate vocab, priors, and likelihoods for a text-based column.
import os
import pandas as pd
from common import SHORT_COLUMN_NAMES, clean_text

# Helper function to build vocabulary from a text column
def build_vocab(text_series):
    vocab = set()
    for text in text_series:
        words = text.split()
        vocab.update(words)
    return sorted(vocab)

def estimate_likelihoods(df, column_name, vocab, class_column, method="mle", alpha=None, beta=None):
    """
    Estimate the likelihoods for each word given each class using MLE, Laplace smoothing, or MAP estimation.

    Parameters:
        df (pd.DataFrame): The training data.
        column_name (str): The text column to process.
        vocab (list): List of words to use as features.
        class_column (str): The column containing the class labels.
        method (str): The estimation method. Options: "mle", "laplace", "map". Defaults to "mle".
        alpha (float): Alpha parameter for Laplace smoothing or MAP estimation.
        beta (float): Beta parameter for MAP estimation.

    Returns:
        likelihoods (dict): A nested dictionary mapping each class to a dictionary of word likelihoods.
    """
    likelihoods = {cls: {} for cls in df[class_column].unique()}
    vocab_size = len(vocab)

    for cls in likelihoods:
        # Filter documents for the current class
        class_docs = df[df[class_column] == cls][column_name]
        total_docs_in_class = len(class_docs)

        for word in vocab:
            # Count the number of documents in the class containing the word
            docs_with_word_in_class = sum(1 for doc in class_docs if word in doc.split())

            # Calculate likelihood based on the specified method
            if method == "mle":
                likelihoods[cls][word] = docs_with_word_in_class / total_docs_in_class
            elif method == "laplace":
                if alpha is None:
                    raise ValueError("Alpha must be provided for Laplace smoothing.")
                likelihoods[cls][word] = (docs_with_word_in_class + alpha) / (total_docs_in_class + alpha * vocab_size)
            elif method == "map":
                if alpha is None or beta is None:
                    raise ValueError("Alpha and beta must be provided for MAP estimation.")
                likelihoods[cls][word] = (docs_with_word_in_class + alpha - 1) / (total_docs_in_class + alpha + beta - 2)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'mle', 'laplace', or 'map'.")

    return likelihoods

def save_to_python_module(vocab, priors, likelihoods, filename, prefix):
    """
    Save vocab, priors, and likelihoods to a Python module file with unique variable names.

    Parameters:
        vocab (list): List of words in the vocabulary.
        priors (dict): Prior probabilities for each class.
        likelihoods (dict): Likelihoods for each word given each class.
        filename (str): Name of the Python module file to save.
        prefix (str): Prefix to add to variable names (e.g., "movies_", "drinks_").
    """
    with open(filename, "w", encoding="utf-8") as f:
        # Save vocab
        f.write(f"{prefix}vocab = [\n")
        for word in vocab:
            f.write(f'    "{word}",\n')
        f.write("]\n\n")

        # Save priors
        f.write(f"{prefix}priors = {{\n")
        for cls, prob in priors.items():
            f.write(f'    "{cls}": {prob},\n')
        f.write("}\n\n")

        # Save likelihoods
        f.write(f"{prefix}likelihoods = {{\n")
        for cls, word_probs in likelihoods.items():
            f.write(f'    "{cls}": {{\n')
            for word, prob in word_probs.items():
                f.write(f'        "{word}": {prob},\n')
            f.write("    },\n")
        f.write("}\n")

if __name__ == "__main__":
    # Load training data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_data_path = os.path.join(project_root, "data", "raw_splits", "train_data.csv")
    
    train_data = pd.read_csv(train_data_path, dtype=str, keep_default_na=False, encoding="utf-8")
    train_data.rename(columns={v: k for k, v in SHORT_COLUMN_NAMES.items()}, inplace=True)
    train_data = train_data.applymap(lambda x: x.lower())
    
    train_data["q2"] = train_data["q2"].apply(clean_text)
    train_data["q5"] = train_data["q5"].apply(clean_text)
    train_data["q6"] = train_data["q6"].apply(clean_text)

    # Build vocabularies
    ingredients_vocab = build_vocab(train_data["q2"])
    movies_vocab = build_vocab(train_data["q5"])
    drinks_vocab = build_vocab(train_data["q6"])

    # Estimate priors (same for all columns)
    priors = train_data["label"].value_counts(normalize=True).to_dict()

    # Estimate likelihoods 
    # ingredients_likelihoods = estimate_likelihoods(train_data, "q2", ingredients_vocab, "label", method="mle", alpha=None, beta=None)
    ingredients_likelihoods = estimate_likelihoods(train_data, "q2", ingredients_vocab, "label", method="laplace", alpha=1, beta=None)
    # ingredients_likelihoods = estimate_likelihoods(train_data, "q2", ingredients_vocab, "label", method="map", alpha=2, beta=2)

    # movies_likelihoods = estimate_likelihoods(train_data, "q5", movies_vocab, "label", method="mle", alpha=None, beta=None)
    movies_likelihoods = estimate_likelihoods(train_data, "q5", movies_vocab, "label", method="laplace", alpha=1, beta=None)
    # movies_likelihoods = estimate_likelihoods(train_data, "q5", movies_vocab, "label", method="map", alpha=2, beta=2)

    # drinks_likelihoods = estimate_likelihoods(train_data, "q6", drinks_vocab, "label", method="mle", alpha=None, beta=None)
    drinks_likelihoods = estimate_likelihoods(train_data, "q6", drinks_vocab, "label", method="laplace", alpha=1, beta=None)
    # drinks_likelihoods = estimate_likelihoods(train_data, "q6", drinks_vocab, "label", method="map", alpha=2, beta=2)

    # Save vocab, priors, and likelihoods to a Python module
    output_dir = os.path.join(project_root, "src", "data")
    save_to_python_module(ingredients_vocab, priors, ingredients_likelihoods, os.path.join(output_dir, "ingredients_bayes_params.py"), "ingredients_")
    save_to_python_module(movies_vocab, priors, movies_likelihoods, os.path.join(output_dir, "movies_bayes_params.py"), "movies_")
    save_to_python_module(drinks_vocab, priors, drinks_likelihoods, os.path.join(output_dir, "drinks_bayes_params.py"), "drinks_")

    print("Vocab, priors, and likelihoods for q2 saved to 'ingredients_bayes_params.py'")
    print("Vocab, priors, and likelihoods for q5 saved to 'movies_bayes_params.py'")
    print("Vocab, priors, and likelihoods for q6 saved to 'drinks_bayes_params.py'")
