import pandas as pd
import numpy as np

from common import SHORT_COLUMN_NAMES, extract_and_aggregate_numbers, extract_ingredient_count, clean_text

# Import movies parameters
from movies_bayes_params import movies_vocab, movies_priors, movies_likelihoods
from drinks_bayes_params import drinks_vocab, drinks_priors, drinks_likelihoods
from ingredients_bayes_params import ingredients_vocab, ingredients_priors, ingredients_likelihoods

# from train_stats_dict import TRAIN_STATS_DICT  # REMOVE AFTER just for automated testing
# Global dictionary to store statistics about training data: (key, value) = (column_name, (mean, std, median))
# Seed 10 Training Data Stats
# TRAIN_STATS_DICT = {'q1_processed': (3.195822454308094, 1.027894887020875, 3.0), 'q2_ingredient_count': (5.991429818000499, 2.8499547594868297, 5.0), 'q2_pizza_prob': (0.2717567312818412, 0.250044877171841, 0.18364402708489455), 'q2_shawarma_prob': (0.34015141204128946, 0.17126375047245213, 0.37268115518607536), 'q2_sushi_prob': (0.3880918566768693, 0.19648928762678364, 0.44367481772903006), 'q3_week_day_lunch': (0.7458659704090513, 0.43556289760440986, 1.0), 'q3_week_day_dinner': (0.6431679721496953, 0.4792732489888412, 1.0), 'q3_weekend_lunch': (0.6814621409921671, 0.4661121947037105, 1.0), 'q3_weekend_dinner': (0.6623150565709313, 0.47312645541458775, 1.0), 'q3_at_a_party': (0.49869451697127937, 0.5002160176007315, 0.0), 'q3_late_night_snack': (0.3907745865970409, 0.48813644296386555, 0.0), 'q4_processed': (11.40549970862471, 7.371762707716392, 10.0), 'q5_pizza_prob': (0.3193449914344474, 0.3577693473289001, 0.1489761731095788), 'q5_shawarma_prob': (0.3525601412574318, 0.3681981854924296, 0.22298621870352328), 'q5_sushi_prob': (0.328094867308121, 0.3703399472149427, 0.15539611026522632), 'q6_pizza_prob': (0.34525547236705223, 0.2876971401902003, 0.258390591477966), 'q6_shawarma_prob': (0.3210725315033917, 0.19039178576197371, 0.3378100131671813), 'q6_sushi_prob': (0.33367199612955617, 0.32935334564148744, 0.17388699447384146), 'q7_parents': (0.36640557006092256, 0.48203189909584915, 0.0), 'q7_siblings': (0.3463881636205396, 0.47602585983599716, 0.0), 'q7_friends': (0.7519582245430809, 0.43206425955602396, 1.0), 'q7_teachers': (0.13141862489120976, 0.3380047363251687, 0.0), 'q7_strangers': (0.2750217580504787, 0.44671967807622964, 0.0), 'q8_processed': (0.9982593559617058, 1.0948073982651472, 1.0)}
# Full Training Data Stats
TRAIN_STATS_DICT = {'q1_processed': (3.1745742092457423, 1.021049667539615, 3.0), 'q2_ingredient_count': (6.045711093700125, 2.8880812079241864, 5.0), 'q2_pizza_prob': (0.2681890937996436, 0.23787194601993517, 0.1906984388802559), 'q2_shawarma_prob': (0.3494519141137362, 0.17066257402297233, 0.38011234589704523), 'q2_sushi_prob': (0.3823589920866203, 0.19020836606838099, 0.4291892152226989), 'q3_week_day_lunch': (0.7475669099756691, 0.43454054176165385, 1.0), 'q3_week_day_dinner': (0.6435523114355232, 0.47909534760896794, 1.0), 'q3_weekend_lunch': (0.6763990267639902, 0.46799209943758335, 1.0), 'q3_weekend_dinner': (0.6587591240875912, 0.47427034557272985, 1.0), 'q3_at_a_party': (0.5060827250608273, 0.5001151253641613, 1.0), 'q3_late_night_snack': (0.3935523114355231, 0.48868615049234715, 0.0), 'q4_processed': (11.386027676027677, 7.255101892495278, 10.0), 'q5_pizza_prob': (0.316391467919315, 0.36421998451180976, 0.14605733686324412), 'q5_shawarma_prob': (0.3576646128129941, 0.37023125454634087, 0.21725601299573502), 'q5_sushi_prob': (0.3259439192676909, 0.3723636414615032, 0.14943500862947182), 'q6_pizza_prob': (0.3469046998692752, 0.29604798423018686, 0.23286692184159644), 'q6_shawarma_prob': (0.317647845679107, 0.19350999153197593, 0.31081558936536774), 'q6_sushi_prob': (0.33544745445161783, 0.3409014949164525, 0.11939974055285463), 'q7_parents': (0.35766423357664234, 0.47945840226139225, 0.0), 'q7_siblings': (0.34367396593673966, 0.4750783702179037, 0.0), 'q7_friends': (0.7627737226277372, 0.4255115798423416, 1.0), 'q7_teachers': (0.12895377128953772, 0.33525074462123405, 0.0), 'q7_strangers': (0.2639902676399027, 0.4409281859347663, 0.0), 'q8_processed': (0.9854014598540146, 1.0865328539666588, 1.0)}

class FeatureBuilder:
    def __init__(self, file_path,  normalize=False, exclude_columns=None):
        """
        Initialize the FeatureBuilder.
        
        Parameters:
        - file_path (str): Path to the CSV data file.
        - normalize (bool): Whether to normalize features.
        - exclude_columns (list): Specific columns to drop.
        """
        self.df = pd.read_csv(file_path, dtype=str, keep_default_na=False)  # Load the data as strings
        self.df.rename(columns={v: k for k, v in SHORT_COLUMN_NAMES.items()}, inplace=True)  # Rename columns
        self.normalize = normalize
        self.exclude_columns = exclude_columns if exclude_columns else []

    def build_features(self):
        # Step 1: Drop excluded columns and lowercase everything
        self.df.drop(self.exclude_columns, axis=1, inplace=True)
        self.df = self.df.applymap(lambda x: x.lower())

        # Step 2: Process Q1–Q8, creating new columns
        self.build_column_features()

        # Step 3: Handle missing values
        self.handle_missing_values()

        # Step 4: Normalize features if enabled
        if self.normalize:
            self.normalize_features()

        self.build_interaction_features()

        # Step 5: Drop original columns
        for col_name in SHORT_COLUMN_NAMES:
            if col_name == 'id' or col_name == 'label':  # Keep 'id' and 'label' columns
                continue
            elif col_name in self.df.columns:
                self.df.drop(col_name, axis=1, inplace=True)

        return self.df

    def build_features2(self):
        """
        Used for building training data set statistics.
        """
        # Step 1: Drop excluded columns and lowercase everything
        self.df.drop(self.exclude_columns, axis=1, inplace=True)
        self.df = self.df.applymap(lambda x: x.lower())

        # Step 2: Process Q1–Q8, creating new columns
        self.build_column_features()
        # self.build_interaction_features()

        # Step 4: Drop original columns
        for col_name in SHORT_COLUMN_NAMES:
            if col_name == 'id' or col_name == 'label':  # Keep 'id' and 'label' columns
                continue
            elif col_name in self.df.columns:
                self.df.drop(col_name, axis=1, inplace=True)

        # code to check if anything is getting dropped
        # rows_with_missing = self.df[self.df.isnull().any(axis=1)]
        # # Print the rows being dropped
        # if not rows_with_missing.empty:
        #     print(f"Dropping {len(rows_with_missing)} rows with missing values:")
        #     print(rows_with_missing)
        # else:
        #     print("No rows with missing values to drop.")

        # Step 6: Drop rows with missing values
        # self.df.dropna(inplace=True)

        return self.df

    def build_column_features(self):
        building_rules = {
            "q1": self.q1_features,
            "q2": self.q2_features,
            "q3": self.q3_features,
            "q4": self.q4_features,
            "q5": self.q5_features,
            "q6": self.q6_features,
            "q7": self.q7_features,
            "q8": self.q8_features,
        }
        for col, func in building_rules.items():
            if col in self.df.columns:
                func()

    def q1_features(self):
        self.df["q1_processed"] = self.df["q1"].apply(lambda x: extract_and_aggregate_numbers(x, aggregation=np.mean)).astype(float)

    def q2_features(self):
        self.df["q2_ingredient_count"] = self.df["q2"].apply(lambda x: extract_ingredient_count(x, aggregation=np.mean)).astype(float)
        self.generate_bayes_features(column_name="q2", vocab=ingredients_vocab, priors=ingredients_priors, likelihoods=ingredients_likelihoods)
        # self.create_bow_features(column_name="q2", vocab=ingredients_vocab)

    def q3_features(self):
        options = ['week day lunch', 'week day dinner', 'weekend lunch', 'weekend dinner', 'at a party', 'late night snack']
        self.one_hot_encode_column(column_name="q3", all_options=options)

    def q4_features(self):
        self.df["q4_processed"] = self.df["q4"].apply(lambda x: extract_and_aggregate_numbers(x, aggregation=np.mean)).astype(float)

    def q5_features(self):
        self.generate_bayes_features(column_name="q5", vocab=movies_vocab, priors=movies_priors, likelihoods=movies_likelihoods)
        # self.create_bow_features(column_name="q5", vocab=movies_vocab)

    def q6_features(self):
        self.generate_bayes_features(column_name="q6", vocab=drinks_vocab, priors=drinks_priors, likelihoods=drinks_likelihoods)
        # self.create_bow_features(column_name="q6", vocab=drinks_vocab)

    def q7_features(self):
        options = ['parents', 'siblings', 'friends', 'teachers', 'strangers']
        self.one_hot_encode_column(column_name="q7", all_options=options)

    def q8_features(self):
        option_mapping = {
            "none": 0,
            "a little (mild)": 1,
            "a moderate amount (medium)": 2,
            "a lot (hot)": 3,
            "i will have some of this food item with my hot sauce": 4
        }

        # Strip whitespace before mapping and note that responses not in option_mapping map to 0 (e.g., empty responses map to 0)
        self.df["q8_processed"] = self.df["q8"].map(lambda x: option_mapping.get(x.strip(), 0)).astype(int)

    def one_hot_encode_column(self, column_name, all_options):
        """
        One-hot encode a column based on a predefined set of possible options.
        
        Parameters:
            column_name (str): The name of the column to encode.
            all_options (list): A list of all possible options (ensures consistency across datasets).
        """
        # Convert column to lists of selected options
        temp = self.df[column_name].astype(str).str.split(",")

        # Apply one-hot encoding using the predefined options
        for option in all_options:
            new_col_name = f"{column_name}_{option.replace(' ', '_')}"
            self.df[new_col_name] = temp.apply(lambda x: 1 if option in [s.strip() for s in x] else 0)

    def create_bow_features(self, column_name, vocab, binary=True):
        """
        Create bag-of-words features for a given column and vocabulary.

        Parameters:
            column_name (str): Name of the column to process.
            vocab (list): List of words to use as features.
            binary (bool): If True, create binary features (1 if word is present, 0 otherwise).
                        If False, create count-based features.
        """

        # Clean the column and split into words
        temp = self.df[column_name].apply(clean_text).apply(lambda x: x.split())

        # Create a dictionary to store the new features
        features = {}

        # Create bag-of-words features
        for word in vocab:
            if binary:
                # Binary features (1 if word is present, 0 otherwise)
                features[f"{column_name}_{word}"] = temp.apply(lambda words: 1 if word in words else 0)
            else:
                # Count-based features (number of occurrences of the word)
                features[f"{column_name}_{word}"] = temp.apply(lambda words: words.count(word))

        # Convert the dictionary to a DataFrame
        bow_df = pd.DataFrame(features)

        # Concatenate the new features with the original DataFrame
        self.df = pd.concat([self.df, bow_df], axis=1)

    def generate_bayes_features(self, column_name, vocab, priors, likelihoods):
        """
        Generate Bayesian features (probabilities of each class) for a given column.

        Parameters:
            column_name (str): Name of the column to process.
            vocab (list): List of words to use as features.
            priors (dict): Prior probabilities for each class.
            likelihoods (dict): Likelihoods for each word given each class.
        """
        # Clean the column and split into words
        temp = self.df[column_name].apply(clean_text).apply(lambda x: x.split())

        # Initialize a DataFrame to store the log probabilities
        log_probs = pd.DataFrame(index=self.df.index)

        # Compute log probabilities for each class
        for cls in priors:
            # Initialize log probabilities for each document
            log_probs[cls] = np.log(priors[cls])

            # Compute log likelihoods for each word in the vocabulary
            for word in vocab:
                # Create a mask for documents containing the word
                word_mask = temp.apply(lambda words: word in words)
                
                # Add the log likelihood for the word if present
                log_probs.loc[word_mask, cls] += np.log(likelihoods[cls][word])
                
                # Add the log likelihood for the word if absent
                log_probs.loc[~word_mask, cls] += np.log(1 - likelihoods[cls][word])

        # Compute the log of the denominator (marginal likelihood)
        log_denominator = np.logaddexp.reduce(log_probs.values, axis=1)

        # Normalize the log probabilities by subtracting the log denominator
        log_probs = log_probs.sub(log_denominator, axis=0)

        # Convert log probabilities back to probabilities
        class_probs = np.exp(log_probs)

        # Add the normalized probabilities to the original DataFrame
        self.df = pd.concat([self.df, class_probs.add_prefix(f"{column_name}_").add_suffix("_prob")], axis=1)

    def handle_missing_values(self):
        # self.df['value_missing'] = self.df.isnull().any(axis=1).astype(int)  # Create a single missing indicator column
        for col, (mean, std, median) in TRAIN_STATS_DICT.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(median)

    def normalize_features(self):
        """
        Normalize features using precomputed training data set means and standard deviations.
        """
        for col, (mean, std, median) in TRAIN_STATS_DICT.items():
            if col in self.df.columns:
                # Standardization (Z-score normalization)
                self.df[col] = ((self.df[col] - mean) / std) if std > 0.0 else 0.0

    def build_interaction_features(self):
        eps = 1e-8
        self.df["q1_times_q2"] = self.df["q1_processed"] * self.df["q2_ingredient_count"]
        self.df["q1_times_q4"] = self.df["q1_processed"] * self.df["q4_processed"]
        self.df["q4_divided_by_q1"] = self.df["q4_processed"] / (self.df["q1_processed"] + eps)
        self.df["q2_times_q4"] = self.df["q2_ingredient_count"] * self.df["q4_processed"]
        self.df["q2_divided_by_q4"] = self.df["q2_ingredient_count"] / (self.df["q4_processed"] + eps)
        self.df["q4_divided_by_q2"] = self.df["q4_processed"] / (self.df["q2_ingredient_count"] + eps)
        self.df["q8_divided_by_q2"] = self.df["q8_processed"] / (self.df["q2_ingredient_count"] + eps)
        self.df["q8_divided_by_q4"] = self.df["q8_processed"] / (self.df["q4_processed"] + eps)

if __name__ == "__main__":
    x=1
    # # Example usage
    # # exclude_columns must be from ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']
    # features_df = FeatureBuilder("./ML Challenge New/cleaned_data_combined.csv", normalize=False, exclude_columns=[]).build_features()
    # print(features_df)

    # # IMPORTANT: Compute statistics based on training data set.
    # # We use build_features2() because it disables handle_missing_values() (otherwise circular logic).
    # train_df = FeatureBuilder("train_data.csv", normalize=False).build_features2()

    # # Compute means, standard deviations, and medians for all columns (except 'id' and 'label')
    # d = {}
    # for col in train_df.columns:
    #     if col not in ['id', 'label']:
    #         d[col] = (train_df[col].mean(), train_df[col].std(), train_df[col].median())

    # # Copy these computed statistics at the top of the file
    # print("TRAIN_STATS_DICT =", d)
