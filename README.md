# 311-ML-Food-Prediction

For this project, a supervised ML model was developed to predict individual food preferences for Pizza, Shawarma, or Sushi. The model was trained using 1644 responses from an 8-question student survey conducted as part of our ML course. The survey included questions ranged from the food's preparation complexity to associated movie choices. A key objective was to ensure the model's effeciveness on new, unseen responses, including those from the teaching team. In a in-class wide competition, our model achieved 86% test accuracy on unseen data, placing it in the top 5 in the class. 

The full project report can be found in repots/ 

This project was developed in collaboration with Mohsin Muzzammil, Imad Syed and Mahmoud Zeidan


## Project Structure

```
311-ML-Food-Prediction/
├── data/
│   ├── raw/                        # Original, immutable data (e.g. student_survey_responses.csv)
│   └── raw_splits/                 # Generated train/val/test splits
│       ├── train_data.csv
│       ├── val_data.csv
│       └── test_data.csv
│
├── notebooks/                      # Jupyter notebooks for exploration and analysis
│   ├── explore_decision_trees.ipynb
│   ├── explore_extra_forest.ipynb
│   ├── explore_knn.ipynb
│   ├── explore_random_forest.ipynb
│   ├── explore_mlp.ipynb
│   ├── explore_logistic_reg.ipynb
│   └── visualize_features.ipynb
│
├── src/
│   ├── data/
│   │   ├── generate_data_splits.py         # Generates train/val/test splits
│   │   ├── create_bayes_params.py          # Generates Bayes parameter files
│   │   ├── extract_forest.py               # Trains and extracts the forest model
│   │   ├── feature_builder.py              # Feature engineering
│   │   ├── common.py                       # Shared utilities
│   │   ├── drinks_bayes_params.py          # Bayes params (generated)
│   │   ├── movies_bayes_params.py          # Bayes params (generated)
│   │   └── ingredients_bayes_params.py     # Bayes params (generated)
│   ├── models/
│   │   └── forest_functions.py             # Random forest model (generated)
│   ├── prediction/
│   │   └── pred.py                         # Prediction script
│   ├── evaluation/
│   │   └── evaluate_data_splits.py         # Evaluate best train/val/test splits
│
├── config/
│   ├── config.yaml
│   └── README.md
│
├── reports/
│   └── ML-Challenge-Final-Report.pdf
│
├── requirements.txt
├── setup.py
└── .gitignore
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow: Training and Prediction

### 1. Data Preparation
- Ensure raw data is located in `data/raw/student_survey_responses.csv`.
- Run the following script to generate train/val/test splits:
  ```bash
  python src/data/generate_data_splits.py
  ```
  This creates `train_data.csv`, `val_data.csv`, and `test_data.csv` in `data/raw_splits/`.

### 2. Generate Bayes Parameters
- Run:
  ```bash
  python src/data/create_bayes_params.py
  ```
  This generates `movies_bayes_params.py`, `drinks_bayes_params.py`, and `ingredients_bayes_params.py` in `src/data/`.

### 3. Train and Extract Forest Model
- Run:
  ```bash
  python src/data/extract_forest.py
  ```
  This generates the model file `forest_functions.py` in `src/models/`.

### 4. Make Predictions
- To predict on new (unseen) data, run:
  ```bash
  python src/prediction/pred.py --input <your_new_data.csv>
  ```
  This uses `feature_builder.py` and the trained model to output predicted food classes (pizza, sushi, shawarma).

## Notebooks
- Use the notebooks in `notebooks/` for exploratory data analysis and different ML model experimentations.

## Configuration
- The `config/` folder is reserved for configuration files (currently, the code does not require any additional configuration).

## License
This project is licensed under the terms of the license included in the LICENSE file.
