"""
train_local_classifier.py

Train a local Logistic Regression classifier on GPT-labeled job 
description sentences in gpt_classified_job_descriptions.csv 
and use it to classify the remaining 19k+ postings without calling GPT.

Usage:
    python train_local_classifier.py --gpt_classified_job_descriptions.csv \
                                     --new_data all_job_postings.csv \
                                     --output classified_20000_descriptions.csv
"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the labeled data
def load_labeled_data(path):
    """Load GPT-labeled CSV file and clean it."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    print(df.head(3))  # preview a few rows

    print("Unique categories:", df["category"].unique())
    return df

# # Testing if loading the data works
# if __name__ == "__main__":
#     labeled_df = load_labeled_data("../data/gpt_classified_job_descriptions.csv")
