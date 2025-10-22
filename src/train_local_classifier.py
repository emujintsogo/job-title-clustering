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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# Load the labeled data
def load_labeled_data(path):
    """Load GPT-labeled CSV file and clean it."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    print(df.head(3))  # preview a few rows

    # Drop missing or empty sentences so the TF-IDF vectorizer won't throw
    # errors when null values are encountered
    df = df.dropna(subset=["sentence", "category"])
    df = df[df["sentence"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)

    print(f"After cleaning: {len(df)} rows remain.")

    print("Unique categories:", df["category"].unique())
    return df


# Split data, create TF-IDF vectors
def prepare_data(df):
    """Split data and create TF-IDF vectors."""
    X = df["sentence"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF vectorizer built.")
    print(f"Train shape: {X_train_tfidf.shape}, Test shape: {X_test_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


