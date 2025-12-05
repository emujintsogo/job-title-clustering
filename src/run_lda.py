import string
import pandas as pd
import os
import pickle
import nltk
from nltk import word_tokenize
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.lda_model
from config import BASE_DIR

vectorizer_ouput_path = os.path.join(BASE_DIR, "models", "count_vectorizer.pkl")
lda_model_output_path = os.path.join(BASE_DIR, "models", "lda_model.pkl")
vis_output_path = os.path.join(BASE_DIR, "models", "lda_vis.html")


class LemmaTokenizer:
    """
    Tokenizer to be passed to CountVectorizer
    """

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stopwords = self.ensure_nltk()
        self.punct = set(string.punctuation)

    def ensure_nltk(self):
        """
        If required NLTK data is unavailable, download it.
        """
        required = {
            "punkt": "tokenizers/punkt",
            "wordnet": "corpora/wordnet",
            "stopwords": "corpora/stopwords",
            "punkt_tab": "tokenizers/punkt_tab",
        }

        for pkg, path in required.items():
            try:
                find(path)
            except LookupError:
                print(f"[NLTK] Missing '{pkg}', downloading...")
                nltk.download(pkg)

        return set(stopwords.words("english"))

    def __call__(self, doc):
        tokens = word_tokenize(doc.lower())
        clean = []
        for t in tokens:
            if t in self.punct or t in self.stopwords or t.isnumeric():
                continue
            clean.append(self.wnl.lemmatize(t))
        return clean


def load_or_fit_vectorizer(docs):
    """
    Load an existing CountVectorizer if it exists.
    Otherwise fit a new one on docs and save it.
    """
    if os.path.exists(vectorizer_ouput_path):
        print(
            f"[Vectorizer] Loading existing vectorizer from {vectorizer_ouput_path}..."
        )
        with open(vectorizer_ouput_path, "rb") as f:
            return pickle.load(f)

    print("Fitting new CountVectorizer...")

    vect = CountVectorizer(
        tokenizer=LemmaTokenizer(),
        min_df=20,  # Remove extremely rare words
        max_df=0.8,  # Remove extremely common words
    )

    vect.fit(docs)

    with open(vectorizer_ouput_path, "wb") as f:
        pickle.dump(vect, f)

    print(f"Saved new vectorizer to {vectorizer_ouput_path}")
    return vect


def run_LDA(n_components, random_state, X):
    """
    Train an LDA model and return it.
    """
    print(f"Training LDA with {n_components} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_components,
        random_state=random_state,
        learning_method="online",
        n_jobs=-1,
    )
    lda.fit(X)
    print("Training complete.")

    save_path = get_next_available_path(lda_model_output_path)
    with open(save_path, "wb") as f:
        pickle.dump(lda, f)
        print(f"LDA model saved to: {save_path}")
    return lda


def print_top_words(model, vectorizer, n_top_words=15):
    """
    Print n_top_words from LDA model results
    """
    feature_names = vectorizer.get_feature_names_out()

    print(f"\nPrinting top {n_top_words} words per topic...")

    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx}")
        top_features = topic.argsort()[-n_top_words:]
        print(" ".join(feature_names[top_features]))

    print("Done.")


def save_visualization(model, dtm, vectorizer):
    """
    Produce an interactive visualization and save it to HTML.
    """
    print("Preparing visualization...")
    vis_data = pyLDAvis.lda_model.prepare(model, dtm, vectorizer)

    pyLDAvis.save_html(vis_data, vis_output_path)
    print(f"Visualization saved to {vis_output_path}")


def get_next_available_path(path):
    """
    If `path` exists, append _01, _02, etc. before the file extension.
    """
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1

    # Increment until a free path exists
    while True:
        new_path = f"{base}_{counter:02d}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def main():
    print("Loading cleaned job postings CSV...")

    prepared_csv_path = os.path.join(BASE_DIR, "data", "all_prepared_job_postings.csv")
    df = pd.read_csv(prepared_csv_path)

    docs = df["description"].astype(str).tolist()

    # Load or train vectorizer
    vect = load_or_fit_vectorizer(docs)

    # Transform documents into a DTM
    print("Transforming documents...")
    X = vect.transform(docs)
    print("Done. Document-term matrix shape:", X.shape)

    # Train LDA
    lda_model = run_LDA(n_components=15, random_state=42, X=X)

    # Visualize and save HTML
    save_visualization(lda_model, X, vect)

    print_top_words()


if __name__ == "__main__":
    main()
