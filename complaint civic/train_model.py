"""
train_model.py
---------------
Trains a text classifier that suggests a complaint category from the
free-text description. Baseline approach: TF-IDF + Logistic Regression —
fast, interpretable, and strong enough for short complaint text.

Run this once (or whenever you refresh the training data) before starting
the Streamlit app:

    python train_model.py
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_PATH = "data/sample_complaints.csv"
MODEL_PATH = "model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "category"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("Evaluation on held-out test set:\n")
    print(classification_report(y_test, preds))

    # Retrain on the full dataset before saving, so the shipped model uses
    # every available labeled example.
    pipeline.fit(df["text"], df["category"])
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
