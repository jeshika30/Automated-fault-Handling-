import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def load_splits():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train, val, test


def train_baseline():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df, val_df, test_df = load_splits()

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["fault_class"].tolist()

    X_val = val_df["text"].astype(str).tolist()
    y_val = val_df["fault_class"].tolist()

    X_test = test_df["text"].astype(str).tolist()
    y_test = test_df["fault_class"].tolist()

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams + bigrams
        stop_words=None,
    )

    print("[INFO] Fitting TF-IDF vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression classifier
    clf = LogisticRegression(
        max_iter=200,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
    )

    print("[INFO] Training Logistic Regression baseline...")
    clf.fit(X_train_tfidf, y_train)

    # Validation performance
    y_val_pred = clf.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    print("\n[VAL] Accuracy:", val_acc)
    print("[VAL] Macro F1:", val_f1)
    print("\n[VAL] Classification report:\n")
    print(classification_report(y_val, y_val_pred))

    # Test performance
    y_test_pred = clf.predict(X_test_tfidf)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    print("\n[TEST] Accuracy:", test_acc)
    print("[TEST] Macro F1:", test_f1)
    print("\n[TEST] Classification report:\n")
    print(classification_report(y_test, y_test_pred))

    print("\n[TEST] Confusion matrix:")
    cm = confusion_matrix(y_test, y_test_pred, labels=sorted(train_df["fault_class"].unique()))
    print(pd.DataFrame(cm, index=sorted(train_df["fault_class"].unique()),
                       columns=sorted(train_df["fault_class"].unique())))

    # Save model + vectorizer
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "logreg_baseline.joblib"))
    print(f"\n[OK] Saved vectorizer and model to {MODEL_DIR}/")


if __name__ == "__main__":
    train_baseline()