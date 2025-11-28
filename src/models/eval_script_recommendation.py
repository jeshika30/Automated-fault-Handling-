import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .fault_to_script import get_script_for_fault

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def load_test_data():
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return test_df


def load_baseline_model():
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    clf_path = os.path.join(MODEL_DIR, "logreg_baseline.joblib")

    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)

    return vectorizer, clf


def evaluate_script_recommendation():
    test_df = load_test_data()
    vectorizer, clf = load_baseline_model()

    texts = test_df["text"].astype(str).tolist()
    y_true_fault = test_df["fault_class"].tolist()

    X_tfidf = vectorizer.transform(texts)
    y_pred_fault = clf.predict(X_tfidf)

    # Stage 2: map faults to scripts
    true_scripts = [get_script_for_fault(lbl)["script_id"] for lbl in y_true_fault]
    pred_scripts = [get_script_for_fault(lbl)["script_id"] for lbl in y_pred_fault]

    # Because mapping is 1-to-1, script accuracy == fault accuracy
    fault_acc = accuracy_score(y_true_fault, y_pred_fault)
    script_acc = accuracy_score(true_scripts, pred_scripts)

    print("[RESULT] Fault classification accuracy:", fault_acc)
    print("[RESULT] Script recommendation accuracy:", script_acc)

    print("\n[RESULT] Fault-class classification report:\n")
    print(classification_report(y_true_fault, y_pred_fault))

    print("\n[RESULT] Fault-class confusion matrix:")
    labels_sorted = sorted(set(y_true_fault))
    cm = confusion_matrix(y_true_fault, y_pred_fault, labels=labels_sorted)
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))

    # Show a few sample predictions
    print("\n[EXAMPLES] Sample script recommendations:")
    for i in range(3):
        print("=" * 60)
        print("Log text snippet:", texts[i][:200].replace("\n", " "))
        print("True fault:      ", y_true_fault[i])
        print("Pred fault:      ", y_pred_fault[i])
        print("Script ID (pred):", pred_scripts[i])
        script = get_script_for_fault(y_pred_fault[i])
        print("Description:     ", script["description"])
        print("Commands (first 3):")
        for cmd in script["commands"][:3]:
            print("  ", cmd)


if __name__ == "__main__":
    evaluate_script_recommendation()