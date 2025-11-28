import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

EMB_DIR = "data/processed/bert_embeddings"


def load_split(split):
    X = np.load(os.path.join(EMB_DIR, f"{split}_embeddings.npy"))
    y = np.load(os.path.join(EMB_DIR, f"{split}_labels.npy"))
    meta = pd.read_csv(os.path.join(EMB_DIR, f"{split}_metadata.csv"))
    return X, y, meta


def load_label_mapping():
    mapping_df = pd.read_csv(os.path.join(EMB_DIR, "label_mapping.csv"))
    id2label = {row["id"]: row["label"] for _, row in mapping_df.iterrows()}
    return id2label


def main():
    id2label = load_label_mapping()

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")

    print("[INFO] Shapes:")
    print("  train:", X_train.shape, y_train.shape)
    print("  val:  ", X_val.shape, y_val.shape)
    print("  test: ", X_test.shape, y_test.shape)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1,
    )

    print("[INFO] Training Logistic Regression on BERT incident embeddings...")
    clf.fit(X_train, y_train)

    # Validation
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    print("\n[VAL] Accuracy:", val_acc)
    print("[VAL] Macro F1:", val_f1)

    # Test
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    print("\n[TEST] Accuracy:", test_acc)
    print("[TEST] Macro F1:", test_f1)

    print("\n[TEST] Classification report:\n")
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    print("\n[TEST] Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))


if __name__ == "__main__":
    main()