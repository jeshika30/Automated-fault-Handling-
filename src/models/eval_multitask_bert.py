import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoConfig

from .multitask_bert import MultiTaskBertForFaultAndScript  # reuse the class

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/multitask_transformer"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


class MultiTaskDataset(Dataset):
    def __init__(self, texts, fault_labels, script_labels, tokenizer, max_length):
        self.texts = texts
        self.fault_labels = fault_labels
        self.script_labels = script_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["fault_labels"] = torch.tensor(self.fault_labels[idx])
        enc["script_labels"] = torch.tensor(self.script_labels[idx])
        return enc


def main():
    # Load test data
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    fault_classes = sorted(test_df["fault_class"].unique())
    fault2id = {c: i for i, c in enumerate(fault_classes)}
    id2fault = {i: c for c, i in fault2id.items()}

    # script ids same as fault ids in this setup
    script2id = dict(fault2id)

    texts = test_df["text"].tolist()
    fault_labels = np.array([fault2id[x] for x in test_df["fault_class"]])
    script_labels = np.array([script2id[x] for x in test_df["fault_class"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = MultiTaskBertForFaultAndScript.from_pretrained(
        MODEL_DIR,
        config=config,
        num_faults=len(fault2id),
        num_scripts=len(script2id),
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    ds = MultiTaskDataset(texts, fault_labels, script_labels, tokenizer, MAX_LENGTH)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    all_fault_logits = []
    all_script_logits = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            all_fault_logits.append(outputs["fault_logits"].cpu().numpy())
            all_script_logits.append(outputs["script_logits"].cpu().numpy())

    fault_logits = np.vstack(all_fault_logits)
    script_logits = np.vstack(all_script_logits)

    fault_pred = np.argmax(fault_logits, axis=-1)
    script_pred = np.argmax(script_logits, axis=-1)

    # Convert ids back to labels
    fault_true_labels = np.array([id2fault[i] for i in fault_labels])
    fault_pred_labels = np.array([id2fault[i] for i in fault_pred])

    # Because script id == fault id, script accuracy == fault accuracy here
    fault_acc = accuracy_score(fault_true_labels, fault_pred_labels)
    fault_macro_f1 = f1_score(fault_true_labels, fault_pred_labels, average="macro")

    print("[TEST] Multi-task BERT fault accuracy:", fault_acc)
    print("[TEST] Multi-task BERT fault macro-F1:", fault_macro_f1)
    print("\n[TEST] Classification report (fault head):\n")
    print(classification_report(fault_true_labels, fault_pred_labels))

    labels_sorted = sorted(fault_classes)
    cm = confusion_matrix(fault_true_labels, fault_pred_labels, labels=labels_sorted)
    print("\n[TEST] Confusion matrix (fault head):")
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))


if __name__ == "__main__":
    main()