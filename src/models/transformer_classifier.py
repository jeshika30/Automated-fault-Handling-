import os
os.environ["TRANSFORMERS_NO_TF"] = "1" # Disable TF in transformers

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/transformer"


# --------- Dataset wrapper ---------
class SyslogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, label2id):
        self.texts = texts
        self.labels = [label2id[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label)
        return enc


# --------- Load data + label mapping ---------
def load_splits():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train, val, test


def build_label_mapping(train_df):
    labels = sorted(train_df["fault_class"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


# --------- Metrics for Trainer ---------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def train_transformer():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df, val_df, test_df = load_splits()

    # ---- print class distributions ----
    print("[INFO] Train class counts:\n", train_df["fault_class"].value_counts())
    print("[INFO] Val class counts:\n", val_df["fault_class"].value_counts())
    print("[INFO] Test class counts:\n", test_df["fault_class"].value_counts())

    # ---- balance training set by oversampling minority classes ----
    labels = train_df["fault_class"].unique().tolist()
    counts = train_df["fault_class"].value_counts()
    max_count = counts.max()

    balanced_dfs = []
    for lbl in labels:
        df_lbl = train_df[train_df["fault_class"] == lbl]
        # sample with replacement up to max_count
        df_bal = df_lbl.sample(max_count, replace=True, random_state=42)
        balanced_dfs.append(df_bal)

    train_df_balanced = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print("[INFO] Balanced train class counts:\n", train_df_balanced["fault_class"].value_counts())

    # build label mapping on (original) train_df
    label2id, id2label = build_label_mapping(train_df)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = 128  # more than enough for windowed syslog text

    train_dataset = SyslogDataset(
        texts=train_df_balanced["text"].tolist(),
        labels=train_df_balanced["fault_class"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
    )
    val_dataset = SyslogDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["fault_class"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
    )
    test_dataset = SyslogDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["fault_class"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
    )

    num_labels = len(label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ---- freeze all BERT encoder parameters; train only classifier head ----
    # Partially unfreeze: only last 2 encoder layers
    if hasattr(model, "bert"):
        for name, param in model.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("[INFO] Unfroze last 2 BERT layers; others frozen.")
    else:
        print("[WARN] Model has no .bert attribute; not freezing encoder.")

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=15,              # more epochs, but tiny data
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,              # gentler LR
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",              # don't save every epoch
        logging_strategy="steps",
        logging_steps=5,
        load_best_model_at_end=False,    
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting BERT fine-tuning...")
    trainer.train()

    print("[INFO] Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("[VAL] metrics:", val_metrics)

    print("[INFO] Evaluating on test set...")
    test_outputs = trainer.predict(test_dataset)
    logits = test_outputs.predictions
    y_true = test_outputs.label_ids
    y_pred = np.argmax(logits, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n[TEST] Accuracy:", acc)
    print("[TEST] Macro F1:", macro_f1)
    print("\n[TEST] Classification report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(id2label))],
        )
    )

    print("\n[TEST] Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Save final model + tokenizer
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\n[OK] Saved transformer model + tokenizer to {MODEL_DIR}")


if __name__ == "__main__":
    train_transformer()