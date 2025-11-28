import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

# Make sure TF is disabled
os.environ["TRANSFORMERS_NO_TF"] = "1"

RAW_PATH = "data/raw/synthetic_syslogs.csv"
PROCESSED_DIR = "data/processed"
INCIDENTS_PATH = os.path.join(PROCESSED_DIR, "incidents.csv")
EMB_DIR = os.path.join(PROCESSED_DIR, "bert_embeddings")

MODEL_NAME = "bert-base-uncased"


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


def encode_texts(texts, tokenizer, model, max_length=128, batch_size=32, device="cpu"):
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            # CLS token representation
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
            all_embeddings.append(cls_emb.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    os.makedirs(EMB_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # ---------- 1. Line-level embeddings (raw logs) ----------
    print("[INFO] Loading raw logs for line-level embeddings...")
    raw_df = pd.read_csv(RAW_PATH)
    line_texts = raw_df["message"].astype(str).tolist()

    print(f"[INFO] Encoding {len(line_texts)} log lines...")
    line_emb = encode_texts(line_texts, tokenizer, model, device=device)

    np.save(os.path.join(EMB_DIR, "lines_embeddings.npy"), line_emb)
    raw_df.to_csv(os.path.join(EMB_DIR, "lines_metadata.csv"), index=False)
    print(f"[OK] Saved line-level embeddings + metadata to {EMB_DIR}")

    # ---------- 2. Window-level embeddings (incidents) ----------
    print("[INFO] Loading incident splits for window-level embeddings...")
    train_df, val_df, test_df = load_splits()
    label2id, id2label = build_label_mapping(train_df)

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        texts = df["text"].astype(str).tolist()
        labels = [label2id[l] for l in df["fault_class"].tolist()]

        print(f"[INFO] Encoding {split_name} incidents: {len(texts)} samples...")
        emb = encode_texts(texts, tokenizer, model, device=device)

        np.save(os.path.join(EMB_DIR, f"{split_name}_embeddings.npy"), emb)
        np.save(os.path.join(EMB_DIR, f"{split_name}_labels.npy"), np.array(labels))
        df.to_csv(os.path.join(EMB_DIR, f"{split_name}_metadata.csv"), index=False)

        print(f"[OK] Saved {split_name} embeddings + labels + metadata to {EMB_DIR}")

    # Also save label mapping for later use
    mapping_path = os.path.join(EMB_DIR, "label_mapping.csv")
    pd.DataFrame(
        [{"label": l, "id": i} for l, i in label2id.items()]
    ).to_csv(mapping_path, index=False)
    print(f"[OK] Saved label mapping to {mapping_path}")


if __name__ == "__main__":
    main()