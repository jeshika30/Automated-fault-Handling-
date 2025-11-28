import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    BertPreTrainedModel,
    BertModel,
    Trainer,
    TrainingArguments,
)

os.environ["TRANSFORMERS_NO_TF"] = "1"

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/multitask_transformer"
MODEL_NAME = "bert-base-uncased"


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


class MultiTaskBertForFaultAndScript(BertPreTrainedModel):
    def __init__(self, config, num_faults, num_scripts, lambda_script=0.5):
        super().__init__(config)
        self.bert = BertModel(config)
        hidden_size = config.hidden_size

        self.fault_head = nn.Linear(hidden_size, num_faults)
        self.script_head = nn.Linear(hidden_size, num_scripts)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_script = lambda_script

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        fault_labels=None,
        script_labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        fault_logits = self.fault_head(cls)
        script_logits = self.script_head(cls)

        loss = None
        if fault_labels is not None and script_labels is not None:
            lf = self.loss_fn(fault_logits, fault_labels)
            ls = self.loss_fn(script_logits, script_labels)
            loss = lf + self.lambda_script * ls

        return {
            "loss": loss,
            "fault_logits": fault_logits,
            "script_logits": script_logits,
        }


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # --- predictions: take fault logits ---
    if isinstance(preds, (tuple, list)):
        fault_logits = preds[0]  # first element = fault head
    else:
        fault_logits = preds
    fault_logits = np.asarray(fault_logits)

    # --- labels: extract fault labels only ---
    labels = np.asarray(labels)

    # Case 1: labels is an array of shape (num_examples,)
    if labels.ndim == 1:
        fault_labels = labels

    # Case 2: labels is stacked for multiple heads, e.g. (num_tasks, num_examples)
    elif labels.ndim == 2:
        # assume row 0 corresponds to fault_labels
        fault_labels = labels[0]

    # Fallback: if something unexpected, just flatten and hope for the best
    else:
        fault_labels = labels.reshape(-1)

    pred_ids = np.argmax(fault_logits, axis=-1)

    acc = accuracy_score(fault_labels, pred_ids)
    macro_f1 = f1_score(fault_labels, pred_ids, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    fault_classes = sorted(train_df["fault_class"].unique())
    fault2id = {c: i for i, c in enumerate(fault_classes)}
    id2fault = {i: c for c, i in fault2id.items()}

    # For now script_id == fault_class; future work can decouple
    script2id = dict(fault2id)
    id2script = dict(id2fault)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = 128

    def make_ds(df):
        texts = df["text"].tolist()
        faults = [fault2id[x] for x in df["fault_class"]]
        scripts = [script2id[x] for x in df["fault_class"]]
        return MultiTaskDataset(texts, faults, scripts, tokenizer, max_length)

    train_ds = make_ds(train_df)
    val_ds = make_ds(val_df)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = MultiTaskBertForFaultAndScript.from_pretrained(
        MODEL_NAME,
        config=config,
        num_faults=len(fault2id),
        num_scripts=len(script2id),
        lambda_script=0.5,
    )

    args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Training multi-task BERT...")
    trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"[OK] Saved multi-task model to {MODEL_DIR}")


if __name__ == "__main__":
    main()