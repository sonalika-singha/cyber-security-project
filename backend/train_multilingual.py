"""
train_multilingual.py
=====================
Fine-tune a multilingual transformer (MuRIL / IndicBERT) for phishing detection
in Hindi, Marathi and code-mixed "Hinglish" text.

Why this is a real novelty: your report leans heavily on the Indian context
(SBI, Aadhaar, KYC, Rs. 7,488 crore fraud), but the current text model is
English-SMS-only TF-IDF. Indian phishing is overwhelmingly code-mixed
("aapka SBI account block ho gaya hai, KYC verify karein"). A TF-IDF English
model degrades badly on this. MuRIL (Google's Multilingual Representations for
Indian Languages) is pre-trained on 17 Indian languages + their transliterated
forms — ideal for exactly this problem, and under-explored for phishing.

Deliverable: a fine-tuned classifier that plugs in as an alternative text
backend, plus honest before/after numbers on a code-mixed test set.

Requirements
------------
    pip install transformers torch datasets scikit-learn pandas
    (GPU strongly recommended; CPU works but is slow.)

Data format (CSV): columns  text,label   with label in {0 legit, 1 phishing}.
Build a code-mixed set by:
  * translating/transliterating a sample of your English phishing SMS to
    Hinglish (many are already code-mixed in the wild),
  * collecting real Hindi/Marathi scam SMS (crowd-sourced, cybercrime forums),
  * keeping an English portion so the model stays bilingual.

Run:
    python train_multilingual.py --data data/multilingual_phishing.csv
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/multilingual_phishing.csv")
    ap.add_argument("--model", default="google/muril-base-cased",
                    help="try 'ai4bharat/indic-bert' as an alternative")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="models/muril_phishing")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df)} rows | phishing={df.label.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"])

    tok = AutoTokenizer.from_pretrained(args.model)

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tok(texts, truncation=True, padding=True, max_length=128)
            self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        pr, rc, f1, _ = precision_recall_fscore_support(
            p.label_ids, preds, average="binary", zero_division=0)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "precision": pr, "recall": rc, "f1": f1}

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="epoch", save_strategy="epoch",
        logging_steps=25, load_best_model_at_end=True,
        metric_for_best_model="f1", report_to="none")

    trainer = Trainer(model=model, args=targs,
                      train_dataset=DS(X_tr, y_tr),
                      eval_dataset=DS(X_te, y_te),
                      compute_metrics=metrics)

    print("Fine-tuning...")
    trainer.train()
    print("Final:", trainer.evaluate())

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved -> {args.out}")
    print("\nInference:")
    print(f"  from transformers import pipeline")
    print(f"  clf = pipeline('text-classification', model='{args.out}')")
    print(f"  clf('aapka SBI account block ho gaya, KYC verify karein')")


if __name__ == "__main__":
    main()
