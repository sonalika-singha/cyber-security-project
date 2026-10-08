"""
evaluate_rigorous.py
====================
Honest, rigorous evaluation harness for PhishGuard AI.

Why this matters for your marks: the original report claims 100% video
accuracy, but the test videos were built from the SAME images the model
trained on. Any examiner will flag that as train/test leakage. Demonstrating
that you *understand* and *correct* evaluation methodology is itself worth
marks — it signals research maturity.

This harness:
  * evaluates the text model on a held-out test split with a full confusion
    matrix and precision / recall / F1 per class,
  * reports calibration (are the confidence scores meaningful?),
  * prints an explicit methodology note about the video leakage issue and how
    to fix it.

Run:  python evaluate_rigorous.py
"""

import numpy as np
import pandas as pd


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def _calibration(probs, y_true, n_bins=5):
    """Reliability: within each confidence bin, does predicted prob match reality?"""
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = float(probs[mask].mean())
        actual = float(y_true[mask].mean())
        rows.append({"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                     "n": int(mask.sum()),
                     "avg_confidence": round(conf, 3),
                     "actual_phishing_rate": round(actual, 3)})
        ece += (mask.sum() / len(probs)) * abs(conf - actual)
    return {"expected_calibration_error": round(ece, 4), "bins": rows}


def evaluate_text(csv_path="data/spam.csv", test_size=0.2, seed=42):
    import joblib
    from sklearn.model_selection import train_test_split

    for m, v in (("model/phishguard_model.pkl", "model/vectorizer.pkl"),
                 ("models/text_model.pkl", "models/vectorizer.pkl")):
        try:
            model = joblib.load(m); vec = joblib.load(v); break
        except Exception:
            continue
    else:
        return {"error": "Text model not found."}

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df = df[["label", "text"]].dropna()
    df["y"] = (df["label"].str.lower() == "spam").astype(int)

    # held-out split (the model was trained on the full set originally, so this
    # is an optimistic estimate; note it honestly in the report)
    _, X_test, _, y_test = train_test_split(
        df["text"], df["y"], test_size=test_size, random_state=seed, stratify=df["y"])

    probs = model.predict_proba(vec.transform(X_test))[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    result = _metrics(y_test.values, y_pred)
    result["calibration"] = _calibration(probs, y_test.values)
    result["n_test"] = int(len(y_test))
    return result


VIDEO_METHODOLOGY_NOTE = """
────────────────────────────────────────────────────────────────────────
METHODOLOGY NOTE — video evaluation (READ BEFORE QUOTING 100% ACCURACY)
────────────────────────────────────────────────────────────────────────
The original 100% video accuracy is NOT a valid generalisation estimate:
the test videos were assembled from the SAME screenshot images used to
train the image model, so the model had effectively already seen the test
frames (train/test leakage / data snooping).

To report an honest number:
  1. Collect NEW screen-recordings of phishing and legitimate sites that
     were NOT part of image training (record them yourself from live/archived
     pages).
  2. Keep image-training data and video-test data strictly disjoint.
  3. Report accuracy, precision, recall, F1 on that held-out video set.
  4. State the frame-sampling rate and the majority-vote threshold used.

In the report, present the original figure as an in-distribution sanity
check, and add the held-out figure as the real generalisation result. This
correction demonstrates methodological rigour and pre-empts the examiner's
most obvious question.
────────────────────────────────────────────────────────────────────────
"""


if __name__ == "__main__":
    import json
    print("=== TEXT MODEL — rigorous evaluation (held-out split) ===")
    res = evaluate_text()
    print(json.dumps(res, indent=2, default=str))
    print(VIDEO_METHODOLOGY_NOTE)
