"""
train_voice_model.py
====================
Turn the heuristic voice detector into a REAL supervised AI-voice / deepfake-
voice detector.

The report honestly states the voice module uses hand-tuned thresholds because
"no standard vishing dataset exists". True for *vishing scripts* — but you do
NOT need one. The detectable signal is "human speech vs machine-generated
speech", and you can build a labelled dataset for that yourself:

    class 0 (legitimate / human) : real recorded speech
        - Mozilla Common Voice  (https://commonvoice.mozilla.org)
        - LibriSpeech           (https://www.openslr.org/12)
        - or your own recordings

    class 1 (synthetic / AI)     : text-to-speech from several engines
        - gTTS, edge-tts, pyttsx3, Coqui-TTS, ElevenLabs samples
        - using MULTIPLE engines is important so the model learns "synthetic"
          in general, not one engine's fingerprint.

This script:
  1. loads every .wav/.mp3 under data/voice/human/ and data/voice/synthetic/
  2. extracts a rich acoustic feature vector (40 MFCC means + MFCC std +
     pitch/RMS/ZCR/spectral statistics) — a superset of the original 40-MFCC
     features, so it stays compatible with app.py's loader,
  3. trains a RandomForest, evaluates with a held-out split,
  4. saves models/voice_model.pkl as {"scaler": ..., "model": ...} — EXACTLY
     the format app.py already expects, so it starts using the trained model
     automatically (no app.py change needed).

Quick start
-----------
    data/voice/human/*.wav        <- real speech clips
    data/voice/synthetic/*.wav    <- TTS-generated clips
    python train_voice_model.py

To generate synthetic samples fast (one engine, for bootstrapping):
    pip install gTTS
    python -c "from gtts import gTTS; \
        [gTTS('Your account is blocked, verify your KYC now').save(f'data/voice/synthetic/s{i}.mp3') for i in range(20)]"
Then record ~20 human clips (read the same sentences aloud) into human/.
Use MORE engines and MORE clips for a stronger, less biased model.

NOTE: app.py extracts 40 MFCC means at inference time. This trainer's DEFAULT
feature mode is 'mfcc40' so the saved model is drop-in compatible. Set
FEATURE_MODE='rich' only if you also upgrade app.py's feature extraction to
match (snippet provided in NOVELTY_UPGRADES.md).
"""

import os
import glob
import pickle

import numpy as np

HUMAN_DIR     = "data/voice/human"
SYNTHETIC_DIR = "data/voice/synthetic"
OUT_PATH      = "models/voice_model.pkl"
SR            = 22050
FEATURE_MODE  = "mfcc40"   # 'mfcc40' (app.py-compatible) or 'rich'


def extract_features(path, mode=FEATURE_MODE):
    import librosa
    y, sr = librosa.load(path, sr=SR, duration=30)
    if len(y) < sr * 0.5:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)          # 40 dims — matches app.py

    if mode == "mfcc40":
        return mfcc_mean

    # 'rich' mode — stronger discrimination (requires app.py feature upgrade)
    mfcc_std = np.std(mfcc.T, axis=0)
    pitch = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    pitch = pitch[(pitch > 60) & (pitch < 400) & ~np.isnan(pitch)]
    extra = np.array([
        np.var(pitch) if len(pitch) > 5 else 0.0,
        np.var(librosa.feature.rms(y=y)[0]),
        np.mean(librosa.feature.zero_crossing_rate(y)[0]),
        np.var(librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
        np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]),
    ])
    return np.concatenate([mfcc_mean, mfcc_std, extra])


def load_dataset():
    X, y = [], []
    for label, folder in ((0, HUMAN_DIR), (1, SYNTHETIC_DIR)):
        files = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"):
            files += glob.glob(os.path.join(folder, ext))
        print(f"  {folder}: {len(files)} files")
        for f in files:
            try:
                feat = extract_features(f)
                if feat is not None:
                    X.append(feat); y.append(label)
            except Exception as e:
                print(f"    [skip] {f}: {e}")
    return np.array(X), np.array(y)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    if not (os.path.isdir(HUMAN_DIR) and os.path.isdir(SYNTHETIC_DIR)):
        print("Create these folders and add audio first:")
        print(f"  {HUMAN_DIR}/      (real human speech)")
        print(f"  {SYNTHETIC_DIR}/  (TTS / AI-generated speech)")
        os.makedirs(HUMAN_DIR, exist_ok=True)
        os.makedirs(SYNTHETIC_DIR, exist_ok=True)
        return

    print("[1/4] Loading + extracting features...")
    X, y = load_dataset()
    if len(X) < 10 or len(set(y)) < 2:
        print(f"Not enough data (got {len(X)} samples, classes={set(y)}). "
              "Add more clips to both folders.")
        return

    print(f"[2/4] Dataset: {len(X)} samples, feature dim {X.shape[1]}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_tr)
    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                 class_weight="balanced")
    clf.fit(scaler.transform(X_tr), y_tr)

    print("[3/4] Evaluation on held-out set:")
    y_pred = clf.predict(scaler.transform(X_te))
    print(classification_report(y_te, y_pred,
          target_names=["human", "synthetic"], zero_division=0))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_te, y_pred))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "model": clf}, f)
    print(f"[4/4] Saved -> {OUT_PATH}  (app.py will use it automatically)")


if __name__ == "__main__":
    main()
