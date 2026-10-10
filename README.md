# PhishGuard AI — Multi-Modal Phishing & Deepfake Detection
**GitHub Repository:** https://github.com/sonalika-singha/cyber-security-project

Cyber-Seccurity project. A cross-modal AI system that detects phishing and
deepfake content across **text, voice, image, and video**, with three flagship
capabilities added on top of the base detectors:

1. **Cross-modal fusion** — combines multiple channels of one incident into a
   single unified verdict (the true multi-modal signal).
2. **Explainable AI** — exact per-word attribution for text, Grad-CAM heat-maps
   for images, acoustic feature attribution for voice.
3. **Adversarial robustness** — evasion attacks + a de-obfuscation defense.

---

## Quick start (extract → install → run)

```bash
# 1. Enter the project
cd backend

# 2. (recommended) create a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. run the app
python app.py
```

Then open **http://localhost:5000** in your browser.

> First run downloads nothing except the Python packages. PyTorch is the
> largest (~200 MB). The trained models are already bundled in `model/` and
> `models/`, so detection works immediately.

---

## Using the dashboard

The UI has three modes (segmented control at the top):

- **Single Scan** — analyze one input (Text / Voice / Image / Video). Text and
  Image show an explainability panel (why it was flagged / Grad-CAM heat-map).
- **Fusion Scan** — provide several channels of the *same* incident (e.g. an SMS
  + a screenshot + a call recording) and get one unified verdict. When ≥2
  channels agree, the threat escalates to **Critical** and the coordinated
  campaign is named.
- **Adversarial Lab** — obfuscate a phishing message to evade the filter, then
  watch the de-obfuscation defense catch it again.

---

## Run the presentation demo (no browser needed)

```bash
cd backend
python demo_novelty.py
```
Prints the fusion, explainability, and adversarial results in the terminal —
handy as a backup during the viva.

## Rigorous evaluation
```bash
cd backend
python evaluate_rigorous.py
```

---

## Project structure

```
phishguard-ai/
├── README.md                  ← you are here
└── backend/
    ├── app.py                 ← Flask server (novelty routes auto-registered)
    ├── templates/index.html   ← the dashboard UI (Single / Fusion / Adversarial)
    │
    ├── fusion_engine.py        ← ★ cross-modal fusion
    ├── explainability.py       ← ★ text attribution + Grad-CAM + voice attribution
    ├── adversarial.py          ← ★ evasion attacks + defense
    ├── routes_advanced.py      ← Flask endpoints for the above
    ├── evaluate_rigorous.py    ← honest metrics (confusion matrix, calibration)
    ├── demo_novelty.py         ← one-command demo of all three
    │
    ├── text_analyzer.py        ← text detector (TF-IDF + Logistic Regression)
    ├── voice_detector.py       ← voice detector (acoustic heuristics)
    ├── image_detector.py       ← image detector (MobileNetV2, PyTorch)
    ├── video_detector.py       ← video detector (frame voting)
    │
    ├── train_model.py          ← train text model
    ├── train_image_model.py    ← train image model
    ├── train_voice_model.py    ← ★ train a REAL supervised AI-voice detector
    ├── train_multilingual.py   ← ★ fine-tune MuRIL for Hinglish phishing
    │
    ├── model/  models/         ← bundled trained models (ready to use)
    ├── data/                   ← spam.csv + small test audio/video
    ├── static/                 ← Grad-CAM overlays are written here at runtime
    ├── requirements.txt
    └── NOVELTY_UPGRADES.md      ← full write-up: methods, report text, viva Q&A
```

★ = new work added on top of the base project.

---

## Notes

- **Training images not bundled.** The 215 MB screenshot training set
  (`data/images/`) was removed to keep the download small. The image model is
  already trained (`model/image_phishing_model.pth`), so image/video detection
  works without it. Re-download from the Kaggle "Phishing and Legit Website
  Screenshots" dataset only if you want to retrain.
- **API endpoints:** `/detect-text`, `/detect-voice`, `/detect-image`,
  `/detect-video`, `/detect-fusion`, `/explain-text`, `/explain-image`,
  `/explain-voice`, `/adversarial-test`.
- See **`backend/NOVELTY_UPGRADES.md`** for the methodology write-up,
  report-ready sections, results tables, and likely viva questions with answers.
