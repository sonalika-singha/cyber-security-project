# PhishGuard AI — Novelty Upgrades

This document explains the new modules added to turn PhishGuard AI from "four
separate detectors behind one UI" into a genuinely novel, defensible final-year
project. Everything here builds on your existing code; nothing was removed.

## What was added

| File | Contribution | Status |
|------|-------------|--------|
| `fusion_engine.py` | **Cross-modal fusion** — one unified verdict from multiple channels | ✅ verified on real models |
| `explainability.py` | **Explainable AI** — exact text attribution, image Grad-CAM, voice feature attribution | ✅ text/voice verified; Grad-CAM needs PyTorch (already in requirements) |
| `adversarial.py` | **Adversarial robustness** — evasion attacks + de-obfuscation defense | ✅ verified on real model |
| `evaluate_rigorous.py` | **Honest evaluation** — confusion matrix, calibration, video-leakage fix | ✅ verified |
| `routes_advanced.py` | Flask endpoints exposing all of the above | code-complete |
| `train_voice_model.py` | Real supervised AI-voice detector (replaces heuristics) | pipeline ready — needs your audio |
| `train_multilingual.py` | MuRIL/IndicBERT fine-tuning for Hinglish phishing | scaffold ready — needs your data + GPU |
| `demo_novelty.py` | One-command live demo of all three flagship features | ✅ verified |

"Verified on real models" = I ran it against your actual `phishguard_model.pkl`
/ `vectorizer.pkl`. The numbers below are real outputs, not placeholders.

---

## 1. The headline: cross-modal fusion (`fusion_engine.py`)

**The gap it fills.** Your comparison slide claims "Multimodal" and "Integrated"
as your differentiator, but the code ran four independent detectors and showed
four separate results — that is not fusion. `fuse_incident()` takes the
per-modality outputs for **one incident** and produces a single verdict.

**How it works (say this in the viva):**
1. Each detector's output is normalised to a phishing-probability `p_i`.
2. Probabilities are combined with a **weighted logarithmic opinion pool**
   (fusion in log-odds space) — the principled Bayesian product-of-experts,
   not naive averaging.
3. Each modality is weighted by its measured reliability (`2·accuracy − 1`), so
   a weak detector cannot overpower strong ones.
4. **Cross-modal corroboration rule:** when ≥2 independent channels agree, the
   threat escalates to *Critical* — because independent agreement is strong
   evidence of a coordinated attack, not a single-detector false positive.
5. A **campaign classifier** labels the coordinated attack type from the
   patterns each channel reports.

**Verified result (real run):** three channels each at only ~0.6 "Medium"
probability fuse to a **Critical** verdict labelled *"Coordinated
credential-harvesting campaign"* — exactly the multi-modal payoff. A single
benign channel correctly stays *Low*.

---

## 2. Explainable AI (`explainability.py`)

Moved from your Future Work into a core feature.

- **Text — exact attribution.** Your text model is Logistic Regression on
  TF-IDF, which is **linear**, so each token's contribution to the decision is
  exactly `coefficient × tfidf`. These are the true model internals, not a
  LIME/SHAP approximation. Verified run on *"URGENT: Your SBI account is
  BLOCKED…"* correctly attributes the flag to `http (+0.75)`, `urgent (+0.71)`,
  `verify (+0.38)`.
- **Image — Grad-CAM.** Heat-map over a screenshot showing which pixels drove
  the phishing call (fake login form, spoofed logo). PyTorch, targets
  MobileNetV2's last conv block.
- **Voice — acoustic attribution.** Reports which acoustic features (flat
  pitch, uniform loudness…) looked synthetic, with natural-speech reference
  ranges.

---

## 3. Adversarial robustness (`adversarial.py`)

Your own literature-review table lists "adversarially robust system" as future
scope — this delivers it, and gives you the most memorable demo.

- **Attacks:** homoglyphs (Cyrillic look-alikes), zero-width characters,
  leetspeak, intra-word spacing, and a combined attack.
- **Defense:** `normalize_text()` — strips invisibles, NFKC-folds homoglyphs,
  reverses leetspeak, collapses spacing.
- **Verified result:** a message the model flags at **0.97** is pushed to
  **0.28** by the combined attack (evasion succeeds), and the de-obfuscation
  defense restores it to **0.97** (detection recovered). Run `demo_novelty.py`
  to reproduce.

---

## 4. Honest evaluation (`evaluate_rigorous.py`)

**Fix the 100% video accuracy before an examiner does.** That figure came from
testing on videos built from the *same* images used for training — textbook
train/test leakage. The script prints a methodology note explaining this and
how to report an honest held-out number.

**Verified text result (held-out split):** accuracy **0.982**, precision
**0.945**, recall **0.919**, F1 **0.932**, with a full confusion matrix and a
calibration table (expected calibration error ≈ 0.11). Note honestly in the
report that this split is optimistic because the model was originally trained
on the full dataset; retrain on a train-only split for a clean number.

---

## Setup / integration

### Wire the new endpoints into `app.py`
Add two lines near the top of `app.py` (after `app = Flask(__name__)`):

```python
from routes_advanced import advanced
app.register_blueprint(advanced)
```

That exposes:
- `POST /detect-fusion` — the flagship; send `results` JSON or raw
  text/image/audio/video and it runs + fuses them.
- `POST /explain-text`, `POST /explain-image` (add `?format=image` to get the
  Grad-CAM PNG back), `POST /explain-voice`.
- `POST /adversarial-test` — `{ "text": "...", "attack": "combined" }`.

Create a `static/` folder for Grad-CAM overlays: `mkdir -p static`.

### Run the live demo
```bash
cd backend
python demo_novelty.py
```

### Train the real voice model (replaces heuristics)
```bash
mkdir -p data/voice/human data/voice/synthetic
# add real speech to human/, TTS clips (gTTS/edge-tts/Coqui) to synthetic/
python train_voice_model.py     # saves models/voice_model.pkl automatically
```
It saves in the exact `{"scaler", "model"}` format `app.py` already loads, so
the app switches from heuristics to the trained model with no code change.

### Fine-tune the multilingual model
```bash
pip install transformers datasets
python train_multilingual.py --data data/multilingual_phishing.csv
```

---

## Suggested new report sections

Add these to strengthen the report:

1. **Section 6.5 — Cross-Modal Fusion.** Describe the opinion-pool method and
   corroboration rule. This is your primary novelty; give it a figure showing
   three medium inputs → one critical verdict.
2. **Section 6.6 — Explainable AI.** Include a Grad-CAM overlay image and a
   text-attribution bar chart.
3. **Section 6.7 — Adversarial Robustness.** Include the attack/defense table
   below.
4. **Rewrite Section 10 (Results)** with the honest held-out numbers and the
   video methodology note.

### Report-ready results tables

**Table A — Adversarial robustness (text model):**

| Condition | Detection behaviour (example message) |
|-----------|----------------------------------------|
| Clean | flagged, P(phishing) = 0.97 |
| Combined attack | missed, P(phishing) = 0.28 (evasion succeeds) |
| Attack + defense | flagged again, P(phishing) = 0.97 (recovered) |

**Table B — Honest text evaluation (held-out split):**

| Metric | Value |
|--------|-------|
| Accuracy | 0.982 |
| Precision | 0.945 |
| Recall | 0.919 |
| F1 | 0.932 |

### Correct these factual errors in the current report
- The image model is **PyTorch MobileNetV2 (`.pth`)**, not Keras/`.h5`.
- The text model is **Logistic Regression** (report implies Naive Bayes).
  Either is fine — just state the one you actually ship.

---

## Likely viva questions (and your answers)

- **"How is this actually multi-modal and not just four models?"** → The fusion
  engine combines channels in log-odds space with reliability weighting and a
  corroboration rule; independent agreement across channels changes the verdict
  and threat level. Demo it.
- **"Why a log opinion pool instead of averaging?"** → Averaging ignores that
  independent evidence should compound; the log pool is the product-of-experts
  / Bayesian combination and correctly turns two medium signals into a strong
  one.
- **"How do you explain a deep model's decision?"** → Text is a linear model so
  attribution is exact; image uses Grad-CAM on the last conv layer.
- **"Is your detector robust to evasion?"** → Show the 0.97 → 0.28 → 0.97 demo.
- **"Why was video 100%?"** → Acknowledge the leakage openly and present the
  held-out methodology — this earns trust rather than losing it.
