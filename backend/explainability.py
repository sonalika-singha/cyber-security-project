"""
explainability.py
=================
Explainable-AI layer for PhishGuard AI.

Moves "explainability" from the report's Future Work into a working, core
feature. Every verdict can now answer the question examiners always ask:
"WHY did the model decide that?"

Three explainers, one per modality that supports it:

  explain_text(message)    -> exact per-word contribution to the phishing
                              decision. The text model is Logistic Regression
                              on TF-IDF, which is LINEAR, so the contribution
                              of each token is EXACTLY coef * tfidf_value.
                              No LIME/SHAP approximation is needed — these are
                              the true model internals. (Strong point for a viva.)

  explain_image(path, out) -> Grad-CAM heat-map overlay showing WHICH pixels
                              of a screenshot drove the phishing classification
                              (e.g. a fake login form). PyTorch MobileNetV2.

  explain_voice(path)      -> attribution over acoustic features, showing which
                              properties (flat pitch, uniform loudness, etc.)
                              looked synthetic.
"""

import numpy as np


# ===========================================================================
# TEXT  —  exact linear attribution (no approximation)
# ===========================================================================
_TXT_MODEL = None
_TXT_VEC = None


def _load_text():
    global _TXT_MODEL, _TXT_VEC
    if _TXT_MODEL is None:
        import joblib
        # match the paths used by text_analyzer.py / app.py
        for m, v in (("model/phishguard_model.pkl", "model/vectorizer.pkl"),
                     ("models/text_model.pkl", "models/vectorizer.pkl")):
            try:
                _TXT_MODEL = joblib.load(m)
                _TXT_VEC = joblib.load(v)
                break
            except Exception:
                continue
    return _TXT_MODEL, _TXT_VEC


def explain_text(message, top_k=8):
    """
    Return the exact tokens that pushed the decision toward 'phishing'
    (positive weight) or 'legitimate' (negative weight).

    For a linear model:  logit = bias + sum_i (coef_i * x_i)
    so token i's signed contribution to the phishing log-odds is coef_i * x_i.
    """
    model, vec = _load_text()
    if model is None:
        return {"error": "Text model not found."}

    x = vec.transform([message])
    feats = np.array(vec.get_feature_names_out())

    # logistic-regression coefficients for the phishing class
    coef = model.coef_[0] if hasattr(model, "coef_") else None
    if coef is None:
        return {"error": "Model is not linear; use a model-agnostic explainer."}

    xarr = x.toarray()[0]
    contributions = coef * xarr                      # exact signed contribution
    nz = np.nonzero(xarr)[0]                          # tokens actually present

    ranked = sorted(nz, key=lambda i: abs(contributions[i]), reverse=True)[:top_k]
    tokens = [{
        "token":        str(feats[i]),
        "contribution": round(float(contributions[i]), 4),
        "direction":    "phishing" if contributions[i] > 0 else "legitimate",
        "tfidf":        round(float(xarr[i]), 4),
    } for i in ranked]

    prob = float(model.predict_proba(x)[0][1])
    pushing_phish = [t for t in tokens if t["direction"] == "phishing"]

    if pushing_phish:
        summary = ("The model flagged this message mainly because of the term(s): "
                   + ", ".join(f"'{t['token']}'" for t in pushing_phish[:4])
                   + f". Combined phishing probability: {prob*100:.1f}%.")
    else:
        summary = (f"No strong phishing terms found; phishing probability "
                   f"{prob*100:.1f}%.")

    return {
        "phishing_probability": round(prob, 4),
        "token_contributions":  tokens,
        "bias":                 round(float(model.intercept_[0]), 4),
        "summary":              summary,
        "method":               "exact_linear_attribution",
    }


# ===========================================================================
# IMAGE  —  Grad-CAM for MobileNetV2 (PyTorch)
# ===========================================================================
def explain_image(image_path, out_path="static/gradcam_overlay.png"):
    """
    Generate a Grad-CAM heat-map overlay highlighting the image regions that
    most influenced the 'phishing' class. Saves an overlay PNG and returns its
    path plus the model's phishing probability.
    """
    import os
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from image_detector import build_model, preprocess, MODEL_PATH, DEVICE

    if not os.path.exists(MODEL_PATH):
        return {"error": f"Image model not found at {MODEL_PATH}."}

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Last convolutional block of MobileNetV2 -> best spatial layer for Grad-CAM
    target_layer = model.features[-1]

    activations, gradients = {}, {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    phish_prob = float(probs[1].item())

    model.zero_grad()
    logits[0, 1].backward()          # gradient of the PHISHING logit

    acts = activations["value"][0]          # (C,H,W)
    grads = gradients["value"][0]           # (C,H,W)
    weights = grads.mean(dim=(1, 2))        # global-average-pool gradients (C,)

    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove(); h2.remove()

    # Overlay heat-map on the original image
    orig = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam, (224, 224))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.55, heat, 0.45, 0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, overlay)

    return {
        "phishing_probability": round(phish_prob, 4),
        "overlay_path":         out_path,
        "summary":              ("Grad-CAM highlights the regions that most "
                                 "influenced the phishing decision (warm colours "
                                 "= high influence — typically fake login forms "
                                 "or spoofed brand areas)."),
        "method":               "grad_cam_mobilenetv2",
    }


# ===========================================================================
# VOICE  —  acoustic feature attribution
# ===========================================================================
def explain_voice(audio_path):
    """
    Explain a voice verdict by reporting the acoustic features that indicate
    synthetic / scripted speech, each with a natural-speech reference range.
    """
    import librosa

    try:
        y, sr = librosa.load(audio_path, duration=10)
    except Exception as e:
        return {"error": f"Could not load audio: {e}"}
    if len(y) < sr * 0.5:
        return {"error": "Audio too short to analyse."}

    pitch = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    pitch = pitch[(pitch > 60) & (pitch < 400) & ~np.isnan(pitch)]
    pitch_var = float(np.var(pitch)) if len(pitch) > 5 else 0.0
    rms_var = float(np.var(librosa.feature.rms(y=y)[0]))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    centroid_var = float(np.var(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))

    # Reference ranges typical of natural conversational speech.
    checks = [
        ("Pitch variance", pitch_var, 200.0,
         "Flat / monotone pitch is characteristic of TTS-generated voices."),
        ("Loudness variance", rms_var, 0.002,
         "Unnaturally uniform loudness suggests synthetic or scripted audio."),
        ("Spectral centroid variance", centroid_var, 50000.0,
         "Low spectral variation indicates limited natural timbre changes."),
    ]

    factors = []
    synthetic_flags = 0
    for name, value, thresh, note in checks:
        looks_synthetic = value < thresh
        if looks_synthetic:
            synthetic_flags += 1
        factors.append({
            "feature":         name,
            "value":           round(value, 5),
            "natural_min":     thresh,
            "looks_synthetic": looks_synthetic,
            "note":            note,
        })

    summary = (f"{synthetic_flags} of {len(checks)} acoustic features fall in the "
               f"synthetic/scripted range. "
               + ("Voice likely AI-generated or scripted."
                  if synthetic_flags >= 2 else
                  "Voice largely consistent with natural human speech."))

    return {
        "synthetic_indicators": synthetic_flags,
        "factors":              factors,
        "extra":                {"zero_crossing_rate": round(zcr, 5)},
        "summary":              summary,
        "method":               "acoustic_feature_attribution",
    }


# ===========================================================================
if __name__ == "__main__":
    from pprint import pprint
    print("=== TEXT explanation ===")
    pprint(explain_text(
        "URGENT: Your SBI account is BLOCKED. Verify KYC immediately: http://sbi-verify.xyz"
    ))
    print("\n=== TEXT explanation (benign) ===")
    pprint(explain_text("Hey, are we still meeting for lunch tomorrow at 1pm?"))
