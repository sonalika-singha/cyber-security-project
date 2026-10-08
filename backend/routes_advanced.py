"""
routes_advanced.py
==================
New API endpoints exposing the novelty modules. Registered as a Flask
Blueprint so your existing app.py stays intact — you add ONE import and ONE
line (see NOVELTY_UPGRADES.md).

Endpoints
---------
POST /detect-fusion      multi-modal incident fusion (the flagship feature)
POST /explain-text       exact per-word attribution for a message
POST /explain-image      Grad-CAM heat-map overlay for a screenshot
POST /explain-voice      acoustic feature attribution for an audio clip
POST /adversarial-test   run an evasion attack + defense on a message
"""

import os
import tempfile
import traceback

from flask import Blueprint, request, jsonify, send_file

from fusion_engine import fuse_incident
from explainability import explain_text, explain_image, explain_voice
from adversarial import ATTACKS, normalize_text, _predict_phishing_prob

advanced = Blueprint("advanced", __name__)


# ---------------------------------------------------------------------------
# FLAGSHIP: cross-modal fusion for a single incident
# ---------------------------------------------------------------------------
@advanced.route("/detect-fusion", methods=["POST"])
def detect_fusion():
    """
    Accepts any combination of channels describing ONE incident and returns a
    single unified verdict. Two ways to call it:

    (A) Pass already-computed detector results as JSON:
        { "results": { "text": {...}, "image": {...}, "voice": {...} } }

    (B) Pass raw inputs and let this endpoint run each detector, then fuse:
        multipart form with any of:
           text  (form field)
           image (file), audio (file), video (file)
    """
    try:
        # Mode A — caller already has per-modality results
        if request.is_json:
            payload = request.get_json()
            results = payload.get("results", {})
            if results:
                return jsonify(fuse_incident(results))

        # Mode B — run detectors here from raw inputs
        results = {}

        text = request.form.get("text", "").strip()
        if text:
            try:
                from text_analyzer import analyze_message
                is_p, threat, kw, expl, patterns = analyze_message(text)
                results["text"] = {
                    "is_phishing": bool(is_p), "threat": str(threat),
                    "keywords": list(kw), "explanation": str(expl),
                    "patterns": list(patterns),
                    "pattern": patterns[0] if patterns else "",
                    "confidence": _predict_phishing_prob(text) if bool(is_p)
                                  else 1 - _predict_phishing_prob(text),
                }
            except Exception:
                p = _predict_phishing_prob(text)
                results["text"] = {"is_phishing": p >= 0.5, "confidence": p,
                                   "scores": {"phishing": p, "legitimate": 1 - p}}

        def _run_file(key, fn, suffix):
            if key in request.files:
                f = request.files[key]
                sfx = os.path.splitext(f.filename)[1] or suffix
                with tempfile.NamedTemporaryFile(suffix=sfx, delete=False) as tmp:
                    f.save(tmp.name); path = tmp.name
                try:
                    return fn(path)
                finally:
                    if os.path.exists(path):
                        os.remove(path)
            return None

        img = _run_file("image", lambda p: __import__("image_detector").detect_phishing_image(p), ".jpg")
        if img:
            results["image"] = img
        vid = _run_file("video", lambda p: __import__("video_detector").detect_phishing_video(p), ".mp4")
        if vid:
            results["video"] = vid
        aud = _run_file("audio", lambda p: __import__("voice_detector").detect_ai_voice(p), ".wav")
        if aud:
            results["voice"] = aud

        if not results:
            return jsonify({"error": "Provide 'results' JSON or at least one of "
                                     "text/image/audio/video inputs."}), 400
        return jsonify(fuse_incident(results))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# EXPLAINABILITY
# ---------------------------------------------------------------------------
@advanced.route("/explain-text", methods=["POST"])
def explain_text_route():
    try:
        text = (request.get_json().get("text") if request.is_json
                else request.form.get("text", "")).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        return jsonify(explain_text(text))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@advanced.route("/explain-image", methods=["POST"])
def explain_image_route():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file (key 'image')"}), 400
        f = request.files["image"]
        sfx = os.path.splitext(f.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=sfx, delete=False) as tmp:
            f.save(tmp.name); path = tmp.name
        try:
            out = os.path.join("static", "gradcam_overlay.png")
            result = explain_image(path, out_path=out)
        finally:
            if os.path.exists(path):
                os.remove(path)
        # return the overlay image directly if requested
        if request.args.get("format") == "image" and "overlay_path" in result:
            return send_file(result["overlay_path"], mimetype="image/png")
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@advanced.route("/explain-voice", methods=["POST"])
def explain_voice_route():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file (key 'audio')"}), 400
        f = request.files["audio"]
        sfx = os.path.splitext(f.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=sfx, delete=False) as tmp:
            f.save(tmp.name); path = tmp.name
        try:
            return jsonify(explain_voice(path))
        finally:
            if os.path.exists(path):
                os.remove(path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# ADVERSARIAL LIVE DEMO
# ---------------------------------------------------------------------------
@advanced.route("/adversarial-test", methods=["POST"])
def adversarial_test():
    """
    Body: { "text": "...", "attack": "combined" }
    Returns clean / attacked / defended probabilities — perfect for a live
    "watch me evade the filter, then watch the defense catch it" demo.
    """
    try:
        data = request.get_json() if request.is_json else request.form
        text = (data.get("text") or "").strip()
        attack = data.get("attack", "combined")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if attack not in ATTACKS:
            return jsonify({"error": f"Unknown attack. Options: {list(ATTACKS)}"}), 400

        adv = ATTACKS[attack](text)
        defended = normalize_text(adv)
        p_clean = _predict_phishing_prob(text)
        p_adv = _predict_phishing_prob(adv)
        p_def = _predict_phishing_prob(defended)

        return jsonify({
            "attack": attack,
            "original":  {"text": text,     "phishing_prob": round(p_clean, 3),
                          "flagged": p_clean >= 0.5},
            "attacked":  {"text": adv,      "phishing_prob": round(p_adv, 3),
                          "flagged": p_adv >= 0.5,
                          "evaded": (p_clean >= 0.5) and (p_adv < 0.5)},
            "defended":  {"text": defended, "phishing_prob": round(p_def, 3),
                          "flagged": p_def >= 0.5,
                          "recovered": (p_adv < 0.5) and (p_def >= 0.5)},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
