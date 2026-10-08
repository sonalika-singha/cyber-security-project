import os
import re
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pickle
import librosa
import tempfile

# IMAGE + VIDEO
from image_detector import detect_phishing_image
from video_detector import detect_phishing_video

app = Flask(__name__)
CORS(app)

# ── Novelty modules: cross-modal fusion, explainability, adversarial ──
try:
    from routes_advanced import advanced
    app.register_blueprint(advanced)
    print("[✓] Advanced routes loaded (fusion / explainability / adversarial)")
except Exception as _e:
    print(f"[~] Advanced routes not loaded: {_e}")

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────

# TEXT MODEL — try both possible save locations
text_model = None
for path in ['models/text_model.pkl', 'model/phishguard_model.pkl']:
    try:
        with open(path, 'rb') as f:
            text_model = pickle.load(f)
        print(f"[✓] Text model loaded from {path}")
        break
    except Exception:
        pass

# Also try loading vectorizer separately (used by older train_model.py)
text_vectorizer = None
for path in ['models/vectorizer.pkl', 'model/vectorizer.pkl']:
    try:
        with open(path, 'rb') as f:
            text_vectorizer = pickle.load(f)
        print(f"[✓] Text vectorizer loaded from {path}")
        break
    except Exception:
        pass

if text_model is None:
    print("[✗] Text model not found — run train_model.py first")

# VOICE MODEL — try both possible save locations
voice_scaler = None
voice_clf    = None
for path in ['models/voice_model.pkl', 'model/voice_model.pkl']:
    try:
        voice_data   = pickle.load(open(path, 'rb'))
        voice_scaler = voice_data['scaler']
        voice_clf    = voice_data['model']
        print(f"[✓] Voice model loaded from {path}")
        break
    except Exception:
        pass

if voice_clf is None:
    try:
        from voice_detector import detect_ai_voice
        print("[✓] Using heuristic voice detector (voice_detector.py)")
        USE_HEURISTIC_VOICE = True
    except Exception as e:
        print(f"[✗] Voice detector not found: {e}")
        USE_HEURISTIC_VOICE = False
else:
    USE_HEURISTIC_VOICE = False

# TEXT ANALYZER — try to load the standalone text_analyzer module
try:
    from text_analyzer import analyze_message
    print("[✓] text_analyzer.py loaded")
    USE_TEXT_ANALYZER = True
except Exception:
    USE_TEXT_ANALYZER = False
    print("[~] text_analyzer.py not found — using built-in text logic")

# ─────────────────────────────────────────
# HELPER — extract confidence % from explanation string
# e.g. "Flagged — ML confidence 86%." → 0.86
# ─────────────────────────────────────────
def _parse_conf(explanation_str):
    try:
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', str(explanation_str))
        if m:
            return round(float(m.group(1)) / 100, 4)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────
# HOME — serves the dashboard UI
# ─────────────────────────────────────────

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception:
        return send_file('templates/index.html')

# ─────────────────────────────────────────
# TEXT DETECTION
# ─────────────────────────────────────────

@app.route('/detect-text', methods=['POST'])
def detect_text():
    try:
        # Accept both JSON and form data
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '').strip()
        else:
            text = request.form.get('text', request.form.get('message', '')).strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # ── Option 1: use text_analyzer.py
        if USE_TEXT_ANALYZER:
            try:
                result = analyze_message(text)

                # text_analyzer returns a TUPLE:
                # (is_phishing, threat, keywords, explanation, patterns)
                # e.g. (np.True_, 'High', ['87121','cup',...], 'Flagged — ML confidence 86%.', ['General Message'])
                if isinstance(result, tuple):
                    is_phishing = bool(result[0])
                    threat      = str(result[1]) if len(result) > 1 else ('High' if is_phishing else 'Low')
                    keywords    = list(result[2]) if len(result) > 2 else []
                    explanation = str(result[3]) if len(result) > 3 else ''
                    pattern     = result[4][0] if len(result) > 4 and result[4] else ''

                    # Try to extract confidence from explanation string
                    conf = _parse_conf(explanation)
                    if conf is None:
                        conf = 0.86 if is_phishing else 0.92  # sensible defaults

                else:
                    # dict response (future-proof)
                    is_phishing = bool(
                        result.get('is_phishing') == True or
                        result.get('prediction') in ['phishing', 'spam'] or
                        result.get('label') in ['phishing', 'spam']
                    )
                    conf        = float(result.get('confidence', result.get('probability', 0.5)))
                    threat      = result.get('threat', result.get('threat_level',
                                    'High' if (is_phishing and conf > 0.7) else
                                    'Medium' if is_phishing else 'Low'))
                    keywords    = result.get('keywords', [])
                    explanation = result.get('explanation', '')
                    pattern     = result.get('pattern', result.get('attack_pattern', ''))

                return jsonify({
                    'is_phishing': is_phishing,
                    'prediction':  'phishing' if is_phishing else 'legitimate',
                    'label':       'phishing' if is_phishing else 'legitimate',
                    'confidence':  conf,
                    'threat':      threat,
                    'explanation': explanation,
                    'keywords':    keywords,
                    'pattern':     pattern
                })

            except Exception as e:
                traceback.print_exc()
                print(f"[!] text_analyzer failed: {e}, falling back to sklearn model")

        # ── Option 2: sklearn pipeline model directly
        if text_model is not None:
            try:
                if text_vectorizer is not None:
                    X          = text_vectorizer.transform([text])
                    prediction = text_model.predict(X)[0]
                    proba      = text_model.predict_proba(X)[0]
                else:
                    prediction = text_model.predict([text])[0]
                    proba      = text_model.predict_proba([text])[0]

                is_phishing = bool(
                    prediction == 1 or
                    str(prediction).lower() in ['phishing', 'spam', '1']
                )
                confidence = float(max(proba))

                phishing_keywords = [
                    'click', 'verify', 'urgent', 'suspended', 'congratulations',
                    'winner', 'prize', 'claim', 'free', 'password', 'login',
                    'account', 'expire', 'immediately', 'limited', 'offer',
                    'confirm', 'blocked', 'kyc', 'otp', 'recharge', 'refund',
                    'aadhaar', 'pan', 'upi', 'bank', 'reward', 'lucky'
                ]
                lower    = text.lower()
                keywords = [kw for kw in phishing_keywords if kw in lower][:6] if is_phishing else []

                return jsonify({
                    'is_phishing': is_phishing,
                    'prediction':  'phishing' if is_phishing else 'legitimate',
                    'label':       'phishing' if is_phishing else 'legitimate',
                    'confidence':  confidence,
                    'threat':      'High' if (is_phishing and confidence > 0.7) else 'Medium' if is_phishing else 'Low',
                    'keywords':    keywords,
                    'pattern':     '',
                    'explanation': (
                        f"Message contains {len(keywords)} phishing indicator(s). "
                        f"Model confidence: {confidence * 100:.1f}%."
                        if is_phishing else
                        f"No significant phishing patterns detected. "
                        f"Model confidence: {confidence * 100:.1f}%."
                    )
                })
            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

        return jsonify({'error': 'Text model not loaded. Run train_model.py first.'}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# /analyze-text alias — works with either endpoint name
@app.route('/analyze-text', methods=['POST'])
def analyze_text_alias():
    return detect_text()

# ─────────────────────────────────────────
# VOICE DETECTION
# ─────────────────────────────────────────

@app.route('/detect-voice', methods=['POST'])
def detect_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided. Use key "audio".'}), 400

        file   = request.files['audio']
        suffix = os.path.splitext(file.filename)[1] or '.wav'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            if USE_HEURISTIC_VOICE:
                from voice_detector import detect_ai_voice
                result = detect_ai_voice(tmp_path)
                return jsonify(result)

            if voice_clf is not None:
                y, sr    = librosa.load(tmp_path, duration=30)
                mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                features = np.mean(mfccs.T, axis=0).reshape(1, -1)

                features_scaled = voice_scaler.transform(features)
                prediction      = voice_clf.predict(features_scaled)[0]
                proba           = voice_clf.predict_proba(features_scaled)[0]

                is_phishing = bool(prediction == 1 or str(prediction).lower() in ['phishing', 'fake', '1'])
                confidence  = float(max(proba))

                return jsonify({
                    'is_phishing': is_phishing,
                    'prediction':  'phishing' if is_phishing else 'legitimate',
                    'label':       'phishing' if is_phishing else 'legitimate',
                    'confidence':  confidence,
                    'threat':      'High' if (is_phishing and confidence > 0.7) else 'Medium' if is_phishing else 'Low',
                    'explanation': (
                        f"Audio shows robotic/scripted speech patterns typical of AI-generated phishing calls. "
                        f"Confidence: {confidence * 100:.1f}%."
                        if is_phishing else
                        f"Audio appears to be natural human speech. No phishing patterns detected. "
                        f"Confidence: {confidence * 100:.1f}%."
                    )
                })

            return jsonify({'error': 'Voice model not loaded.'}), 500

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────
# IMAGE DETECTION
# ─────────────────────────────────────────

@app.route('/detect-image', methods=['POST'])
def detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided. Use key "image".'}), 400

        file   = request.files['image']
        suffix = os.path.splitext(file.filename)[1] or '.jpg'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = detect_phishing_image(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────
# VIDEO DETECTION
# ─────────────────────────────────────────

@app.route('/detect-video', methods=['POST'])
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided. Use key "video".'}), 400

        file   = request.files['video']
        suffix = os.path.splitext(file.filename)[1] or '.mp4'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = detect_phishing_video(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────
# URL ANALYSIS (NEW — V2)
# ─────────────────────────────────────────

@app.route('/analyze-url', methods=['POST'])
def analyze_url_route():
    try:
        if request.is_json:
            data = request.get_json()
            url = data.get('url', '').strip()
            text = data.get('text', '').strip()
        else:
            url = request.form.get('url', '').strip()
            text = request.form.get('text', '').strip()

        if not url and not text:
            return jsonify({'error': 'Provide a URL or text containing URLs.'}), 400

        from url_analyzer import analyze_url, analyze_text_and_url

        if text:
            return jsonify(analyze_text_and_url(text, url))
        else:
            return jsonify(analyze_url(url))

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# REAL-TIME AUDIO CHUNK ANALYSIS (NEW — V2)
# ─────────────────────────────────────────

@app.route('/analyze-audio-chunk', methods=['POST'])
def analyze_audio_chunk():
    """
    Accepts a short audio chunk (from browser MediaRecorder),
    runs voice analysis on it, and returns risk signals.
    For real-time call guardian mode.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio chunk provided'}), 400

        file = request.files['audio']
        suffix = os.path.splitext(file.filename)[1] or '.webm'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # The browser may also send a transcript it produced with the Web
        # Speech API (more reliable in-browser than server-side STT on webm).
        client_transcript = request.form.get('transcript', '').strip()

        try:
            from realtime_scorer import score_transcript, combine_risk

            # ── 1. Get a transcript ──────────────────────────────────
            transcript = client_transcript
            if not transcript:
                # Try server-side STT (needs a WAV; convert webm if needed)
                try:
                    import speech_recognition as sr
                    wav_path = tmp_path
                    if not tmp_path.lower().endswith('.wav'):
                        # Convert to wav via pydub/ffmpeg if available
                        try:
                            from pydub import AudioSegment
                            seg = AudioSegment.from_file(tmp_path)
                            wav_path = tmp_path + '.wav'
                            seg.export(wav_path, format='wav')
                        except Exception:
                            wav_path = tmp_path
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(wav_path) as source:
                        audio_data = recognizer.record(source)
                    transcript = recognizer.recognize_google(audio_data, language='en-IN')
                    if wav_path != tmp_path and os.path.exists(wav_path):
                        os.remove(wav_path)
                except Exception:
                    transcript = ""

            # ── 2. Voice-synthesis analysis (secondary signal) ───────
            voice_result = {}
            voice_synthetic = False
            voice_conf = 0.0
            try:
                if USE_HEURISTIC_VOICE:
                    from voice_detector import detect_ai_voice
                    voice_result = detect_ai_voice(tmp_path)
                    voice_synthetic = voice_result.get('prediction') == 'phishing'
                    voice_conf = voice_result.get('confidence', 0.0)
                elif voice_clf is not None:
                    y, sr_rate = librosa.load(tmp_path, duration=30)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40)
                    features = np.mean(mfccs.T, axis=0).reshape(1, -1)
                    features_scaled = voice_scaler.transform(features)
                    prediction = voice_clf.predict(features_scaled)[0]
                    proba = voice_clf.predict_proba(features_scaled)[0]
                    voice_synthetic = bool(prediction == 1)
                    voice_conf = float(max(proba))
                    voice_result = {'is_phishing': voice_synthetic,
                                    'confidence': voice_conf,
                                    'prediction': 'suspicious' if voice_synthetic else 'legitimate'}
            except Exception:
                pass

            # ── 3. Score the CONTENT (primary driver) ────────────────
            text_score = score_transcript(transcript)
            final = combine_risk(text_score, voice_synthetic, voice_conf)

            return jsonify({
                'transcript': transcript,
                'voice_analysis': voice_result,
                'text_analysis': text_score,
                'combined_risk': final['risk'],
                'risk_level': final['level'],
                'signals': final['signals'],
                'category': final['category'],
            })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# COMBINED TEXT + URL ANALYSIS (NEW — V2)
# ─────────────────────────────────────────

@app.route('/detect-text-url', methods=['POST'])
def detect_text_url():
    """Combined text phishing + URL analysis in one call."""
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '').strip()
            url = data.get('url', '').strip()
        else:
            text = request.form.get('text', '').strip()
            url = request.form.get('url', '').strip()

        if not text and not url:
            return jsonify({'error': 'Provide text and/or URL'}), 400

        results = {}

        # Text analysis
        if text:
            try:
                if USE_TEXT_ANALYZER:
                    result = analyze_message(text)
                    if isinstance(result, tuple):
                        is_phishing = bool(result[0])
                        conf = _parse_conf(str(result[3])) if len(result) > 3 else None
                        if conf is None:
                            conf = 0.86 if is_phishing else 0.92
                        results['text'] = {
                            'is_phishing': is_phishing,
                            'confidence': conf,
                            'threat': str(result[1]) if len(result) > 1 else 'Low',
                            'keywords': list(result[2]) if len(result) > 2 else [],
                            'explanation': str(result[3]) if len(result) > 3 else '',
                            'pattern': result[4][0] if len(result) > 4 and result[4] else '',
                        }
                elif text_model is not None:
                    pred = text_model.predict([text])[0]
                    proba = text_model.predict_proba([text])[0]
                    is_p = bool(pred == 1 or str(pred).lower() in ['phishing', 'spam', '1'])
                    results['text'] = {
                        'is_phishing': is_p,
                        'confidence': float(max(proba)),
                        'threat': 'High' if is_p and max(proba) > 0.7 else 'Medium' if is_p else 'Low',
                    }
            except Exception as e:
                results['text'] = {'error': str(e)}

        # URL analysis
        from url_analyzer import analyze_url as _analyze_url, analyze_text_and_url
        if url or text:
            try:
                url_result = analyze_text_and_url(text, url)
                results['url'] = url_result
            except Exception as e:
                results['url'] = {'error': str(e)}

        # Combined verdict
        text_phish = results.get('text', {}).get('is_phishing', False)
        text_conf = results.get('text', {}).get('confidence', 0)
        url_suspicious = results.get('url', {}).get('any_suspicious', False)
        url_conf = results.get('url', {}).get('highest_risk', 0)

        if text_phish and url_suspicious:
            combined_risk = 'High'
            combined_explanation = 'Both the message text and URL show phishing indicators — this is very likely a scam.'
        elif text_phish:
            combined_risk = 'High' if text_conf > 0.7 else 'Medium'
            combined_explanation = 'The message text contains phishing indicators.'
        elif url_suspicious:
            combined_risk = 'Medium'
            combined_explanation = 'The URL shows suspicious characteristics.'
        else:
            combined_risk = 'Low'
            combined_explanation = 'No significant phishing indicators found.'

        is_threat = text_phish or url_suspicious

        results['combined'] = {
            'is_phishing': is_threat,
            'risk_level': combined_risk,
            'explanation': combined_explanation,
            'confidence': round(max(text_conf if text_phish else 0,
                                    url_conf if url_suspicious else 0, 0.1), 3),
        }

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# SAFETY ADVICE ENGINE (NEW — V2)
# ─────────────────────────────────────────

@app.route('/safety-advice', methods=['POST'])
def safety_advice():
    """Returns threat-specific action guidance based on detection results."""
    try:
        data = request.get_json() if request.is_json else {}
        threat_type = data.get('threat_type', 'general')
        risk_level = data.get('risk_level', 'Medium')

        advice_map = {
            'bank_scam': {
                'title': 'Possible Bank Scam',
                'actions': [
                    'Do NOT share OTP, PIN, CVV, or password with anyone.',
                    'End the call or ignore the message immediately.',
                    'Contact your bank directly using the number on the back of your card.',
                    'Report the number/message to your bank\'s fraud department.',
                    'Never click links in suspicious messages — type the bank URL directly.',
                ],
            },
            'impersonation': {
                'title': 'Possible Impersonation / Deepfake',
                'actions': [
                    'Call the person back using a number you already know.',
                    'Ask a personal question only they would know.',
                    'Do NOT transfer money based on voice/video calls alone.',
                    'Be especially wary of urgent financial requests from "family".',
                    'Report the incident to local cyber-crime authorities.',
                ],
            },
            'malicious_url': {
                'title': 'Suspicious Link Detected',
                'actions': [
                    'Do NOT open the link or enter any credentials.',
                    'Do not download any files from the link.',
                    'If you already clicked, change your passwords immediately.',
                    'Run a security scan on your device.',
                    'Report the URL to your browser\'s safe browsing programme.',
                ],
            },
            'threat_coercion': {
                'title': 'Threatening / Coercive Content',
                'actions': [
                    'Do NOT respond or engage with the sender.',
                    'Save/screenshot the evidence before it disappears.',
                    'Contact a trusted person for support.',
                    'Report to local police or cyber-crime cell.',
                    'Block the sender after preserving evidence.',
                ],
            },
            'general': {
                'title': 'Safety Precautions',
                'actions': [
                    'Do not click suspicious links or share personal information.',
                    'Verify the source independently using official channels.',
                    'Do not share OTP/PIN/password with anyone.',
                    'Contact a trusted person if you are unsure.',
                    'Report suspicious activity to authorities.',
                ],
            },
        }

        advice = advice_map.get(threat_type, advice_map['general'])
        return jsonify(advice)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# SCORE TRANSCRIPT (NEW — V3)
# Used when the browser does its own speech-to-text (Web Speech API)
# and sends the text for scam-content scoring. Fast + reliable.
# ─────────────────────────────────────────

@app.route('/score-transcript', methods=['POST'])
def score_transcript_route():
    try:
        data = request.get_json() if request.is_json else request.form
        text = (data.get('transcript') or data.get('text') or '').strip()
        if not text:
            return jsonify({'combined_risk': 0.0, 'risk_level': 'LOW',
                            'signals': [], 'category': 'general', 'transcript': ''})

        from realtime_scorer import score_transcript
        result = score_transcript(text)

        # Also blend in the ML text model if available (belt and braces)
        try:
            if text_model is not None:
                proba = text_model.predict_proba([text])[0]
                ml_phish = float(proba[1]) if len(proba) > 1 else 0.0
                # Take the stronger of keyword-score and ML score
                result['risk'] = round(max(result['risk'], ml_phish * 0.7), 3)
                if result['risk'] >= 0.75:
                    result['level'] = 'DANGER'
                elif result['risk'] >= 0.55:
                    result['level'] = 'HIGH'
                elif result['risk'] >= 0.30:
                    result['level'] = 'SUSPICIOUS'
                else:
                    result['level'] = 'LOW'
        except Exception:
            pass

        return jsonify({
            'transcript': text,
            'combined_risk': result['risk'],
            'risk_level': result['level'],
            'signals': result['signals'],
            'category': result['category'],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# SCREEN SCAN ANALYSIS (NEW — V3)
# Accepts a screenshot frame + any OCR text the browser extracted,
# runs image detection AND text scam-scoring, returns a clear verdict.
# ─────────────────────────────────────────

@app.route('/analyze-screen', methods=['POST'])
def analyze_screen():
    try:
        ocr_text = request.form.get('text', '').strip()
        image_result = {}
        image_phish = False
        image_conf = 0.0

        # Image analysis (if a frame was sent)
        if 'image' in request.files:
            file = request.files['image']
            suffix = os.path.splitext(file.filename)[1] or '.jpg'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                image_result = detect_phishing_image(tmp_path)
                image_phish = (image_result.get('is_phishing') or
                               image_result.get('prediction') == 'phishing')
                image_conf = image_result.get('confidence', 0.0)
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Text/OCR scam scoring (if the browser extracted on-screen text)
        text_score = {'risk': 0.0, 'level': 'LOW', 'signals': [], 'category': 'general'}
        if ocr_text:
            from realtime_scorer import score_transcript
            text_score = score_transcript(ocr_text)

        # Combine: whichever signal is stronger drives the verdict
        img_risk = image_conf if image_phish else 0.0
        combined = max(img_risk, text_score['risk'])
        signals = list(text_score.get('signals', []))
        if image_phish:
            signals.insert(0, 'Suspicious visual content on screen (fake page / deepfake indicators)')

        if combined >= 0.7:
            level = 'HIGH'
        elif combined >= 0.45:
            level = 'SUSPICIOUS'
        else:
            level = 'LOW'

        category = text_score.get('category', 'general')
        if category == 'general' and image_phish:
            category = 'malicious_url'

        return jsonify({
            'image_analysis': image_result,
            'text_analysis': text_score,
            'combined_risk': round(combined, 3),
            'risk_level': level,
            'signals': signals,
            'category': category,
            'is_scam': combined >= 0.45,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)