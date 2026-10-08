"""
adversarial.py
==============
ADVERSARIAL ROBUSTNESS TOOLKIT for PhishGuard AI.

Real attackers deliberately obfuscate phishing text to slip past filters. Your
own literature-review table lists an "adversarially robust system" as future
scope — this module delivers it, and gives you a memorable live demo:

    1. Take a phishing message the model correctly flags.
    2. Apply an evasion attack (homoglyphs, zero-width chars, leetspeak...).
    3. Show the UNDEFENDED model now MISSES it (evasion succeeded).
    4. Run the same input through the DEFENSE (de-obfuscation normaliser).
    5. Show the model catches it again.

Contents
--------
Attacks (attacker's perspective):
    homoglyph_attack, zero_width_attack, leetspeak_attack,
    spacing_attack, combined_attack
Defense (our perspective):
    normalize_text  — folds look-alikes, strips invisibles, undoes leetspeak
Evaluation:
    evaluate_robustness — measures clean vs. attacked vs. defended accuracy
"""

import re
import unicodedata

import numpy as np


# ---------------------------------------------------------------------------
# Homoglyph map: visually identical look-alikes from other Unicode scripts.
# "pаypal" with a Cyrillic 'а' looks identical to "paypal" but is a different
# string, so a naive classifier treats it as an unknown token.
# ---------------------------------------------------------------------------
LATIN_TO_HOMOGLYPH = {
    "a": "а",  # Cyrillic a  (U+0430)
    "e": "е",  # Cyrillic e  (U+0435)
    "o": "о",  # Cyrillic o  (U+043E)
    "p": "р",  # Cyrillic er (U+0440)
    "c": "с",  # Cyrillic es (U+0441)
    "x": "х",  # Cyrillic ha (U+0445)
    "y": "у",  # Cyrillic u  (U+0443)
    "i": "і",  # Cyrillic i  (U+0456)
}
HOMOGLYPH_TO_LATIN = {v: k for k, v in LATIN_TO_HOMOGLYPH.items()}

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "g": "9"}
LEET_REVERSE = {v: k for k, v in LEET_MAP.items()}

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]  # invisible characters


# ===========================================================================
# ATTACKS  (attacker's perspective — for building the test set)
# ===========================================================================
def homoglyph_attack(text, rate=0.3):
    out = []
    for ch in text:
        low = ch.lower()
        if low in LATIN_TO_HOMOGLYPH and np.random.rand() < rate:
            g = LATIN_TO_HOMOGLYPH[low]
            out.append(g.upper() if ch.isupper() else g)
        else:
            out.append(ch)
    return "".join(out)


def zero_width_attack(text, rate=0.15):
    """Insert invisible characters inside words to break token matching."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and np.random.rand() < rate:
            out.append(np.random.choice(ZERO_WIDTH))
    return "".join(out)


def leetspeak_attack(text, rate=0.4):
    out = []
    for ch in text:
        low = ch.lower()
        if low in LEET_MAP and np.random.rand() < rate:
            out.append(LEET_MAP[low])
        else:
            out.append(ch)
    return "".join(out)


def spacing_attack(text, rate=0.2):
    """Insert spaces inside words: 'verify' -> 'ver ify'."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and np.random.rand() < rate:
            out.append(" ")
    return "".join(out)


def combined_attack(text):
    """Stack several evasion techniques — the realistic worst case."""
    t = homoglyph_attack(text, rate=0.25)
    t = leetspeak_attack(t, rate=0.25)
    t = zero_width_attack(t, rate=0.1)
    return t


ATTACKS = {
    "homoglyph":  homoglyph_attack,
    "zero_width": zero_width_attack,
    "leetspeak":  leetspeak_attack,
    "spacing":    spacing_attack,
    "combined":   combined_attack,
}


# ===========================================================================
# DEFENSE  (our perspective — the de-obfuscation normaliser)
# ===========================================================================
def normalize_text(text):
    """
    Undo common obfuscations before classification:
      * strip zero-width / invisible characters
      * NFKC-normalise and fold homoglyphs back to Latin
      * reverse leetspeak substitutions
      * collapse intra-word spacing artefacts
    This is a preprocessing DEFENSE — call it before the text model.
    """
    # 1) remove invisible characters
    for zw in ZERO_WIDTH:
        text = text.replace(zw, "")

    # 2) unicode normalise + fold homoglyphs
    text = unicodedata.normalize("NFKC", text)
    text = "".join(HOMOGLYPH_TO_LATIN.get(ch, ch) for ch in text)

    # 3) reverse leetspeak, but only inside alphabetic contexts so we don't
    #    destroy legitimate numbers (amounts, OTPs, phone numbers).
    def deleet(match):
        w = match.group(0)
        return "".join(LEET_REVERSE.get(c, c) for c in w)
    text = re.sub(r"\b[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*\b", deleet, text)

    # 4) collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================================================================
# EVALUATION HARNESS
# ===========================================================================
def _predict_phishing_prob(message):
    """P(phishing) from the project's real text model."""
    import joblib
    global _M, _V
    try:
        _M
    except NameError:
        _M = _V = None
    if _M is None:
        for m, v in (("model/phishguard_model.pkl", "model/vectorizer.pkl"),
                     ("models/text_model.pkl", "models/vectorizer.pkl")):
            try:
                _M = joblib.load(m); _V = joblib.load(v); break
            except Exception:
                continue
    return float(_M.predict_proba(_V.transform([message]))[0][1])


def evaluate_robustness(phishing_messages, attack="combined", threshold=0.5, seed=42):
    """
    Measure detection rate on phishing messages that are:
      (a) clean, (b) attacked, (c) attacked-then-defended.

    Returns a dict of detection rates + per-message detail. A large gap between
    (a) and (b), closed by (c), is the headline robustness result.
    """
    np.random.seed(seed)
    attack_fn = ATTACKS[attack]

    clean_hits = attacked_hits = defended_hits = 0
    detail = []
    for msg in phishing_messages:
        p_clean = _predict_phishing_prob(msg)
        adv = attack_fn(msg)
        p_adv = _predict_phishing_prob(adv)
        p_def = _predict_phishing_prob(normalize_text(adv))

        c = p_clean >= threshold
        a = p_adv >= threshold
        d = p_def >= threshold
        clean_hits += c; attacked_hits += a; defended_hits += d
        detail.append({
            "message":        msg[:60] + ("..." if len(msg) > 60 else ""),
            "clean_prob":     round(p_clean, 3),
            "attacked_prob":  round(p_adv, 3),
            "defended_prob":  round(p_def, 3),
        })

    n = max(len(phishing_messages), 1)
    return {
        "attack":                 attack,
        "n_messages":             n,
        "clean_detection_rate":   round(clean_hits / n, 3),
        "attacked_detection_rate": round(attacked_hits / n, 3),
        "defended_detection_rate": round(defended_hits / n, 3),
        "evasion_success_rate":   round((clean_hits - attacked_hits) / n, 3),
        "defense_recovery_rate":  round((defended_hits - attacked_hits) / n, 3),
        "detail":                 detail,
    }


# ===========================================================================
if __name__ == "__main__":
    from pprint import pprint

    samples = [
        "URGENT: Your SBI account is blocked. Verify KYC immediately to avoid suspension.",
        "Congratulations! You won a prize. Click here to claim your reward now.",
        "Your account has been suspended. Please verify your password immediately.",
        "Dear customer, your KYC update is pending. Update now to avoid deactivation.",
        "Your debit card is blocked. Confirm your OTP to reactivate immediately.",
    ]

    print("=== Example of one attacked message ===")
    print("original :", samples[0])
    print("attacked :", combined_attack(samples[0]))
    print("defended :", normalize_text(combined_attack(samples[0])))

    for atk in ("homoglyph", "leetspeak", "combined"):
        print(f"\n=== Robustness under '{atk}' attack ===")
        r = evaluate_robustness(samples, attack=atk)
        print(f"clean    detection rate: {r['clean_detection_rate']*100:.0f}%")
        print(f"attacked detection rate: {r['attacked_detection_rate']*100:.0f}%  "
              f"(evasion success {r['evasion_success_rate']*100:.0f}%)")
        print(f"defended detection rate: {r['defended_detection_rate']*100:.0f}%  "
              f"(defense recovered {r['defense_recovery_rate']*100:.0f}%)")
