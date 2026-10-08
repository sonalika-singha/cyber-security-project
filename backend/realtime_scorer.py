"""
realtime_scorer.py
==================
Reliable scam-risk scoring for real-time call/screen analysis.

The problem with pure voice-AI detection: a real human scammer's voice
won't trigger a "synthetic voice" detector, so an obvious scam call would
show LOW risk. This module scores based on WHAT IS BEING SAID (the scam
content), which is the real signal, combined with voice synthesis as a
secondary signal.
"""

import re

# ── High-signal scam phrases (strong indicators) ──────────────────────
CRITICAL_PHRASES = [
    r'\botp\b', r'one[\s-]?time[\s-]?password', r'\bpin\b', r'\bcvv\b',
    r'share.{0,20}(otp|pin|password|code)', r'tell me.{0,20}(otp|pin|code)',
    r'read.{0,15}(otp|code|number)', r'password',
]

HIGH_PHRASES = [
    r'transfer.{0,20}(money|rupees|rs|inr|amount|\u20b9|\$)',
    r'send.{0,20}(money|rupees|payment)', r'pay.{0,20}(now|immediately|fee)',
    r'account.{0,20}(blocked|suspended|frozen|compromised|hacked)',
    r'verify.{0,20}(account|identity|kyc|details|card)',
    r'kyc.{0,20}(update|expire|pending|verify)', r'kyc has expired',
    r'refund.{0,20}(pending|process|claim)', r'lottery|you.{0,10}won|prize|reward',
    r'arrest|legal action|police|court|fir', r'gift card|google play card',
    r'bitcoin|crypto|usdt', r'remote.{0,10}access|anydesk|teamviewer',
    r'gpay|phonepe|paytm.{0,20}(transfer|send)',
]

MEDIUM_PHRASES = [
    r'urgent|immediately|right now|hurry|quickly|expire',
    r'limited time|act now|last chance|final notice',
    r'click.{0,15}(link|here|below)', r'suspicious activity',
    r'confirm.{0,15}(details|identity|information)',
    r'bank|customer care|helpline|support team',
    r'update.{0,15}(details|information|kyc)',
    r'your.{0,10}(card|account|number)', r'security.{0,10}(reason|purpose|check)',
]


def score_transcript(text: str) -> dict:
    """
    Score a transcript for scam risk based on content.
    Returns risk (0-1), level, matched signals, and category.
    """
    if not text or not text.strip():
        return {"risk": 0.0, "level": "LOW", "signals": [], "category": "general"}

    lower = text.lower()
    score = 0.0
    signals = []
    category = "general"

    # Critical phrases (OTP/PIN/password) — the biggest red flags
    for pat in CRITICAL_PHRASES:
        if re.search(pat, lower):
            score += 0.80
            signals.append("Request for OTP/PIN/password")
            category = "bank_scam"
            break

    # High-signal phrases
    high_hits = 0
    for pat in HIGH_PHRASES:
        if re.search(pat, lower):
            high_hits += 1
            if re.search(r'transfer|send.{0,20}money|pay', pat):
                signals.append("Financial/money-transfer request")
                if category == "general":
                    category = "bank_scam"
            elif re.search(r'account|verify|kyc', pat):
                signals.append("Account/KYC verification pressure")
                if category == "general":
                    category = "bank_scam"
            elif re.search(r'arrest|legal|police', pat):
                signals.append("Threat of legal action / arrest")
                category = "threat_coercion"
            elif re.search(r'lottery|won|prize|reward', pat):
                signals.append("Lottery / prize bait")
                if category == "general":
                    category = "bank_scam"
            elif re.search(r'remote|anydesk|teamviewer', pat):
                signals.append("Request for remote device access")
                category = "bank_scam"
    score += min(high_hits * 0.40, 0.75)

    # Medium-signal phrases (urgency, generic pressure)
    med_hits = 0
    for pat in MEDIUM_PHRASES:
        if re.search(pat, lower):
            med_hits += 1
    if med_hits >= 1:
        score += min(med_hits * 0.08, 0.30)
        if med_hits >= 2 and "Urgency / pressure tactics" not in signals:
            signals.append("Urgency / pressure tactics")

    score = min(score, 1.0)

    # De-duplicate signals preserving order
    seen = set()
    signals = [s for s in signals if not (s in seen or seen.add(s))]

    if score >= 0.75:
        level = "DANGER"
    elif score >= 0.55:
        level = "HIGH"
    elif score >= 0.30:
        level = "SUSPICIOUS"
    else:
        level = "LOW"

    return {
        "risk": round(score, 3),
        "level": level,
        "signals": signals,
        "category": category,
    }


def combine_risk(text_score: dict, voice_is_synthetic: bool = False,
                 voice_conf: float = 0.0) -> dict:
    """
    Combine transcript-content risk with voice-synthesis signal.
    Content is the primary driver; synthetic voice adds to it.
    """
    risk = text_score.get("risk", 0.0)
    signals = list(text_score.get("signals", []))
    category = text_score.get("category", "general")

    # A synthetic/cloned voice is itself suspicious — boost the score
    if voice_is_synthetic and voice_conf >= 0.35:
        risk = min(risk + 0.25, 1.0)
        signals.append("Possible AI-generated / cloned voice")
        if category == "general":
            category = "impersonation"

    if risk >= 0.75:
        level = "DANGER"
    elif risk >= 0.55:
        level = "HIGH"
    elif risk >= 0.30:
        level = "SUSPICIOUS"
    else:
        level = "LOW"

    return {
        "risk": round(risk, 3),
        "level": level,
        "signals": signals,
        "category": category,
    }
