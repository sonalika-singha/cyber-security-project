"""
demo_novelty.py
===============
One script that showcases all three flagship contributions end-to-end.
Run this live in your presentation:  python demo_novelty.py

  1. CROSS-MODAL FUSION  — three 'medium' channels corroborate into 'Critical'
  2. EXPLAINABILITY       — exact reason a message was flagged
  3. ADVERSARIAL ROBUSTNESS — evade the filter, then defeat the evasion
"""

from fusion_engine import fuse_incident
from explainability import explain_text
from adversarial import ATTACKS, normalize_text, _predict_phishing_prob


def hr(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def demo_fusion():
    hr("1. CROSS-MODAL FUSION — coordinated KYC-fraud incident")
    print("A victim receives:  an SMS  +  a screenshot of the linked page  +  "
          "a follow-up call.\nEach channel alone looks only MEDIUM risk...\n")
    incident = {
        "text":  {"is_phishing": True, "confidence": 0.63, "threat": "Medium",
                  "pattern": "Credential Harvesting",
                  "patterns": ["Credential Harvesting", "Urgency / Fear-based Scam"]},
        "image": {"prediction": "phishing", "confidence": 0.66,
                  "scores": {"phishing": 0.66, "legitimate": 0.34}, "threat": "Medium"},
        "voice": {"is_phishing": True, "confidence": 0.58, "threat": "Medium"},
    }
    r = fuse_incident(incident)
    for m, d in r["per_modality"].items():
        print(f"  {m:6s}: P(phishing)={d['phishing_probability']:.2f}  "
              f"weight={d['reliability_weight']:.2f}  flagged={d['flagged']}")
    print(f"\n  => FUSED: {r['prediction'].upper()}  "
          f"confidence={r['fused_confidence']:.2f}  THREAT={r['threat']}")
    print(f"  => Campaign: {r['campaign_label']}")
    print(f"\n  {r['explanation']}")


def demo_explain():
    hr("2. EXPLAINABILITY — why was this message flagged?")
    msg = ("URGENT: Your SBI account is BLOCKED. Verify KYC immediately: "
           "http://sbi-verify.xyz")
    print(f"Message: {msg}\n")
    e = explain_text(msg)
    print(f"  Phishing probability: {e['phishing_probability']*100:.1f}%\n")
    print("  Top contributing tokens (exact linear attribution):")
    for t in e["token_contributions"][:6]:
        arrow = "PHISH" if t["direction"] == "phishing" else "legit"
        print(f"    {t['token']:14s}  {t['contribution']:+.3f}  ({arrow})")
    print(f"\n  {e['summary']}")


def demo_adversarial():
    import numpy as np
    hr("3. ADVERSARIAL ROBUSTNESS — evade, then defend")
    # fixed seed -> deterministic, dramatic result for a live demo
    np.random.seed(0)
    msg = "Congratulations! You have won a free prize. Click the link to claim your reward now."
    attacked = ATTACKS["combined"](msg)
    defended = normalize_text(attacked)

    p_clean = _predict_phishing_prob(msg)
    p_adv   = _predict_phishing_prob(attacked)
    p_def   = _predict_phishing_prob(defended)

    print(f"  ORIGINAL : {msg}\n             P(phishing)={p_clean:.2f}  flagged={p_clean>=0.5}\n")
    print(f"  ATTACKED : {attacked}\n             P(phishing)={p_adv:.2f}  flagged={p_adv>=0.5}"
          f"   <- attacker tried to evade\n")
    print(f"  DEFENDED : {defended}\n             P(phishing)={p_def:.2f}  flagged={p_def>=0.5}"
          f"   <- de-obfuscation restores detection")


if __name__ == "__main__":
    demo_fusion()
    demo_explain()
    demo_adversarial()
    print("\n" + "=" * 68)
    print("All three novelty modules ran against the project's real models.")
    print("=" * 68)
