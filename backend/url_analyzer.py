"""
url_analyzer.py
===============
Lightweight URL reputation / phishing-indicator analyser.
No external API keys required — uses heuristic + pattern analysis.

Returns a dict with:
  is_suspicious, risk_level, confidence, indicators[], explanation
"""

import re
import math
from urllib.parse import urlparse

# ── Known suspicious TLDs ──────────────────────────────────────────────
SUSPICIOUS_TLDS = {
    '.xyz', '.top', '.club', '.work', '.click', '.loan', '.gq', '.cf',
    '.tk', '.ml', '.ga', '.buzz', '.icu', '.cam', '.rest', '.surf',
    '.monster', '.cyou', '.pw', '.cc', '.ws', '.bid', '.stream',
    '.racing', '.download', '.win', '.date', '.trade', '.accountant',
    '.science', '.party', '.review', '.cricket', '.faith', '.zip',
    '.mov', '.nexus',
}

# ── Brands attackers commonly impersonate ──────────────────────────────
BRAND_KEYWORDS = [
    'sbi', 'hdfc', 'icici', 'axis', 'paytm', 'phonepe', 'gpay',
    'paypal', 'apple', 'google', 'microsoft', 'amazon', 'netflix',
    'facebook', 'instagram', 'whatsapp', 'telegram', 'razorpay',
    'flipkart', 'myntra', 'aadhaar', 'pan-card', 'uidai', 'irctc',
    'wells-fargo', 'chase', 'citibank', 'hsbc', 'barclays',
]

# ── Suspicious path/param patterns ─────────────────────────────────────
PHISH_PATH_PATTERNS = [
    r'login', r'verify', r'secure', r'account', r'confirm', r'update',
    r'signin', r'auth', r'password', r'reset', r'unlock', r'suspend',
    r'kyc', r'otp', r'reward', r'claim', r'prize', r'wallet',
    r'banking', r'refund', r'aadhaar', r'pan',
]

# ── URL shortener domains ─────────────────────────────────────────────
SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd', 'buff.ly',
    'ow.ly', 'rebrand.ly', 'cutt.ly', 'shorturl.at', 'rb.gy',
}


def analyze_url(url: str) -> dict:
    """
    Analyse a single URL for phishing indicators.
    Returns a dict suitable for direct JSON serialisation.
    """
    url = url.strip()
    if not url:
        return {"error": "No URL provided"}

    # Normalise — add scheme if missing so urlparse works
    parse_url = url
    if not re.match(r'https?://', url, re.I):
        parse_url = 'http://' + url

    try:
        parsed = urlparse(parse_url)
    except Exception:
        return {
            "url": url, "is_suspicious": True, "risk_level": "High",
            "confidence": 0.85, "indicators": ["Malformed URL"],
            "explanation": "The URL could not be parsed — likely malformed or crafted to confuse."
        }

    domain = (parsed.hostname or '').lower()
    path = (parsed.path or '').lower()
    query = (parsed.query or '').lower()
    full_lower = parse_url.lower()

    indicators = []
    score = 0.0  # cumulative suspicion score

    # 1. IP address instead of domain
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain):
        indicators.append("Uses raw IP address instead of domain name")
        score += 0.30

    # 2. Suspicious TLD
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            indicators.append(f"Suspicious top-level domain: {tld}")
            score += 0.20
            break

    # 3. Excessive subdomains (>3 levels)
    parts = domain.split('.')
    if len(parts) > 3:
        indicators.append(f"Unusually many subdomains ({len(parts)} levels)")
        score += 0.15

    # 4. Domain length
    if len(domain) > 40:
        indicators.append(f"Very long domain name ({len(domain)} chars)")
        score += 0.10

    # 5. Hyphens in domain (common in phishing)
    hyphen_count = domain.count('-')
    if hyphen_count >= 2:
        indicators.append(f"Multiple hyphens in domain ({hyphen_count})")
        score += 0.15

    # 6. Brand impersonation
    for brand in BRAND_KEYWORDS:
        if brand in domain and not domain.endswith(f'{brand}.com') and not domain.endswith(f'{brand}.in') and not domain.endswith(f'{brand}.co.in'):
            indicators.append(f"Possible brand impersonation: '{brand}' in domain")
            score += 0.30
            break

    # 7. Suspicious path keywords
    path_hits = [kw for kw in PHISH_PATH_PATTERNS if re.search(kw, path + '?' + query)]
    if path_hits:
        indicators.append(f"Suspicious path keywords: {', '.join(path_hits[:4])}")
        score += 0.10 * min(len(path_hits), 3)

    # 8. URL shortener
    if domain in SHORTENERS:
        indicators.append("Uses URL shortener (hides real destination)")
        score += 0.20

    # 9. @ symbol in URL (credential smuggling)
    if '@' in parse_url:
        indicators.append("Contains '@' — may redirect to a different host")
        score += 0.25

    # 10. HTTPS missing
    if parsed.scheme != 'https':
        indicators.append("No HTTPS — connection is not encrypted")
        score += 0.10

    # 11. Homoglyph / look-alike characters in domain
    lookalikes = set('аеіоруАЕІОРУ')  # Cyrillic look-alikes
    if any(c in lookalikes for c in domain):
        indicators.append("Domain contains look-alike (homoglyph) characters")
        score += 0.35

    # 12. Double extension tricks
    if re.search(r'\.\w{2,4}\.\w{2,4}/', full_lower):
        double_ext = re.search(r'(\.\w{2,4}\.\w{2,4}/)', full_lower)
        if double_ext:
            indicators.append(f"Possible double-extension trick: {double_ext.group(1)}")
            score += 0.15

    # ── Clamp and convert to confidence ────────────────────────────────
    score = min(score, 1.0)
    # Apply a sigmoid-like scaling for more nuanced scores
    confidence = 1 / (1 + math.exp(-6 * (score - 0.3)))
    confidence = round(confidence, 3)

    is_suspicious = confidence >= 0.5

    if confidence >= 0.8:
        risk_level = "High"
    elif confidence >= 0.5:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if indicators:
        explanation = f"Found {len(indicators)} suspicious indicator(s): {'; '.join(indicators[:3])}."
    else:
        explanation = "No obvious phishing indicators found in the URL structure."

    return {
        "url": url,
        "domain": domain,
        "is_suspicious": is_suspicious,
        "risk_level": risk_level,
        "confidence": confidence,
        "indicators": indicators,
        "explanation": explanation,
    }


def analyze_text_and_url(text: str, url: str = "") -> dict:
    """
    Combined analysis: extract URLs from text, analyse them alongside
    the text content. Returns unified result.
    """
    # Extract URLs from text
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?'
    found_urls = re.findall(url_pattern, text)

    # Add explicit URL if provided
    if url and url.strip():
        found_urls.append(url.strip())

    # De-duplicate
    found_urls = list(dict.fromkeys(found_urls))

    url_results = [analyze_url(u) for u in found_urls[:5]]  # cap at 5

    return {
        "urls_found": len(found_urls),
        "urls_analyzed": url_results,
        "any_suspicious": any(r.get("is_suspicious") for r in url_results),
        "highest_risk": max((r.get("confidence", 0) for r in url_results), default=0),
    }
