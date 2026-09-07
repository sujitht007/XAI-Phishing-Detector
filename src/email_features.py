import re

EMAIL_FEATURE_COLUMNS = [
    "subject_length",
    "body_length",
    "num_links",
    "sender_domain_age_days",
    "has_suspicious_words",
    "has_urgent_words",
    "num_uppercase_words",
    "sender_is_free_email",
    "has_html_tags",
    "has_money_mention",
    "is_trusted_sender",
]

# Columns for sender-only prediction (used when only an email address is provided)
EMAIL_SENDER_FEATURE_COLUMNS = [
    "sender_domain_age_days",
    "sender_is_free_email",
    "is_trusted_sender",
    "local_part_length",
    "local_has_digits",
    "local_has_special",
    "domain_has_hyphen",
    "domain_length",
]

# Columns specifically for email-address-only analysis pipeline
EMAIL_ADDRESS_FEATURE_COLUMNS = [
    "local_part_length",
    "local_has_digits",
    "local_has_special",
    "local_has_plus_tag",
    "local_has_many_dots",
    "domain_length",
    "domain_has_hyphen",
    "domain_has_digits",
    "sender_is_free_email",
    "is_trusted_sender",
    "sender_domain_age_days",
    "has_invalid_chars",
    "has_very_long_local",
]

SUSPICIOUS_WORDS = [
    "login", "verify", "password", "bank", "paypal", "confirm", "account",
    "suspend", "unusual", "security", "update", "credential", "invoice",
    "payment", "refund", "bitcoin", "wallet", "ssn", "social security",
]

URGENT_WORDS = [
    "urgent", "immediately", "asap", "act now", "expire", "deadline",
    "within 24 hours", "final notice", "last chance", "warning",
]

FREE_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "proton.me",
    "icloud.com", "aol.com", "mail.com", "yandex.com",
]

TRUSTED_SENDER_SUFFIXES = [
    "ac.in", "edu", "edu.in", "gov.in", "gov", "gov.uk", "ac.uk", "nic.in",
]


def _extract_sender_domain(sender: str) -> str:
    sender = (sender or "").strip().lower()
    match = re.search(r"@[\w\.-]+", sender)
    if not match:
        return ""
    return match.group(0)[1:]


def _is_trusted_sender_domain(domain: str) -> int:
    if not domain:
        return 0
    for suffix in sorted(TRUSTED_SENDER_SUFFIXES, key=len, reverse=True):
        if domain == suffix or domain.endswith(f".{suffix}"):
            return 1
    return 0


def _count_links(text: str) -> int:
    return len(re.findall(r"https?://[^\s<>\"']+", text, flags=re.IGNORECASE))


def _has_word_list(text: str, words: list[str]) -> int:
    lowered = text.lower()
    for word in words:
        if word in lowered:
            return 1
    return 0


def _count_uppercase_words(text: str) -> int:
    return sum(1 for word in re.findall(r"\b[A-Z]{2,}\b", text))


def _estimate_domain_age_from_sender(domain: str) -> int:
    """Deterministic, bounded estimate of sender domain 'age' to help model.

    This mirrors the URL domain age heuristic but is conservative and deterministic.
    """
    if not domain:
        return 0
    seed = sum(ord(c) for c in domain) % 10000
    # Slight boost for institutional/trusted suffixes
    for suffix in TRUSTED_SENDER_SUFFIXES:
        if domain == suffix or domain.endswith(f".{suffix}"):
            return 500 + (seed % 1500)
    # Generic spread
    return 200 + (seed % 2000)


def is_clearly_legitimate_email(features: dict) -> bool:
    # Primary: trusted institutional sender with no red flags
    if (
        features.get("is_trusted_sender") == 1
        and features.get("has_suspicious_words") == 0
        and features.get("has_urgent_words") == 0
        and features.get("num_links") <= 2
    ):
        return True

    # Secondary: non-free email domains (corporate/institutional) with no red flags
    if (
        features.get("sender_is_free_email") == 0
        and features.get("has_suspicious_words") == 0
        and features.get("has_urgent_words") == 0
        and features.get("num_links") <= 1
    ):
        return True

    return False


def extract_features_from_email(subject: str, sender: str, body: str) -> dict:
    subject = (subject or "").strip()
    sender = (sender or "").strip()
    body = (body or "").strip()
    combined = f"{subject}\n{body}"

    sender_domain = _extract_sender_domain(sender)
    has_suspicious_words = _has_word_list(combined, SUSPICIOUS_WORDS)
    has_urgent_words = _has_word_list(combined, URGENT_WORDS)
    is_trusted_sender = _is_trusted_sender_domain(sender_domain)
    sender_domain_age = _estimate_domain_age_from_sender(sender_domain)

    return {
        "subject_length": len(subject),
        "body_length": len(body),
        "num_links": _count_links(combined),
        "sender_domain_age_days": sender_domain_age,
        "has_suspicious_words": has_suspicious_words,
        "has_urgent_words": has_urgent_words,
        "num_uppercase_words": _count_uppercase_words(combined),
        "sender_is_free_email": 1 if sender_domain in FREE_EMAIL_DOMAINS else 0,
        "has_html_tags": 1 if re.search(r"<[a-z][\s\S]*>", body, re.IGNORECASE) else 0,
        "has_money_mention": 1 if re.search(r"(\$|€|£|rs\.?\s*\d|usd|inr|payment due)", combined, re.IGNORECASE) else 0,
        "is_trusted_sender": is_trusted_sender,
    }


def extract_features_from_sender(sender: str) -> dict:
    """Extract a compact set of features from a sender email address only.

    This is used when only the email address is provided (no subject/body).
    """
    sender = (sender or "").strip()
    domain = _extract_sender_domain(sender)
    local_part = sender.split("@")[0] if "@" in sender else sender
    sender_domain_age = _estimate_domain_age_from_sender(domain)
    sender_is_free = 1 if domain in FREE_EMAIL_DOMAINS else 0
    is_trusted = _is_trusted_sender_domain(domain)
    local_has_digits = 1 if re.search(r"\d", local_part) else 0
    local_has_special = 1 if re.search(r"[^A-Za-z0-9._+-]", local_part) else 0
    domain_has_hyphen = 1 if "-" in domain else 0
    domain_length = len(domain)

    return {
        "sender_domain_age_days": sender_domain_age,
        "sender_is_free_email": sender_is_free,
        "is_trusted_sender": is_trusted,
        "local_part_length": len(local_part),
        "local_has_digits": local_has_digits,
        "local_has_special": local_has_special,
        "domain_has_hyphen": domain_has_hyphen,
        "domain_length": domain_length,
    }


def heuristic_sender_check(sender: str) -> dict:
    """Simple rule-based check returning prediction, confidence, explanation and features.

    Rules (priority):
    1. Missing/invalid domain -> Phishing (0.99)
    2. Trusted institutional domain -> Legit (0.99)
    3. Free email domains (gmail, yahoo, etc.) -> Phishing (0.85)
    4. Domain contains hyphen or short domain with digits in local part -> Phishing (0.80)
    5. Otherwise -> Legit (0.90)
    """
    sender = (sender or "").strip()
    features = extract_features_from_sender(sender)
    domain = _extract_sender_domain(sender)

    if not domain:
        return {
            "prediction": "Phishing",
            "confidence": 0.99,
            "explanation": "No valid sender domain detected; marked as Phishing by heuristic.",
            "features": features,
        }

    if features.get("is_trusted_sender") == 1:
        return {
            "prediction": "Legit",
            "confidence": 0.99,
            "explanation": "Trusted institutional sender domain detected.",
            "features": features,
        }

    if features.get("sender_is_free_email") == 1:
        return {
            "prediction": "Phishing",
            "confidence": 0.85,
            "explanation": "Sender uses a free email provider which is treated as higher risk by this heuristic.",
            "features": features,
        }

    if features.get("domain_has_hyphen") == 1 or (
        features.get("local_has_digits") == 1 and features.get("local_part_length", 0) < 6
    ):
        return {
            "prediction": "Phishing",
            "confidence": 0.80,
            "explanation": "Sender exhibits suspicious formatting (hyphen/short numeric local part).",
            "features": features,
        }

    return {
        "prediction": "Legit",
        "confidence": 0.90,
        "explanation": "No heuristic phishing indicators found; treated as Legit.",
        "features": features,
    }


def analyze_email_address(sender: str) -> dict:
    """Analyze a single email address string and return a structured result.

    Returns labels from the set: "Invalid Email Address", "Suspicious Email Address",
    "Likely Legitimate Email Address" along with a confidence [0..1], explanation, and features.
    This is a deterministic, local-only pipeline and does not require message content.
    """
    sender = (sender or "").strip()

    # Basic syntactic validation
    # Accept typical forms: local@domain
    if not sender or "@" not in sender:
        return {
            "label": "Invalid Email Address",
            "confidence": 0.99,
            "explanation": "Missing `@` or empty input; not a valid email address.",
            "features": {},
        }

    local, domain = sender.split("@", 1)
    # Basic character checks
    has_invalid_chars = 1 if re.search(r"[\s\\(),:;<>\[\]\\\"]", sender) else 0

    # Domain must look like domain.tld (at least one dot and tld letters)
    domain_valid = bool(re.search(r"[a-z0-9\-]+\.[a-z]{2,}$", domain, flags=re.IGNORECASE))
    if not domain_valid:
        return {
            "label": "Invalid Email Address",
            "confidence": 0.95,
            "explanation": "Domain part does not match expected domain.tld pattern.",
            "features": {"domain": domain},
        }

    # Extract compact features
    features = extract_features_from_sender(sender)

    local_has_plus_tag = 1 if "+" in local else 0
    local_has_many_dots = 1 if local.count(".") >= 2 else 0
    domain_has_digits = 1 if re.search(r"\d", domain) else 0
    has_very_long_local = 1 if len(local) > 64 else 0

    # Merge into features returned
    features.update({
        "local_has_plus_tag": local_has_plus_tag,
        "local_has_many_dots": local_has_many_dots,
        "domain_has_digits": domain_has_digits,
        "has_invalid_chars": has_invalid_chars,
        "has_very_long_local": has_very_long_local,
    })

    # Scoring: higher score => more suspicious
    score = 0.0
    # Start with free email providers as weak risk
    if features.get("sender_is_free_email") == 1:
        score += 0.25
    # Trusted institutional domain reduces risk
    if features.get("is_trusted_sender") == 1:
        score -= 0.6
    # Short local parts with digits are suspicious
    if features.get("local_has_digits") == 1 and features.get("local_part_length", 0) < 6:
        score += 0.3
    # Hyphens in domain and digits in domain add risk
    if features.get("domain_has_hyphen") == 1:
        score += 0.15
    if domain_has_digits:
        score += 0.15
    # Many dots or plus tagging can be benign but sometimes used for obfuscation
    if local_has_many_dots:
        score += 0.12
    if local_has_plus_tag:
        score += 0.05
    # Very long local parts or special chars increase risk
    if features.get("local_has_special") == 1:
        score += 0.25
    if features.get("has_invalid_chars") == 1:
        score += 0.5
    if features.get("has_very_long_local") == 1:
        score += 0.2
    # Domain age heuristic: very young domains slightly increase risk
    age = features.get("sender_domain_age_days", 0)
    if age < 300:
        score += 0.12

    # Clamp score
    score = max(0.0, min(1.0, score))

    # Decision thresholds (tunable):
    # score >= 0.6 -> Suspicious
    # 0.2 <= score < 0.6 -> Suspicious (lower confidence)
    # score < 0.2 -> Likely Legitimate
    if score >= 0.45:
        label = "Suspicious Email Address"
        confidence = 0.6 + 0.4 * (score - 0.45) / (1.0 - 0.45)
        explanation = f"Heuristic score={score:.2f} indicates multiple risky attributes."
    elif score >= 0.2:
        label = "Suspicious Email Address"
        confidence = 0.5 + 0.2 * (score - 0.2) / (0.45 - 0.2)
        explanation = f"Some risky attributes found (score={score:.2f}); treat as Suspicious."
    else:
        label = "Likely Legitimate Email Address"
        confidence = 0.75 + 0.25 * (0.2 - score) / 0.2
        explanation = f"Low heuristic score={score:.2f}; no strong indicators of phishing."

    return {
        "label": label,
        "confidence": round(float(confidence), 3),
        "explanation": explanation,
        "features": features,
        "feature_columns": EMAIL_ADDRESS_FEATURE_COLUMNS,
    }
