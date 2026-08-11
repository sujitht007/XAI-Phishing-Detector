import re

EMAIL_FEATURE_COLUMNS = [
    "subject_length",
    "body_length",
    "num_links",
    "num_exclamation",
    "has_suspicious_words",
    "has_urgent_words",
    "num_uppercase_words",
    "sender_is_free_email",
    "has_html_tags",
    "has_money_mention",
    "is_trusted_sender",
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


def is_clearly_legitimate_email(features: dict) -> bool:
    return (
        features.get("is_trusted_sender") == 1
        and features.get("has_suspicious_words") == 0
        and features.get("has_urgent_words") == 0
        and features.get("num_links") <= 2
    )


def extract_features_from_email(subject: str, sender: str, body: str) -> dict:
    subject = (subject or "").strip()
    sender = (sender or "").strip()
    body = (body or "").strip()
    combined = f"{subject}\n{body}"

    sender_domain = _extract_sender_domain(sender)
    has_suspicious_words = _has_word_list(combined, SUSPICIOUS_WORDS)
    has_urgent_words = _has_word_list(combined, URGENT_WORDS)
    is_trusted_sender = _is_trusted_sender_domain(sender_domain)

    return {
        "subject_length": len(subject),
        "body_length": len(body),
        "num_links": _count_links(combined),
        "num_exclamation": combined.count("!"),
        "has_suspicious_words": has_suspicious_words,
        "has_urgent_words": has_urgent_words,
        "num_uppercase_words": _count_uppercase_words(combined),
        "sender_is_free_email": 1 if sender_domain in FREE_EMAIL_DOMAINS else 0,
        "has_html_tags": 1 if re.search(r"<[a-z][\s\S]*>", body, re.IGNORECASE) else 0,
        "has_money_mention": 1 if re.search(r"(\$|€|£|rs\.?\s*\d|usd|inr|payment due)", combined, re.IGNORECASE) else 0,
        "is_trusted_sender": is_trusted_sender,
    }
