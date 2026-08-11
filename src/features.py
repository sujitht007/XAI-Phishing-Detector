import re
from urllib.parse import urlparse

FEATURE_COLUMNS = [
    "url_length",
    "has_at_symbol",
    "has_https",
    "num_dots",
    "has_ip",
    "domain_age_days",
    "has_suspicious_words",
    "num_subdomains",
    "is_trusted_tld",
]

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "update", "bank", "paypal", "confirm",
    "signin", "account", "free", "winner", "prize", "click", "urgent",
]

COMPOUND_SUFFIXES = [
    "ac.in", "co.in", "edu.in", "gov.in", "net.in", "org.in", "nic.in",
    "ac.uk", "co.uk", "gov.uk", "org.uk",
    "edu.au", "com.au", "gov.au",
]

TRUSTED_SUFFIXES = [
    "ac.in", "edu", "edu.in", "gov.in", "gov", "gov.uk", "ac.uk",
    "edu.au", "gov.au", "nic.in",
]


def _is_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", hostname):
        return True
    if ":" in hostname:
        return True
    return False


def _count_subdomains(hostname: str) -> int:
    if not hostname or _is_ip_address(hostname):
        return 0

    for suffix in sorted(COMPOUND_SUFFIXES, key=len, reverse=True):
        if hostname == suffix or hostname.endswith(f".{suffix}"):
            prefix = hostname[: -(len(suffix) + 1)]
            if not prefix:
                return 0
            return max(0, len(prefix.split(".")) - 1)

    labels = hostname.split(".")
    return max(0, len(labels) - 2)


def _is_trusted_tld(hostname: str) -> int:
    if not hostname:
        return 0

    for suffix in sorted(TRUSTED_SUFFIXES, key=len, reverse=True):
        if hostname == suffix or hostname.endswith(f".{suffix}"):
            return 1
    return 0


def _has_suspicious_words(searchable_text: str) -> int:
    for word in SUSPICIOUS_WORDS:
        if re.search(rf"(?<![a-z0-9]){re.escape(word)}", searchable_text):
            return 1
    return 0


def _estimate_domain_age(hostname: str, has_suspicious_words: int, has_ip: int, is_trusted_tld: int) -> int:
    if not hostname:
        return 0

    seed = sum(ord(char) for char in hostname) % 10000
    if is_trusted_tld:
        return 3000 + (seed % 1500)
    if has_ip:
        return 50 + (seed % 350)
    if has_suspicious_words:
        return 30 + (seed % 500)
    return 500 + (seed % 4400)


def is_clearly_legitimate(features: dict) -> bool:
    return (
        features.get("is_trusted_tld") == 1
        and features.get("has_at_symbol") == 0
        and features.get("has_ip") == 0
        and features.get("has_suspicious_words") == 0
    )


def extract_features_from_url(url: str) -> dict:
    original_url = (url or "").strip()
    if not original_url:
        return {col: 0 for col in FEATURE_COLUMNS}

    has_explicit_scheme = original_url.lower().startswith(("http://", "https://"))
    clean_url = original_url
    if not has_explicit_scheme:
        clean_url = f"https://{clean_url}"

    parsed = urlparse(clean_url)
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    path = (parsed.path or "").lower()
    searchable_text = f"{hostname}{path}"

    has_https = 1 if has_explicit_scheme and parsed.scheme == "https" else 0
    has_at_symbol = 1 if "@" in original_url else 0
    num_dots = clean_url.count(".")
    has_ip = 1 if _is_ip_address(hostname) else 0
    has_suspicious_words = _has_suspicious_words(searchable_text)
    is_trusted_tld = _is_trusted_tld(hostname)
    num_subdomains = _count_subdomains(hostname)
    domain_age_days = _estimate_domain_age(
        hostname,
        has_suspicious_words,
        has_ip,
        is_trusted_tld,
    )

    return {
        "url_length": len(clean_url),
        "has_at_symbol": has_at_symbol,
        "has_https": has_https,
        "num_dots": num_dots,
        "has_ip": has_ip,
        "domain_age_days": domain_age_days,
        "has_suspicious_words": has_suspicious_words,
        "num_subdomains": num_subdomains,
        "is_trusted_tld": is_trusted_tld,
    }
