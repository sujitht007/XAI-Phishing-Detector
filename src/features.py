import csv
import os
import re
from functools import lru_cache
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
    "is_malformed_url",
    "is_known_legit_domain",
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

# Fallback list used when seed file is unavailable.
KNOWN_LEGIT_DOMAINS_FALLBACK = [
    "kongu.ac.in", "kongu.edu", "iitm.ac.in", "vit.ac.in", "annauniv.edu",
    "mit.edu", "stanford.edu", "harvard.edu",
    "google.com", "microsoft.com", "github.com", "wikipedia.org",
    "gov.in", "india.gov.in", "nic.in",
    "amazon.in", "flipkart.com", "stackoverflow.com", "python.org", "ubuntu.com",
    "apple.com", "ibm.com", "nasa.gov", "education.microsoft.com",
    "bbc.co.uk", "ted.com", "cnn.com", "medium.com",
    "chatgpt.com", "openai.com", "youtube.com", "linkedin.com",
    "reddit.com", "netflix.com", "spotify.com", "zoom.us",
]


@lru_cache(maxsize=1)
def _get_known_legit_domains() -> tuple[str, ...]:
    domains = set(KNOWN_LEGIT_DOMAINS_FALLBACK)
    seeds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "url_seeds.csv")
    if not os.path.exists(seeds_path):
        return tuple(sorted(domains))

    with open(seeds_path, newline="", encoding="utf-8") as seed_file:
        for row in csv.DictReader(seed_file):
            if int(row.get("label", 1)) != 0:
                continue
            url = (row.get("url") or "").strip()
            if not url:
                continue
            parsed = urlparse(url if "://" in url else f"https://{url}")
            hostname = (parsed.hostname or "").lower()
            if hostname.startswith("www."):
                hostname = hostname[4:]
            if hostname:
                domains.add(hostname)
    return tuple(sorted(domains))


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


def _is_malformed_url(hostname: str) -> int:
    if not hostname:
        return 1
    if "." not in hostname:
        return 1
    labels = hostname.split(".")
    if len(labels) < 2 or any(not label for label in labels):
        return 1
    return 0


def _is_known_legit_domain(hostname: str) -> int:
    if not hostname:
        return 0
    for domain in _get_known_legit_domains():
        if hostname == domain or hostname.endswith(f".{domain}"):
            return 1
    return 0


def _has_url_red_flags(features: dict) -> bool:
    return any(
        features.get(flag) == 1
        for flag in ("is_malformed_url", "has_at_symbol", "has_ip", "has_suspicious_words")
    )


def _has_suspicious_words(searchable_text: str) -> int:
    for word in SUSPICIOUS_WORDS:
        if re.search(rf"(?<![a-z0-9]){re.escape(word)}", searchable_text):
            return 1
    return 0


def _estimate_domain_age(hostname: str, has_suspicious_words: int, has_ip: int, is_trusted_tld: int) -> int:
    if not hostname:
        return 0

    # Produce a deterministic but conservative age estimate based on hostname
    # Keep values bounded so domain age does not dominate other signals.
    seed = sum(ord(char) for char in hostname) % 10000

    # Trusted TLDs get a modest boost, not an overwhelming one.
    if is_trusted_tld:
        return 500 + (seed % 1500)

    # IP addresses are usually short-lived in this context.
    if has_ip:
        return 30 + (seed % 120)

    # If suspicious words are present, assume a newer/riskiest domain.
    if has_suspicious_words:
        return 10 + (seed % 300)

    # Generic case: spread across a reasonable range.
    return 200 + (seed % 2000)


def is_clearly_legitimate(features: dict) -> bool:
    return (
        features.get("is_known_legit_domain") == 1
        and features.get("is_malformed_url") == 0
        and features.get("has_at_symbol") == 0
        and features.get("has_ip") == 0
        and features.get("has_suspicious_words") == 0
    )


def is_clearly_phishing(features: dict) -> bool:
    if features.get("is_malformed_url") == 1:
        return True
    if features.get("is_known_legit_domain") == 0 and features.get("is_trusted_tld") == 1:
        return True
    if features.get("has_suspicious_words") == 1 and features.get("is_trusted_tld") == 1:
        return True
    return False


def resolve_url_prediction(model_results: list[dict], features: dict) -> tuple[str, float, dict]:
    """Combine model votes conservatively to reduce false positives on clean URLs."""
    if is_clearly_phishing(features):
        return "Phishing", 0.95, model_results[0]

    if is_clearly_legitimate(features):
        best_legit = max(
            (item for item in model_results if item["prediction"] == "Legit"),
            key=lambda item: item["confidence"],
            default=model_results[0],
        )
        return "Legit", max(best_legit["confidence"], 0.95), best_legit

    phishing_votes = [item for item in model_results if item["prediction"] == "Phishing"]
    legit_votes = [item for item in model_results if item["prediction"] == "Legit"]

    if not _has_url_red_flags(features):
        confident_legit = [item for item in legit_votes if item["confidence"] >= 0.85]
        if confident_legit:
            best = max(confident_legit, key=lambda item: item["confidence"])
            return "Legit", best["confidence"], best

    if len(phishing_votes) >= 2:
        best = max(phishing_votes, key=lambda item: item["confidence"])
        return "Phishing", best["confidence"], best

    if len(legit_votes) >= 2:
        best = max(legit_votes, key=lambda item: item["confidence"])
        return "Legit", best["confidence"], best

    best = max(model_results, key=lambda item: item["score"])
    return best["prediction"], best["confidence"], best


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
    is_malformed_url = _is_malformed_url(hostname)
    is_known_legit_domain = _is_known_legit_domain(hostname)
    num_subdomains = _count_subdomains(hostname)
    domain_age_days = _estimate_domain_age(
        hostname,
        has_suspicious_words,
        has_ip,
        is_trusted_tld if is_known_legit_domain else 0,
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
        "is_malformed_url": is_malformed_url,
        "is_known_legit_domain": is_known_legit_domain,
    }
