import re

PHONE_FEATURE_COLUMNS = [
    "phone_length",
    "has_plus_prefix",
    "num_separators",
    "has_repeated_digits",
    "has_suspicious_prefix",
    "country_code_known",
    "digit_diversity",
    "starts_with_0",
    "contains_alpha",
    "has_short_code",
]

SUSPICIOUS_PREFIXES = [
    "1800",
    "1900",
    "900",
    "808",
    "070",
    "15600",
    "999",
    "911",
    "123",
    "666",
]

KNOWN_COUNTRY_CODES = [
    "+1",
    "+44",
    "+91",
    "+61",
    "+81",
    "+49",
    "+33",
    "+86",
    "+7",
]


def _normalize_phone(phone: str) -> str:
    return re.sub(r"[\s\-().]+", "", phone.strip())


def _has_repeated_digits(phone: str) -> int:
    return 1 if re.search(r"(\d)\1{2,}", phone) else 0


def _has_suspicious_prefix(phone: str) -> int:
    for prefix in SUSPICIOUS_PREFIXES:
        if phone.startswith(prefix) or phone.startswith(f"+{prefix}"):
            return 1
    return 0


def _country_code_known(phone: str) -> int:
    for code in KNOWN_COUNTRY_CODES:
        if phone.startswith(code):
            return 1
    return 0


def _digit_diversity(phone: str) -> float:
    digits = re.sub(r"\D", "", phone)
    if not digits:
        return 0.0
    return len(set(digits)) / len(digits)


def _has_short_code(phone: str) -> int:
    digits = re.sub(r"\D", "", phone)
    return 1 if 3 <= len(digits) <= 6 else 0


def is_clearly_legitimate_phone(features: dict) -> bool:
    return (
        features.get("contains_alpha") == 0
        and features.get("has_suspicious_prefix") == 0
        and features.get("has_repeated_digits") == 0
        and features.get("digit_diversity", 0) >= 0.5
        and features.get("country_code_known") == 1
    )


def extract_features_from_phone(phone: str) -> dict:
    original_phone = (phone or "").strip()
    normalized = _normalize_phone(original_phone)
    digits_only = re.sub(r"\D", "", normalized)

    return {
        "phone_length": len(normalized),
        "has_plus_prefix": 1 if normalized.startswith("+") else 0,
        "num_separators": len(re.findall(r"[\s\-().]", original_phone)),
        "has_repeated_digits": _has_repeated_digits(digits_only),
        "has_suspicious_prefix": _has_suspicious_prefix(normalized),
        "country_code_known": _country_code_known(normalized),
        "digit_diversity": _digit_diversity(digits_only),
        "starts_with_0": 1 if digits_only.startswith("0") else 0,
        "contains_alpha": 1 if re.search(r"[A-Za-z]", original_phone) else 0,
        "has_short_code": _has_short_code(original_phone),
    }
