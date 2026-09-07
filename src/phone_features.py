import re

import phonenumbers
from phonenumbers import NumberParseException

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

DEFAULT_PHONE_REGION = "IN"

# Keep product-specific metadata here; phonenumbers remains the source of truth
# for country numbering plans and supports adding countries without new logic.
COUNTRY_NUMBERING_RULES = {
    "IN": {
        "country_name": "India",
        "iso_code": "IN",
        "calling_code": "+91",
        "national_number_lengths": (10,),
        "national_prefix": "0",
    },
    "US": {
        "country_name": "United States",
        "iso_code": "US",
        "calling_code": "+1",
        "national_number_lengths": (10,),
        "national_prefix": "1",
    },
    "CA": {
        "country_name": "Canada",
        "iso_code": "CA",
        "calling_code": "+1",
        "national_number_lengths": (10,),
        "national_prefix": "1",
    },
    "GB": {
        "country_name": "United Kingdom",
        "iso_code": "GB",
        "calling_code": "+44",
        "national_number_lengths": None,
        "national_prefix": "0",
    },
    "AE": {
        "country_name": "United Arab Emirates",
        "iso_code": "AE",
        "calling_code": "+971",
        "national_number_lengths": (9,),
        "national_prefix": "0",
    },
}

KNOWN_COUNTRY_CODES = [
    rule["calling_code"] for rule in COUNTRY_NUMBERING_RULES.values()
]
COUNTRY_CALLING_CODES = sorted(
    {code[1:] for code in KNOWN_COUNTRY_CODES}, key=len, reverse=True
)


def normalize_phone_number(phone: str) -> str:
    """Return a canonical international form while preserving a leading '+'."""
    if phone is None:
        return ""
    value = phone.strip()
    if value.startswith("00"):
        value = "+" + value[2:]
    if value.startswith("+"):
        return "+" + re.sub(r"\D", "", value[1:])
    digits = re.sub(r"\D", "", value)
    for calling_code in COUNTRY_CALLING_CODES:
        if not digits.startswith(calling_code):
            continue
        region_rules = [
            rule
            for rule in COUNTRY_NUMBERING_RULES.values()
            if rule["calling_code"] == f"+{calling_code}"
        ]
        national_length = len(digits) - len(calling_code)
        if any(
            rule["national_number_lengths"]
            and national_length in rule["national_number_lengths"]
            for rule in region_rules
        ):
            return f"+{digits}"
    return digits


def _normalize_phone(phone: str) -> str:
    return normalize_phone_number(phone)


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


def _country_code_digit_count(phone: str) -> int:
    for code in KNOWN_COUNTRY_CODES:
        if phone.startswith(code):
            return len(code) - 1
    return 0


def _parse_phone(phone: str):
    normalized = normalize_phone_number(phone)
    if not normalized:
        return None
    try:
        return phonenumbers.parse(
            normalized,
            None if normalized.startswith("+") else DEFAULT_PHONE_REGION,
        )
    except NumberParseException:
        return None


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
        and features.get("digit_diversity", 0) >= 0.5
        and features.get("country_code_known") == 1
        and features.get("has_short_code") == 0
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


def is_valid_phone(phone: str) -> bool:
    """Validate a phone number using the numbering plan for its country.

    Rules:
    - No alphabetic characters
    - Optional leading '+' only at start, with ``00`` converted to '+'
    - Country and national-number rules are checked by ``phonenumbers``
    """
    if phone is None:
        return False
    s = phone.strip()
    if re.search(r"[A-Za-z]", s):
        return False
    if "+" in s[1:] or ("+" in s and not s.startswith("+")):
        return False
    normalized = normalize_phone_number(s)
    if not normalized or normalized.startswith("+") and len(normalized) == 1:
        return False
    parsed = _parse_phone(normalized)
    if parsed is None:
        return False
    if not phonenumbers.is_possible_number(parsed):
        return False
    if not phonenumbers.is_valid_number(parsed):
        return False

    region = phonenumbers.region_code_for_number(parsed)
    rules = COUNTRY_NUMBERING_RULES.get(region)
    if rules and rules["national_number_lengths"]:
        national_length = len(str(parsed.national_number))
        if national_length not in rules["national_number_lengths"]:
            return False
    return True
