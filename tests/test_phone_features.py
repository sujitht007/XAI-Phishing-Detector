from src.phone_features import is_valid_phone, normalize_phone_number


def test_normalizes_international_phone_formats():
    assert normalize_phone_number("+91 987-654-3210") == "+919876543210"
    assert normalize_phone_number("0091 (987) 6543210") == "+919876543210"


def test_validates_numbers_by_country_numbering_rules():
    assert is_valid_phone("+91 9876543210")
    assert is_valid_phone("+1 2025550123")
    assert is_valid_phone("+44 7911123456")
    assert is_valid_phone("+971 501234567")
    assert is_valid_phone("919876543210")
    assert is_valid_phone("9876543210")


def test_rejects_invalid_country_numbers():
    assert not is_valid_phone("+91 987654321")
    assert not is_valid_phone("+1 202555012")
    assert not is_valid_phone("+971 50123456")