"""Tests for fishbone.classify_keyword."""
from src.fishbone import classify_keyword, ISHIKAWA_CATEGORIES


def test_known_pattern_maps_to_its_category():
    first_cat = next(iter(ISHIKAWA_CATEGORIES))
    first_pat = ISHIKAWA_CATEGORIES[first_cat][0]
    assert classify_keyword(first_pat) == first_cat


def test_unknown_keyword_defaults_to_technology():
    assert classify_keyword("zzqwx") == "Technology"


def test_classification_is_case_insensitive():
    first_pat = ISHIKAWA_CATEGORIES[next(iter(ISHIKAWA_CATEGORIES))][0]
    assert classify_keyword(first_pat.upper()) == classify_keyword(first_pat)


def test_non_string_input_defaults_to_technology():
    assert classify_keyword(12345) == "Technology"
