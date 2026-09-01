"""Tests for src.preprocessing pure text helpers."""
from src.preprocessing import (
    clean_ticket_text,
    clean_resolution_text,
    remove_boilerplate,
    preprocess_document,
)

# Minimal settings that leave text intact except the toggle under test.
BASE = {
    "remove_emails": False,
    "remove_urls": False,
    "remove_ticket_ids": False,
    "remove_timestamps": False,
    "remove_phone_numbers": False,
    "remove_ip_addresses": False,
    "remove_file_paths": False,
    "remove_special_chars": False,
    "min_token_length": 1,
}


def _with(**overrides):
    s = dict(BASE)
    s.update(overrides)
    return s


def test_non_string_returns_empty():
    assert clean_ticket_text(None) == ""
    assert clean_ticket_text(123) == ""


def test_lowercases():
    assert clean_ticket_text("HELLO World", _with()) == "hello world"


def test_remove_emails_toggle():
    text = "email alice@example.com please"
    assert "@" not in clean_ticket_text(text, _with(remove_emails=True))
    assert "@" in clean_ticket_text(text, _with(remove_emails=False))


def test_remove_urls():
    out = clean_ticket_text("see http://example.com/x now", _with(remove_urls=True))
    assert "http" not in out and "now" in out


def test_remove_ticket_ids():
    out = clean_ticket_text("ticket INC0001234 raised", _with(remove_ticket_ids=True))
    assert "inc0001234" not in out and "raised" in out


def test_remove_ip_addresses():
    out = clean_ticket_text("host 192.168.0.1 down", _with(remove_ip_addresses=True))
    assert "192.168" not in out


def test_min_token_length_filters_short_tokens():
    out = clean_ticket_text("a bb ccc dddd", _with(min_token_length=3))
    assert out == "ccc dddd"


def test_custom_regex_invalid_pattern_is_ignored():
    # An invalid regex must be skipped, not raise.
    out = clean_ticket_text("keep this text", _with(custom_regex_patterns=["(unclosed"]))
    assert "keep" in out


def test_default_settings_used_when_none():
    # settings=None should fall back to DEFAULTS and still return a lowercased str.
    out = clean_ticket_text("Hello World Example")
    assert isinstance(out, str) and out == out.lower()


def test_clean_resolution_text_strips_journal_metadata():
    raw = "2024-01-15 10:30:00 - John Doe (Work notes)\nReset the account password"
    out = clean_resolution_text(raw)
    assert "john doe" not in out.lower()
    assert "work notes" not in out.lower()
    assert "reset the account password" in out.lower()


def test_clean_resolution_text_strips_cr_artifacts_and_non_string():
    assert "_x000d_" not in clean_resolution_text("line1_x000D_line2").lower()
    assert clean_resolution_text(None) == ""


def test_remove_boilerplate_strips_greetings_and_signoffs():
    out = remove_boilerplate("Dear team, my VPN is broken. Thanks and regards").lower()
    assert "dear team" not in out
    assert "regards" not in out
    assert "vpn" in out


def test_preprocess_document_short_result_falls_back_to_original():
    # "ok" cleans to empty (too short) -> fallback returns the minimally-cleaned original.
    assert preprocess_document("ok") == "ok"


def test_preprocess_document_empty_input():
    assert preprocess_document("") == ""
    assert preprocess_document(None) == ""
