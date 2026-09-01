"""Tests for src.llm_utils.parse_llm_sections."""
from src.llm_utils import parse_llm_sections

HEADERS = ["TITLE", "SYMPTOMS", "CAUSE", "RESOLUTION"]


def test_parses_simple_sections():
    text = "TITLE: Password Reset\nSYMPTOMS: cannot log in\nCAUSE: expired credential"
    out = parse_llm_sections(text, HEADERS)
    assert out["TITLE"] == "Password Reset"
    assert out["SYMPTOMS"] == "cannot log in"
    assert out["CAUSE"] == "expired credential"


def test_multiline_section_content_is_joined():
    text = "SYMPTOMS: line one\nline two\nCAUSE: root"
    out = parse_llm_sections(text, HEADERS)
    assert out["SYMPTOMS"] == "line one\nline two"
    assert out["CAUSE"] == "root"


def test_header_match_is_case_insensitive_and_key_uppercased():
    out = parse_llm_sections("title: hello", HEADERS)
    assert out == {"TITLE": "hello"}


def test_header_with_empty_first_line_then_body():
    text = "RESOLUTION:\nstep 1\nstep 2"
    out = parse_llm_sections(text, HEADERS)
    assert out["RESOLUTION"] == "step 1\nstep 2"


def test_no_headers_returns_empty_dict():
    assert parse_llm_sections("just some prose with no headers", HEADERS) == {}
