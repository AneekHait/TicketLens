"""Tests for the relaxed label cleanup used to name Repetitive Sub/Category."""
from src.clustering import TicketClusterer

clean = TicketClusterer()._clean_llm_output


def test_preserves_source_detail_colon():
    assert clean("Nagios Alert : Disk Space", max_words=8) == "Nagios Alert : Disk Space"


def test_titlecases_each_slash_segment():
    assert clean("order: release/update/delete/other", max_words=8) == "Order: Release/Update/Delete/Other"


def test_preserves_acronyms():
    assert clean("ICM errored transactions", max_words=8) == "ICM Errored Transactions"
    assert clean("SSRS report failure", max_words=8) == "SSRS Report Failure"


def test_strips_wrapping_quotes_and_markdown():
    assert clean('"**Password Reset**"') == "Password Reset"


def test_strips_label_prefix_and_list_marker():
    assert clean("Label: 1. VPN Issues") == "VPN Issues"


def test_respects_explicit_max_words():
    assert clean("one two three four five six", max_words=4) == "One Two Three Four"


def test_default_cap_is_four_words():
    assert clean("alpha beta gamma delta epsilon") == "Alpha Beta Gamma Delta"


def test_empty_input():
    assert clean("") == ""
    assert clean(None) == ""
