"""Tests for the LLM category audit (src.category_audit.CategoryAuditor).

A FakeLLM stands in for llama-cpp: its create_chat_completion returns a canned
reply chosen by matching the cluster's subcategory in the prompt. This lets us
assert the auditor's proposal logic (resolve, dedupe, skip-if-same) without a
real model.
"""
import pytest

from src.category_audit import CategoryAuditor


class FakeLLM:
    """Minimal stand-in for a llama-cpp Llama; replies keyed by prompt substring."""

    def __init__(self, replies):
        self.replies = replies          # {substring_in_prompt: reply_text}
        self.calls = 0

    def create_chat_completion(self, messages, **kwargs):
        self.calls += 1
        prompt = messages[0]["content"]
        reply = "KEEP"
        for key, val in self.replies.items():
            if key in prompt:
                reply = val
                break
        return {"choices": [{"message": {"content": reply}}]}


def _cluster_data():
    return {
        -1: {"category": "Non-Repetitive", "subcategory": "Non-Repetitive",
             "keywords": [], "sample_docs": []},
        1: {"category": "General IT Support", "subcategory": "Login Failures",
            "keywords": ["login", "password"], "sample_docs": ["cannot login to portal"]},
        2: {"category": "Software & Applications", "subcategory": "SAP GUI Crash",
            "keywords": ["sap", "gui"], "sample_docs": ["sap gui crashes on launch"]},
        3: {"category": "Network & Connectivity", "subcategory": "VPN Drop",
            "keywords": ["vpn"], "sample_docs": ["vpn keeps dropping"]},
        4: {"category": "Access & Authorization", "subcategory": "Account Unlock",
            "keywords": ["unlock"], "sample_docs": ["please unlock my account"]},
    }


def test_proposes_moves_and_respects_keep():
    llm = FakeLLM({
        "SAP GUI Crash": "Business Application",       # plural variant -> Business Applications
        "VPN Drop": "KEEP",                            # no move
    })
    proposals = CategoryAuditor(llm, _cluster_data()).audit()
    by_id = {p["cluster_id"]: p for p in proposals}

    assert set(by_id) == {1, 2}, "only the two real moves should be proposed"
    # cid 1 ("Login Failures", login/password) is claimed by the access RULE, not the LLM.
    assert by_id[1]["proposed_category"] == "Access & Authorization"
    assert by_id[1]["current_category"] == "General IT Support"
    assert by_id[1]["source"] == "rule"
    # cid 2 ("SAP GUI Crash") has no access intent -> LLM decides.
    assert by_id[2]["proposed_category"] == "Business Applications"  # canonicalized
    assert by_id[2]["source"] == "ai"


def test_same_category_under_variant_spelling_is_not_a_move():
    # cid 4 is already Access & Authorization (and access-claimed) -> kept, no move.
    llm = FakeLLM({"Account Unlock": "Authorization and Access"})
    proposals = CategoryAuditor(llm, _cluster_data()).audit()
    assert all(p["cluster_id"] != 4 for p in proposals)


def test_junk_reply_yields_no_move():
    # Use a NON-access cluster so the LLM (not the rule) is in charge.
    llm = FakeLLM({"SAP GUI Crash": "banana pancakes"})
    proposals = CategoryAuditor(llm, _cluster_data()).audit()
    assert all(p["cluster_id"] != 2 for p in proposals)


def test_access_intent_keeps_cluster_and_skips_llm():
    # An access cluster already in A&A must be kept WITHOUT calling the LLM,
    # even if the LLM would have moved it (this is the screenshot bug fix).
    cd = {1: {"category": "Access & Authorization",
              "subcategory": "ServiceNow Provisioning And Maintenance Requests",
              "keywords": ["servicenow", "maintenance", "request", "application"],
              "sample_docs": ["ServiceNow application maintenance request"]}}
    llm = FakeLLM({"ServiceNow": "Software & Applications"})  # would wrongly move it
    proposals = CategoryAuditor(llm, cd).audit()
    assert proposals == [], "access-claimed cluster must be kept"
    assert llm.calls == 0, "the LLM must not be consulted for access-claimed clusters"


def test_access_intent_moves_misfiled_cluster_by_rule():
    # Access intent but filed elsewhere -> deterministic rule corrects it to A&A.
    cd = {1: {"category": "Software & Applications",
              "subcategory": "2FA Reset / Keeper Account Management",
              "keywords": ["keeper", "security", "compliance"],
              "sample_docs": ["Keeper security compliance review"]}}
    llm = FakeLLM({"2FA": "Security & Compliance"})  # would go the wrong way
    proposals = CategoryAuditor(llm, cd).audit()
    assert len(proposals) == 1
    assert proposals[0]["proposed_category"] == "Access & Authorization"
    assert proposals[0]["source"] == "rule"
    assert llm.calls == 0


def test_llm_cannot_move_nonaccess_cluster_into_access():
    # A non-access cluster: even if the LLM says "Access & Authorization",
    # the deterministic boundary rejects moving it in.
    cd = {1: {"category": "Software & Applications", "subcategory": "SAP GUI Crash",
              "keywords": ["sap", "gui"], "sample_docs": ["sap gui crash"]}}
    llm = FakeLLM({"SAP GUI Crash": "Access & Authorization"})
    proposals = CategoryAuditor(llm, cd).audit()
    assert proposals == [], "LLM must not move a non-access cluster into A&A"


def test_noise_cluster_is_skipped():
    llm = FakeLLM({"Non-Repetitive": "Access & Authorization"})
    proposals = CategoryAuditor(llm, _cluster_data()).audit()
    assert all(p["cluster_id"] != -1 for p in proposals)


def test_no_llm_returns_empty():
    assert CategoryAuditor(llm=None, cluster_data=_cluster_data()).audit() == []


def test_missing_or_none_fields_do_not_crash():
    # cluster_data with None keywords/sample_docs must not raise in _build_prompt.
    cd = {
        1: {"category": "General IT Support", "subcategory": None,
            "keywords": None, "sample_docs": None},
        2: {"category": "Network & Connectivity"},   # keys entirely absent
    }
    llm = FakeLLM({"General IT Support": "Access & Authorization"})
    proposals = CategoryAuditor(llm, cd).audit()   # should not raise
    # cid 1 has no subcategory but still gets classified from keywords/current.
    assert isinstance(proposals, list)


def test_callback_is_invoked():
    seen = []
    llm = FakeLLM({})   # everything KEEP
    CategoryAuditor(llm, _cluster_data()).audit(callback=lambda m, p: seen.append((m, p)))
    assert seen, "progress callback should fire at least once"
    assert seen[-1][1] == 1.0, "final progress should be 1.0"


def test_available_categories_dedupes_and_includes_discovered():
    cd = {
        1: {"category": "Order Related", "subcategory": "x", "keywords": [], "sample_docs": []},
        2: {"category": "Software & Applications", "subcategory": "y", "keywords": [], "sample_docs": []},
    }
    avail = CategoryAuditor(llm=None, cluster_data=cd)._available_categories()
    assert "Order Related" in avail          # discovered category surfaced
    assert "Software & Applications" in avail
    # No two options share a canonical signature.
    from src.clustering import _cat_signature
    sigs = [_cat_signature(c) for c in avail]
    assert len(sigs) == len(set(sigs))


@pytest.mark.parametrize("reply,expected", [
    ("KEEP", None),
    ("Access & Authorization", "Access & Authorization"),
    ("Business Application", "Business Applications"),
    ("Category: Network Connectivity", "Network & Connectivity"),  # echoed prefix stripped
    ('"Data & Reporting"', "Data & Reporting"),
    ("", None),
    ("something unrelated", None),
])
def test_resolve_reply(reply, expected):
    aud = CategoryAuditor(llm=None, cluster_data={})
    avail = aud._available_categories()
    assert aud._resolve_reply(reply, avail) == expected
