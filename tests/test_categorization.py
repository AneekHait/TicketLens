"""Tests for macro-category assignment (src.clustering._match_category_template).

Covers the priority-based rewrite that fixed access/authorization tickets
leaking into other categories, and the removal of the "Service Requests" bucket.
"""
import pytest

from src.clustering import (
    TicketClusterer,
    STANDARD_CATEGORIES,
    CATEGORY_TEMPLATES,
    ACCESS_INTENT_RE,
    ACCESS_HARD_SIGNAL_RE,
    _cat_signature,
    _STANDARD_CATEGORY_SIGNATURES,
)

match = TicketClusterer()._match_category_template


# --- Access intent beats the system a ticket happens to name ---------------
# These are the exact leak patterns seen in the Diebold output: the label
# clearly says "access" but the old longest-substring rule filed them under the
# system noun (sharepoint/connectivity/application/dashboard/vm).
@pytest.mark.parametrize("label", [
    "Auth SPO SharePoint Hub Access Issues",              # was Email & Collaboration
    "Salesforce Access And Connectivity Issues",           # was Network & Connectivity
    "EBS Application Access And Data Handling Issues",      # was Software & Applications
    "Dashboard Access And Functionality Issues",           # was Data & Reporting
    "MFA Configuration And Reset Requests / VM Support",    # was Infrastructure & Servers
    "PPM Authorization And Role Access Issues",            # was Business Applications
])
def test_access_intent_wins_over_system(label):
    assert match([label]) == "Access & Authorization"


@pytest.mark.parametrize("label", [
    "Password Reset",
    "Account Creation Request",
    "Account Unlock Request",
    "PPM New Access Provisioning Issues",   # "provisioning" is access, not Service Requests
    "User Deactivation / Termination",
    "SSO Login Failure",
    "Active Directory Group Membership",
])
def test_plain_access_labels(label):
    assert match([label]) == "Access & Authorization"


# --- Non-access clusters still route to their real domain ------------------
@pytest.mark.parametrize("label,expected", [
    ("Control-M Batch Job Abend", "Jobs & Batch Processing"),
    ("Power BI Dashboard Refresh Failure", "Data & Reporting"),
    ("Outlook Mailbox Migration", "Email & Collaboration"),
    ("VPN Connectivity Drop", "Network & Connectivity"),
    ("SAP HANA Transaction Error", "Business Applications"),
    ("Printer Toner Replacement", "Device & Hardware"),
    ("Netcool Disk Threshold Alert", "Monitoring & Alerts"),
    ("BitLocker Encryption Compliance", "Security & Compliance"),
])
def test_non_access_routes_to_domain(label, expected):
    assert match([label]) == expected


# --- Word-boundary guards: no mid-word false positives ---------------------
@pytest.mark.parametrize("label", [
    "Payroll Run Correction",     # "role" must not match inside "payroll"
    "Accounting Period Close",    # "account" must not match inside "accounting"
    "GL Account Posting Error",   # finance "account" is not access intent
])
def test_no_false_access_positives(label):
    assert match([label]) != "Access & Authorization"


# --- "Service Requests" was removed as a category --------------------------
def test_service_requests_not_a_standard_category():
    assert "Service Requests" not in STANDARD_CATEGORIES


def test_no_template_maps_to_service_requests():
    assert "Service Requests" not in CATEGORY_TEMPLATES.values()


def test_ticketing_tool_words_do_not_force_a_category():
    # servicenow/portal/catalog/ritm no longer template-match to anything.
    assert match(["ServiceNow Portal Catalog RITM Item"]) is None


# --- Scoring picks the domain with the most signal -------------------------
def test_multi_keyword_scoring():
    # Two Jobs hits ("batch", "scheduler") outweigh one Data hit ("report").
    assert match(["batch", "scheduler", "report"]) == "Jobs & Batch Processing"


# --- Sample docs must NOT influence the match (old contamination bug) ------
def test_sample_docs_are_ignored():
    # A pure password-reset label stays Access even if a sample ticket mentions
    # a longer competing keyword like "transaction".
    assert match(["Password Reset"], docs=["SAP transaction su01 failed"]) == \
        "Access & Authorization"


# --- Hard-signal safety-net regex ------------------------------------------
@pytest.mark.parametrize("text", [
    "user needs a password reset asap",
    "please reset password for jsmith",
    "account is locked, unlock the account",
    "MFA not working after phone change",
])
def test_hard_signal_matches_credential_ops(text):
    assert ACCESS_HARD_SIGNAL_RE.search(text)


@pytest.mark.parametrize("text", [
    "order stuck in the queue",
    "invoice cancellation failed",
    "batch job abended overnight",
])
def test_hard_signal_ignores_unrelated(text):
    assert not ACCESS_HARD_SIGNAL_RE.search(text)


def test_access_intent_regex_is_word_bounded():
    assert not ACCESS_INTENT_RE.search("payroll")
    assert not ACCESS_INTENT_RE.search("accounting")
    assert ACCESS_INTENT_RE.search("grant access to the folder")


# --- Category-name canonicalization (fold LLM variants onto one spelling) ---
def _canonicalize(cleaned):
    """Mirror of the standard-category match in _generate_category_label."""
    sig = _cat_signature(cleaned)
    if sig:
        for std_sig, std_cat in _STANDARD_CATEGORY_SIGNATURES:
            if sig == std_sig:
                return std_cat
    for std_cat in STANDARD_CATEGORIES:
        if std_cat.lower() in cleaned.lower() or cleaned.lower() in std_cat.lower():
            return std_cat
    return None


@pytest.mark.parametrize("variant,expected", [
    # The exact duplicates reported: word order swapped, and singular.
    ("Application & Software", "Software & Applications"),
    ("Business Application", "Business Applications"),
    # Connector / casing / plural variance also folds in.
    ("Software and Applications", "Software & Applications"),
    ("Access and Authorization", "Access & Authorization"),
    ("Network Connectivity", "Network & Connectivity"),
    ("business applications", "Business Applications"),
])
def test_category_variants_canonicalize_to_standard(variant, expected):
    assert _canonicalize(variant) == expected


def test_distinct_standard_categories_keep_distinct_signatures():
    # The intentional split (general Software vs line-of-business apps) must
    # survive — otherwise the fold would merge two real categories.
    assert _cat_signature("Software & Applications") != \
        _cat_signature("Business Applications")
    sigs = [s for s, _ in _STANDARD_CATEGORY_SIGNATURES]
    assert len(sigs) == len(set(sigs)), "standard category signatures collide"


def test_novel_category_is_not_forced_into_standard():
    # A genuinely new business category has no standard signature match.
    assert _canonicalize("Order Related") is None
