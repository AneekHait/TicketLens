"""Tests for the KBA and SOP generators — previously no test files at all.

These call the model as ``llm(prompt, ...)`` and read ``choices[0]["text"]`` (unlike
category_audit/disposition, which use ``create_chat_completion``). The shared
``FakeLLM`` in conftest implements both shapes, so one stub drives every generator.
"""
import pytest

from src.kba_generator import KBAGenerator
from src.sop_generator import SOPGenerator

KBA_REPLY = """TITLE: Password Reset Failures
SYMPTOMS: Users cannot sign in after a scheduled rotation.
CAUSE: The directory rejects the cached credential.
RESOLUTION: Reset the password and clear the credential cache.
PREVENTION: Pre-expiry reminders and self-service reset."""

SOP_REPLY = """TITLE: SOP: Access & Authorization
PURPOSE: Standardise handling of access requests.
SCOPE: All access and credential tickets.
STEPS: 1. Verify identity. 2. Reset. 3. Confirm with the user.
ESCALATION: Route to IAM if the account is locked at the domain level."""


# --- KBA --------------------------------------------------------------------
def test_kba_generates_one_article_per_real_cluster(fake_llm, cluster_data):
    llm = fake_llm(default=KBA_REPLY)
    articles = KBAGenerator(llm, cluster_data).generate_all()
    assert len(articles) == len(cluster_data)          # cluster_data has no -1 entry
    assert llm.calls == len(cluster_data)


def test_kba_skips_the_noise_cluster(fake_llm, cluster_data):
    data = dict(cluster_data)
    data[-1] = {"keywords": [], "subcategory": "Non-Repetitive",
                "category": "Non-Repetitive", "sample_docs": []}
    articles = KBAGenerator(fake_llm(default=KBA_REPLY), data).generate_all()
    assert all(a["cluster_id"] != -1 for a in articles)
    assert len(articles) == len(cluster_data)


def test_kba_parses_the_model_sections(fake_llm, cluster_data):
    article = KBAGenerator(fake_llm(default=KBA_REPLY), cluster_data).generate_article(0)
    assert article["title"] == "Password Reset Failures"
    assert "cannot sign in" in article["symptoms"]
    assert "credential cache" in article["resolution"]
    assert article["prevention"]


def test_kba_carries_cluster_identity_through(fake_llm, cluster_data):
    article = KBAGenerator(fake_llm(default=KBA_REPLY), cluster_data).generate_article(1)
    assert article["cluster_id"] == 1
    assert article["subcategory"] == "VPN Drops"
    assert article["category"] == "Network & Connectivity"


def test_kba_falls_back_when_the_model_returns_junk(fake_llm, cluster_data):
    # No recognisable headers -> a usable stub article, not an exception.
    article = KBAGenerator(fake_llm(default="banana"), cluster_data).generate_article(0)
    assert article is not None
    assert article["cluster_id"] == 0
    assert article["title"]          # falls back to the subcategory


def test_kba_survives_an_llm_that_raises(cluster_data):
    class Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("model died")

    article = KBAGenerator(Boom(), cluster_data).generate_article(0)
    assert article is not None, "a model failure must degrade, not propagate"
    assert article["resolution"]     # the documented fallback text


def test_kba_reports_progress(fake_llm, cluster_data):
    seen = []
    KBAGenerator(fake_llm(default=KBA_REPLY), cluster_data).generate_all(
        callback=lambda msg, prog: seen.append((msg, prog)))
    assert seen, "callback should fire"
    assert seen[-1][1] == 1.0, "final progress should be 1.0"


def test_kba_empty_cluster_data_yields_no_articles(fake_llm):
    assert KBAGenerator(fake_llm(), {}).generate_all() == []


def test_kba_honours_sample_and_keyword_limits(fake_llm, cluster_data):
    gen = KBAGenerator(fake_llm(default=KBA_REPLY), cluster_data,
                       settings={"max_sample_tickets": 2, "max_keywords": 3})
    assert gen.max_samples == 2 and gen.max_keywords == 3
    assert gen.generate_article(0) is not None


# --- SOP --------------------------------------------------------------------
def test_sop_generates_documents(fake_llm, cluster_data):
    sops = SOPGenerator(fake_llm(default=SOP_REPLY), cluster_data).generate_all()
    assert len(sops) > 0
    assert all("title" in s for s in sops)


def test_sop_parses_the_model_sections(fake_llm, cluster_data):
    sops = SOPGenerator(fake_llm(default=SOP_REPLY), cluster_data).generate_all()
    joined = " ".join(str(v) for s in sops for v in s.values())
    assert "Verify identity" in joined or "Standardise" in joined


def test_sop_survives_an_llm_that_raises(cluster_data):
    class Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("model died")

    sops = SOPGenerator(Boom(), cluster_data).generate_all()
    assert isinstance(sops, list)    # degrades rather than propagating


def test_sop_reports_progress(fake_llm, cluster_data):
    seen = []
    SOPGenerator(fake_llm(default=SOP_REPLY), cluster_data).generate_all(
        callback=lambda msg, prog: seen.append((msg, prog)))
    assert seen and seen[-1][1] == 1.0


def test_sop_empty_cluster_data(fake_llm):
    assert SOPGenerator(fake_llm(), {}).generate_all() == []


# --- the fake itself --------------------------------------------------------
def test_fake_llm_supports_both_calling_conventions(fake_llm):
    """Guards the conftest stub: the app uses two different llama-cpp APIs, so a
    single fake must answer both or generator tests silently diverge."""
    llm = fake_llm(replies={"alpha": "A"}, default="D")

    assert llm("... alpha ...")["choices"][0]["text"] == "A"
    assert llm("... other ...")["choices"][0]["text"] == "D"

    chat = llm.create_chat_completion(messages=[{"role": "user", "content": "alpha"}])
    assert chat["choices"][0]["message"]["content"] == "A"
    assert llm.calls == 3


@pytest.mark.parametrize("generator_cls", [KBAGenerator, SOPGenerator])
def test_generators_share_the_llm_cluster_data_settings_signature(generator_cls, fake_llm):
    """Both take (llm, cluster_data, settings=None) — worth pinning, since the
    analysis modules otherwise have 8 divergent constructor shapes."""
    gen = generator_cls(fake_llm(), {}, settings={})
    assert gen.llm is not None
    assert gen.cluster_data == {}
