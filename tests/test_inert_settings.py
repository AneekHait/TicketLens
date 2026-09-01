"""Tests for settings that were gathered, saved, and then ignored.

Each of these presented a control (a checkbox, a config key, a Reset button) that had
no effect on behaviour. They fail silently by construction — the value is read with a
``.get(..., default)`` that succeeds against the wrong dict, or the flag is tested
against a function parameter nobody passes — so nothing logs and nothing raises. Only a
test that asserts the *effect* catches them.
"""
import io
import json
import os

import pytest

from src.config import DEFAULTS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- "Remove Boilerplate" was inert ----------------------------------------
BOILERPLATE_SAMPLE = (
    "Hi team, Please could you reset the VPN token for the finance server. "
    "Thanks and regards, Dave"
)


def test_remove_boilerplate_defaults_to_on():
    from src.preprocessing import preprocess_document

    out = preprocess_document(BOILERPLATE_SAMPLE, settings={})
    assert "thanks and regards" not in out.lower()


def test_remove_boilerplate_false_actually_leaves_it_in():
    """The whole bug: preprocess_documents never passed the flag, and the function only
    consulted its own parameter, so unchecking the box changed nothing."""
    from src.preprocessing import preprocess_document

    kept = preprocess_document(BOILERPLATE_SAMPLE,
                               settings={"remove_boilerplate": False})
    stripped = preprocess_document(BOILERPLATE_SAMPLE,
                                   settings={"remove_boilerplate": True})
    assert kept != stripped, "the setting still has no effect"
    assert "regards" in kept.lower()
    assert "regards" not in stripped.lower()


def test_the_setting_reaches_the_batch_entry_point():
    """preprocess_documents is what clustering actually calls."""
    from src.preprocessing import preprocess_documents

    kept = preprocess_documents([BOILERPLATE_SAMPLE],
                                settings={"remove_boilerplate": False})[0]
    stripped = preprocess_documents([BOILERPLATE_SAMPLE],
                                    settings={"remove_boilerplate": True})[0]
    assert kept != stripped
    assert "regards" in kept.lower()


def test_an_explicit_parameter_still_wins_over_settings():
    """Direct callers (and the old signature) must keep working."""
    from src.preprocessing import preprocess_document

    out = preprocess_document(BOILERPLATE_SAMPLE, remove_boilerplate_text=False,
                              settings={"remove_boilerplate": True})
    assert "regards" in out.lower()


# --- the word cloud ignored the stopwords section --------------------------
def test_wordcloud_passes_the_stopwords_section_not_the_whole_config():
    """get_stopwords() reads use_it_stopwords etc., which live under config["stopwords"].
    Handed the whole config it found none of them and used every default, so turning IT
    stopwords off never affected the cloud."""
    with io.open(os.path.join(REPO, "src", "wordcloud_studio.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'get_stopwords(\n' in src or 'get_stopwords(' in src
    assert '.get("stopwords", {})' in src, "still passing the whole config"


def test_get_stopwords_honours_the_it_toggle():
    from src.stopwords import IT_STOPWORDS, get_stopwords

    on = set(get_stopwords({"use_it_stopwords": True, "use_english_stopwords": False}))
    off = set(get_stopwords({"use_it_stopwords": False, "use_english_stopwords": False}))
    sample = next(iter(IT_STOPWORDS))
    assert sample in on
    assert sample not in off


def test_whole_config_shaped_dict_would_have_used_defaults():
    """Documents the failure mode: nesting matters, and nothing raises when it's wrong."""
    # The "wrong nesting" path leaves use_english_stopwords at its default of True, which
    # pulls in sklearn — so this particular test genuinely needs it.
    pytest.importorskip("sklearn", reason="get_stopwords loads ENGLISH_STOP_WORDS")
    from src.stopwords import IT_STOPWORDS, get_stopwords

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    whole_config = {"stopwords": {"use_it_stopwords": False},
                    "clustering": {"min_cluster_size": 15}}
    # Read at the wrong level, use_it_stopwords is simply absent -> default True.
    wrong = set(get_stopwords(whole_config))
    right = set(get_stopwords(whole_config["stopwords"]))
    # Must be an IT-only word: 35 of IT_STOPWORDS are also English stopwords (including
    # "please", the first one iteration yields), and those stay in either way because
    # use_english_stopwords is still True. Picking one of those made this assertion
    # unsatisfiable -- it only ever "passed" because sklearn was absent and the whole
    # test was skipped.
    sample = next(w for w in sorted(IT_STOPWORDS) if w not in ENGLISH_STOP_WORDS)
    assert sample in wrong, "the wrong nesting silently keeps the default"
    assert sample not in right, (
        f"{sample!r} is IT-only, so turning off IT stopwords must drop it")


# --- save_config could not clear a stale override --------------------------
@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    import src.config as cfg

    monkeypatch.setattr(cfg, "_CONFIG_DIR", str(tmp_path), raising=True)
    monkeypatch.setattr(cfg, "_USER_CONFIG_PATH",
                        str(tmp_path / "user_settings.json"), raising=True)
    return cfg, tmp_path / "user_settings.json"


def test_reverting_to_defaults_clears_the_override_file(isolated_config):
    """save_config skipped the write when the diff was empty, so the previous override
    file survived and load_config merged it straight back on the next launch."""
    cfg, path = isolated_config

    changed = {"clustering": dict(DEFAULTS["clustering"])}
    changed["clustering"]["umap_n_neighbors"] = 40
    cfg.save_config(changed)
    assert cfg.load_config()["clustering"]["umap_n_neighbors"] == 40

    # Now put it back to the default, as "Reset to defaults" does.
    cfg.save_config({"clustering": dict(DEFAULTS["clustering"])})
    assert json.loads(path.read_text()) == {}, "a stale override was left behind"
    assert (cfg.load_config()["clustering"]["umap_n_neighbors"]
            == DEFAULTS["clustering"]["umap_n_neighbors"])


def test_reverting_one_key_does_not_resurrect_it(isolated_config):
    cfg, _path = isolated_config

    two = {"clustering": dict(DEFAULTS["clustering"])}
    two["clustering"]["umap_n_neighbors"] = 40
    two["clustering"]["min_cluster_size"] = 25
    cfg.save_config(two)

    one = {"clustering": dict(DEFAULTS["clustering"])}
    one["clustering"]["min_cluster_size"] = 25       # neighbours back to default
    cfg.save_config(one)

    loaded = cfg.load_config()["clustering"]
    assert loaded["min_cluster_size"] == 25
    assert loaded["umap_n_neighbors"] == DEFAULTS["clustering"]["umap_n_neighbors"]


# --- keys with no widget were dropped by _gather_settings ------------------
NO_WIDGET_KEYS = [
    ("preprocessing", "custom_regex_patterns"),
    ("preprocessing", "custom_boilerplate_patterns"),
    ("stopwords", "custom_stopwords"),
    ("clustering", "hdbscan_cluster_selection_epsilon"),
]


@pytest.mark.parametrize("section,key", NO_WIDGET_KEYS)
def test_no_widget_keys_survive_gather_settings(qapp, section, key):
    """_gather_settings rebuilds these blocks from the UI, so anything without a widget
    was silently reset to the default before it could reach the engine."""
    from src.gui import ClusterApp

    w = ClusterApp()
    sentinel = {"custom_regex_patterns": [r"\bZZTOP\b"],
                "custom_boilerplate_patterns": ["kind regards zztop"],
                "custom_stopwords": ["zztop"],
                "hdbscan_cluster_selection_epsilon": 0.25}[key]
    w.config.setdefault(section, {})[key] = sentinel

    got = w._gather_settings()
    assert got[section][key] == sentinel, f"{section}.{key} was dropped"


def test_gather_settings_still_prefers_the_widget_for_widget_backed_keys(qapp):
    """The carry-through must not shadow a real control."""
    from src.gui import ClusterApp

    w = ClusterApp()
    w.config.setdefault("clustering", {})["min_cluster_size"] = 999
    w.settings_page.spin_cluster_size.setValue(33)
    assert w._gather_settings()["clustering"]["min_cluster_size"] == 33


# --- metadata_mapping was never read or persisted -------------------------
def test_metadata_mapping_is_persisted(qapp):
    from src.gui import ClusterApp

    w = ClusterApp()
    got = w._gather_settings()
    assert "metadata_mapping" in got, "the mapping is still not saved anywhere"
    # Every known role, not just the 8 that have a dropdown.
    assert set(got["metadata_mapping"]) == set(DEFAULTS["metadata_mapping"])


# Roles with no dropdown on the Settings page. disposition.py reads reopen_col and
# reassignment_col for the Automatability score, so persisting only the dropdown roles
# would silently erase a hand-edited value on the next save.
CONFIG_ONLY_ROLES = ["effort_col", "reopen_col", "reassignment_col",
                     "close_code_col", "symptom_col"]


@pytest.mark.parametrize("role", CONFIG_ONLY_ROLES)
def test_config_only_roles_are_not_lost(qapp, role):
    from src.gui import ClusterApp

    w = ClusterApp()
    w.config.setdefault("metadata_mapping", {})[role] = "my_column"
    assert w._gather_metadata_mapping()[role] == "my_column", \
        f"{role} has no dropdown, so it must be carried through from config"


def test_the_dropdowns_still_win_for_the_roles_they_cover(qapp):
    """Config carry-through must not shadow a live control."""
    pd = pytest.importorskip("pandas")
    from src.gui import ClusterApp

    w = ClusterApp()
    w.df = pd.DataFrame({"priority": ["3"], "text": ["x"]})
    w._update_metadata_dropdowns()                       # auto-detects "priority"
    w.config["metadata_mapping"] = {"priority_col": "stale_value"}
    assert w._gather_metadata_mapping()["priority_col"] == "priority"


def test_a_saved_mapping_is_restored_over_auto_detection(qapp):
    """auto_map's name patterns can't know about an unrecognised column, so a mapping the
    user chose deliberately must win."""
    pd = pytest.importorskip("pandas")
    from src.gui import ClusterApp

    w = ClusterApp()
    # "urgency" is not in auto_map's priority patterns; "priority" is.
    w.df = pd.DataFrame({"priority": ["3"], "urgency": ["high"], "text": ["x"]})
    w.config["metadata_mapping"] = {"priority_col": "urgency"}
    w._update_metadata_dropdowns()
    assert w._gather_metadata_mapping()["priority_col"] == "urgency"


def test_auto_detection_still_applies_when_the_saved_column_is_absent(qapp):
    """Switching to a workbook without the remembered column must fall back, not blank."""
    pd = pytest.importorskip("pandas")
    from src.gui import ClusterApp

    w = ClusterApp()
    w.df = pd.DataFrame({"priority": ["3"], "text": ["x"]})
    w.config["metadata_mapping"] = {"priority_col": "not_in_this_sheet"}
    w._update_metadata_dropdowns()
    assert w._gather_metadata_mapping()["priority_col"] == "priority"


# --- the audit no longer invents a 50% ------------------------------------
def test_zero_checks_reports_nothing_measured():
    """Load a file, go straight to the audit with no columns mapped: every ticket used to
    score exactly 50.0, indistinguishable from a genuinely mediocre dataset."""
    pd = pytest.importorskip("pandas")
    from src.quality_audit import TicketQualityAuditor

    df = pd.DataFrame({"short_description": ["vpn down", "printer jam"]})
    auditor = TicketQualityAuditor(df, {})          # no mapping at all
    scores = auditor.audit_completeness(text_columns=[])   # and no text columns

    assert scores == [None, None], scores


def test_zero_checks_propagates_to_the_summary():
    pd = pytest.importorskip("pandas")
    from src.quality_audit import TicketQualityAuditor

    df = pd.DataFrame({"short_description": ["vpn down", "printer jam"]})
    _out, summary = TicketQualityAuditor(df, {}).generate_report(text_columns=[])

    assert summary["overall_score"] is None
    assert summary["completeness_avg"] is None
    assert summary["tickets_scored"] == 0


def test_real_checks_still_score(tickets_df):
    """The change must not turn working audits into None."""
    from src.quality_audit import TicketQualityAuditor

    auditor = TicketQualityAuditor(tickets_df, {"priority_col": "priority"})
    scores = auditor.audit_completeness(text_columns=["short_description"])
    assert all(s is not None for s in scores)


# --- the vagueness check was a near-constant ------------------------------
def test_vague_language_penalises_a_long_vague_description():
    """vague_hits counted matching *patterns* (max 3), not vague words, so the ratio
    could only reach the 0.5 threshold at ~6 words or fewer — every longer description
    scored a full 100 however vague."""
    pd = pytest.importorskip("pandas")
    from src.quality_audit import TicketQualityAuditor

    vague = " ".join(["issue problem error broken not working"] * 3)   # 18 vague words
    precise = ("VPN tunnel drops after the Cisco AnyConnect client renews its "
               "certificate on the finance subnet gateway")
    df = pd.DataFrame({"short_description": [vague, precise]})

    scores = TicketQualityAuditor(df, {}).audit_completeness(
        text_columns=["short_description"])

    assert scores[0] < scores[1], (
        f"the vague description did not score lower: {scores}")


# --- whole config sections with no UI were wiped on every RUN ---------------
# _compute_diff only walks the keys present in the dict it is given, and save_config
# truncates the file. _gather_settings returned 7 of the 13 DEFAULTS sections, so the
# six with no widget behind them were silently deleted from user_settings.json the next
# time the user pressed RUN. Anything hand-edited there was lost.
UI_LESS_SECTIONS = ["analysis", "audit", "category_audit", "disposition", "kba", "sop"]


def test_gather_settings_covers_every_defaults_section(qapp):
    """The gathered dict must name every DEFAULTS section, or save_config drops it."""
    from src.gui import ClusterApp

    w = ClusterApp()
    missing = sorted(set(DEFAULTS) - set(w._gather_settings()))
    assert not missing, f"_gather_settings omits {missing}; save_config would delete them"


@pytest.mark.parametrize("section", UI_LESS_SECTIONS)
def test_ui_less_section_survives_a_save(isolated_config, qapp, section):
    """Hand-edit a UI-less section, press RUN, and it must still be there."""
    from src.gui import ClusterApp

    cfg, path = isolated_config

    # An override no widget can represent.
    key = next(iter(DEFAULTS[section]))
    original = DEFAULTS[section][key]
    if isinstance(original, bool):
        tweaked = not original
    elif isinstance(original, (int, float)):
        tweaked = original + 1
    else:
        tweaked = "USER_SET"
    path.write_text(json.dumps({section: {key: tweaked}}), encoding="utf-8")

    w = ClusterApp()
    w.config = cfg.load_config()          # as the app does at startup
    assert w.config[section][key] == tweaked

    cfg.save_config(w._gather_settings())  # as pressing RUN does

    assert cfg.load_config()[section][key] == tweaked, (
        f"{section}.{key} was wiped by the save")
