"""Session save/reload round-trips a whole run.

The point of the feature is that reopening a session must not re-pay the LLM cost, so
these tests care most about the things that would silently break a lookup after a
JSON round-trip: cluster-ID keys becoming strings, numpy scalars refusing to encode,
and unserialisable values (DataFrames, Plotly figures) taking the whole save down.
"""
import json
import zipfile

import pytest

from src.session import (SESSION_EXT, SESSION_FORMAT, PERSISTED_ANALYSIS_KEYS,
                         RECOMPUTED_ANALYSIS_KEYS, SessionError, load_session,
                         read_manifest, save_session)


@pytest.fixture
def run_state():
    """The state a finished run leaves behind, with the awkward types included."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")

    df = pd.DataFrame({
        "Short Description": ["vpn drops", "sap slow", "password reset", "vpn drops too"],
        "Cluster_ID": [0, 1, -1, 0],
        "Repetitive Subcategory": ["VPN Instability", "SAP Latency",
                                   "Non-Repetitive", "VPN Instability"],
        "Repetitive Category": ["Network & Connectivity", "Business Applications",
                                "Non-Repetitive", "Network & Connectivity"],
    })
    cluster_data = {
        np.int64(0): {
            "keywords": ["vpn", "tunnel"], "subcategory": "VPN Instability",
            "category": "Network & Connectivity", "sample_docs": ["vpn drops"],
            "sample_doc_indices": [np.int64(0), np.int64(3)],
            "topic_embedding": np.zeros(768, dtype="float32"),
        },
        np.int64(1): {
            "keywords": ["sap"], "subcategory": "SAP Latency",
            "category": "Business Applications", "sample_docs": ["sap slow"],
            "sample_doc_indices": [np.int64(1)],
            "topic_embedding": np.ones(768, dtype="float32"),
        },
        np.int64(-1): {
            "keywords": [], "subcategory": "Non-Repetitive", "category": "",
            "sample_docs": [], "sample_doc_indices": [],
        },
    }
    analysis = {
        "disposition": [{"cluster_id": 0, "disposition": "Automate",
                         "score": np.float64(0.81), "recommendation": "self-service"}],
        "disposition_summary": {"Automate": np.int64(1)},
        "audit_summary": {"overall_score": np.float64(72.5), "tickets_scored": 4,
                          "clusters": {np.int64(0): {"avg_quality": 80.0, "count": 2}}},
        "main_theme": {"headline": "VPN dominates"},
        # Not persistable: a DataFrame and a stand-in for a Plotly figure.
        "problem_clusters": {"volume_stats": df},
        "impact_figs": [object()],
    }
    return df, cluster_data, analysis


def _save(tmp_path, df, cluster_data, analysis, **kw):
    path = str(tmp_path / f"run{SESSION_EXT}")
    dropped = save_session(
        path, df=df, cluster_data=cluster_data, analysis_results=analysis,
        kba_articles=[{"cluster_id": 0, "title": "VPN drops"}],
        sop_documents=[{"category": "Network & Connectivity", "purpose": "p"}],
        selected_text_cols=["Short Description"],
        settings={"clustering": {"min_cluster_size": 15}},
        source_file="C:/data/tickets.xlsx", sheet_name="Incidents",
        app_version="1.2.0", **kw)
    return path, dropped


# --- the round trip ---------------------------------------------------------
def test_round_trip_restores_every_field(tmp_path, run_state):
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    got = load_session(path)

    assert list(got["df"].columns) == list(df.columns)
    assert len(got["df"]) == len(df)
    assert got["selected_text_cols"] == ["Short Description"]
    assert got["kba_articles"][0]["title"] == "VPN drops"
    assert got["sop_documents"][0]["category"] == "Network & Connectivity"


def test_cluster_ids_come_back_as_ints(tmp_path, run_state):
    """JSON stringifies keys. If they stayed strings every cluster_data[cid] would miss."""
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    got = load_session(path)

    assert set(got["cluster_data"]) == {0, 1, -1}
    assert all(isinstance(k, int) for k in got["cluster_data"])
    # The noise bucket in particular, since -1 is the one that looks least like a key.
    assert got["cluster_data"][-1]["subcategory"] == "Non-Repetitive"
    assert got["cluster_data"][0]["keywords"] == ["vpn", "tunnel"]


def test_audit_summary_clusters_are_also_int_keyed(tmp_path, run_state):
    """The per-cluster quality table reads this back by ID."""
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    clusters = load_session(path)["analysis_results"]["audit_summary"]["clusters"]
    assert set(clusters) == {0}
    assert all(isinstance(k, int) for k in clusters)


def test_numpy_scalars_survive_as_plain_numbers(tmp_path, run_state):
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    res = load_session(path)["analysis_results"]

    assert res["disposition"][0]["score"] == pytest.approx(0.81)
    assert type(res["disposition"][0]["score"]) is float
    assert res["disposition_summary"]["Automate"] == 1
    assert res["audit_summary"]["overall_score"] == pytest.approx(72.5)


# --- what is deliberately excluded ------------------------------------------
def test_unserialisable_results_are_excluded_not_fatal(tmp_path, run_state):
    """A DataFrame and a Plotly figure in analysis_results must not fail the save."""
    df, cluster_data, analysis = run_state
    path, dropped = _save(tmp_path, df, cluster_data, analysis)
    res = load_session(path)["analysis_results"]

    for key in RECOMPUTED_ANALYSIS_KEYS:
        assert key not in res, f"{key} should be recomputed, not persisted"
    for key in ("disposition", "audit_summary", "main_theme"):
        assert key in res
    # They are filtered by the allowlist before serialising, so nothing had to be
    # dropped by the normaliser.
    assert dropped == []


def test_topic_embedding_is_not_stored(tmp_path, run_state):
    """768 floats per cluster that nothing outside clustering.py reads."""
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    got = load_session(path)
    assert all("topic_embedding" not in c for c in got["cluster_data"].values())


def test_persisted_and_recomputed_keys_do_not_overlap():
    assert not set(PERSISTED_ANALYSIS_KEYS) & set(RECOMPUTED_ANALYSIS_KEYS)


# --- provenance -------------------------------------------------------------
def test_manifest_records_provenance(tmp_path, run_state):
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    m = read_manifest(path)

    assert m["format"] == SESSION_FORMAT
    assert m["app_version"] == "1.2.0"
    assert m["rows"] == 4
    assert m["clusters"] == 3
    assert m["sheet_name"] == "Incidents"
    assert m["source_file"] == "tickets.xlsx", "should store the basename, not the path"
    assert m["saved_at"].endswith("Z")
    assert m["recompute_on_load"] == list(RECOMPUTED_ANALYSIS_KEYS)


def test_settings_are_stored_for_provenance(tmp_path, run_state):
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    assert load_session(path)["settings"]["clustering"]["min_cluster_size"] == 15


def test_source_path_is_not_leaked(tmp_path, run_state):
    """Sessions get emailed around; the author's directory layout should not ride along."""
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    with zipfile.ZipFile(path) as z:
        blob = z.read("manifest.json").decode()
    assert "C:/data" not in blob


# --- refusing bad input -----------------------------------------------------
def test_a_newer_format_is_refused_with_a_clear_message(tmp_path, run_state):
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)

    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    m = json.loads(parts["manifest.json"])
    m["format"] = SESSION_FORMAT + 5
    parts["manifest.json"] = json.dumps(m).encode()
    with zipfile.ZipFile(path, "w") as z:
        for n, b in parts.items():
            z.writestr(n, b)

    with pytest.raises(SessionError) as e:
        load_session(path)
    assert "newer version" in str(e.value)
    assert str(SESSION_FORMAT + 5) in str(e.value)


def test_a_non_session_zip_is_refused(tmp_path):
    path = str(tmp_path / f"bogus{SESSION_EXT}")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("hello.txt", "not a session")
    with pytest.raises(SessionError) as e:
        load_session(path)
    assert "not a TicketLens session" in str(e.value)


def test_a_truncated_session_is_refused(tmp_path, run_state):
    """Missing members must give a clear error, not a KeyError traceback."""
    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)
    with zipfile.ZipFile(path) as z:
        manifest = z.read("manifest.json")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", manifest)      # manifest only
    with pytest.raises(SessionError) as e:
        load_session(path)
    assert "missing part of its contents" in str(e.value)


def test_garbage_file_is_refused(tmp_path):
    path = tmp_path / f"junk{SESSION_EXT}"
    path.write_bytes(b"this is not a zip at all")
    with pytest.raises(SessionError):
        load_session(str(path))


def test_saving_without_data_is_refused(tmp_path):
    with pytest.raises(SessionError):
        save_session(str(tmp_path / f"x{SESSION_EXT}"), df=None)


def test_a_failed_save_leaves_no_partial_file(tmp_path, run_state):
    """The write goes to .part and is renamed, so a crash cannot replace a good session
    with a truncated one."""
    df, cluster_data, analysis = run_state
    target = tmp_path / "sub" / f"run{SESSION_EXT}"      # parent does not exist
    with pytest.raises(OSError):
        save_session(str(target), df=df, cluster_data=cluster_data)
    assert not target.exists()
    assert not (tmp_path / "sub").exists()
    assert not list(tmp_path.glob("*.part"))


# --- the GUI wiring ---------------------------------------------------------
def test_gui_restores_a_session_end_to_end(tmp_path, run_state, qapp, monkeypatch):
    """Save from a populated window, restore into a fresh one, and confirm the state the
    analysis features guard on is actually back."""
    import src.gui as gui_mod
    from src.gui import ClusterApp

    df, cluster_data, analysis = run_state
    path, _ = _save(tmp_path, df, cluster_data, analysis)

    monkeypatch.setattr(gui_mod.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(gui_mod.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(gui_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))

    w = ClusterApp()
    assert not w.cluster_data          # nothing loaded yet
    w._open_session()

    assert w.df is not None and len(w.df) == 4
    assert set(w.cluster_data) == {0, 1, -1}
    assert w.selected_text_cols == ["Short Description"]
    assert w.kba_articles and w.sop_documents
    assert w.analysis_results["main_theme"]["headline"] == "VPN dominates"
    # The results card is what gates the post-run actions.
    assert w.clustering_page.results_card.isVisibleTo(w.clustering_page)
    # And the fishbone scope must be repopulated, not left on its placeholder.
    labels = [w.analysis_page.fishbone_scope.itemText(i)
              for i in range(w.analysis_page.fishbone_scope.count())]
    assert labels and not labels[0].startswith("--"), labels


def test_gui_refuses_to_save_a_session_before_clustering(tmp_path, qapp, monkeypatch):
    """A session without clusters is just a worse Excel export."""
    import src.gui as gui_mod
    from src.gui import ClusterApp

    pd = pytest.importorskip("pandas")
    seen = []
    monkeypatch.setattr(gui_mod.QMessageBox, "information",
                        staticmethod(lambda p, t, m, *a, **k: seen.append((t, m))))
    monkeypatch.setattr(gui_mod.QMessageBox, "warning",
                        staticmethod(lambda p, t, m, *a, **k: seen.append((t, m))))
    monkeypatch.setattr(gui_mod.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("reached the file dialog"))))

    w = ClusterApp()
    w.df = pd.DataFrame({"a": [1, 2]})
    w.cluster_data = None
    w._save_session()

    assert seen and "Run clustering first" in seen[-1][1]


def test_session_is_refused_while_a_run_is_in_progress(tmp_path, qapp, monkeypatch):
    import src.gui as gui_mod
    from src.gui import ClusterApp

    seen = []
    monkeypatch.setattr(gui_mod.QMessageBox, "information",
                        staticmethod(lambda p, t, m, *a, **k: seen.append((t, m))))
    monkeypatch.setattr(gui_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("reached the file dialog during a run"))))

    w = ClusterApp()
    w._run_in_progress = True
    w._open_session()
    w._run_in_progress = False

    assert seen and "clustering run is in progress" in seen[-1][1]
