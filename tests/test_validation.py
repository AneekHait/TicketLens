"""Tests for src.validation."""
from src.validation import (
    validate_excel_file,
    validate_model_path,
    validate_clustering_settings,
)


def test_validate_excel_file_rejects_missing_and_bad_ext(tmp_path):
    assert validate_excel_file("")[0] is False
    assert validate_excel_file(str(tmp_path / "nope.xlsx"))[0] is False  # not found
    txt = tmp_path / "data.txt"
    txt.write_text("x")
    ok, err = validate_excel_file(str(txt))
    assert ok is False and ".txt" in err


def test_validate_excel_file_accepts_existing_xlsx(tmp_path):
    f = tmp_path / "book.xlsx"
    f.write_text("")  # existence + extension only
    ok, err = validate_excel_file(str(f))
    assert ok is True and err == ""


def test_validate_model_path(tmp_path):
    assert validate_model_path(None)[0] is False
    assert validate_model_path(str(tmp_path / "missing.gguf"))[0] is False
    bad = tmp_path / "model.txt"
    bad.write_text("x")
    ok, err = validate_model_path(str(bad))
    assert ok is False and ".gguf" in err
    good = tmp_path / "model.gguf"
    good.write_text("x")
    assert validate_model_path(str(good)) == (True, "")


def test_validate_clustering_settings_ok():
    ok, err = validate_clustering_settings(num_docs=100, min_cluster_size=5, umap_n_neighbors=15)
    assert ok is True and err == ""


def test_validate_clustering_settings_flags_bad_values():
    ok, err = validate_clustering_settings(num_docs=1, min_cluster_size=1, umap_n_neighbors=1)
    assert ok is False
    assert "at least 2 documents" in err

    ok, err = validate_clustering_settings(num_docs=10, min_cluster_size=50, umap_n_neighbors=5)
    assert ok is False and "cannot exceed" in err

    ok, err = validate_clustering_settings(num_docs=10, min_cluster_size=3, umap_n_neighbors=10)
    assert ok is False and "must be less than" in err
