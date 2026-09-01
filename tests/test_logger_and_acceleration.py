"""Coverage for two modules the suite never touched.

`logger` is the more pressing of the two: the `.log.N` rollover fix shipped with no test
at all, and the retention arithmetic is easy to get subtly wrong (the keep/reserve split
is off-by-one-prone by design, because pruning runs *before* the new handler opens its
file).

`acceleration.resolve_backend` is the backstop that forces PyTorch when OpenVINO is
missing or the model is incompatible, regardless of what the user persisted. It is a pure
function, so it needs no OpenVINO install to test.
"""
import os

import pytest

from src import logger as log_mod
from src.acceleration import resolve_backend


# --- logger: which files count as ours -------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("clustering_20260817_120057.log", True),
    ("clustering_20260817_120057.log.1", True),    # RotatingFileHandler rollover
    ("clustering_20260817_120057.log.12", True),
    ("notes.txt", False),
    ("tickets.xlsx", False),
    ("logfile", False),
    ("something.logical", False),                  # must not match on a prefix
])
def test_is_log_file(name, expected):
    assert log_mod._is_log_file(name) is expected


def test_rollovers_are_pruned_too(tmp_path, monkeypatch):
    """The whole point of the fix: .log.N files are up to MAX_LOG_BYTES each, and an
    endswith('.log') filter left every one of them on disk forever."""
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path))
    for i in range(6):
        (tmp_path / f"clustering_2026081{i}_000000.log").write_text("x")
        (tmp_path / f"clustering_2026081{i}_000000.log.1").write_text("x")
    unrelated = tmp_path / "keepme.txt"
    unrelated.write_text("not a log")

    log_mod._prune_old_logs(keep=4, reserve=1)

    left = sorted(p.name for p in tmp_path.iterdir())
    # Counted by pattern, deliberately NOT via _is_log_file: asserting through the same
    # helper under test let a broken filter mark its own homework (verified -- this test
    # passed against the pre-fix endswith('.log') version until the count was inlined).
    remaining_logs = [n for n in left if ".log" in n]
    # 12 log files in, keep=4/reserve=1 => 3 left. The count is the assertion that
    # matters: with the pre-fix endswith('.log') filter the 6 rollovers were invisible
    # to pruning and 9 files survived. Which 3 remain is just newest-first, and a
    # .log.1 legitimately can be among them.
    assert len(remaining_logs) == 3, remaining_logs
    assert "keepme.txt" in left, "pruning must not touch unrelated files"


def test_pruning_keeps_the_newest(tmp_path, monkeypatch):
    """Names are timestamped, so lexical order is chronological order."""
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path))
    for stamp in ("20260101", "20260601", "20261231"):
        (tmp_path / f"clustering_{stamp}_000000.log").write_text("x")

    log_mod._prune_old_logs(keep=2, reserve=1)

    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["clustering_20261231_000000.log"], left


def test_reserve_leaves_room_for_the_launch_about_to_start(tmp_path, monkeypatch):
    """Pruning runs before the new handler opens its file, so keep=N/reserve=1 must
    settle at N files once that one is created -- not N+1."""
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path))
    for i in range(10):
        (tmp_path / f"clustering_202601{i:02d}_000000.log").write_text("x")

    log_mod._prune_old_logs(keep=5, reserve=1)

    remaining = len(list(tmp_path.iterdir()))
    assert remaining == 4, remaining
    assert remaining + 1 == 5, "plus this launch's file == keep"


def test_pruning_a_missing_directory_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path / "nope"))
    log_mod._prune_old_logs(keep=3)      # must not raise


def test_keep_zero_removes_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path))
    (tmp_path / "clustering_20260101_000000.log").write_text("x")
    (tmp_path / "clustering_20260101_000000.log.1").write_text("x")

    log_mod._prune_old_logs(keep=0)

    assert not list(tmp_path.iterdir())


# --- logger: which file is "the latest" ------------------------------------
def test_latest_log_ignores_rollovers(tmp_path, monkeypatch):
    """A .log.1 holds the OLDER half of a launch's output, so offering it as the latest
    log would hand the user the wrong file. Pruning must see them; this must not."""
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path))
    (tmp_path / "clustering_20260101_000000.log").write_text("older")
    (tmp_path / "clustering_20260202_000000.log").write_text("newest")
    (tmp_path / "clustering_20260202_000000.log.1").write_text("rollover")

    latest = log_mod.get_latest_log_file()

    assert os.path.basename(latest) == "clustering_20260202_000000.log"


def test_latest_log_is_none_when_there_is_no_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path / "absent"))
    assert log_mod.get_latest_log_file() is None


def test_latest_log_is_none_when_only_rollovers_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(log_mod, "_LOG_DIR", str(tmp_path))
    (tmp_path / "clustering_20260101_000000.log.1").write_text("x")
    assert log_mod.get_latest_log_file() is None


def test_retention_constants_are_coherent():
    assert log_mod.MAX_LOG_FILES > 0
    assert log_mod.MAX_LOG_BYTES > 0
    # The count cap alone left file *size* unbounded, which is what the byte cap fixed.
    assert log_mod.LOG_BACKUP_COUNT >= 1


# --- acceleration: the downgrade backstop ----------------------------------
def test_pytorch_is_passed_through_without_a_reason():
    for value in (None, "", "pytorch"):
        assert resolve_backend(value, "BAAI/bge-base-en-v1.5") == ("pytorch", None, None)


def test_an_unknown_backend_falls_back_and_says_so():
    backend, precision, reason = resolve_backend("cuda_fp8", "BAAI/bge-base-en-v1.5")
    assert (backend, precision) == ("pytorch", None)
    assert "unknown acceleration" in reason and "cuda_fp8" in reason


def test_openvino_downgrades_when_not_installed(monkeypatch):
    import src.acceleration as accel

    monkeypatch.setattr(accel, "openvino_available", lambda: False)
    backend, precision, reason = accel.resolve_backend(
        "openvino_int8_cpu", "BAAI/bge-base-en-v1.5")
    assert (backend, precision) == ("pytorch", None)
    assert "not installed" in reason


def test_openvino_downgrades_for_an_incompatible_model(monkeypatch):
    """EmbeddingGemma is deliberately excluded; the persisted choice must not win."""
    import src.acceleration as accel

    monkeypatch.setattr(accel, "openvino_available", lambda: True)
    backend, precision, reason = accel.resolve_backend(
        "openvino_int8_cpu", "unsloth/embeddinggemma-300m")
    assert (backend, precision) == ("pytorch", None)
    assert "not OpenVINO-compatible" in reason


def test_openvino_is_selected_when_everything_lines_up(monkeypatch):
    import src.acceleration as accel

    monkeypatch.setattr(accel, "openvino_available", lambda: True)
    monkeypatch.setattr(accel, "model_supports_acceleration", lambda p: True)
    assert accel.resolve_backend("openvino_int8_cpu", "BAAI/bge-base-en-v1.5") == (
        "openvino", "int8", None)


def test_every_curated_embedding_model_resolves_without_raising():
    """resolve_backend is called for whatever is in the dropdown, so none of the
    catalog entries may blow up in it."""
    from src.config import EMBEDDING_MODELS

    for hf_path in EMBEDDING_MODELS.values():
        backend, _precision, _reason = resolve_backend("openvino_int8_cpu", hf_path)
        assert backend in ("pytorch", "openvino")
