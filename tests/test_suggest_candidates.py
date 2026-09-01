"""Tests for the Min Cluster Size suggestion candidate sweep and the config
round-trip that feeds it.

`_candidate_sizes` became load-bearing once `min_cluster_size` was added to
_gather_settings(): the sweep reads `current` from the settings dict to pin the
user's actual value into the candidate list, so it must always appear.
"""
from src.clustering import _candidate_sizes
from src.config import DEFAULTS


# --- _candidate_sizes ------------------------------------------------------
def test_returns_sorted_nonempty():
    sizes = _candidate_sizes(3000)
    assert sizes == sorted(sizes)
    assert sizes
    assert all(isinstance(s, int) for s in sizes)


def test_never_below_two():
    for n in (0, 1, 2, 5, 10, 100000):
        assert all(s >= 2 for s in _candidate_sizes(n))


def test_clamped_to_a_third_of_n():
    # hi = n // 3, so with n=30 nothing above 10 from the base list survives.
    assert max(_candidate_sizes(30)) <= 10


def test_current_is_always_included():
    sizes = _candidate_sizes(3000, current=37)
    assert 37 in sizes, sizes


def test_current_included_even_above_the_clamp():
    # current is added AFTER the n//3 clamp, so an unusual value still gets swept
    # (this is what lets the user's own setting show up in the picker).
    sizes = _candidate_sizes(30, current=25)   # hi == 10, yet 25 <= n
    assert 25 in sizes, sizes


def test_current_beyond_n_is_ignored():
    sizes = _candidate_sizes(50, current=999)   # 999 > n, not a usable size
    assert 999 not in sizes


def test_current_below_two_is_ignored():
    for bad in (0, 1):
        sizes = _candidate_sizes(3000, current=bad)
        assert bad not in sizes
        assert all(s >= 2 for s in sizes)


def test_tiny_dataset_still_yields_a_candidate():
    assert _candidate_sizes(1) == [2]


# --- config: min_cluster_size is a real, persistable setting ---------------
def test_default_exists():
    assert "min_cluster_size" in DEFAULTS["clustering"]
    assert isinstance(DEFAULTS["clustering"]["min_cluster_size"], int)


def test_save_load_round_trip_preserves_non_default(tmp_path, monkeypatch):
    """A non-default min_cluster_size must survive save -> load.

    Regression guard for the bug where the value never reached save_config at all
    (it was missing from _gather_settings), so it silently reset to 15 each launch.
    """
    import src.config as cfg

    # raising=True on purpose: if these private names are ever renamed the test must
    # fail loudly rather than silently writing to the real user config file.
    monkeypatch.setattr(cfg, "_CONFIG_DIR", str(tmp_path), raising=True)
    monkeypatch.setattr(cfg, "_USER_CONFIG_PATH",
                        str(tmp_path / "user_settings.json"), raising=True)

    settings = {"clustering": dict(DEFAULTS["clustering"])}
    settings["clustering"]["min_cluster_size"] = 42
    cfg.save_config(settings)

    loaded = cfg.load_config()
    assert loaded["clustering"]["min_cluster_size"] == 42


def test_round_trip_test_does_not_touch_the_real_config(tmp_path, monkeypatch):
    """Guard the guard: saving under a patched path must create the file there."""
    import src.config as cfg

    monkeypatch.setattr(cfg, "_CONFIG_DIR", str(tmp_path), raising=True)
    target = tmp_path / "user_settings.json"
    monkeypatch.setattr(cfg, "_USER_CONFIG_PATH", str(target), raising=True)

    settings = {"clustering": dict(DEFAULTS["clustering"])}
    settings["clustering"]["min_cluster_size"] = 7
    cfg.save_config(settings)
    assert target.exists(), "save_config wrote somewhere other than the patched path"
