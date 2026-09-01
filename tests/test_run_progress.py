"""The progress value run() reports must never go backwards.

Guarding this at runtime rather than by reading the source, because the source has
several mutually-exclusive if/else branches that report different values (cache hit vs
generate, reuse-projection vs fresh UMAP) — static ordering cannot tell those apart from
a genuine regression, and reported 3 false positives when tried.

The bug this locks down: run() logged 0.68 after topic modelling, then 0.55 for the
intermediate save and 0.60 for "Naming Subcategories". The 0.60 was unconditional, so
the bar jumped backwards on *every* run. A third, narrower one: the shared
"UMAP: <label>" line reported 0.42, below the 0.43 the sampling branch had just logged.
"""
import pytest


def _drive_run(monkeypatch, n=40, use_sampling_threshold=None, reuse_projection=False,
               intermediate=False):
    """Run the whole pipeline against fakes and return the list of progress values."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    import src.clustering as clustering

    monkeypatch.setattr(clustering, "_lazy_import_ml", lambda: None)
    monkeypatch.setattr(clustering, "_np", np, raising=False)

    class FakeUMAP:
        def __init__(self, **kw):
            self.n_components = kw.get("n_components", 5)

        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X)[:, : self.n_components]

        def fit_transform(self, X):
            return self.transform(X)

    class FakeHDBSCAN:
        def __init__(self, **kw):
            self.relative_validity_ = 0.5

        def fit_predict(self, X):
            labels = np.array([i % 3 for i in range(len(X))])
            labels[0] = -1
            return labels

    class FakeTopicModel:
        def __init__(self, **kw):
            self._topics = None

        def fit_transform(self, docs, embeddings):
            self._topics = [i % 3 for i in range(len(docs))]
            self._topics[0] = -1
            return self._topics, None

        def get_topic_info(self):
            return pd.DataFrame({"Topic": [-1, 0, 1, 2],
                                 "Name": ["-1_noise", "0_a_b", "1_c_d", "2_e_f"]})

        def get_topic(self, tid):
            return [("alpha", 0.9), ("beta", 0.5), ("gamma", 0.2)]

    class FakeVectorizer:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(clustering, "_UMAP", FakeUMAP, raising=False)
    monkeypatch.setattr(clustering, "_HDBSCAN", FakeHDBSCAN, raising=False)
    monkeypatch.setattr(clustering, "_BERTopic", FakeTopicModel, raising=False)
    monkeypatch.setattr(clustering, "_CountVectorizer", FakeVectorizer, raising=False)
    monkeypatch.setattr(clustering, "_ClassTfidfTransformer", FakeVectorizer,
                        raising=False)
    # get_stopwords pulls in sklearn's ENGLISH_STOP_WORDS, which isn't installed in the
    # dev venv; the vectorizer is a fake here anyway.
    monkeypatch.setattr(clustering, "get_stopwords", lambda *a, **k: ["the", "a"])

    embeddings = np.random.default_rng(0).random((n, 8)).astype("float32")
    monkeypatch.setattr(
        clustering, "get_embedding_model_resolved",
        lambda *a, **k: (object(), "pytorch", None))

    class FakeEmbModel:
        def encode(self, batch, **kw):
            return embeddings[: len(batch)]

    monkeypatch.setattr(
        clustering, "get_embedding_model_resolved",
        lambda *a, **k: (FakeEmbModel(), "pytorch", None))

    if use_sampling_threshold is not None:
        # Force the "large dataset" UMAP branch without building a large dataset.
        monkeypatch.setattr(clustering, "_sample_indices",
                            lambda total, size: list(range(min(size, total))))

    progress = []

    def log(msg, prog=None):
        if prog is not None:
            progress.append(float(prog))

    c = clustering.TicketClusterer()
    c._cache_enabled = False
    docs = [f"vpn tunnel drops for user {i}" for i in range(n)]

    hook = (lambda emb, cur: (cur, emb[:, :5])) if reuse_projection else None
    inter = (lambda topics, names: None) if intermediate else None
    c.run(docs, min_cluster_size=2, use_preprocessing=False,
          label_categories=False, callback=log, on_embeddings_ready=hook,
          intermediate_callback=inter)
    return progress


def _assert_monotonic(progress):
    assert progress, "run() reported no progress at all"
    drops = [(i, progress[i - 1], progress[i])
             for i in range(1, len(progress)) if progress[i] < progress[i - 1]]
    assert not drops, (
        f"progress went backwards at {drops} in sequence {progress}")
    assert progress[-1] == pytest.approx(1.0), "run() should finish at 1.0"


def test_progress_never_goes_backwards(monkeypatch):
    _assert_monotonic(_drive_run(monkeypatch))


def test_progress_never_goes_backwards_when_reusing_the_projection(monkeypatch):
    """The mid-run picker path skips UMAP and reports 0.52 instead."""
    _assert_monotonic(_drive_run(monkeypatch, reuse_projection=True))


def test_the_intermediate_save_does_not_rewind_the_bar(monkeypatch):
    """The regression that started this: 0.68 -> 0.55 -> 0.60."""
    progress = _drive_run(monkeypatch, intermediate=True)
    after_topics = [p for p in progress if p >= 0.68]
    assert after_topics == sorted(after_topics), (
        f"values after topic modelling are not ascending: {after_topics}")


# --- the precomputed cluster-size table must agree with the old scan ---------
def test_precomputed_topic_sizes_match_the_scan():
    """_is_cluster_repetitive now prefers a Counter built once in run(); it must give
    exactly the same verdict the per-cluster O(N) scan did."""
    from collections import Counter
    import src.clustering as clustering

    c = clustering.TicketClusterer()
    topics = [0] * 7 + [1] * 4 + [2] * 5 + [-1] * 3

    scanned = {tid: c._is_cluster_repetitive(tid, topics)
               for tid in (-1, 0, 1, 2)}

    c._topic_sizes = Counter(topics)
    precomputed = {tid: c._is_cluster_repetitive(tid, topics)
                   for tid in (-1, 0, 1, 2)}

    assert scanned == precomputed
    # Sanity-check the boundary itself so the test can't pass by both being wrong.
    assert scanned[0] is True and scanned[2] is True     # >= MIN_REPETITIVE_SIZE (5)
    assert scanned[1] is False                           # 4 tickets
    assert scanned[-1] is False                          # noise is never repetitive


def test_numpy_int_keys_still_resolve():
    """topics comes from HDBSCAN as numpy ints; lookups must not silently miss."""
    from collections import Counter
    np = pytest.importorskip("numpy")
    import src.clustering as clustering

    c = clustering.TicketClusterer()
    topics = np.array([0] * 6 + [1] * 2)
    c._topic_sizes = Counter(topics.tolist())
    assert c._is_cluster_repetitive(np.int64(0), topics) is True
    assert c._is_cluster_repetitive(np.int64(1), topics) is False
