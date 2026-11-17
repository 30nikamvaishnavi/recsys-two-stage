import numpy as np
from recsys.eval.metrics import evaluate_ranked
from recsys.index.faiss_index import HAVE_FAISS, AnnIndex, NumpyFlatIndex, build
from recsys.retrieval import retrievers as R


def test_topk_from_scores_sorted():
    s = np.array([[0.1, 0.9, 0.5, 0.7], [3.0, 1.0, 2.0, 0.0]])
    i, v = R.topk_from_scores(s, 2)
    assert i.tolist() == [[1, 3], [0, 2]] and (np.diff(v, axis=1) <= 0).all()


def test_retrievers_never_return_padding_or_seen(world):
    st, seen = world.states["val"], world.seen["val"]
    sub = st.subsample(60, 0)
    for ret in (
        R.PopularityRetriever(world.tables["val"]),
        R.ContentRetriever(world.tables["val"], 0.7),
    ):
        items, _ = ret.retrieve(sub, 50, seen)
        assert (items != 0).all()
        for r in range(len(sub)):
            assert not (set(items[r].tolist()) & set(seen.seen(int(sub.user_idx[r])).tolist()))


def test_svd_mf_beats_popularity_on_validation(world):
    """Sanity: the personalised baseline must beat non-personalised popularity on held-out future."""
    ex = world.train_ex
    mf = R.SVDRetriever.fit(
        ex["user_idx"].to_numpy(), ex["item_idx"].to_numpy(), world.n_users, world.n_items, 16
    )
    st = world.states["val"]
    warm = st.take(np.where(st.n_pos >= 5)[0])
    seen = world.seen["val"]
    pop = R.evaluate_retriever(R.PopularityRetriever(world.tables["val"]), warm, seen, (10, 50))
    svd = R.evaluate_retriever(mf, warm, seen, (10, 50))
    assert svd["recall@50"] > pop["recall@50"] * 1.2, (svd, pop)


def test_embedding_retriever_matches_manual(world):
    d = 8
    E = np.random.default_rng(0).normal(size=(world.n_items + 1, d)).astype(np.float32)
    U = np.random.default_rng(1).normal(size=(len(world.states["val"]), d)).astype(np.float32)
    st = world.states["val"]
    ret = R.EmbeddingRetriever(lambda s: U[: len(s)], E)
    sub = st.take(np.arange(5))
    items, scores = ret.retrieve(sub, 5, None)
    manual = U[:5] @ E.T
    manual[:, 0] = -np.inf
    assert np.array_equal(items[:, 0], manual.argmax(1))


def test_numpy_index_matches_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 16)).astype(np.float32)
    q = rng.normal(size=(5, 16)).astype(np.float32)
    idx = NumpyFlatIndex(16)
    idx.add(x)
    s, i = idx.search(q, 10)
    assert np.array_equal(i[:, 0], (q @ x.T).argmax(1)) and (np.diff(s, axis=1) <= 1e-6).all()


def test_ann_roundtrip_and_id_mapping(tmp_path):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)  # unit norm => self is the max inner product
    a = build(
        "flat",
        emb,
        {"hnsw_m": 8, "hnsw_ef_construction": 40, "ivf_nlist": 8, "pq_m": 4, "pq_nbits": 6},
    )
    q = emb[:3]
    items, _ = a.search(q, 5)
    assert (items[:, 0] == np.array([1, 2, 3])).all()  # item ids are 1-based; row r <-> item r+1
    a.save(tmp_path)
    b = AnnIndex.load(tmp_path)
    items2, _ = b.search(q, 5)
    assert np.array_equal(items, items2)


def test_faiss_hnsw_and_ivfpq_recall():
    import pytest

    if not HAVE_FAISS:
        pytest.skip("faiss not installed")
    from recsys.index.benchmark import run_benchmark

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(3000, 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cfg = {
        "hnsw_m": 16,
        "hnsw_ef_construction": 100,
        "hnsw_ef_search": [64],
        "ivf_nlist": 32,
        "ivf_nprobe": [16],
        "pq_m": 8,
        "pq_nbits": 8,
        "target_recall": 0.5,
    }
    rows, chosen, _ = run_benchmark(emb, emb[:100], cfg, 10, 20)
    by = {r["name"]: r for r in rows}
    assert by["flat"]["recall@100"] == 1.0 and by["hnsw_ef64"]["recall@100"] > 0.8
    assert by["ivfpq_np16"]["memory_mb"] < by["flat"]["memory_mb"]


def test_evaluate_retriever_uses_future_targets_only(world):
    st = world.states["val"].subsample(80, 0)
    # an oracle that returns the targets must score ~1.0 -> proves the metric pipeline wiring
    items = np.zeros((len(st), 100), dtype=np.int64)
    for i, t in enumerate(st.targets):
        items[i, : min(len(t), 100)] = t[:100]
    assert evaluate_ranked(items, st.targets, (100,))["recall@100"] > 0.95
