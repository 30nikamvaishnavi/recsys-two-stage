"""Online pipeline end-to-end with a numpy fake model server (no TF / Triton / Redis needed)."""

import numpy as np
import pytest
from helpers import FakeBackend
from recsys.cache.redis_cache import make_cache
from recsys.coldstart.router import ColdStartRouter
from recsys.index.faiss_index import build
from recsys.retrieval.retrievers import SVDRetriever
from recsys.serving.pipeline import Recommender
from recsys.serving.state_store import UserStateStore


@pytest.fixture(scope="session")
def stack(world):
    import scipy.sparse as sp
    from recsys.features import pandas_features as pf

    ex, cfg = world.train_ex, world.cfg
    mf = SVDRetriever.fit(
        ex["user_idx"].to_numpy(), ex["item_idx"].to_numpy(), world.n_users, world.n_items, 16
    )
    vecs = mf.v
    tab = world.tables["test"]
    ann = build(
        "flat",
        vecs[1:] / np.maximum(np.linalg.norm(vecs[1:], axis=1, keepdims=True), 1e-9),
        cfg["index"],
    )
    b = world.ds.boundaries
    states = pf.user_state(world.ds.interactions, b["t3"], 2**62, cfg["features"]["history_len"])
    seen = world.seen["test"].m
    store = UserStateStore(
        world.ds.user_map, states, sp.csr_matrix(seen), cfg["features"]["history_len"], b["t3"]
    )
    router = ColdStartRouter(tab, 5)
    return dict(
        world=world, vecs=vecs, ann=ann, tab=tab, store=store, router=router, cfg=cfg, states=states
    )


def make_rec(stack, backend=None, cache=None):
    w = stack["world"]
    emb = stack["vecs"] / np.maximum(np.linalg.norm(stack["vecs"], axis=1, keepdims=True), 1e-9)
    return Recommender(backend or FakeBackend(stack["vecs"]), stack["ann"], emb, stack["tab"], w.ds.items, stack["store"],
                       cache or make_cache(force_memory=True), stack["router"], stack["cfg"])  # fmt: skip


def ext_id(stack, idx):
    inv = dict(
        zip(
            stack["world"].ds.user_map["user_idx"],
            stack["world"].ds.user_map["user_id"],
            strict=True,
        )
    )
    return int(inv[idx])


def warm_user(stack):
    st = stack["states"]
    return int(st.loc[st["n_pos"] >= 15, "user_idx"].iloc[0])


def test_warm_user_two_stage_filters_seen_and_reports_latency(stack):
    rec = make_rec(stack)
    idx = warm_user(stack)
    out = rec.recommend(ext_id(stack, idx), 10, "dcn_v2")
    assert out["meta"]["strategy"] == "two_stage" and out["meta"]["reason_code"] == "WARM_USER"
    assert len(out["items"]) == 10 and len({i["item_id"] for i in out["items"]}) == 10
    seen_movies = {int(rec.movie_id[i]) for i in stack["store"].seen(stack["store"].states[idx])}
    assert not ({i["item_id"] for i in out["items"]} & seen_movies)
    lat = out["meta"]["latency_ms"]
    assert {"user_tower", "ann_search", "features", "ranker", "postprocess", "total"} <= set(
        lat
    ) and lat["total"] >= lat["ranker"]
    assert all(
        i["retrieval_score"] is not None and 0 <= i["ranker_score"] <= 1 for i in out["items"]
    )


def test_cache_hit_on_second_call_and_invalidation_on_feedback(stack):
    rec = make_rec(stack)
    u = ext_id(stack, warm_user(stack))
    a = rec.recommend(u, 10, "dcn_v2")
    b = rec.recommend(u, 10, "dcn_v2")
    assert not a["meta"]["cache_hit"] and b["meta"]["cache_hit"]
    assert [i["item_id"] for i in a["items"]] == [i["item_id"] for i in b["items"]]
    assert rec.cache.get_recs("wide_deep", u, 10) is None  # other variant not polluted
    top = a["items"][0]["item_id"]
    fb = rec.feedback(u, top, "click")
    assert fb["cache_entries_invalidated"] >= 1
    c = rec.recommend(u, 10, "dcn_v2")
    assert not c["meta"]["cache_hit"] and top not in [
        i["item_id"] for i in c["items"]
    ]  # consumed item now filtered


def test_new_user_routes_to_onboarding_then_global_popularity(stack):
    rec = make_rec(stack)
    g = rec.recommend(10**9, 10, "dcn_v2", onboarding_genres=["Comedy", "Romance"])
    assert (
        g["meta"]["reason_code"] == "NEW_USER_ONBOARDING_GENRES"
        and g["meta"]["strategy"] == "genre_popularity"
    )
    t = stack["tab"]
    assert all(t.genres[rec.idx_of_movie[i["item_id"]], [2 + 2, 14]].sum() >= 1 for i in g["items"])
    p = rec.recommend(10**9 + 1, 10, "wide_deep")
    assert (
        p["meta"]["reason_code"] == "NEW_USER_NO_SIGNAL" and rec.backend.calls == 0
    )  # models never called


def test_sparse_history_uses_content_blend(stack):
    rec = make_rec(stack)
    st = stack["states"]
    idx = (
        int(st.loc[(st["n_pos"] >= 1) & (st["n_pos"] < 5), "user_idx"].iloc[0])
        if ((st["n_pos"] >= 1) & (st["n_pos"] < 5)).any()
        else None
    )
    if idx is None:
        pytest.skip("no sparse user in this synthetic world")
    out = rec.recommend(ext_id(stack, idx), 10, "dcn_v2")
    assert out["meta"]["reason_code"] == "SPARSE_HISTORY_CONTENT_BLEND"


def test_model_failure_degrades_to_popularity_not_error(stack):
    rec = make_rec(stack, FakeBackend(stack["vecs"], fail=True))
    out = rec.recommend(ext_id(stack, warm_user(stack)), 10, "dcn_v2")
    assert out["meta"]["reason_code"] == "MODEL_UNAVAILABLE_POPULARITY" and len(out["items"]) == 10


def test_similar_items_and_unknown_item(stack):
    rec = make_rec(stack)
    mid = int(rec.movie_id[5])
    sims = rec.similar_items(mid, 5)
    assert (
        len(sims) == 5
        and mid not in [s["item_id"] for s in sims]
        and sims[0]["similarity"] >= sims[-1]["similarity"]
    )
    with pytest.raises(KeyError):
        rec.similar_items(-1, 5)


def test_invalid_variant_rejected(stack):
    with pytest.raises(ValueError):
        make_rec(stack).recommend(1, 10, "nope")
