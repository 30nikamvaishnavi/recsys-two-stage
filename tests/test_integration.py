"""Whole online path + simulated A/B using the real Recommender and a numpy fake model server."""

import numpy as np
from helpers import FakeBackend
from recsys.abtest import simulate as SIM
from test_pipeline import make_rec, stack  # noqa: F401


def test_end_to_end_ab_replay_on_real_pipeline(stack):  # noqa: F811
    rec = make_rec(stack)
    w, st = stack["world"], stack["states"]
    um = dict(zip(w.ds.user_map["user_idx"], w.ds.user_map["user_id"], strict=True))
    items = w.ds.items.set_index("item_idx")["movie_id"]
    import pandas as pd

    sub = st.sample(min(300, len(st)), random_state=0)
    users = pd.DataFrame(
        {
            "user_id": [int(um[u]) for u in sub["user_idx"]],
            "relevant": [set(items.loc[t].tolist()) for t in sub["targets"]],
        }
    )
    acfg = {**w.cfg["abtest"], "n_bootstrap": 200}

    def rec_fn(u, variant, k):
        r = rec.recommend(u, k, variant)
        return [x["item_id"] for x in r["items"]], r["meta"]

    df = SIM.run_replay(rec_fn, users, acfg)
    res = SIM.analyze(df, acfg)
    assert (
        res["simulation"]
        and set(res["arms"]) == {"A", "B"}
        and res["n_users"]["A"] + res["n_users"]["B"] == len(users)
    )
    assert 0 <= res["arms"]["A"]["ctr"] <= 1 and res["decision"]["decision"] in {
        "SHIP_B",
        "KEEP_A",
        "INCONCLUSIVE",
    }
    assert (
        df["strategy"]
        .isin(["two_stage", "content_blend", "genre_popularity", "global_popularity"])
        .all()
    )
    # personalised retrieval must beat random slate on held-out future positives
    assert df["recall@10"].mean() > 10 / w.n_items * 3


def test_throughput_smoke_with_cache(stack):  # noqa: F811
    rec = make_rec(stack, FakeBackend(stack["vecs"]))
    ids = list(stack["world"].ds.user_map["user_id"][:60])
    for u in ids:
        rec.recommend(int(u), 10, "dcn_v2")
    for u in ids:
        rec.recommend(int(u), 10, "dcn_v2")
    assert rec.cache.stats()["hit_rate"] >= 0.45
    assert np.isfinite(rec.recommend(int(ids[0]), 10, "dcn_v2")["meta"]["latency_ms"]["total"])
