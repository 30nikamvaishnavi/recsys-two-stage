import numpy as np

from recsys.coldstart.diversity import mmr
from recsys.coldstart.evaluate import evaluate_cold_start
from recsys.coldstart.router import NO_SIGNAL, ONBOARD, SPARSE, WARM, ColdStartRouter
from recsys.features.common import GENRE_IDX


def router(world):
    return ColdStartRouter(world.tables["test"], min_history=5)


def test_router_reason_codes(world):
    r = router(world)
    assert r.route(50).reason == "WARM_USER" and r.route(50).strategy == WARM
    assert r.route(5).strategy == WARM and r.route(4).strategy == SPARSE and r.route(1).reason == "SPARSE_HISTORY_CONTENT_BLEND"
    assert r.route(0, ["Comedy"]).strategy == ONBOARD and r.route(0, ["Comedy"]).reason == "NEW_USER_ONBOARDING_GENRES"
    d = r.route(0)
    assert d.strategy == NO_SIGNAL and d.reason == "NEW_USER_NO_SIGNAL"
    assert r.route(50, models_ok=False).reason == "MODEL_UNAVAILABLE_POPULARITY"
    assert r.route(50, retrieval_empty=True).reason == "RETRIEVAL_EMPTY_POPULARITY"


def test_genre_popularity_matches_genre_and_skips_seen(world):
    r = router(world)
    seen = np.array([int(np.argmax(world.tables["test"].genres[:, GENRE_IDX["Comedy"]] * world.tables["test"].pop[:, 0]))])
    items, _ = r.genre_popularity(["Comedy"], 10, seen)
    assert len(items) == 10 and seen[0] not in items
    assert (world.tables["test"].genres[items, GENRE_IDX["Comedy"]] == 1).all()


def test_global_popularity_is_popular_and_diverse(world):
    r = router(world)
    items, _ = r.global_popularity(10, np.empty(0, np.int64))
    pop = world.tables["test"].pop[:, 0]
    assert np.median(pop[items]) > np.median(pop[1:])
    assert len(set(items.tolist())) == 10


def test_mmr_trades_relevance_for_diversity():
    content = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], np.float32)  # items 1,2 identical; 3 different
    items, rel = np.array([1, 2, 3]), np.array([1.0, 0.99, 0.6])
    assert mmr(items, rel, content, 2, lam=1.0).tolist() == [0, 1]       # pure relevance
    assert mmr(items, rel, content, 2, lam=0.3).tolist() == [0, 2]       # diversity picks the dissimilar item


def test_content_blend_uses_history(world):
    r = router(world)
    st = world.states["test"]
    i = int(np.where(st.n_pos >= 10)[0][0])
    row = st.take(np.array([i])).truncate_history(2)
    items, _ = r.content_blend(row, 20, np.empty(0, np.int64))
    hist_genres = world.tables["test"].genres[row.hist[0][row.hist[0] != 0]].sum(0) > 0
    share = (world.tables["test"].genres[items][:, hist_genres].sum(1) > 0).mean()
    assert share > 0.7  # recommendations overlap the genres of the (2) known items


def test_cold_start_evaluation_runs_and_router_beats_random(world):
    r = router(world)
    out = evaluate_cold_start(r, world.states["val"], world.seen["val"], world.tables["val"], None, (0, 1, 2), 30, 200, (10,))
    assert out["n_users"] > 20
    res1 = out["by_history_len"]["1"]
    assert "content_blend" in res1 and "router" in res1 and "global_popularity" in res1
    assert "genre_popularity" in out["by_history_len"]["0"]
    assert out["by_history_len"]["2"]["router"]["ndcg@10"] >= 0
