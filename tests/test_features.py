import numpy as np
import pandas as pd

from recsys.features import common as C
from recsys.features import pandas_features as pf
from recsys.features.item_table import ItemTable
from recsys.features.state import SeenIndex, StateArrays

DAY = 86400


def tiny():
    rows = [  # user, item, rating, ts, stage
        (1, 1, 5.0, 0 * DAY, 0), (1, 2, 3.0, 1 * DAY, 0), (1, 3, 4.5, 2 * DAY, 0), (1, 4, 4.0, 10 * DAY, 1), (1, 2, 5, 11 * DAY, 1),
        (2, 1, 4.0, 3 * DAY, 0), (2, 3, 2.0, 4 * DAY, 0), (2, 5, 4.0, 12 * DAY, 1),
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "ts", "stage"])
    df = df.drop_duplicates(["user_idx", "item_idx"], keep="first")  # (1,2) twice -> keep first like real cleaning
    df["is_pos"] = df["rating"] >= 4.0
    return df.astype({"user_idx": "int32", "item_idx": "int32", "rating": "float32", "ts": "int64"})


def test_pad_history_and_masked_mean():
    h = C.pad_history([5, 6, 7, 8], 3)
    assert h.tolist() == [6, 7, 8]
    assert C.pad_history([5], 3).tolist() == [5, 0, 0]
    table = np.arange(12, dtype=np.float32).reshape(6, 2)
    m = C.masked_mean(np.array([[1, 2, 0]]), table)
    assert np.allclose(m, table[[1, 2]].mean(0, keepdims=True))
    assert np.allclose(C.masked_mean(np.zeros((1, 3), int), table), 0)


def test_parse_year_genres_buckets():
    assert C.parse_year("Toy Story (1995)") == 1995
    assert np.isnan(C.parse_year("No year"))
    assert C.parse_genres("Comedy|Drama") == [C.GENRE_IDX["Comedy"], C.GENRE_IDX["Drama"]]
    assert C.parse_genres(np.nan) == [C.GENRE_IDX["(no genres listed)"]]
    yb = C.year_bucket(np.array([np.nan, 1940.0, 1995.0, 2020.0]))
    assert yb[0] == 0 and yb[1] == 1 and yb[3] == C.N_YEAR_BUCKETS - 1 and yb[1] < yb[2] < yb[3]


def test_popularity_buckets_monotone():
    pop = np.array([0, 0, 1, 2, 5, 10, 100, 1000], float)
    edges = C.popularity_edges(pop, 5)
    b = C.bucketize(pop, edges)
    assert b[0] == 0 and (np.diff(b) >= 0).all() and b.max() <= 4


def test_user_state_hand_computed():
    d = tiny()
    st = pf.user_state(d, cutoff=5 * DAY, window_end=20 * DAY, history_len=3).set_index("user_idx")
    u1 = st.loc[1]
    assert u1["n_all"] == 3 and u1["n_pos"] == 2
    assert np.isclose(u1["avg_rating"], (5 + 3 + 4.5) / 3)
    assert [i for i in u1["hist_items"] if i] == [1, 3]
    assert np.isclose(u1["days_since_last"], 3.0)
    assert sorted(u1["targets"]) == [4]
    assert sorted(st.loc[2]["targets"]) == [5]


def test_new_user_with_no_history_has_zero_state():
    d = tiny()
    d = pd.concat([d, pd.DataFrame({"user_idx": [3], "item_idx": [1], "rating": [5.0], "ts": [12 * DAY], "stage": [1], "is_pos": [True]})])
    st = pf.user_state(d, 5 * DAY, 20 * DAY, 3).set_index("user_idx")
    assert st.loc[3, "n_all"] == 0 and st.loc[3, "n_pos"] == 0 and (st.loc[3, "hist_items"] == 0).all()


def test_item_pop_windows():
    d = tiny()
    p = pf.item_pop(d, cutoff=12 * DAY, windows_days=[7, 30], n_items=5).set_index("item_idx")
    assert p.loc[1, "pop_all"] == 2 and p.loc[3, "pop_all"] == 1 and p.loc[5, "pop_all"] == 0  # (5 @12d is NOT < cutoff)
    assert p.loc[4, "pop_7d"] == 1 and p.loc[1, "pop_7d"] == 0 and p.loc[1, "pop_30d"] == 2


def test_train_examples_counts_prior_activity():
    d = tiny()
    ex = pf.train_examples(d, 3).sort_values(["user_idx", "ts"]).reset_index(drop=True)
    assert (ex["stage"] if "stage" in ex else pd.Series([0])).max() == 0
    r = ex[(ex.user_idx == 1) & (ex.item_idx == 3)].iloc[0]
    assert r["n_prior"] == 2 and np.isclose(r["avg_prior_rating"], 4.0) and np.isclose(r["days_since_prev"], 1.0)
    assert [i for i in r["hist_items"] if i] == [1]  # item 2 was rated 3.0 -> not a positive


def test_seen_index_masks_scores():
    d = tiny()
    seen = SeenIndex(d, 5 * DAY, n_users=3, n_items=5)
    sc = np.ones((2, 6), np.float32)
    seen.mask_scores(sc, np.array([1, 2]))
    assert np.isinf(sc[0, [1, 2, 3]]).all() and np.isfinite(sc[0, [4, 5]]).all() and np.isinf(sc[1, [1, 3]]).all()


def test_truncate_history_keeps_most_recent():
    s = StateArrays(np.array([1]), np.array([[1, 2, 3, 4, 0]], np.int32), np.zeros((1, 3), np.float32), np.array([4], np.int32))
    t = s.truncate_history(2)
    assert t.hist[0].tolist() == [3, 4, 0, 0, 0] and t.n_pos[0] == 2


def test_item_table_roundtrip(world, tmp_path):
    t = world.tables["test"]
    t.save(tmp_path / "t.npz")
    t2 = ItemTable.load(tmp_path / "t.npz")
    assert np.array_equal(t.genres, t2.genres) and t2.n_pop_buckets == t.n_pop_buckets
    assert t.genres[0].sum() == 0 and t.pop[0].sum() == 0  # padding row
    assert t.cold_mask.sum() >= 0 and not t.cold_mask[0]
    v = t.content_vectors()
    assert np.allclose(np.linalg.norm(v[1:], axis=1), 1.0, atol=1e-4) or (np.linalg.norm(v[1:], axis=1) > 0).all()
