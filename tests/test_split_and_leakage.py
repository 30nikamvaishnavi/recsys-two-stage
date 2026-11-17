"""Time-based split + point-in-time (no-leakage) guarantees."""

import numpy as np
from recsys.data.validate import validate_interactions
from recsys.features import pandas_features as pf


def test_split_is_temporal_not_random(world):
    inter = world.ds.interactions
    mx = [inter.loc[inter["stage"] == s, "ts"].max() for s in range(4)]
    mn = [inter.loc[inter["stage"] == s, "ts"].min() for s in range(4)]
    for s in range(3):
        assert mx[s] <= mn[s + 1], f"stage {s} overlaps stage {s + 1} in time"
    b = world.ds.boundaries
    assert b["t1"] < b["t2"] < b["t3"]


def test_split_proportions(world):
    frac = world.ds.interactions["stage"].value_counts(normalize=True).sort_index()
    assert abs(frac[0] - 0.70) < 0.02 and abs(frac[3] - 0.10) < 0.02


def test_validate_passes(world):
    validate_interactions(world.ds.interactions, world.ds.boundaries)


def _scramble_future(inter, cutoff, seed=1):
    """Counterfactual world: rewrite everything at/after the cutoff (random items, ratings)."""
    rng = np.random.default_rng(seed)
    x = inter.copy()
    m = (x["ts"] >= cutoff).to_numpy()
    x.loc[m, "item_idx"] = rng.integers(1, x["item_idx"].max() + 1, m.sum()).astype("int32")
    x.loc[m, "rating"] = rng.choice([1.0, 5.0], m.sum()).astype("float32")
    x["is_pos"] = x["rating"] >= 4.0
    return x.drop_duplicates(["user_idx", "item_idx"])


def test_item_popularity_uses_only_past(world):
    inter, cut = world.ds.interactions, world.ds.boundaries["t2"]
    a = pf.item_pop(inter, cut, [7, 30], world.n_items)
    b = pf.item_pop(_scramble_future(inter, cut), cut, [7, 30], world.n_items)
    assert a.equals(b), "popularity features changed when the FUTURE changed -> leakage"


def test_user_state_history_uses_only_past(world):
    inter, cut = world.ds.interactions, world.ds.boundaries["t2"]
    N = world.cfg["features"]["history_len"]
    a = pf.user_state(inter, cut, cut + 10**9, N).set_index("user_idx")
    sc = _scramble_future(inter, cut)
    b = pf.user_state(sc, cut, cut + 10**9, N).set_index("user_idx")
    common = a.index.intersection(b.index)
    assert len(common) > 50
    for col in ("n_all", "n_pos", "avg_rating", "days_since_last"):
        assert np.allclose(
            a.loc[common, col].astype(float), b.loc[common, col].astype(float), equal_nan=True
        ), col
    assert all(
        np.array_equal(x, y)
        for x, y in zip(a.loc[common, "hist_items"], b.loc[common, "hist_items"], strict=True)
    )


def test_train_examples_only_stage0_and_history_strictly_earlier(world):
    inter, ex = world.ds.interactions, world.train_ex
    assert ex["ts"].max() < world.ds.boundaries["t1"]
    # history of an example must be positives with earlier-or-equal order of the same user
    u = ex["user_idx"].iloc[len(ex) // 2]
    rows = inter[(inter["user_idx"] == u) & inter["is_pos"] & (inter["stage"] == 0)].sort_values(
        ["ts", "item_idx"]
    )
    e = ex[ex["user_idx"] == u].sort_values(["ts", "item_idx"])
    for pos, (_, r) in enumerate(e.iterrows()):
        expected = rows["item_idx"].iloc[max(0, pos - 20) : pos].tolist()
        got = [i for i in r["hist_items"] if i != 0]
        assert got == expected


def test_targets_are_not_in_history_or_seen(world):
    st, seen = world.states["val"], world.seen["val"]
    for i in range(0, len(st), 15):
        s = set(seen.seen(int(st.user_idx[i])).tolist())
        assert not (set(st.targets[i].tolist()) & s), "target already seen before the cutoff"


def test_eda_summary_and_figures(world, tmp_path):
    from recsys.eval.eda import make_figures, summarize

    s = summarize(world.ds)
    assert s["n_users"] == world.n_users and 0 < s["positive_rate"] < 1 and s["sparsity"] > 0.8
    assert s["cold_items_in_test_window_pct"] > 0 and s["top1pct_items_share"] > 0.01
    files = make_figures(world.ds, tmp_path)
    assert len(files) == 4 and all(f.stat().st_size > 1000 for f in files)
