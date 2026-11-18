import numpy as np
from recsys.ranking.analysis import (
    ACTIVITY_NAMES,
    activity_buckets,
    bucket_means,
    item_popularity_buckets,
    list_stats,
    paired_bootstrap_diff,
    target_recall_by_bucket,
)


def test_paired_bootstrap_constant_difference_and_nan_handling():
    b = np.linspace(0.0, 1.0, 50)
    out = paired_bootstrap_diff(b + 0.1, b, 500)
    assert np.isclose(out["mean_diff"], 0.1) and out["ci_low"] > 0.09 and out["ci_high"] < 0.11
    a = np.array([0.3, np.nan, 0.5])
    assert paired_bootstrap_diff(a, np.array([0.1, 0.2, np.nan]), 50)["n_users"] == 1
    assert paired_bootstrap_diff([], [], 10)["n_users"] == 0


def test_paired_bootstrap_no_difference_includes_zero():
    rng = np.random.default_rng(0)
    x = rng.random(200)
    out = paired_bootstrap_diff(x, x[::-1].copy(), 500)
    assert out["ci_low"] < 0 < out["ci_high"]


def test_activity_buckets_are_roughly_equal_terciles():
    idx, (low, high) = activity_buckets(np.arange(30))
    assert [int((idx == b).sum()) for b in range(3)] == [10, 10, 10] and low < high
    rows = bucket_means({"m": np.arange(30.0)}, idx, ACTIVITY_NAMES)
    assert rows["light"]["m"] < rows["medium"]["m"] < rows["heavy"]["m"]
    assert rows["light"]["n_users"] == 10


def test_item_popularity_buckets_cold_tail_torso_head():
    n_pos = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    b = item_popularity_buckets(n_pos)
    assert b[0] == b[1] == 0 and b[-1] == 3 and b[2] == 1
    assert set(b.tolist()) == {0, 1, 2, 3}
    assert item_popularity_buckets(np.zeros(5)).tolist() == [0] * 5


def test_target_recall_by_bucket_hand_example():
    bucket_of_item = np.array([0, 3, 3, 1, 1, 2])
    ranked = np.array([[1, 3, 5, 0], [2, 4, 0, 0]])
    targets = [np.array([1, 2]), np.array([3, 4])]
    out = target_recall_by_bucket(ranked, targets, bucket_of_item, 3)
    # user 0 finds item 1 (head) but not 2 (head); user 1 finds item 4 (tail) but not 3 (tail)
    assert out["head"]["n_targets"] == 2 and out["head"]["target_recall@3"] == 0.5
    assert out["tail"]["n_targets"] == 2 and out["tail"]["target_recall@3"] == 0.5
    assert out["cold"]["n_targets"] == 0 and np.isnan(out["cold"]["target_recall@3"])


def test_list_stats_hand_example():
    genres = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)  # item 0 = padding
    n_pos = np.array([0, 4, 4, 1])
    out = list_stats(np.array([[1, 2], [1, 3]]), n_pos, genres, n_items=3)
    assert out["distinct_items"] == 3 and np.isclose(out["catalog_coverage"], 1.0)
    # list [1, 2]: identical genres -> distance 0; list [1, 3]: orthogonal -> distance 1
    assert np.isclose(out["intra_list_diversity"], 0.5)
    assert out["novelty_bits"] > 0
