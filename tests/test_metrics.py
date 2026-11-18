import math

import numpy as np
from recsys.eval import metrics as M


def test_recall_ndcg_map_hand_computed():
    ranked = np.array([[10, 20, 30, 40, 50]])
    targets = [np.array([20, 50, 99])]
    assert M.recall_at_k(ranked, targets, 3) == 1 / 3
    assert M.recall_at_k(ranked, targets, 5) == 2 / 3
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(M.ndcg_at_k(ranked, targets, 5), dcg / idcg)
    ap = (0.5 + 2 / 5) / 3
    assert math.isclose(M.map_at_k(ranked, targets, 5), ap)
    assert M.hit_at_k(ranked, targets, 1) == 0.0 and M.hit_at_k(ranked, targets, 2) == 1.0


def test_perfect_and_empty():
    ranked = np.array([[1, 2, 3], [4, 5, 6]])
    assert M.ndcg_at_k(ranked, [np.array([1, 2]), np.array([4])], 3) == 1.0
    assert math.isnan(M.recall_at_k(ranked, [np.array([]), np.array([])], 3))
    # users without targets are ignored, not counted as zeros
    assert M.recall_at_k(ranked, [np.array([1]), np.array([])], 3) == 1.0


def test_evaluate_ranked_keys():
    r = M.evaluate_ranked(np.arange(1, 101)[None, :], [np.array([1])], (10, 50, 100))
    assert {"recall@10", "ndcg@50", "hit@100", "map@10", "n_users"} <= set(r)


def test_detailed_metrics_per_user_and_bootstrap_on_tiny_example():
    ranked = np.array([[10, 20], [30, 40]])
    targets = [np.array([10, 99]), np.array([40])]

    result = M.evaluate_ranked_detailed(ranked, targets, (2,), n_bootstrap=2000, seed=3)

    assert result["per_user"]["recall@2"] == [0.5, 1.0]
    assert result["per_user"]["ndcg@2"] == [
        1 / (1 + 1 / math.log2(3)),
        1 / math.log2(3),
    ]
    assert result["recall@2"] == 0.75
    for metric in ("recall@2", "ndcg@2"):
        ci = result["bootstrap_ci"][metric]
        assert ci["ci_low"] <= result[metric] <= ci["ci_high"]
        assert (
            ci
            == M.evaluate_ranked_detailed(ranked, targets, (2,), n_bootstrap=2000, seed=3)[
                "bootstrap_ci"
            ][metric]
        )


def test_auc_logloss_calibration():
    y = np.array([0, 0, 1, 1])
    assert M.auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert M.auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert math.isclose(M.logloss(np.array([1, 0]), np.array([0.5, 0.5])), math.log(2))
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    yy = (rng.random(20000) < p).astype(float)  # perfectly calibrated by construction
    assert M.calibration_table(yy, p)["ece"] < 0.02
    assert M.calibration_table(np.zeros(1000), np.full(1000, 0.9))["ece"] > 0.8
