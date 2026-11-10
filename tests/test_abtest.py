import math

import numpy as np
import pandas as pd

from recsys.abtest import simulate as SIM
from recsys.abtest import stats as S

ACFG = {"salt": "t", "arms": {"A": "wide_deep", "B": "dcn_v2"}, "slate_size": 10, "relevant_click_prob": 0.65, "noise_click_prob": 0.02,
        "examination_decay": 1.0, "n_bootstrap": 300, "alpha": 0.05}  # fmt: skip


def test_bucketing_deterministic_balanced_and_salted():
    a = [S.bucket(u, "s") for u in range(20000)]
    assert a == [S.bucket(u, "s") for u in range(20000)]
    assert abs(np.mean(a) - 0.5) < 0.02
    b = [S.bucket(u, "other") for u in range(20000)]
    assert 0.4 < np.mean(np.array(a) == np.array(b)) < 0.6  # different salt => independent assignment


def test_two_proportion_ztest_known_value():
    r = S.two_proportion_ztest(100, 1000, 130, 1000)
    pooled = 230 / 2000
    z = 0.03 / math.sqrt(pooled * (1 - pooled) * 2 / 1000)
    assert math.isclose(r["z"], z, rel_tol=1e-9) and 0.03 < r["p_value"] < 0.06
    assert S.two_proportion_ztest(100, 1000, 100, 1000)["p_value"] == 1.0


def test_bootstrap_ci_covers_truth_and_diff_detects_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1.0, 2000)
    m, lo, hi = S.bootstrap_ci(x, 500)
    assert lo < 0.3 < hi and lo < m < hi
    d = S.bootstrap_diff(rng.normal(0, 1, 3000), rng.normal(0.2, 1, 3000), 500)
    assert d["ci_low"] > 0 and d["p_value_bootstrap"] < 0.05
    n = S.bootstrap_diff(rng.normal(0, 1, 3000), rng.normal(0, 1, 3000), 500)
    assert n["ci_low"] < 0 < n["ci_high"]


def test_aa_false_positive_rate_is_near_alpha():
    x = np.random.default_rng(0).binomial(10, 0.2, 4000) / 10
    assert S.aa_false_positive_rate(x, 400, 0.05) < 0.10


def test_mde_shrinks_with_sample_size():
    assert S.min_detectable_effect(0.1, 1000) > S.min_detectable_effect(0.1, 100000)


def test_decision_rules():
    pos = {"ci_low": 0.01, "ci_high": 0.03, "diff": 0.02}
    neg = {"ci_low": -0.03, "ci_high": -0.01, "diff": -0.02}
    flat = {"ci_low": -0.01, "ci_high": 0.02, "diff": 0.005}
    ok_ndcg = {"ci_low": -0.01, "ci_high": 0.02, "diff": 0.0}
    assert S.decide(pos, ok_ndcg)["decision"] == "SHIP_B"
    assert S.decide(neg, ok_ndcg)["decision"] == "KEEP_A"
    assert S.decide(flat, ok_ndcg)["decision"] == "INCONCLUSIVE"
    assert S.decide(pos, {"ci_low": -0.05, "ci_high": -0.01, "diff": -0.03})["decision"] == "INCONCLUSIVE"


def test_click_model_is_position_biased_and_relevance_driven():
    rng = np.random.default_rng(0)
    slate = list(range(10))
    pos_clicks = np.zeros(10)
    rel_clicks = 0
    irr_clicks = 0
    for _ in range(4000):
        pos_clicks += SIM.simulate_clicks(slate, set(slate), ACFG, rng)
        rel_clicks += sum(SIM.simulate_clicks(slate, set(slate), ACFG, rng))
        irr_clicks += sum(SIM.simulate_clicks(slate, set(), ACFG, rng))
    assert pos_clicks[0] > pos_clicks[4] > pos_clicks[9]
    assert rel_clicks > 10 * irr_clicks


def test_replay_is_deterministic_and_assigns_arms_consistently():
    users = pd.DataFrame({"user_id": np.arange(1, 801), "relevant": [set(range(u % 7, u % 7 + 5)) for u in range(1, 801)]})

    def rec_fn(u, variant, k):  # variant B recommends relevant items higher
        base = list(range(u % 7, u % 7 + 5)) if variant == "dcn_v2" else list(range(50, 55))
        return (base + list(range(100, 100 + k - 5)))[:k], {"strategy": "fake"}

    d1 = SIM.run_replay(rec_fn, users, ACFG)
    d2 = SIM.run_replay(rec_fn, users, ACFG)
    assert d1.equals(d2)
    assert ((d1["arm"] == "A") == (d1["variant"] == "wide_deep")).all()
    res = SIM.analyze(d1, ACFG)
    assert res["simulation"] is True and res["decision"]["decision"] == "SHIP_B"
    assert res["arms"]["B"]["ctr"] > res["arms"]["A"]["ctr"] and res["tests"]["ctr_two_proportion_z"]["p_value"] < 0.01
