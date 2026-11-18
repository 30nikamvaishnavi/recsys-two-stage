import numpy as np

from scripts.aggregate_ranking import aggregate, render_markdown


def _run(seed: int, shift: float) -> dict:
    def part(base):
        return {"ndcg@10": base, "recall@10": base / 2, "hit@10": base * 2, "recall@50": base * 3}

    det = {
        "ranker": {**part(0.2 + shift), "auc": 0.7},
        "retrieval_only": part(0.1),
        "paired_diff_vs_retrieval": {
            "ndcg@10": {"mean_diff": 0.1 + shift},
            "recall@10": {"mean_diff": 0.05 + shift},
        },
    }
    abl = {"full": {"ndcg@10": 0.2 + shift, "recall@10": 0.1}}
    return {
        "seed": seed,
        "detailed": {"val": {"dcn_v2": det}, "test": {"dcn_v2": det}},
        "ablations": abl,
        "ablations_test": abl,
    }


def test_aggregate_mean_std_and_markdown():
    out = aggregate([_run(0, 0.0), _run(1, 0.02), _run(2, 0.04)])
    row = out["detailed"]["test"]["dcn_v2"]
    assert out["n_runs"] == 3 and out["seeds"] == [0, 1, 2]
    assert np.isclose(row["ranker"]["ndcg@10"]["mean"], 0.22)
    assert np.isclose(row["ranker"]["ndcg@10"]["std"], 0.02)
    assert np.isclose(row["paired_diff"]["ndcg@10"]["mean"], 0.12)
    assert np.isclose(out["ablations"]["test"]["full"]["ndcg@10"]["mean"], 0.22)
    text = render_markdown(out, "test")
    assert "dcn_v2" in text and "±" in text and "full" in text


def test_aggregate_single_run_has_nan_std_and_plain_format():
    out = aggregate([_run(0, 0.0)])
    assert np.isnan(out["detailed"]["val"]["dcn_v2"]["ranker"]["ndcg@10"]["std"])
    assert "±" not in render_markdown(out, "val")
