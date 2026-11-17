import numpy as np
from recsys.config import load_config, path
from recsys.promote import gate
from recsys.serving.export import pbtxt
from recsys.serving.state_store import UserStateStore  # noqa: F401
from recsys.utils import percentiles, save_json


def test_pbtxt_contains_required_triton_settings():
    t = pbtxt("ranker_dcn", [("item_idx", 7, [1]), ("dense", 1, [98])], [("score", 1, [1])])
    for needle in (
        'backend: "onnxruntime"',
        "max_batch_size",
        "dynamic_batching",
        "KIND_CPU",
        "TYPE_INT64",
        "TYPE_FP32",
        "dims: [ 98 ]",
    ):
        assert needle in t


def test_config_modes_and_overrides():
    s, f = load_config(mode="smoke"), load_config(mode="full")
    assert (
        s.data.dataset == "synthetic"
        and f.data.dataset == "movielens-25m"
        and f.data.user_sample_frac == 1.0
    )
    assert s.retrieval.embedding_dim == 16 and f.retrieval.embedding_dim == 64
    assert load_config(mode="sample", overrides={"retrieval": {"epochs": 1}}).retrieval.epochs == 1
    assert path(s, "features").name == "smoke" and path(f, "features").name == "features"


def test_percentiles():
    p = percentiles(list(range(1, 101)))
    assert abs(p["p50"] - 50.5) < 1e-9 and p["p99"] > 98


def test_promotion_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("RECSYS_HOME", str(tmp_path))
    cfg = load_config(mode="smoke")
    rep = path(cfg, "reports", create=True)
    rep.mkdir(parents=True, exist_ok=True)

    def write(recall, ndcg):
        save_json({"two_tower": {"recall@100": recall}}, rep / "retrieval_metrics.json")
        save_json(
            {"models": {"dcn_v2": {"ranker": {"ndcg@10": ndcg, "auc": 0.7}}}},
            rep / "ranking_metrics.json",
        )

    write(0.30, 0.20)
    ok, info = gate(cfg)
    assert ok and info["production"] is None  # first model always promoted
    prod = path(cfg, "production", create=True)
    prod.mkdir(parents=True, exist_ok=True)
    save_json(
        {"retrieval_recall@100": 0.30, "ranker_ndcg@10": 0.20, "ranker_auc": 0.7},
        prod / "metrics.json",
    )
    write(0.31, 0.21)
    assert gate(cfg)[0]  # better -> promote
    write(0.20, 0.21)
    assert not gate(cfg)[0]  # retrieval regressed -> blocked
    write(0.31, 0.10)
    assert not gate(cfg)[0]  # ranker regressed -> blocked
    write(0.298, 0.199)
    assert gate(cfg)[0]  # within tolerance


def test_synthetic_data_has_cold_items_and_users(world):
    inter = world.ds.interactions
    first_item = inter.groupby("item_idx")["stage"].min()
    assert (
        first_item >= 2
    ).any(), "synthetic data should contain items first seen after training (cold items)"
    first_user = inter.groupby("user_idx")["stage"].min()
    assert (
        first_user >= 1
    ).any(), "synthetic data should contain users that appear after the first window (cold users)"
    assert np.isfinite(world.tables["test"].genome).all()
