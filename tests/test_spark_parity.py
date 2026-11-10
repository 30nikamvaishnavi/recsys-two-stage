"""The Spark pipeline must produce the same tables as the pandas reference engine."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("pyarrow")

from recsys.config import path  # noqa: E402
from recsys.data.prepare import save_dataset  # noqa: E402
from recsys.features import pandas_features as pf  # noqa: E402
from recsys.features.spark_features import get_spark, run  # noqa: E402


def test_spark_matches_pandas(world, tmp_path, monkeypatch):
    monkeypatch.setenv("RECSYS_HOME", str(tmp_path))
    cfg = world.cfg
    save_dataset(world.ds, path(cfg, "data_processed"))
    spark = get_spark(cfg)
    out = run(cfg, spark)
    N, b, inter = cfg["features"]["history_len"], world.ds.boundaries, world.ds.interactions

    for stage, (cut, end) in pf.stage_windows(b).items():
        a = pf.user_state(inter, cut, end, N).set_index("user_idx").sort_index()
        s = pd.read_parquet(out / f"user_state_{stage}.parquet").set_index("user_idx").sort_index()
        assert list(a.index) == list(s.index)
        assert (a["n_pos"].to_numpy() == s["n_pos"].to_numpy()).all()
        assert all(sorted(x) == sorted(y) for x, y in zip(a["targets"], s["targets"], strict=True))
        assert all(np.array_equal(x, y) for x, y in zip(a["hist_items"], s["hist_items"], strict=True))
        pa = pf.item_pop(inter, cut, [7, 30], world.n_items).set_index("item_idx").sort_index()
        ps = pd.read_parquet(out / f"item_pop_{stage}.parquet").set_index("item_idx").sort_index()
        assert (pa[["pop_all", "pop_7d", "pop_30d"]].to_numpy() == ps[["pop_all", "pop_7d", "pop_30d"]].to_numpy()).all()

    ex_p = pf.train_examples(inter, N).sort_values(["user_idx", "ts", "item_idx"]).reset_index(drop=True)
    ex_s = pd.read_parquet(out / "train_examples.parquet").sort_values(["user_idx", "ts", "item_idx"]).reset_index(drop=True)
    assert len(ex_p) == len(ex_s)
    assert all(np.array_equal(x, y) for x, y in zip(ex_p["hist_items"], ex_s["hist_items"], strict=True))
    assert np.allclose(ex_p["n_prior"], ex_s["n_prior"])
    for f in ("cold_items", "cold_users", "pop_negatives_rank", "genome_vectors"):
        assert len(pd.read_parquet(out / f"{f}.parquet")) > 0
