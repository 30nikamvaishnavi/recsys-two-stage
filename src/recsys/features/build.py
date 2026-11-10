"""Run the feature pipeline (Spark by default) and assemble the dense item tables.

    python -m recsys.features.build --engine spark     # default, needs Java 17
    python -m recsys.features.build --engine pandas    # reference/fallback engine
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import parse_cfg, path
from ..data.prepare import load_dataset
from ..utils import get_logger, load_json, save_json, timed
from . import pandas_features as pf
from .item_table import build_item_table
from .loader import STAGES

log = get_logger("features")


def run_pandas(cfg) -> Path:
    ds = load_dataset(Path(path(cfg, "data_processed")))
    out = path(cfg, "features", create=True)
    out.mkdir(parents=True, exist_ok=True)
    inter, N = ds.interactions, cfg["features"]["history_len"]
    n_items, n_users = len(ds.items), len(ds.user_map)
    pf.train_examples(inter, N).to_parquet(out / "train_examples.parquet", index=False)
    pops = {}
    for stage, (cut, end) in pf.stage_windows(ds.boundaries).items():
        pf.user_state(inter, cut, end, N).to_parquet(out / f"user_state_{stage}.parquet", index=False)
        pops[stage] = pf.item_pop(inter, cut, cfg["features"]["pop_windows_days"], n_items)
        pops[stage].to_parquet(out / f"item_pop_{stage}.parquet", index=False)
    pf.genome_vectors(ds.genome, cfg["features"]["genome_dim"]).to_parquet(out / "genome_vectors.parquet", index=False)
    c = cfg["coldstart"]
    ci, cu = pf.cold_lists(inter, ds.boundaries["t3"], n_items, n_users, c["min_history"], c["cold_item_min_interactions"])
    ci.to_parquet(out / "cold_items.parquet", index=False)
    cu.to_parquet(out / "cold_users.parquet", index=False)
    rank_states = pd.read_parquet(out / "user_state_rank.parquet")
    pf.pop_negatives(
        inter, rank_states, ds.boundaries["t1"], pops["rank"], cfg["spark"]["pop_negatives_pool"],
        cfg["spark"]["pop_negatives_per_user"], cfg["data"]["seed"],
    ).to_parquet(out / "pop_negatives_rank.parquet", index=False)  # fmt: skip
    return out


def assemble(cfg) -> None:
    """Dense item tables per stage + sampling distribution for logQ correction + meta.json."""
    proc, out = Path(path(cfg, "data_processed")), Path(path(cfg, "features"))
    items = pd.read_parquet(proc / "items.parquet")
    genome = pd.read_parquet(out / "genome_vectors.parquet")
    for stage in STAGES:
        pop = pd.read_parquet(out / f"item_pop_{stage}.parquet")
        tab = build_item_table(items, pop, genome, cfg)
        tab.save(out / f"item_table_{stage}.npz")
    inter = pd.read_parquet(proc / "interactions.parquet", columns=["item_idx", "is_pos", "stage"])
    n_items = int(items["item_idx"].max())
    cnt = np.bincount(inter.loc[inter["is_pos"] & (inter["stage"] == 0), "item_idx"], minlength=n_items + 1).astype(np.float64)
    q = np.maximum(cnt, 1.0) / cnt.sum()  # sampling prob. of an item as in-batch positive
    np.save(out / "item_q.npy", q.astype(np.float32))
    n_users = int(pd.read_parquet(proc / "user_map.parquet")["user_idx"].max())
    save_json(
        {"n_items": n_items, "n_users": n_users, "history_len": cfg["features"]["history_len"],
         "genome_dim": cfg["features"]["genome_dim"], "boundaries": load_json(proc / "boundaries.json")},
        out / "meta.json",
    )  # fmt: skip


def main():
    cfg, args = parse_cfg("Build features", [("--engine", {"default": "spark", "choices": ["spark", "pandas"]})])
    engine = args.engine
    if engine == "spark":
        try:
            import pyspark  # noqa: F401
        except ImportError:
            log.warning("pyspark missing -> falling back to the pandas engine")
            engine = "pandas"
    with timed(log, f"feature engine={engine}"):
        if engine == "spark":
            from .spark_features import run

            run(cfg)
        else:
            run_pandas(cfg)
        assemble(cfg)
    log.info("features ready in %s", path(cfg, "features"))


if __name__ == "__main__":
    main()
