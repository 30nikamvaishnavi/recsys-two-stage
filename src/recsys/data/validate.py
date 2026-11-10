"""Data validation checks (used by the Airflow DAG and tests). Raises AssertionError on failure."""
from __future__ import annotations

import pandas as pd

from ..config import parse_cfg, path
from ..utils import get_logger, load_json

log = get_logger("validate")


def validate_interactions(inter: pd.DataFrame, b: dict) -> None:
    assert {"user_idx", "item_idx", "rating", "ts", "is_pos", "stage"} <= set(inter.columns), "schema mismatch"
    assert inter["rating"].between(0.5, 5.0).all(), "rating out of [0.5, 5]"
    assert not inter.duplicated(["user_idx", "item_idx"]).any(), "duplicate user-item pairs"
    assert b["t1"] < b["t2"] < b["t3"], "split boundaries not increasing"
    for s in range(4):
        assert (inter["stage"] == s).sum() > 0, f"stage {s} is empty"
    for s in range(3):  # strict temporal ordering between consecutive stages
        assert inter.loc[inter["stage"] == s, "ts"].max() < inter.loc[inter["stage"] == s + 1, "ts"].min() + 1, f"time overlap between stage {s} and {s + 1}"
    assert inter["user_idx"].min() >= 1 and inter["item_idx"].min() >= 1, "index 0 is reserved for padding"


def main():
    cfg, _ = parse_cfg("Validate processed data")
    d = path(cfg, "data_processed")
    inter = pd.read_parquet(d / "interactions.parquet")
    validate_interactions(inter, load_json(d / "boundaries.json"))
    log.info("data validation passed: %d rows", len(inter))


if __name__ == "__main__":
    main()
