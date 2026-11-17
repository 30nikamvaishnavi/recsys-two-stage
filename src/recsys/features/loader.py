"""Read-side API for everything the feature pipeline writes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import path
from ..utils import load_json
from .item_table import ItemTable
from .pandas_features import stage_windows
from .state import SeenIndex, StateArrays, examples_to_arrays, states_from_frame

STAGES = ("rank", "val", "test")


class FeatureStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = Path(path(cfg, "data_processed"))
        self.dir = Path(path(cfg, "features"))
        self.meta = load_json(self.dir / "meta.json")
        self.N = cfg["features"]["history_len"]
        self.boundaries = load_json(self.proc / "boundaries.json")
        self.windows = stage_windows(self.boundaries)

    n_items = property(lambda self: self.meta["n_items"])
    n_users = property(lambda self: self.meta["n_users"])

    def cutoff(self, stage: str) -> int:
        return self.windows[stage][0]

    @lru_cache(maxsize=4)  # noqa: B019
    def item_table(self, stage: str) -> ItemTable:
        return ItemTable.load(self.dir / f"item_table_{stage}.npz")

    @lru_cache(maxsize=1)  # noqa: B019
    def items(self) -> pd.DataFrame:
        return pd.read_parquet(self.proc / "items.parquet")

    @lru_cache(maxsize=1)  # noqa: B019
    def user_map(self) -> pd.DataFrame:
        return pd.read_parquet(self.proc / "user_map.parquet")

    def interactions(self, columns=None) -> pd.DataFrame:
        return pd.read_parquet(self.proc / "interactions.parquet", columns=columns)

    def train_arrays(self) -> dict[str, np.ndarray]:
        df = pd.read_parquet(self.dir / "train_examples.parquet")
        u, i, hist, num = examples_to_arrays(df, self.N)
        q = np.load(self.dir / "item_q.npy")
        return {
            "user_idx": u,
            "item_idx": i,
            "hist_items": hist,
            "user_numeric": num,
            "item_q": q[i].astype(np.float32),
        }

    def states(self, stage: str) -> StateArrays:
        return states_from_frame(pd.read_parquet(self.dir / f"user_state_{stage}.parquet"), self.N)

    def seen(self, stage: str, inter: pd.DataFrame | None = None) -> SeenIndex:
        inter = inter if inter is not None else self.interactions(["user_idx", "item_idx", "ts"])
        return SeenIndex(inter, self.cutoff(stage), self.n_users, self.n_items)

    def pop_negatives(self) -> pd.DataFrame | None:
        p = self.dir / "pop_negatives_rank.parquet"
        return pd.read_parquet(p) if p.exists() else None
