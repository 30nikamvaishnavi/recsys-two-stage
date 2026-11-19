"""Load a ready-to-serve Recommender from artifacts/serving + model_repository."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..cache.redis_cache import make_cache
from ..coldstart.router import ColdStartRouter
from ..config import path
from ..features.item_table import ItemTable
from ..index.faiss_index import AnnIndex
from ..utils import load_json
from .backends import make_backend
from .pipeline import Recommender
from .state_store import UserStateStore


def load_recommender(
    cfg, backend_kind: str | None = None, force_memory_cache: bool = False
) -> Recommender:
    s = Path(path(cfg, "serving"))
    meta = load_json(s / "assets_meta.json")
    table = ItemTable.load(s / "item_table.npz")
    store = UserStateStore(
        pd.read_parquet(s / "user_map.parquet"), pd.read_parquet(s / ("user_states_all.parquet" if (s / "user_states_all.parquet").exists() else "user_states.parquet")), sp.load_npz(s / "seen.npz").tocsr(),
        meta["history_len"], meta["cutoff_ts"], meta["positive_threshold"],
    )  # fmt: skip
    c = cfg["coldstart"]
    router = ColdStartRouter(table, c["min_history"], c["content_alpha"], c["mmr_lambda"])
    backend = make_backend(
        cfg, Path(path(cfg, "model_repository")), s / "serving_meta.json", backend_kind
    )
    return Recommender(
        backend, AnnIndex.load(s), np.load(s / "item_emb.npy"), table, pd.read_parquet(s / "items.parquet"), store,
        make_cache(cfg, force_memory=force_memory_cache), router, cfg, Path(path(cfg, "logs")),
    )  # fmt: skip
