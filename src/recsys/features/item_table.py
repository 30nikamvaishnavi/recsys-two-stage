"""ItemTable: dense per-item feature arrays as of a cutoff timestamp (row 0 = padding/unknown)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import G, bucketize, multi_hot, parse_genres, popularity_edges, year_bucket


@dataclass
class ItemTable:
    genres: np.ndarray  # (n+1, G) float32 multi-hot
    year_bucket: np.ndarray  # (n+1,) int32
    pop: np.ndarray  # (n+1, 3) float32  log1p[all, 7d, 30d] positive counts before cutoff
    pop_bucket: np.ndarray  # (n+1,) int32
    pop_rank_pct: np.ndarray  # (n+1,) float32 in [0,1], 1 = most popular
    genome: np.ndarray  # (n+1, D) float32 (zeros where unavailable)
    has_genome: np.ndarray  # (n+1,) float32
    n_pos: np.ndarray  # (n+1,) int32 raw positive count before cutoff (0 => cold item)
    n_pop_buckets: int

    @property
    def n_items(self) -> int:
        return self.genres.shape[0] - 1

    @property
    def cold_mask(self) -> np.ndarray:
        m = self.n_pos == 0
        m[0] = False
        return m

    @property
    def raw_pop(self) -> np.ndarray:
        return self.n_pos.astype(np.float32)

    def content_vectors(self, w_genre: float = 1.0, w_genome: float = 1.0) -> np.ndarray:
        """L2-normalised content vectors (genres + genome); used for cold-start kNN and MMR."""
        g = self.genres / np.sqrt(np.maximum(self.genres.sum(1, keepdims=True), 1.0))
        gn = self.genome / np.maximum(np.linalg.norm(self.genome, axis=1, keepdims=True), 1e-6)
        v = np.concatenate([w_genre * g, w_genome * gn], axis=1)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        v[0] = 0
        return v.astype(np.float32)

    # ---- (de)serialisation -------------------------------------------------------------------
    def save(self, p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            n_pop_buckets=self.n_pop_buckets,
            **{k: v for k, v in self.__dict__.items() if k != "n_pop_buckets"},
        )

    @classmethod
    def load(cls, p: Path) -> ItemTable:
        z = np.load(p)
        return cls(
            **{k: (int(z[k]) if k == "n_pop_buckets" else z[k]) for k in cls.__dataclass_fields__}
        )


def build_item_table(
    items: pd.DataFrame, pop_df: pd.DataFrame, genome_df: pd.DataFrame | None, cfg
) -> ItemTable:
    """Assemble from: processed items, popularity counts (item_idx,pop_all,pop_7d,pop_30d) and
    genome vectors (item_idx, genome: array<float>). Identical for the Spark and pandas engines."""
    n = int(items["item_idx"].max())
    D = cfg["features"]["genome_dim"]
    genres = np.zeros((n + 1, G), dtype=np.float32)
    genres[items["item_idx"].to_numpy()] = multi_hot([parse_genres(s) for s in items["genres"]])
    yb = np.zeros(n + 1, dtype=np.int32)
    yb[items["item_idx"].to_numpy()] = year_bucket(items["year"].to_numpy())

    raw = np.zeros((n + 1, 3), dtype=np.float64)
    p = pop_df.set_index("item_idx")
    ids = p.index.to_numpy()
    raw[ids, 0], raw[ids, 1], raw[ids, 2] = (
        p["pop_all"].to_numpy(),
        p["pop_7d"].to_numpy(),
        p["pop_30d"].to_numpy(),
    )
    n_pos = raw[:, 0].astype(np.int32)
    edges = popularity_edges(raw[1:, 0], cfg["features"]["n_pop_buckets"])
    pb = bucketize(raw[:, 0], edges)
    pb[0] = 0
    order = raw[:, 0].argsort(kind="stable").argsort() / max(n, 1)
    pct = order.astype(np.float32)
    pct[0] = 0

    genome = np.zeros((n + 1, D), dtype=np.float32)
    has = np.zeros(n + 1, dtype=np.float32)
    if genome_df is not None and len(genome_df):
        gi = genome_df["item_idx"].to_numpy()
        vecs = np.stack(genome_df["genome"].to_numpy()).astype(np.float32)[:, :D]
        genome[gi, : vecs.shape[1]] = vecs / (vecs.std() + 1e-6)
        has[gi] = 1.0
    return ItemTable(
        genres,
        yb,
        np.log1p(raw).astype(np.float32),
        pb,
        pct,
        genome,
        has,
        n_pos,
        int(cfg["features"]["n_pop_buckets"]),
    )
