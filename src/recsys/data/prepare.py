"""Clean raw MovieLens, build id maps, and split BY TIME (never randomly).

Output (data/processed/<mode>/):
  interactions.parquet  user_idx, item_idx, rating, ts, is_pos, stage
  items.parquet         item_idx, movie_id, title, year, genres (pipe string)
  genome.parquet        item_idx, tag_id, relevance
  user_map.parquet      user_idx, user_id            (external MovieLens id <-> dense idx)
  boundaries.json       t1, t2, t3 timestamps

stage: 0 = retrieval-train [0,t1) | 1 = ranker-train [t1,t2) | 2 = val [t2,t3) | 3 = test [t3,inf)
Index 0 is reserved (padding / unknown) for BOTH users and items.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import parse_cfg, path
from ..features.common import parse_genres, parse_year
from ..utils import get_logger, save_json
from .download import ensure_raw

log = get_logger("prepare")
STAGE_NAMES = {0: "retr", 1: "rank", 2: "val", 3: "test"}


@dataclass
class Dataset:
    interactions: pd.DataFrame
    items: pd.DataFrame
    genome: pd.DataFrame
    user_map: pd.DataFrame
    boundaries: dict[str, int]
    stats: dict


def load_raw(raw: Path):
    ratings = pd.read_csv(
        raw / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(raw / "movies.csv")
    genome = pd.read_csv(
        raw / "genome-scores.csv",
        dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    )
    return ratings, movies, genome


def assign_stage(ts: np.ndarray, boundaries: dict[str, int]) -> np.ndarray:
    """stage by global timestamp: t < t1 -> 0, < t2 -> 1, < t3 -> 2, else 3."""
    return np.searchsorted(
        [boundaries["t1"], boundaries["t2"], boundaries["t3"]], ts, side="right"
    ).astype(np.int8)


def compute_boundaries(ts: np.ndarray, split_cfg) -> dict[str, int]:
    q = np.quantile(ts, [split_cfg["retrieval_end"], split_cfg["ranker_end"], split_cfg["val_end"]])
    return {"t1": int(q[0]), "t2": int(q[1]), "t3": int(q[2])}


def clean_and_split(
    ratings: pd.DataFrame, movies: pd.DataFrame, genome: pd.DataFrame, cfg
) -> Dataset:
    d = cfg["data"]
    n0 = len(ratings)
    ratings = ratings.sort_values("timestamp").drop_duplicates(["userId", "movieId"], keep="last")
    ratings = ratings.dropna(subset=["rating", "timestamp"])

    frac = d["user_sample_frac"]
    if frac < 1.0:  # deterministic user subsample (keeps full user histories)
        users = np.sort(ratings["userId"].unique())
        rng = np.random.default_rng(d["seed"])
        keep = rng.choice(users, size=max(int(len(users) * frac), 1), replace=False)
        ratings = ratings[ratings["userId"].isin(keep)]

    counts = ratings.groupby("userId")["movieId"].transform("size")
    ratings = ratings[counts >= d["min_user_interactions"]]
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]

    uids = np.sort(ratings["userId"].unique())
    user_map = pd.DataFrame(
        {"user_idx": np.arange(1, len(uids) + 1, dtype=np.int32), "user_id": uids}
    )
    mids = np.sort(ratings["movieId"].unique())
    items = movies[movies["movieId"].isin(mids)].set_index("movieId").loc[mids].reset_index()
    items.insert(0, "item_idx", np.arange(1, len(items) + 1, dtype=np.int32))
    items = pd.DataFrame(
        {
            "item_idx": items["item_idx"],
            "movie_id": items["movieId"].astype("int32"),
            "title": items["title"].fillna(""),
            "year": items["title"].map(parse_year).astype("float32"),
            "genres": items["genres"].fillna("(no genres listed)"),
        }
    )

    inter = pd.DataFrame(
        {
            "user_idx": ratings["userId"]
            .map(dict(zip(user_map["user_id"], user_map["user_idx"], strict=True)))
            .astype("int32"),
            "item_idx": ratings["movieId"]
            .map(dict(zip(items["movie_id"], items["item_idx"], strict=True)))
            .astype("int32"),
            "rating": ratings["rating"].astype("float32"),
            "ts": ratings["timestamp"].astype("int64"),
        }
    )
    inter["is_pos"] = inter["rating"] >= d["positive_threshold"]
    boundaries = compute_boundaries(inter["ts"].to_numpy(), cfg["split"])
    inter["stage"] = assign_stage(inter["ts"].to_numpy(), boundaries)
    inter = inter.sort_values(["user_idx", "ts", "item_idx"]).reset_index(drop=True)

    g = genome[genome["movieId"].isin(mids)].copy()
    g["item_idx"] = (
        g["movieId"]
        .map(dict(zip(items["movie_id"], items["item_idx"], strict=True)))
        .astype("int32")
    )
    g = g[["item_idx", "tagId", "relevance"]].rename(columns={"tagId": "tag_id"})

    stats = {
        "raw_interactions": int(n0),
        "interactions": int(len(inter)),
        "users": int(len(user_map)),
        "items": int(len(items)),
        "positive_rate": float(inter["is_pos"].mean()),
        "per_stage": {STAGE_NAMES[s]: int((inter["stage"] == s).sum()) for s in range(4)},
        "boundaries": boundaries,
        "genome_items": int(g["item_idx"].nunique()),
    }
    return Dataset(inter, items, g, user_map, boundaries, stats)


def save_dataset(ds: Dataset, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ds.interactions.to_parquet(out / "interactions.parquet", index=False)
    ds.items.to_parquet(out / "items.parquet", index=False)
    ds.genome.to_parquet(out / "genome.parquet", index=False)
    ds.user_map.to_parquet(out / "user_map.parquet", index=False)
    save_json(ds.boundaries, out / "boundaries.json")
    save_json(ds.stats, out / "dataset_stats.json")


def load_dataset(out: Path) -> Dataset:
    from ..utils import load_json

    return Dataset(
        pd.read_parquet(out / "interactions.parquet"),
        pd.read_parquet(out / "items.parquet"),
        pd.read_parquet(out / "genome.parquet"),
        pd.read_parquet(out / "user_map.parquet"),
        load_json(out / "boundaries.json"),
        load_json(out / "dataset_stats.json"),
    )


def genre_lists(items: pd.DataFrame) -> list[list[int]]:
    return [parse_genres(s) for s in items["genres"]]


def main():
    cfg, _ = parse_cfg("Clean, time-split and index the dataset")
    raw = ensure_raw(cfg)
    ratings, movies, genome = load_raw(raw)
    ds = clean_and_split(ratings, movies, genome, cfg)
    out = path(cfg, "data_processed", create=True)
    save_dataset(ds, out)
    log.info("saved to %s | %s", out, ds.stats)


if __name__ == "__main__":
    main()
