"""Feature constants and pure-numpy transforms shared by training, Spark output readers and serving.

Keeping these in ONE place avoids train/serve skew: the online gateway builds the ranker's dense
vector with the same functions used to build the training set.
"""

from __future__ import annotations

import re

import numpy as np

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "(no genres listed)",
]  # fmt: skip
G = len(GENRES)
GENRE_IDX = {g: i for i, g in enumerate(GENRES)}
YEAR_EDGES = np.array([1950, 1970, 1980, 1990, 1995, 2000, 2005, 2010, 2015])
N_YEAR_BUCKETS = len(YEAR_EDGES) + 2  # 0 = unknown
NUM_USER = 3
SECONDS_PER_DAY = 86400.0
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def parse_year(title: str) -> float:
    m = _YEAR_RE.search(title or "")
    return float(m.group(1)) if m else float("nan")


def parse_genres(s: str | float) -> list[int]:
    if not isinstance(s, str) or not s.strip():
        return [GENRE_IDX["(no genres listed)"]]
    out = sorted({GENRE_IDX[g] for g in s.split("|") if g in GENRE_IDX})
    return out or [GENRE_IDX["(no genres listed)"]]


def multi_hot(genre_lists, n_rows: int | None = None) -> np.ndarray:
    n = n_rows if n_rows is not None else len(genre_lists)
    m = np.zeros((n, G), dtype=np.float32)
    for i, gs in enumerate(genre_lists):
        m[i, list(gs)] = 1.0
    return m


def year_bucket(years: np.ndarray) -> np.ndarray:
    y = np.asarray(years, dtype=float)
    b = np.searchsorted(YEAR_EDGES, np.nan_to_num(y, nan=0.0), side="right") + 1
    b[np.isnan(y)] = 0
    return b.astype(np.int32)


def user_numeric(n_prior, avg_rating, days_since) -> np.ndarray:
    """(B,3): log activity, centered mean rating, log recency. Fixed transform => no fitted stats."""
    n = np.asarray(n_prior, dtype=np.float32)
    r = np.nan_to_num(np.asarray(avg_rating, dtype=np.float32), nan=3.5)
    d = np.clip(np.nan_to_num(np.asarray(days_since, dtype=np.float32), nan=0.0), 0, None)
    return np.stack([np.log1p(n) / 6.0, (r - 3.0) / 1.5, np.log1p(d) / 6.0], axis=1).astype(
        np.float32
    )


def masked_mean(hist: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Mean of table[hist] over non-padding (id != 0) entries. hist: (B,N) -> (B,table.shape[1])."""
    mask = (hist != 0).astype(np.float32)[..., None]
    s = (table[hist] * mask).sum(axis=1)
    return (s / np.maximum(mask.sum(axis=1), 1.0)).astype(np.float32)


def pad_history(items: list[int] | np.ndarray, n: int) -> np.ndarray:
    """Oldest->newest, keep the most recent n, zero-pad at the END (same layout as Spark output)."""
    items = list(items)[-n:]
    out = np.zeros(n, dtype=np.int32)
    out[: len(items)] = items
    return out


def popularity_edges(pop_all: np.ndarray, n_buckets: int) -> np.ndarray:
    """Bucket edges from quantiles of positive counts. bucket = #(edges <= pop) in [0, n_buckets-1]."""
    pos = pop_all[pop_all > 0]
    if pos.size == 0:
        return np.array([1.0])
    qs = np.quantile(pos, np.linspace(0, 1, n_buckets)[1:-1])
    return np.unique(np.concatenate([[1.0], qs]))


def bucketize(pop_all: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, pop_all, side="right").astype(np.int32)
