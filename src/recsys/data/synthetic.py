"""Synthetic dataset in the exact MovieLens-25M file format (movies/ratings/genome-*).

Used for smoke tests, CI and offline development. It has real structure: user taste vectors,
item latent factors tied to genres, power-law popularity, items released over time (=> cold items),
users joining over time (=> cold users), and genome tag relevances derived from item factors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]  # fmt: skip
T0 = 1_420_070_400  # 2015-01-01 UTC


def generate(
    out_dir: str | Path, n_users=3000, n_items=1500, n_tags=40, mean_events=55, seed=0, days=1200
):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    K = 8
    g_emb = rng.normal(size=(len(GENRES), K))

    # ---- items, sorted by release time so "released by t" is a prefix -----------------------
    release = np.sort(np.where(rng.random(n_items) < 0.4, 0.0, rng.random(n_items) * 0.9))
    item_genres = []
    z = np.zeros((n_items, K))
    for i in range(n_items):
        k = rng.choice([0, 1, 2, 3], p=[0.02, 0.38, 0.40, 0.20])
        gs = rng.choice(len(GENRES), size=k, replace=False) if k else np.array([], dtype=int)
        item_genres.append(gs)
        z[i] = g_emb[gs].sum(0) / max(np.sqrt(len(gs)), 1) + rng.normal(scale=0.5, size=K)
    pop = rng.lognormal(0.0, 1.0, size=n_items)
    year = (1980 + (release * 39)).astype(int)
    movie_ids = 100 + np.arange(n_items) * 3
    movies = pd.DataFrame(
        {
            "movieId": movie_ids,
            "title": [f"Movie {i} ({year[i]})" for i in range(n_items)],
            "genres": [
                "|".join(GENRES[g] for g in gs) if len(gs) else "(no genres listed)"
                for gs in item_genres
            ],
        }
    )

    # ---- users ------------------------------------------------------------------------------
    taste = np.zeros((n_users, K))
    for u in range(n_users):
        pick = rng.choice(len(GENRES), size=2, replace=False)
        taste[u] = g_emb[pick].sum(0) + rng.normal(scale=0.5, size=K)
    join = np.where(rng.random(n_users) < 0.5, 0.0, rng.random(n_users) * 0.85)
    n_events = np.clip(rng.lognormal(np.log(mean_events * 0.7), 0.8, size=n_users), 6, 400).astype(
        int
    )

    rows = []
    for u in range(n_users):
        aff = z @ taste[u]
        aff = (aff - aff.mean()) / (aff.std() + 1e-9)
        w = pop * np.exp(1.2 * aff)
        cum = np.cumsum(w)
        n = n_events[u]
        tau = np.sort(join[u] + rng.random(n) * (1 - join[u]))
        m = np.maximum(np.searchsorted(release, tau, side="right"), 1)
        idx = np.searchsorted(cum, rng.random(n) * cum[m - 1] * 0.999999)
        idx = np.minimum(idx, m - 1)
        _, first = np.unique(idx, return_index=True)  # no duplicate (user, item)
        first.sort()
        idx, tau = idx[first], tau[first]
        r = 2.7 + 1.0 * aff[idx] + rng.normal(scale=0.6, size=len(idx))
        r = np.clip(np.round(r * 2) / 2, 0.5, 5.0)
        rows.append(
            np.column_stack(
                [
                    np.full(len(idx), u + 1),
                    movie_ids[idx],
                    r,
                    (T0 + tau * days * 86400).astype(np.int64),
                ]
            )
        )
    arr = np.vstack(rows)
    ratings = pd.DataFrame(arr, columns=["userId", "movieId", "rating", "timestamp"])
    ratings = ratings.astype(
        {"userId": "int64", "movieId": "int64", "rating": "float32", "timestamp": "int64"}
    )

    # ---- genome (85% of items have it, like the real data) ---------------------------------
    basis = rng.normal(size=(n_tags, K))
    rel = 1 / (1 + np.exp(-(z @ basis.T) / 2 + rng.normal(scale=0.3, size=(n_items, n_tags))))
    has = rng.random(n_items) < 0.85
    gi, ti = np.nonzero(has[:, None] & np.ones((1, n_tags), bool))
    genome = pd.DataFrame(
        {"movieId": movie_ids[gi], "tagId": ti + 1, "relevance": rel[gi, ti].astype("float32")}
    )
    tags = pd.DataFrame(
        {"tagId": np.arange(1, n_tags + 1), "tag": [f"tag_{i}" for i in range(n_tags)]}
    )

    movies.to_csv(out / "movies.csv", index=False)
    ratings.to_csv(out / "ratings.csv", index=False)
    genome.to_csv(out / "genome-scores.csv", index=False)
    tags.to_csv(out / "genome-tags.csv", index=False)
    return out
