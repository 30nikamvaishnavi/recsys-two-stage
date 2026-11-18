"""Pure-NumPy helpers for ranker analysis (no TensorFlow): paired bootstrap, user and item
breakdowns, and list-level statistics. Everything here works on arrays, so it is unit-tested.

Bucket definitions (documented so the numbers can be defended):
* user activity  = terciles of each user's prior positive count -> light / medium / heavy
* item popularity = by prior positive count among items that have any: the least-popular 60%
  are "tail", the next 30% "torso", the top 10% "head"; items with zero prior positives are "cold"
"""

from __future__ import annotations

import numpy as np

ACTIVITY_NAMES = ("light", "medium", "heavy")
POPULARITY_NAMES = ("cold", "tail", "torso", "head")


def paired_bootstrap_diff(a, b, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """Mean of (a - b) over users with a percentile 95% bootstrap CI (users resampled in pairs).

    a and b are per-user metric arrays for two systems on the SAME users. If the CI excludes 0,
    the difference is unlikely to be user-sampling noise.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = (a - b)[np.isfinite(a) & np.isfinite(b)]
    if len(diff) == 0:
        nan = float("nan")
        return {"mean_diff": nan, "ci_low": nan, "ci_high": nan, "n_users": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    step = min(128, n_bootstrap)
    for start in range(0, n_bootstrap, step):
        stop = min(start + step, n_bootstrap)
        sample = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        means[start:stop] = diff[sample].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_users": int(len(diff)),
    }


def activity_buckets(n_pos) -> tuple[np.ndarray, tuple[float, float]]:
    """Bucket index 0/1/2 (light/medium/heavy) per user from tercile cut points."""
    n_pos = np.asarray(n_pos, dtype=np.float64)
    low, high = np.quantile(n_pos, [1 / 3, 2 / 3])
    idx = np.where(n_pos <= low, 0, np.where(n_pos <= high, 1, 2))
    return idx, (float(low), float(high))


def bucket_means(values: dict, bucket_idx: np.ndarray, names=ACTIVITY_NAMES) -> dict:
    """Mean of each per-user array inside each bucket (NaN-safe), plus the bucket size."""
    out = {}
    for b, name in enumerate(names):
        mask = bucket_idx == b
        row = {"n_users": int(mask.sum())}
        for key, vals in values.items():
            v = np.asarray(vals, dtype=np.float64)[mask]
            v = v[np.isfinite(v)]
            row[key] = float(v.mean()) if len(v) else float("nan")
        out[name] = row
    return out


def item_popularity_buckets(n_pos) -> np.ndarray:
    """Bucket id per item: 0 cold, 1 tail, 2 torso, 3 head (see module docstring)."""
    n_pos = np.asarray(n_pos)
    out = np.zeros(len(n_pos), dtype=np.int8)
    warm = n_pos > 0
    if warm.any():
        c_tail, c_torso = np.quantile(n_pos[warm], [0.6, 0.9])
        w = n_pos[warm]
        out[warm] = np.where(w <= c_tail, 1, np.where(w <= c_torso, 2, 3))
    return out


def target_recall_by_bucket(ranked, targets, bucket_of_item, k: int = 10) -> dict:
    """Of all held-out positives in each popularity bucket, the share found in the top-k lists.

    Micro-averaged over (user, target item) pairs, so it answers "which kinds of items does this
    ordering actually surface?".
    """
    found = np.zeros(len(POPULARITY_NAMES))
    total = np.zeros(len(POPULARITY_NAMES))
    for row, t in zip(np.asarray(ranked)[:, :k], targets, strict=True):
        if len(t) == 0:
            continue
        t = np.asarray(t)
        b = np.asarray(bucket_of_item)[t]
        np.add.at(total, b, 1.0)
        np.add.at(found, b, np.isin(t, row).astype(np.float64))
    out = {}
    for i, name in enumerate(POPULARITY_NAMES):
        rec = float(found[i] / total[i]) if total[i] else float("nan")
        out[name] = {"n_targets": int(total[i]), f"target_recall@{k}": rec}
    return out


def list_stats(top_items, n_pos, genres, n_items: int) -> dict:
    """Catalog coverage, novelty and intra-list diversity of the top-k lists (B, k).

    * coverage = distinct recommended items / catalog size
    * novelty = mean -log2(item popularity share); higher means less mainstream
    * diversity = mean pairwise cosine DISTANCE of genre vectors inside a list (items with no
      genre vector count as maximally different, distance 1)
    """
    top = np.asarray(top_items)
    shown = top[top > 0]
    distinct = np.unique(shown)
    share = np.maximum(np.asarray(n_pos, dtype=np.float64), 1.0)
    share = share / share.sum()
    g = np.asarray(genres, dtype=np.float64)[top]  # (B, k, G)
    norm = np.linalg.norm(g, axis=2, keepdims=True)
    unit = np.divide(g, norm, out=np.zeros_like(g), where=norm > 0)
    cos = unit @ unit.transpose(0, 2, 1)  # (B, k, k)
    off_diagonal = ~np.eye(top.shape[1], dtype=bool)
    return {
        "distinct_items": int(len(distinct)),
        "catalog_coverage": float(len(distinct) / n_items),
        "novelty_bits": float(np.mean(-np.log2(share[shown]))),
        "intra_list_diversity": float(np.mean(1.0 - cos[:, off_diagonal])),
    }
