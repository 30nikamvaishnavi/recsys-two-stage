"""Reference (pandas) implementation of the feature pipeline.

The Spark job in spark_features.py produces tables with the SAME schema; this engine is the
fallback for machines without Java, the engine used by unit tests, and the oracle the Spark output
is compared against (tests/test_spark_parity.py). All features are POINT-IN-TIME: a feature for
cutoff c only reads rows with ts < c.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import SECONDS_PER_DAY, pad_history

INF_TS = 2**62


def stage_windows(b: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Per-stage (state cutoff, window end). State is frozen at the window start."""
    return {"rank": (b["t1"], b["t2"]), "val": (b["t2"], b["t3"]), "test": (b["t3"], INF_TS)}


def item_pop(
    inter: pd.DataFrame, cutoff: int, windows_days: list[int], n_items: int
) -> pd.DataFrame:
    """Positive-interaction counts strictly before `cutoff`: all-time and trailing windows."""
    pos = inter[inter["is_pos"] & (inter["ts"] < cutoff)]
    out = pd.DataFrame({"item_idx": np.arange(1, n_items + 1, dtype=np.int32)}).set_index(
        "item_idx"
    )
    out["pop_all"] = pos.groupby("item_idx").size()
    for d, name in zip(windows_days, ["pop_7d", "pop_30d"], strict=False):
        w = pos[pos["ts"] >= cutoff - d * SECONDS_PER_DAY]
        out[name] = w.groupby("item_idx").size()
    return out.fillna(0).astype("int64").reset_index()


def genome_vectors(genome: pd.DataFrame, dim: int, seed: int = 0) -> pd.DataFrame:
    """PCA of the (item x tag) genome matrix -> `dim` content features per item. Content-only (no
    interaction data), so it cannot leak labels."""
    from sklearn.decomposition import PCA

    piv = genome.pivot_table(index="item_idx", columns="tag_id", values="relevance", fill_value=0.0)
    k = min(dim, piv.shape[1], piv.shape[0])
    x = piv.to_numpy(np.float32)
    z = PCA(n_components=k, random_state=seed).fit_transform(x - x.mean(0))
    if k < dim:
        z = np.pad(z, ((0, 0), (0, dim - k)))
    return pd.DataFrame(
        {"item_idx": piv.index.to_numpy(np.int32), "genome": list(z.astype(np.float32))}
    )


def train_examples(inter: pd.DataFrame, history_len: int) -> pd.DataFrame:
    """Retrieval training examples = stage-0 positives, each with the user's previous <=N positives
    and prior-activity stats (all strictly earlier events of that user)."""
    d = (
        inter[inter["stage"] == 0]
        .sort_values(["user_idx", "ts", "item_idx"])
        .reset_index(drop=True)
    )
    g = d.groupby("user_idx", sort=False)
    d["n_prior"] = g.cumcount().astype("int32")
    csum = g["rating"].cumsum()
    d["avg_prior_rating"] = np.where(
        d["n_prior"] > 0, (csum - d["rating"]) / d["n_prior"].clip(lower=1), np.nan
    )
    d["days_since_prev"] = (d["ts"] - g["ts"].shift(1)) / SECONDS_PER_DAY

    p = d[d["is_pos"]].reset_index(drop=True)
    items = p["item_idx"].to_numpy()
    within = p.groupby("user_idx", sort=False).cumcount().to_numpy()
    row = np.arange(len(p))
    cnt = np.minimum(within, history_len)
    j = np.arange(history_len)[None, :]
    idx = np.clip(row[:, None] - cnt[:, None] + j, 0, max(len(p) - 1, 0))
    hist = np.where(j < cnt[:, None], items[idx], 0).astype(np.int32)
    out = p[["user_idx", "item_idx", "ts", "n_prior", "avg_prior_rating", "days_since_prev"]].copy()
    out["hist_items"] = list(hist)
    return out


def user_state(inter: pd.DataFrame, cutoff: int, window_end: int, history_len: int) -> pd.DataFrame:
    """State of every user that has >=1 positive in [cutoff, window_end), frozen at `cutoff`."""
    tgt = (
        inter[(inter["ts"] >= cutoff) & (inter["ts"] < window_end) & inter["is_pos"]]
        .groupby("user_idx")["item_idx"].agg(list).rename("targets")
    )  # fmt: skip
    before = inter[inter["ts"] < cutoff]
    agg = before.groupby("user_idx").agg(
        n_all=("rating", "size"),
        avg_rating=("rating", "mean"),
        last_ts=("ts", "max"),
        n_pos=("is_pos", "sum"),
    )
    pos_b = before[before["is_pos"]].sort_values(["user_idx", "ts", "item_idx"])
    hist = (
        pos_b.groupby("user_idx")
        .tail(history_len)
        .groupby("user_idx")["item_idx"]
        .agg(list)
        .rename("hist_items")
    )
    st = tgt.to_frame().join(agg, how="left").join(hist, how="left")
    st["n_all"] = st["n_all"].fillna(0).astype("int64")
    st["n_pos"] = st["n_pos"].fillna(0).astype("int64")
    st["days_since_last"] = ((cutoff - st["last_ts"]) / SECONDS_PER_DAY).fillna(0.0)
    st["hist_items"] = [
        pad_history(h if isinstance(h, list) else [], history_len) for h in st["hist_items"]
    ]
    return st.reset_index().drop(columns=["last_ts"])


def cold_lists(
    inter: pd.DataFrame, cutoff: int, n_items: int, n_users: int, min_hist: int, min_item: int
):
    """Cold items (< min_item positives before cutoff) and cold users (< min_hist positives)."""
    pos = inter[inter["is_pos"] & (inter["ts"] < cutoff)]
    ic = pos.groupby("item_idx").size().reindex(np.arange(1, n_items + 1), fill_value=0)
    uc = pos.groupby("user_idx").size().reindex(np.arange(1, n_users + 1), fill_value=0)
    items = pd.DataFrame(
        {
            "item_idx": ic.index[ic < min_item].to_numpy(np.int32),
            "n_pos": ic[ic < min_item].to_numpy(),
        }
    )
    users = pd.DataFrame(
        {
            "user_idx": uc.index[uc < min_hist].to_numpy(np.int32),
            "n_pos": uc[uc < min_hist].to_numpy(),
        }
    )
    return items, users


def pop_negatives(
    inter,
    states: pd.DataFrame,
    cutoff: int,
    pop_df: pd.DataFrame,
    pool: int,
    per_user: int,
    seed: int = 0,
):
    """Popularity hard negatives: for each user, items sampled from the global top-`pool` popular
    that the user has NOT interacted with before the cutoff."""
    top = pop_df.sort_values("pop_all", ascending=False).head(pool)["item_idx"].to_numpy()
    seen = inter[inter["ts"] < cutoff].groupby("user_idx")["item_idx"].agg(set)
    rng = np.random.default_rng(seed)
    rows_u, rows_i = [], []
    for u in states["user_idx"]:
        s = seen.get(u, set())
        cand = np.array([i for i in top if i not in s])
        if len(cand):
            pick = rng.choice(cand, size=min(per_user, len(cand)), replace=False)
            rows_u.extend([u] * len(pick))
            rows_i.extend(pick.tolist())
    return pd.DataFrame(
        {"user_idx": np.array(rows_u, dtype=np.int32), "item_idx": np.array(rows_i, dtype=np.int32)}
    )
