"""Serving-time user states for EVERY user, frozen at a cutoff.

`features.pandas_features.user_state` only returns users that have >=1 positive in the target
window (it exists to build evaluation sets), so serving from it would treat most real users as
brand-new. This builds the same columns for all users with any interaction before the cutoff,
and has no `targets` column.
"""

from __future__ import annotations

import pandas as pd

from ..features.common import SECONDS_PER_DAY, pad_history


def user_state_all(inter: pd.DataFrame, cutoff: int, history_len: int) -> pd.DataFrame:
    """State of every user with an interaction before `cutoff` (same columns as user_state)."""
    before = inter[inter["ts"] < cutoff]
    agg = before.groupby("user_idx").agg(
        n_all=("rating", "size"),
        avg_rating=("rating", "mean"),
        last_ts=("ts", "max"),
        n_pos=("is_pos", "sum"),
    )
    pos = before[before["is_pos"]].sort_values(["user_idx", "ts", "item_idx"])
    hist = (
        pos.groupby("user_idx")
        .tail(history_len)
        .groupby("user_idx")["item_idx"]
        .agg(list)
        .rename("hist_items")
    )
    st = agg.join(hist, how="left")
    st["n_all"] = st["n_all"].astype("int64")
    st["n_pos"] = st["n_pos"].astype("int64")
    st["days_since_last"] = (cutoff - st["last_ts"]) / SECONDS_PER_DAY
    st["hist_items"] = [
        pad_history(h if isinstance(h, list) else [], history_len) for h in st["hist_items"]
    ]
    return st.reset_index().drop(columns=["last_ts"])
