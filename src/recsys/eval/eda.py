"""Exploratory data analysis: figures + a JSON summary used by the notebook and docs.

    python -m recsys.eval.eda --mode sample     ->  docs/figures/<mode>/*.png , artifacts/reports/<mode>/eda_summary.json
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, parse_cfg, path
from ..data.prepare import Dataset, load_dataset
from ..features.common import GENRES, parse_genres
from ..utils import get_logger, save_json

log = get_logger("eda")


def summarize(ds: Dataset) -> dict:
    i = ds.interactions
    per_user = i.groupby("user_idx").size()
    per_item = i.groupby("item_idx").size()
    first_item_stage = i.groupby("item_idx")["stage"].min()
    first_user_stage = i.groupby("user_idx")["stage"].min()
    n_users, n_items = len(ds.user_map), len(ds.items)
    return {
        "n_users": n_users, "n_items": n_items, "n_interactions": len(i),
        "sparsity": 1 - len(i) / (n_users * n_items),
        "rating_mean": float(i["rating"].mean()), "positive_rate": float(i["is_pos"].mean()),
        "interactions_per_user": {"median": float(per_user.median()), "p90": float(per_user.quantile(0.9)), "max": int(per_user.max())},
        "interactions_per_item": {"median": float(per_item.median()), "p90": float(per_item.quantile(0.9)), "max": int(per_item.max())},
        "top1pct_items_share": float(np.sort(per_item.to_numpy())[::-1][: max(n_items // 100, 1)].sum() / len(i)),
        "cold_items_in_test_window_pct": float((first_item_stage >= 3).mean() * 100),
        "new_users_after_train_pct": float((first_user_stage >= 2).mean() * 100),
        "genome_coverage_pct": float(ds.genome["item_idx"].nunique() / n_items * 100),
        "per_stage": {int(k): int(v) for k, v in i["stage"].value_counts().sort_index().items()},
    }  # fmt: skip


def make_figures(ds: Dataset, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    i, b = ds.interactions, ds.boundaries
    files = []

    def save(fig, name):
        fig.tight_layout()
        p = out / name
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    i["rating"].value_counts().sort_index().plot.bar(ax=ax[0], color="#4c78a8")
    ax[0].set(title="Rating distribution", xlabel="rating", ylabel="count")
    pu = i.groupby("user_idx").size().sort_values(ascending=False).to_numpy()
    ax[1].loglog(np.arange(1, len(pu) + 1), pu)
    ax[1].set(
        title="Interactions per user (rank-size)", xlabel="user rank", ylabel="# interactions"
    )
    pi = i.groupby("item_idx").size().sort_values(ascending=False).to_numpy()
    ax[2].loglog(np.arange(1, len(pi) + 1), pi, color="#e45756")
    ax[2].set(title="Item popularity long tail", xlabel="item rank", ylabel="# interactions")
    save(fig, "distributions.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    t = pd.to_datetime(i["ts"], unit="s")
    t.dt.to_period("M").value_counts().sort_index().plot(ax=ax, color="#4c78a8")
    ax.set(
        title="Interactions per month (split boundaries in text below)",
        xlabel="month",
        ylabel="count",
    )
    ax.text(
        0.01,
        0.95,
        "split: " + ", ".join(f"{k}={pd.to_datetime(v, unit='s').date()}" for k, v in b.items()),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    save(fig, "time_split.png")

    gl = [parse_genres(s) for s in ds.items["genres"]]
    counts = np.zeros(len(GENRES))
    for gs in gl:
        counts[gs] += 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(GENRES, counts, color="#72b7b2")
    ax.set(title="Items per genre", ylabel="# items")
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right", fontsize=8)
    save(fig, "genres.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    first = i.groupby("item_idx")["stage"].min().value_counts().sort_index()
    ax.bar(
        ["retr", "rank", "val", "test"][: len(first)],
        first.reindex(range(4), fill_value=0).to_numpy()[: len(first)],
        color="#f58518",
    )
    ax.set(title="Stage in which each item FIRST appears (cold items)", ylabel="# items")
    save(fig, "cold_items.png")
    return files


def main():
    cfg, _ = parse_cfg("EDA")
    ds = load_dataset(Path(path(cfg, "data_processed")))
    out = ROOT / "docs" / "figures" / cfg["mode"]
    files = make_figures(ds, out)
    s = summarize(ds)
    save_json(s, path(cfg, "reports", "eda_summary.json", create=True))
    log.info("EDA: %s | figures: %s", s, [f.name for f in files])


if __name__ == "__main__":
    main()
