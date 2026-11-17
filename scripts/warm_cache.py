"""Warm Redis: popularity lists (global + per genre) and recommendations for the most active users."""

from __future__ import annotations

import numpy as np
from recsys.config import parse_cfg
from recsys.features.common import GENRES
from recsys.serving.assets import load_recommender
from recsys.utils import get_logger

log = get_logger("warm_cache")


def main():
    cfg, args = parse_cfg("Warm the cache", [("--top-users", {"type": int, "default": 500})])
    rec = load_recommender(cfg)
    seen = np.empty(0, dtype=np.int64)
    ids, _ = rec.router.global_popularity(100, seen)
    rec.cache.set_list("global", [int(rec.movie_id[i]) for i in ids])
    for g in GENRES[:-1]:
        ids, _ = rec.router.genre_popularity([g], 50, seen)
        rec.cache.set_list(f"genre:{g}", [int(rec.movie_id[i]) for i in ids])
    users = sorted(rec.store.states.values(), key=lambda s: -s.n_all)[: args.top_users]
    inv = {v: k for k, v in rec.store.ext2idx.items()}
    for st in users:
        for variant in ("wide_deep", "dcn_v2"):
            rec.recommend(inv[st.user_idx], cfg["serving"]["default_k"], variant)
    log.info("warmed %d users x 2 variants + popularity lists | %s", len(users), rec.cache.stats())


if __name__ == "__main__":
    main()
