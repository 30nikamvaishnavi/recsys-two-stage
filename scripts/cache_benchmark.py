"""Latency with vs without cache (in-process recommender; Redis if REDIS_URL reachable)."""

from __future__ import annotations

import json

import numpy as np
from recsys.config import parse_cfg, path
from recsys.serving.assets import load_recommender
from recsys.utils import percentiles


def main():
    cfg, args = parse_cfg("Cache benchmark", [("--n", {"type": int, "default": 300})])
    rec = load_recommender(cfg)
    inv = {v: k for k, v in rec.store.ext2idx.items()}
    users = [inv[u] for u in list(rec.store.states)[: args.n]]
    nocache = [
        rec.recommend(u, 10, "dcn_v2", use_cache=False)["meta"]["latency_ms"]["total"]
        for u in users
    ]
    cold = [
        rec.recommend(u, 10, "dcn_v2")["meta"]["latency_ms"]["total"] for u in users
    ]  # fills cache
    warm = [rec.recommend(u, 10, "dcn_v2")["meta"]["latency_ms"]["total"] for u in users]  # hits
    res = {"cache_backend": rec.cache.backend_name, "no_cache": percentiles(nocache), "first_pass_miss": percentiles(cold),
           "second_pass_hit": percentiles(warm), "speedup_p50": float(np.median(nocache) / max(np.median(warm), 1e-9)), **rec.cache.stats()}  # fmt: skip
    path(cfg, "reports", "cache_benchmark.json", create=True).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
