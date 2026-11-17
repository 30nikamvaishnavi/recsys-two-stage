"""Async load test against the running API. Reports QPS and latency p50/p95/p99, cache hit rate.

    python scripts/load_test.py --url http://localhost:8080 --requests 2000 --concurrency 16 --repeat-frac 0.3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from recsys.config import load_config, path
from recsys.utils import percentiles


async def main_async(a):
    cfg = load_config(mode=a.mode)
    um = pd.read_parquet(Path(path(cfg, "serving")) / "user_states.parquet")
    users = (
        pd.read_parquet(Path(path(cfg, "serving")) / "user_map.parquet")
        .set_index("user_idx")
        .loc[um["user_idx"], "user_id"]
        .to_numpy()
    )
    rng = np.random.default_rng(0)
    pool = rng.choice(users, size=min(len(users), 2000), replace=False)
    seq = [
        int(
            rng.choice(pool[: max(10, int(len(pool) * 0.05))])
            if rng.random() < a.repeat_frac
            else rng.choice(pool)
        )
        for _ in range(a.requests)
    ]
    sem, lat, hits, errs = asyncio.Semaphore(a.concurrency), [], 0, 0
    lat_hit, lat_miss = [], []

    async with httpx.AsyncClient(base_url=a.url, timeout=30) as c:

        async def one(u):
            nonlocal hits, errs
            async with sem:
                t = time.perf_counter()
                try:
                    r = await c.post(
                        "/recommend", json={"user_id": u, "k": 10, "model_variant": a.variant}
                    )
                    r.raise_for_status()
                    ms = (time.perf_counter() - t) * 1000
                    lat.append(ms)
                    if r.json()["meta"]["cache_hit"]:
                        hits += 1
                        lat_hit.append(ms)
                    else:
                        lat_miss.append(ms)
                except Exception:
                    errs += 1

        t0 = time.perf_counter()
        await asyncio.gather(*[one(u) for u in seq])
        wall = time.perf_counter() - t0
    res = {"requests": a.requests, "concurrency": a.concurrency, "errors": errs, "qps": len(lat) / wall, "latency_ms": percentiles(lat),
           "cache_hit_rate": hits / max(len(lat), 1), "latency_ms_cache_hit": percentiles(lat_hit), "latency_ms_cache_miss": percentiles(lat_miss)}  # fmt: skip
    out = path(cfg, "reports", "load_test.json", create=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--requests", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--repeat-frac", type=float, default=0.3)
    p.add_argument("--variant", default="dcn_v2")
    p.add_argument("--mode", default=None)
    asyncio.run(main_async(p.parse_args()))
