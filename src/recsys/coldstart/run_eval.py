"""Evaluate cold-start strategies on validation users and write reports/coldstart_metrics.json.

    python -m recsys.coldstart.run_eval --mode sample
"""
from __future__ import annotations

import numpy as np

from ..config import parse_cfg, path
from ..features.loader import FeatureStore
from ..retrieval import retrievers as R
from ..utils import get_logger, save_json
from .evaluate import evaluate_cold_start
from .router import ColdStartRouter

log = get_logger("coldstart_eval")


def main():
    from ..retrieval import two_tower as tt

    cfg, _ = parse_cfg("Cold-start evaluation")
    fs = FeatureStore(cfg)
    c = cfg["coldstart"]
    tab = fs.item_table("val")
    router = ColdStartRouter(tab, c["min_history"], c["content_alpha"], c["mmr_lambda"])
    mdir = path(cfg, "models", "retrieval")
    ut, _ = tt.load_towers(cfg, fs.meta, fs.item_table("rank").genres, fs.item_table("rank").n_pop_buckets, mdir)
    tower = R.EmbeddingRetriever(lambda s: tt.embed_users(ut, s), np.load(mdir / "item_emb_val.npy"))
    seen = fs.seen("val")
    users = evaluate_cold_start(router, fs.states("val"), seen, tab, tower, c["eval_truncations"], 50, 1500)
    save_json({"users": users, "note": "history truncated to k most recent positives; k=0 uses onboarding genres"}, path(cfg, "reports", "coldstart_metrics.json", create=True))
    log.info("cold-start NDCG@10: %s", {k: {s: round(m["ndcg@10"], 4) for s, m in v.items()} for k, v in users["by_history_len"].items()})


if __name__ == "__main__":
    main()
