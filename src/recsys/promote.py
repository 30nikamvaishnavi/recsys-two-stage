"""Evaluation gate + promotion of a newly trained model set to 'production'.

    python -m recsys.promote gate      # exit 0 if the candidate may be promoted, 1 otherwise
    python -m recsys.promote promote   # snapshot candidate metrics + reload Triton models
A candidate is promoted only if retrieval recall@100 and the best ranker NDCG@10 are not worse than
the current production snapshot (tolerance configurable). First run always promotes.
"""

from __future__ import annotations

import shutil
import sys

from .config import parse_cfg, path
from .utils import get_logger, load_json, save_json

log = get_logger("promote")
TOL = 0.005


def candidate_metrics(cfg) -> dict:
    r = load_json(path(cfg, "reports", "retrieval_metrics.json"))
    k = load_json(path(cfg, "reports", "ranking_metrics.json"))
    return {
        "retrieval_recall@100": r["two_tower"]["recall@100"],
        "ranker_ndcg@10": max(m["ranker"]["ndcg@10"] for m in k["models"].values()),
        "ranker_auc": max(m["ranker"]["auc"] for m in k["models"].values()),
    }


def gate(cfg) -> tuple[bool, dict]:
    cand = candidate_metrics(cfg)
    prod_f = path(cfg, "production") / "metrics.json"
    if not prod_f.exists():
        return True, {"candidate": cand, "production": None, "reason": "no production model yet"}
    prod = load_json(prod_f)
    checks = {k: cand[k] >= prod[k] - TOL for k in ("retrieval_recall@100", "ranker_ndcg@10")}
    return all(checks.values()), {"candidate": cand, "production": prod, "checks": checks}


def promote(cfg) -> None:
    ok, info = gate(cfg)
    if not ok:
        raise SystemExit(f"gate failed: {info}")
    pdir = path(cfg, "production", create=True)
    pdir.mkdir(parents=True, exist_ok=True)
    save_json(info["candidate"], pdir / "metrics.json")
    shutil.copytree(path(cfg, "models"), pdir / "models", dirs_exist_ok=True)
    try:
        import os

        from .serving.backends import TritonBackend

        meta = load_json(path(cfg, "serving") / "serving_meta.json")
        TritonBackend(os.environ.get("TRITON_URL", cfg["serving"]["triton_url"]), meta).reload()
        log.info("Triton models reloaded")
    except Exception as e:  # Triton may not be running (offline promotion)
        log.warning("could not reload Triton (%s); it will pick up the repository on next start", e)
    log.info("promoted: %s", info["candidate"])


def main():
    cfg, args = parse_cfg("Gate / promote", [("action", {"choices": ["gate", "promote"]})])
    if args.action == "gate":
        ok, info = gate(cfg)
        log.info("gate %s: %s", "PASSED" if ok else "FAILED", info)
        sys.exit(0 if ok else 1)
    promote(cfg)


if __name__ == "__main__":
    main()
