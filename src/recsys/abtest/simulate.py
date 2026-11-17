"""Simulated online A/B test: replay test-window users through the LIVE recommender (HTTP API or
in-process), simulate clicks with a position-biased model driven by held-out interactions.

THIS IS A SIMULATION: there are no real users. A user "clicks" an examined slot with probability
relevant_click_prob if the item is one of their real held-out test-period positives, otherwise
noise_click_prob. Examination probability decays with rank: 1 / (1 + decay*log2(1+rank)).

    python -m recsys.abtest.simulate --mode sample [--api-url http://localhost:8080]
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import parse_cfg, path
from ..utils import get_logger, save_json
from . import stats as S

log = get_logger("abtest")


def simulate_clicks(slate: list[int], relevant: set, acfg, rng: np.random.Generator) -> list[int]:
    """slate: ranked item ids -> list of 0/1 clicks (position-biased cascade-free click model)."""
    clicks = []
    for r, it in enumerate(slate):
        exam = 1.0 / (1.0 + acfg["examination_decay"] * np.log2(1 + r))
        p = acfg["relevant_click_prob"] if it in relevant else acfg["noise_click_prob"]
        clicks.append(int(rng.random() < exam * p))
    return clicks


def _ndcg(slate, relevant, k=10):
    if not relevant:
        return np.nan
    dcg = sum(1 / np.log2(i + 2) for i, it in enumerate(slate[:k]) if it in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def run_replay(
    recommend_fn,
    users: pd.DataFrame,
    acfg,
    out_jsonl: Path | None = None,
    feedback_fn=None,
    workers: int = 1,
) -> pd.DataFrame:
    """users: columns user_id, relevant (set of movie ids). recommend_fn(user_id, variant, k)-> (ids, meta)."""
    arms = acfg["arms"]
    K = acfg["slate_size"]

    def one(row):
        arm = "AB"[S.bucket(row.user_id, acfg["salt"])]
        variant = arms[arm]
        ids, meta = recommend_fn(int(row.user_id), variant, K)
        seed = int(
            hashlib.sha256(f"{acfg['salt']}:click:{row.user_id}".encode()).hexdigest()[:8], 16
        )
        clicks = simulate_clicks(ids, row.relevant, acfg, np.random.default_rng(seed))
        imps = [{"user_id": int(row.user_id), "arm": arm, "variant": variant, "position": r, "item_id": int(i), "clicked": c}
                for r, (i, c) in enumerate(zip(ids, clicks, strict=True))]  # fmt: skip
        if feedback_fn:
            for e in imps:
                feedback_fn(e)
        return {
            "user_id": int(row.user_id), "arm": arm, "variant": variant, "impressions": len(ids), "clicks": int(sum(clicks)),
            "ndcg@10": _ndcg(ids, row.relevant, 10), "recall@10": len(set(ids[:10]) & row.relevant) / max(len(row.relevant), 1),
            "hit@10": float(len(set(ids[:10]) & row.relevant) > 0), "strategy": meta.get("strategy"), "_imps": imps,
        }  # fmt: skip

    rows = (
        list(ThreadPoolExecutor(workers).map(one, users.itertuples(index=False)))
        if workers > 1
        else [one(r) for r in users.itertuples(index=False)]
    )
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w") as f:
            for r in rows:
                for e in r["_imps"]:
                    f.write(json.dumps(e) + "\n")
    return pd.DataFrame([{k: v for k, v in r.items() if k != "_imps"} for r in rows])


def analyze(df: pd.DataFrame, acfg) -> dict:
    nb, al = acfg["n_bootstrap"], acfg["alpha"]
    A, B = df[df["arm"] == "A"], df[df["arm"] == "B"]
    res: dict = {"simulation": True, "n_users": {"A": int(len(A)), "B": int(len(B))}, "arms": {}}
    for name, d in (("A", A), ("B", B)):
        ctr_u = (d["clicks"] / d["impressions"].clip(lower=1)).to_numpy()
        arm = {
            "variant": d["variant"].iloc[0] if len(d) else None,
            "impressions": int(d["impressions"].sum()),
            "clicks": int(d["clicks"].sum()),
        }
        arm["ctr"] = arm["clicks"] / max(arm["impressions"], 1)
        arm["ctr_user_level"] = dict(
            zip(("mean", "ci_low", "ci_high"), S.bootstrap_ci(ctr_u, nb, al), strict=True)
        )
        for m in ("ndcg@10", "recall@10", "hit@10"):
            arm[m] = dict(
                zip(
                    ("mean", "ci_low", "ci_high"),
                    S.bootstrap_ci(d[m].to_numpy(), nb, al),
                    strict=True,
                )
            )
        res["arms"][name] = arm
    ctr_a = (A["clicks"] / A["impressions"].clip(lower=1)).to_numpy()
    ctr_b = (B["clicks"] / B["impressions"].clip(lower=1)).to_numpy()
    res["tests"] = {
        "ctr_two_proportion_z": S.two_proportion_ztest(res["arms"]["A"]["clicks"], res["arms"]["A"]["impressions"], res["arms"]["B"]["clicks"], res["arms"]["B"]["impressions"]),
        "ctr_bootstrap_user_level": S.bootstrap_diff(ctr_a, ctr_b, nb, al),
        "ndcg@10_bootstrap": S.bootstrap_diff(A["ndcg@10"].to_numpy(), B["ndcg@10"].to_numpy(), nb, al),
        "recall@10_bootstrap": S.bootstrap_diff(A["recall@10"].to_numpy(), B["recall@10"].to_numpy(), nb, al),
    }  # fmt: skip
    res["mde_ctr_abs"] = S.min_detectable_effect(
        max(res["arms"]["A"]["ctr"], 1e-6), res["arms"]["A"]["impressions"]
    )
    res["aa_false_positive_rate"] = S.aa_false_positive_rate(ctr_a, 300, al)
    res["decision"] = S.decide(
        res["tests"]["ctr_bootstrap_user_level"], res["tests"]["ndcg@10_bootstrap"], al
    )
    return res


def load_users(sdir: Path, n: int, seed: int = 0) -> pd.DataFrame:
    st = pd.read_parquet(sdir / "user_states.parquet")
    um = pd.read_parquet(sdir / "user_map.parquet").set_index("user_idx")["user_id"]
    items = pd.read_parquet(sdir / "items.parquet").set_index("item_idx")["movie_id"]
    st = st.sample(min(n, len(st)), random_state=seed)
    return pd.DataFrame(
        {
            "user_id": um.loc[st["user_idx"]].to_numpy(),
            "relevant": [set(items.loc[t].tolist()) for t in st["targets"]],
        }
    )


def main():
    cfg, args = parse_cfg(
        "Simulated A/B test",
        [("--api-url", {"default": None}), ("--workers", {"type": int, "default": 4})],
    )
    a, sdir = cfg["abtest"], Path(path(cfg, "serving"))
    users = load_users(sdir, a["n_users"])
    if args.api_url:
        import httpx

        client = httpx.Client(base_url=args.api_url, timeout=30)

        def rec_fn(u, variant, k):
            r = client.post(
                "/recommend", json={"user_id": u, "k": k, "model_variant": variant}
            ).json()
            return [x["item_id"] for x in r["items"]], r["meta"]

        def fb(e):
            client.post("/feedback", json={"user_id": e["user_id"], "item_id": e["item_id"], "event": "click" if e["clicked"] else "impression",
                                           "position": e["position"], "variant": e["variant"]})  # fmt: skip

        df = run_replay(
            rec_fn,
            users,
            a,
            path(cfg, "reports", "ab_impressions.jsonl", create=True),
            fb,
            args.workers,
        )
    else:
        from ..serving.assets import load_recommender

        rec = load_recommender(cfg, force_memory_cache=True)

        def rec_fn(u, variant, k):
            r = rec.recommend(u, k, variant)
            return [x["item_id"] for x in r["items"]], r["meta"]

        df = run_replay(rec_fn, users, a, path(cfg, "reports", "ab_impressions.jsonl", create=True))
    res = analyze(df, a)
    res["mode"] = "http_api" if args.api_url else "in_process"
    save_json(res, path(cfg, "reports", "ab_results.json", create=True))
    df.to_parquet(path(cfg, "reports", "ab_user_level.parquet", create=True), index=False)
    log.info(
        "A/B decision: %s | %s", res["decision"], {k: v["ctr"] for k, v in res["arms"].items()}
    )


if __name__ == "__main__":
    main()
