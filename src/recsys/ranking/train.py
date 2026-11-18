"""Train DCN-V2 and Wide&Deep rankers on retrieval candidates; evaluate vs retrieval-only ordering.

    python -m recsys.ranking.train --mode sample
    python -m recsys.ranking.train --seed 1 --tag s1      # isolated model/report paths

Candidates: top-K from the two-tower (exact search) for users whose positives fall in the ranker
window [t1,t2); state frozen at t1. Evaluation on the val window (state frozen at t2) and on the
untouched test window (state frozen at t3). Nothing is selected or tuned on the test window.
Untagged runs keep the production paths used by export/serving; tagged runs are experiments.
"""

from __future__ import annotations

import argparse
import re
from functools import partial

import numpy as np
import yaml

from .. import tracking
from ..config import load_config, path
from ..eval.metrics import (
    auc_score,
    calibration_table,
    evaluate_ranked,
    evaluate_ranked_detailed,
    logloss,
    ndcg_per_user,
    recall_per_user,
)
from ..features.loader import FeatureStore
from ..retrieval import retrievers as R
from ..utils import get_logger, save_json, set_seed, timed
from .analysis import (
    ACTIVITY_NAMES,
    activity_buckets,
    bucket_means,
    item_popularity_buckets,
    list_stats,
    paired_bootstrap_diff,
    target_recall_by_bucket,
)
from .data import ABLATIONS, Layout, build_rank_train, candidate_dense, labels_for

log = get_logger("train_ranker")

KS = (10, 50)
N_BOOTSTRAP = 2000


def _arguments():
    parser = argparse.ArgumentParser(description="Train rankers")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", default=None, choices=["smoke", "sample", "full"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value")
    args = parser.parse_args()
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.tag):
        parser.error("--tag must start with a letter/number and contain only letters, numbers, _ -")
    base = load_config(args.config, args.mode)
    overrides: dict = {}
    for spec in args.set:
        key, sep, raw = spec.partition("=")
        parts = key.split(".")
        if not sep or len(parts) < 2 or not all(parts):
            parser.error(f"invalid override {spec!r}; expected section.key=value")
        node = base
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict) or parts[-1] not in node:
            parser.error(f"unknown config key in override {key!r}")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(raw)
    if args.seed is not None:
        overrides.setdefault("data", {})["seed"] = args.seed
    return load_config(args.config, args.mode, overrides), args


def split_by_user(rd, frac: float, seed: int):
    from .data import RankData

    users = np.unique(rd.group)
    hold = set(
        np.random.default_rng(seed)
        .choice(users, max(int(len(users) * frac), 1), replace=False)
        .tolist()
    )
    m = np.array([g in hold for g in rd.group])

    def mk(s):
        return RankData(rd.dense[s], rd.item_idx[s], rd.label[s], rd.group[s])

    return mk(~m), mk(m)


def eval_ranker(predict_fn, states, items, scores, table, ks=(10, 50)) -> dict:
    """Candidates (B,K) re-scored; compare model ordering to retrieval-only ordering."""
    B, K = items.shape
    dense = candidate_dense(states, items, scores, table)
    p = predict_fn(items.reshape(-1), dense).reshape(B, K)
    order = np.argsort(-p, axis=1)
    reranked = np.take_along_axis(items, order, 1)
    y = labels_for(items, states.targets)
    out = {
        "ranker": evaluate_ranked(reranked, states.targets, ks),
        "retrieval_only": evaluate_ranked(items, states.targets, ks),
    }
    ok = y.sum(1) > 0
    out["ranker"].update(
        auc=auc_score(y[ok].ravel(), p[ok].ravel()), logloss=logloss(y[ok].ravel(), p[ok].ravel())
    )
    out["calibration"] = calibration_table(y[ok].ravel(), p[ok].ravel())
    out["candidate_recall"] = float(
        np.mean([np.isin(t, items[i]).mean() for i, t in enumerate(states.targets) if len(t)])
    )
    return out


def detailed_eval(
    predict_fn, states, items, scores, table, n_items: int, n_bootstrap: int = N_BOOTSTRAP
) -> dict:
    """Bootstrap CIs, a PAIRED ranker-vs-retrieval difference, and user/item breakdowns."""
    B, K = items.shape
    dense = candidate_dense(states, items, scores, table)
    p = predict_fn(items.reshape(-1), dense).reshape(B, K)
    reranked = np.take_along_axis(items, np.argsort(-p, axis=1), 1)
    targets = states.targets

    out = {
        "ranker": evaluate_ranked_detailed(reranked, targets, KS, n_bootstrap),
        "retrieval_only": evaluate_ranked_detailed(items, targets, KS, n_bootstrap),
    }
    for part in out.values():  # per-user arrays are only needed here; keep the report small
        part.pop("per_user", None)
    y = labels_for(items, targets)
    ok = y.sum(1) > 0
    out["ranker"].update(
        auc=auc_score(y[ok].ravel(), p[ok].ravel()), logloss=logloss(y[ok].ravel(), p[ok].ravel())
    )

    rec_rank, rec_ret = recall_per_user(reranked, targets, 10), recall_per_user(items, targets, 10)
    ndcg_rank, ndcg_ret = ndcg_per_user(reranked, targets, 10), ndcg_per_user(items, targets, 10)
    out["paired_diff_vs_retrieval"] = {
        "recall@10": paired_bootstrap_diff(rec_rank, rec_ret, n_bootstrap),
        "ndcg@10": paired_bootstrap_diff(ndcg_rank, ndcg_ret, n_bootstrap),
    }
    bucket_idx, edges = activity_buckets(states.n_pos)
    out["by_user_activity"] = {
        "edges_prior_positives": list(edges),
        "buckets": bucket_means(
            {
                "ranker_ndcg@10": ndcg_rank,
                "retrieval_ndcg@10": ndcg_ret,
                "ranker_recall@10": rec_rank,
                "retrieval_recall@10": rec_ret,
            },
            bucket_idx,
            ACTIVITY_NAMES,
        ),
    }
    item_bucket = item_popularity_buckets(table.n_pos)
    out["by_item_popularity"] = {
        "ranker": target_recall_by_bucket(reranked, targets, item_bucket, 10),
        "retrieval_only": target_recall_by_bucket(items, targets, item_bucket, 10),
    }
    out["list_stats@10"] = {
        "ranker": list_stats(reranked[:, :10], table.n_pos, table.genres, n_items),
        "retrieval_only": list_stats(items[:, :10], table.n_pos, table.genres, n_items),
    }
    return out


def main():
    from ..retrieval import two_tower as tt
    from . import models as M

    cfg, args = _arguments()
    suffix = f"_{args.tag}" if args.tag else ""
    set_seed(cfg["data"]["seed"])
    rc = cfg["ranking"]
    fs = FeatureStore(cfg)
    mdir = path(cfg, "models", "retrieval")
    rdir = path(cfg, "models", f"ranking{suffix}", create=True)
    rdir.mkdir(parents=True, exist_ok=True)
    layout = Layout(cfg["features"]["genome_dim"])
    ut, _ = tt.load_towers(
        cfg, fs.meta, fs.item_table("rank").genres, fs.item_table("rank").n_pop_buckets, mdir
    )
    inter = fs.interactions(["user_idx", "item_idx", "ts"])

    def retriever(stage):
        return R.EmbeddingRetriever(
            lambda s: tt.embed_users(ut, s), np.load(mdir / f"item_emb_{stage}.npy")
        )

    with timed(log, "build ranker training set"):
        st = fs.states("rank")
        st = st.take(np.where(st.n_pos >= 1)[0]).subsample(rc["max_train_users"], 0)
        negs = fs.pop_negatives() if rc["use_pop_negatives"] else None
        rd = build_rank_train(
            retriever("rank"),
            st,
            fs.seen("rank", inter),
            fs.item_table("rank"),
            rc["n_candidates_train"],
            negs,
            fs.n_items,
        )
        train, hold = split_by_user(rd, rc["holdout_user_frac"], 1)
        log.info(
            "ranker rows train %d (pos rate %.3f) | holdout %d",
            len(train.label),
            train.label.mean(),
            len(hold.label),
        )

    min_history = cfg["coldstart"]["min_history"]

    def eval_split(stage):
        s = fs.states(stage)
        s = s.take(np.where(s.n_pos >= min_history)[0]).subsample(rc["eval_users"], 0)
        its, scs = retriever(stage).retrieve(s, rc["n_candidates_serve"], fs.seen(stage, inter))
        return s, its, scs, fs.item_table(stage)

    evals = {"val": eval_split("val"), "test": eval_split("test")}
    vs, items, scores, tab_val = evals["val"]
    report = {
        "mode": cfg["mode"],
        "seed": cfg["data"]["seed"],
        "tag": args.tag,
        "train_rows": int(len(train.label)),
        "n_eval_users": int(len(vs)),
        "n_eval_users_test": int(len(evals["test"][0])),
        "models": {},
        "detailed": {"val": {}, "test": {}},
        "ablations": {},
        "ablations_test": {},
    }

    with tracking.run(cfg, f"ranking{suffix}") as mlf:
        mlf.params(dict(rc))
        for kind in ("wide_deep", "dcn_v2"):
            with timed(log, f"fit {kind}"):
                model = M.build_ranker(kind, layout, fs.n_items, rc)
                M.fit_ranker(model, train, hold, rc)
            model.save_weights(str(rdir / f"{kind}.weights.h5"))
            predict = partial(M.predict, model)
            res = eval_ranker(predict, vs, items, scores, tab_val)
            report["models"][kind] = res
            mlf.metrics({f"{kind}_{k}": v for k, v in res["ranker"].items()})
            log.info(
                "%s val: %s | retrieval-only ndcg@10 %.4f",
                kind,
                res["ranker"],
                res["retrieval_only"]["ndcg@10"],
            )
            for split, (s, its, scs, tab) in evals.items():
                det = detailed_eval(predict, s, its, scs, tab, fs.n_items)
                report["detailed"][split][kind] = det
                d = det["paired_diff_vs_retrieval"]["ndcg@10"]
                log.info(
                    "%s %s: ranker ndcg@10 %.4f vs retrieval %.4f | paired diff %.4f [%.4f, %.4f]",
                    kind,
                    split,
                    det["ranker"]["ndcg@10"],
                    det["retrieval_only"]["ndcg@10"],
                    d["mean_diff"],
                    d["ci_low"],
                    d["ci_high"],
                )
        abl_val, abl_test = _ablate(M, layout, fs, rc, train, hold, evals)
        report["ablations"], report["ablations_test"] = abl_val, abl_test
        save_json({"layout_dim": layout.dim, "genome_dim": layout.genome_dim}, rdir / "layout.json")
        save_json(report, path(cfg, "reports", f"ranking_metrics{suffix}.json", create=True))
        plot_path = path(cfg, "reports", f"ranker_calibration{suffix}.png", create=True)
        _plot_calibration(report, plot_path)


def _ablate(M, layout, fs, rc, train, hold, evals) -> tuple[dict, dict]:
    """Retrain DCN-V2 with feature groups removed; evaluate on val (selection) and test."""
    val_out, test_out = {}, {}
    keys = ("ndcg@10", "recall@10", "map@10", "auc")
    for name in rc["ablations"]:
        m = M.build_ranker("dcn_v2", layout, fs.n_items, rc, ABLATIONS[name])
        M.fit_ranker(m, train, hold, rc, verbose=0)
        predict = partial(M.predict, m)
        for split, target in (("val", val_out), ("test", test_out)):
            s, its, scs, tab = evals[split]
            r = eval_ranker(predict, s, its, scs, tab)["ranker"]
            target[name] = {k: r[k] for k in keys}
        log.info("ablation %-20s val %s | test %s", name, val_out[name], test_out[name])
    return val_out, test_out


def _plot_calibration(report, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for k, v in report["models"].items():
        b = v["calibration"]["bins"]
        ax.plot(
            [x["mean_pred"] for x in b],
            [x["frac_pos"] for x in b],
            "o-",
            label=f"{k} (ECE {v['calibration']['ece']:.3f})",
        )
    ax.set(
        xlabel="mean predicted P(click)",
        ylabel="observed positive rate",
        title="Ranker calibration (val candidates)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
