"""Train DCN-V2 and Wide&Deep rankers on retrieval candidates; evaluate vs retrieval-only ordering.

    python -m recsys.ranking.train --mode sample
Candidates: top-K from the two-tower (exact search) for users whose positives fall in the ranker
window [t1,t2); state frozen at t1. Evaluation on the val window with state frozen at t2.
"""

from __future__ import annotations

import numpy as np

from .. import tracking
from ..config import parse_cfg, path
from ..eval.metrics import auc_score, calibration_table, evaluate_ranked, logloss
from ..features.loader import FeatureStore
from ..retrieval import retrievers as R
from ..utils import get_logger, save_json, set_seed, timed
from .data import ABLATIONS, Layout, build_rank_train, candidate_dense, labels_for

log = get_logger("train_ranker")


def split_by_user(rd, frac: float, seed: int):
    from .data import RankData

    users = np.unique(rd.group)
    hold = set(
        np.random.default_rng(seed)
        .choice(users, max(int(len(users) * frac), 1), replace=False)
        .tolist()
    )
    m = np.array([g in hold for g in rd.group])
    mk = lambda s: RankData(rd.dense[s], rd.item_idx[s], rd.label[s], rd.group[s])  # noqa: E731
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


def main():
    from ..retrieval import two_tower as tt
    from . import models as M

    cfg, _ = parse_cfg("Train rankers")
    set_seed(cfg["data"]["seed"])
    rc = cfg["ranking"]
    fs = FeatureStore(cfg)
    mdir = path(cfg, "models", "retrieval")
    rdir = path(cfg, "models", "ranking", create=True)
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

    vs = fs.states("val")
    vs = vs.take(np.where(vs.n_pos >= cfg["coldstart"]["min_history"])[0]).subsample(
        rc["eval_users"], 0
    )
    tab_val = fs.item_table("val")
    items, scores = retriever("val").retrieve(vs, rc["n_candidates_serve"], fs.seen("val", inter))
    report = {
        "mode": cfg["mode"],
        "train_rows": int(len(train.label)),
        "n_eval_users": int(len(vs)),
        "models": {},
        "ablations": {},
    }

    with tracking.run(cfg, "ranking") as mlf:
        mlf.params(dict(rc))
        for kind in ("wide_deep", "dcn_v2"):
            with timed(log, f"fit {kind}"):
                model = M.build_ranker(kind, layout, fs.n_items, rc)
                M.fit_ranker(model, train, hold, rc)
            model.save_weights(str(rdir / f"{kind}.weights.h5"))
            res = eval_ranker(lambda i, d, m=model: M.predict(m, i, d), vs, items, scores, tab_val)
            report["models"][kind] = res
            mlf.metrics({f"{kind}_{k}": v for k, v in res["ranker"].items()})
            log.info(
                "%s val: %s | retrieval-only ndcg@10 %.4f",
                kind,
                res["ranker"],
                res["retrieval_only"]["ndcg@10"],
            )
        report["ablations"] = _ablate(M, layout, fs, rc, train, hold, vs, items, scores, tab_val)
        save_json({"layout_dim": layout.dim, "genome_dim": layout.genome_dim}, rdir / "layout.json")
        save_json(report, path(cfg, "reports", "ranking_metrics.json", create=True))
        _plot_calibration(report, path(cfg, "reports", "ranker_calibration.png", create=True))


def _ablate(M, layout, fs, rc, train, hold, vs, items, scores, tab_val) -> dict:
    out = {}
    for name in rc["ablations"]:
        m = M.build_ranker("dcn_v2", layout, fs.n_items, rc, ABLATIONS[name])
        M.fit_ranker(m, train, hold, rc, verbose=0)
        r = eval_ranker(lambda i, d, mm=m: M.predict(mm, i, d), vs, items, scores, tab_val)[
            "ranker"
        ]
        out[name] = {k: r[k] for k in ("ndcg@10", "recall@10", "map@10", "auc")}
        log.info("ablation %-20s %s", name, out[name])
    return out


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
