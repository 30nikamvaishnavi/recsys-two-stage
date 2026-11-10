"""Train + evaluate retrieval: popularity and SVD-MF baselines vs the TFRS two-tower.

    python -m recsys.retrieval.train --mode sample
Writes artifacts/models/retrieval/{user,item}_tower.weights.h5, item embeddings per stage and
artifacts/reports/retrieval_metrics.json. Metrics are computed on the TIME-BASED validation window.
"""
from __future__ import annotations

import numpy as np

from .. import tracking
from ..config import parse_cfg, path
from ..features.loader import FeatureStore
from ..utils import get_logger, save_json, set_seed, timed
from . import retrievers as R

log = get_logger("train_retrieval")


def main():
    import tensorflow as tf

    from . import two_tower as tt

    cfg, _ = parse_cfg("Train two-tower retrieval")
    set_seed(cfg["data"]["seed"])
    r = cfg["retrieval"]
    fs = FeatureStore(cfg)
    ks = r["ks"]
    mdir = path(cfg, "models", "retrieval", create=True)
    mdir.mkdir(parents=True, exist_ok=True)

    tab_rank, tab_val = fs.item_table("rank"), fs.item_table("val")
    arrs = fs.train_arrays()
    log.info("training examples: %d | items %d | users %d", len(arrs["item_idx"]), fs.n_items, fs.n_users)
    val_states, val_seen = fs.states("val"), fs.seen("val")
    warm = val_states.take(np.where(val_states.n_pos >= cfg["coldstart"]["min_history"])[0])
    report: dict = {"mode": cfg["mode"], "eval_window": "val", "n_eval_users_warm": int(len(warm))}

    with tracking.run(cfg, "retrieval") as mlf:
        mlf.params({**r, "n_train": len(arrs["item_idx"])})

        # ---- baselines -------------------------------------------------------------------------
        with timed(log, "baselines"):
            pop = R.PopularityRetriever(tab_val)
            mf = R.SVDRetriever.fit(arrs["user_idx"], arrs["item_idx"], fs.n_users, fs.n_items, r["mf_rank"])
            report["popularity"] = R.evaluate_retriever(pop, warm, val_seen, ks, r["eval_users"])
            report["svd_mf"] = R.evaluate_retriever(mf, warm, val_seen, ks, r["eval_users"])

        # ---- two-tower -------------------------------------------------------------------------
        genre_matrix = tab_rank.genres
        ut = tt.build_user_tower(fs.n_users, fs.n_items, genre_matrix, fs.N, r["embedding_dim"], r["hidden"])
        it = tt.build_item_tower(fs.n_items, cfg["features"]["genome_dim"], tab_rank.n_pop_buckets, r["embedding_dim"], r["hidden"])
        model = tt.TwoTowerModel(ut, it, tab_rank, cfg)
        model.compile(optimizer=tf.keras.optimizers.Adam(r["learning_rate"]))
        ds = tt.make_dataset(arrs, r["batch_size"], cfg["data"]["seed"])
        quick = warm.subsample(min(2000, len(warm)), 1)
        best = {"recall": -1.0, "w_u": None, "w_i": None}

        class Track(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                emb = tt.embed_items(it, tab_val)
                rec = R.evaluate_retriever(R.EmbeddingRetriever(lambda s: tt.embed_users(ut, s), emb), quick, val_seen, (100,))
                mlf.metrics({"loss": logs["loss"], "val_recall@100": rec["recall@100"]}, step=epoch)
                log.info("epoch %d loss %.4f val recall@100 %.4f", epoch, logs["loss"], rec["recall@100"])
                if rec["recall@100"] > best["recall"]:
                    best.update(recall=rec["recall@100"], w_u=ut.get_weights(), w_i=it.get_weights())

        with timed(log, "two-tower fit"):
            hist = model.fit(ds, epochs=r["epochs"], callbacks=[Track()], verbose=2)
        ut.set_weights(best["w_u"])
        it.set_weights(best["w_i"])
        tt.save_towers(ut, it, mdir)

        emb_val = tt.embed_items(it, tab_val)
        tower = R.EmbeddingRetriever(lambda s: tt.embed_users(ut, s), emb_val)
        report["two_tower"] = R.evaluate_retriever(tower, warm, val_seen, ks, r["eval_users"])
        report["train_loss"] = [float(x) for x in hist.history["loss"]]
        report["best_epoch_val_recall@100_quick"] = best["recall"]
        # Cold items: items with no positives before the val cutoff, embedded from content only.
        report["cold_item_embeddings"] = _cold_item_eval(tower, tab_val, warm, val_seen, ks)
        for stage in ("rank", "val", "test"):
            np.save(mdir / f"item_emb_{stage}.npy", tt.embed_items(it, fs.item_table(stage)))

        for name in ("popularity", "svd_mf", "two_tower"):
            mlf.metrics({f"{name}_{k}": v for k, v in report[name].items()})
        save_json(report, path(cfg, "reports", "retrieval_metrics.json", create=True))
    log.info("retrieval metrics (val, warm users): %s", {k: report[k] for k in ("popularity", "svd_mf", "two_tower")})


def _cold_item_eval(retriever, table, states, seen, ks) -> dict:
    """Of the val targets that are cold items, what fraction does the two-tower surface in the
    top-K (popularity can never surface them: their count is 0)."""
    cold = table.cold_mask
    sub = states.take(np.where([cold[t].any() for t in states.targets])[0]) if len(states) else states
    if len(sub) == 0:
        return {"n_users": 0}
    sub.targets = [t[cold[t]] for t in sub.targets]
    items, _ = retriever.retrieve(sub, max(ks), seen)
    from ..eval.metrics import evaluate_ranked

    out = evaluate_ranked(items, sub.targets, ks)
    out["n_cold_items"] = int(cold.sum())
    return out


if __name__ == "__main__":
    main()
