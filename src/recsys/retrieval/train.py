"""Train + evaluate popularity and SVD-MF baselines against the TFRS two-tower.

    python -m recsys.retrieval.train --mode sample
    python -m recsys.retrieval.train --seed 0 --tag logq-on --set retrieval.logq_correction=true
Tagged runs use isolated model and report paths; untagged paths remain the production defaults.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import yaml

from .. import tracking
from ..config import load_config, path
from ..features.loader import FeatureStore
from ..utils import get_logger, save_json, set_seed, timed
from . import retrievers as R

log = get_logger("train_retrieval")


def _arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate retrieval models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", default=None, choices=["smoke", "sample", "full"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value")
    args = parser.parse_args()

    base = load_config(args.config, args.mode)
    overrides = {}
    for spec in args.set:
        key, separator, raw_value = spec.partition("=")
        parts = key.split(".")
        node = base
        if not separator or not all(parts):
            parser.error(f"invalid override {spec!r}; expected section.key=value")
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                parser.error(f"unknown config key in override {key!r}")
            node = node[part]
        value = yaml.safe_load(raw_value)
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    if args.seed is not None:
        overrides.setdefault("data", {})["seed"] = args.seed
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        parser.error(
            "--tag must start with a letter/number and contain only letters, numbers, . _ -"
        )

    return load_config(args.config, args.mode, overrides), args, overrides


def _scalar_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def main():
    import tensorflow as tf

    from . import two_tower as tt

    cfg, args, overrides = _arguments()
    set_seed(cfg["data"]["seed"])
    r = cfg["retrieval"]
    fs = FeatureStore(cfg)
    ks = r["ks"]
    model_dir = f"retrieval_{args.tag}" if args.tag else "retrieval"
    report_name = f"retrieval_metrics_{args.tag}.json" if args.tag else "retrieval_metrics.json"
    mdir = path(cfg, "models", model_dir, create=True)
    mdir.mkdir(parents=True, exist_ok=True)

    tab_rank = fs.item_table("rank")
    tab_val = fs.item_table("val")
    tab_test = fs.item_table("test")
    arrs = fs.train_arrays()
    log.info(
        "training examples: %d | items %d | users %d", len(arrs["item_idx"]), fs.n_items, fs.n_users
    )
    val_states, val_seen = fs.states("val"), fs.seen("val")
    test_states, test_seen = fs.states("test"), fs.seen("test")
    min_history = cfg["coldstart"]["min_history"]
    warm_val = val_states.take(np.where(val_states.n_pos >= min_history)[0])
    warm_test = test_states.take(np.where(test_states.n_pos >= min_history)[0])
    experiment = {
        "tag": args.tag,
        "seed": int(cfg["data"]["seed"]),
        "overrides": overrides,
        "settings": dict(r),
    }
    report: dict = {
        "mode": cfg["mode"],
        "eval_window": "val",
        "n_eval_users_warm": int(len(warm_val)),
        "experiment": experiment,
        "val": {},
        "test": {},
    }

    with tracking.run(cfg, "retrieval") as mlf:
        mlf.params(
            {
                **r,
                "seed": cfg["data"]["seed"],
                "tag": args.tag or "",
                "n_train": len(arrs["item_idx"]),
            }
        )

        with timed(log, "baselines"):
            mf = R.SVDRetriever.fit(
                arrs["user_idx"],
                arrs["item_idx"],
                fs.n_users,
                fs.n_items,
                r["mf_rank"],
                seed=cfg["data"]["seed"],
            )
            pop_val = R.PopularityRetriever(tab_val)
            pop_test = R.PopularityRetriever(tab_test)
            for name, retriever in (("popularity", pop_val), ("svd_mf", mf)):
                report["val"][name] = R.evaluate_retriever_detailed(
                    retriever, warm_val, val_seen, ks, r["eval_users"]
                )
                test_retriever = pop_test if name == "popularity" else mf
                report["test"][name] = R.evaluate_retriever_detailed(
                    test_retriever, warm_test, test_seen, ks, r["eval_users"]
                )

        genre_matrix = tab_rank.genres
        ut = tt.build_user_tower(
            fs.n_users, fs.n_items, genre_matrix, fs.N, r["embedding_dim"], r["hidden"]
        )
        it = tt.build_item_tower(
            fs.n_items,
            cfg["features"]["genome_dim"],
            tab_rank.n_pop_buckets,
            r["embedding_dim"],
            r["hidden"],
        )
        model = tt.TwoTowerModel(ut, it, tab_rank, cfg)
        model.compile(optimizer=tf.keras.optimizers.Adam(r["learning_rate"]))
        ds = tt.make_dataset(arrs, r["batch_size"], cfg["data"]["seed"])
        quick = warm_val.subsample(min(2000, len(warm_val)), 1)
        best = {"recall": -1.0, "w_u": None, "w_i": None}

        class Track(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                emb = tt.embed_items(it, tab_val)
                rec = R.evaluate_retriever(
                    R.EmbeddingRetriever(lambda s: tt.embed_users(ut, s), emb),
                    quick,
                    val_seen,
                    (100,),
                )
                mlf.metrics({"loss": logs["loss"], "val_recall@100": rec["recall@100"]}, step=epoch)
                log.info(
                    "epoch %d loss %.4f val recall@100 %.4f", epoch, logs["loss"], rec["recall@100"]
                )
                if rec["recall@100"] > best["recall"]:
                    best.update(
                        recall=rec["recall@100"], w_u=ut.get_weights(), w_i=it.get_weights()
                    )

        with timed(log, "two-tower fit"):
            hist = model.fit(ds, epochs=r["epochs"], callbacks=[Track()], verbose=2)
        ut.set_weights(best["w_u"])
        it.set_weights(best["w_i"])
        tt.save_towers(ut, it, mdir)

        emb_val = tt.embed_items(it, tab_val)
        tower_val = R.EmbeddingRetriever(lambda s: tt.embed_users(ut, s), emb_val)
        report["val"]["two_tower"] = R.evaluate_retriever_detailed(
            tower_val, warm_val, val_seen, ks, r["eval_users"]
        )
        emb_test = tt.embed_items(it, tab_test)
        tower_test = R.EmbeddingRetriever(lambda s: tt.embed_users(ut, s), emb_test)
        report["test"]["two_tower"] = R.evaluate_retriever_detailed(
            tower_test, warm_test, test_seen, ks, r["eval_users"]
        )

        report["train_loss"] = [float(x) for x in hist.history["loss"]]
        report["best_epoch_val_recall@100_quick"] = best["recall"]
        report["cold_item_embeddings"] = _cold_item_eval(tower_val, tab_val, warm_val, val_seen, ks)
        for stage in ("rank", "val", "test"):
            np.save(mdir / f"item_emb_{stage}.npy", tt.embed_items(it, fs.item_table(stage)))

        for name, metrics in report["val"].items():
            report[name] = _scalar_metrics(metrics)
        logged_metrics = {
            f"{split}_{name}_{key}": value
            for split in ("val", "test")
            for name, metrics in report[split].items()
            for key, value in _scalar_metrics(metrics).items()
        }
        mlf.metrics(logged_metrics)
        save_json(report, path(cfg, "reports", report_name, create=True))

    log.info(
        "retrieval metrics (val and test, warm users): %s",
        {split: report[split] for split in ("val", "test")},
    )


def _cold_item_eval(retriever, table, states, seen, ks) -> dict:
    """Of the val targets that are cold items, what fraction does the two-tower surface in the
    top-K (popularity can never surface them: their count is 0)."""
    cold = table.cold_mask
    sub = (
        states.take(np.where([cold[t].any() for t in states.targets])[0]) if len(states) else states
    )
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
