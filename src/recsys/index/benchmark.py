"""Benchmark Flat vs HNSW vs IVF-PQ: recall@100 vs exact, p50/p95 single-query latency, memory.
Selects the cheapest config (lowest p95) that reaches the target recall, saves it, plots the trade-off.

    python -m recsys.index.build --mode sample
"""
from __future__ import annotations

import time

import numpy as np

from ..config import parse_cfg, path
from ..features.loader import FeatureStore
from ..utils import get_logger, percentiles, save_json
from .faiss_index import HAVE_FAISS, AnnIndex, _apply_params, build, index_bytes

log = get_logger("index")


def _latency(a: AnnIndex, q: np.ndarray, k: int) -> dict:
    ts = []
    a.search(q[:5], k)  # warm up
    for i in range(len(q)):
        t = time.perf_counter()
        a.search(q[i : i + 1], k)
        ts.append((time.perf_counter() - t) * 1000)
    b = time.perf_counter()
    a.search(q, k)
    qps = len(q) / (time.perf_counter() - b)
    return {**percentiles(ts), "batch_qps": qps}


def _recall(a: AnnIndex, q: np.ndarray, truth: np.ndarray, k: int) -> float:
    got, _ = a.search(q, k)
    return float(np.mean([len(set(got[i]) & set(truth[i])) / k for i in range(len(q))]))


def run_benchmark(emb: np.ndarray, queries: np.ndarray, icfg, k: int = 100, lat_n: int = 300):
    """emb: (n,d) item embeddings (items 1..n). Returns (rows, chosen_row, {name: AnnIndex})."""
    flat = build("flat", emb, icfg)
    truth, _ = flat.search(queries, k)
    rows, built = [], {}

    def add(name, a, params):
        if HAVE_FAISS and a.kind != "flat":
            _apply_params(a.index, a.kind, params)
        a.params = {**a.params, **params}
        r = {"name": name, "kind": a.kind, "params": a.params, "recall@100": _recall(a, queries, truth, k) if a.kind != "flat" else 1.0,
             "memory_mb": index_bytes(a) / 1e6, **_latency(a, queries[:lat_n], k)}  # fmt: skip
        rows.append(r)
        built[name] = a
        log.info("%s recall %.3f p95 %.3f ms mem %.1f MB", name, r["recall@100"], r["p95"], r["memory_mb"])

    add("flat", flat, {})
    if HAVE_FAISS:
        hn = build("hnsw", emb, icfg)
        for ef in icfg["hnsw_ef_search"]:
            add(f"hnsw_ef{ef}", hn, {"efSearch": ef})
        iv = build("ivfpq", emb, icfg)
        for npb in icfg["ivf_nprobe"]:
            add(f"ivfpq_np{npb}", iv, {"nprobe": npb})
    # NB: sweeps reuse one index object, so rebuild the chosen one fresh below.
    ok = [r for r in rows if r["recall@100"] >= icfg["target_recall"]]
    chosen = min(ok, key=lambda r: r["p95"]) if ok else max(rows, key=lambda r: r["recall@100"])
    return rows, chosen, truth


def plot(rows, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for kind, mk in (("flat", "s"), ("hnsw", "o"), ("ivfpq", "^")):
        rr = [r for r in rows if r["kind"] == kind]
        if rr:
            ax[0].scatter([r["p95"] for r in rr], [r["recall@100"] for r in rr], marker=mk, s=60, label=kind)
            ax[1].scatter([r["memory_mb"] for r in rr], [r["recall@100"] for r in rr], marker=mk, s=60, label=kind)
    ax[0].set(xlabel="p95 latency per query (ms)", ylabel="recall@100 vs exact", title="Recall vs latency")
    ax[1].set(xlabel="index memory (MB)", ylabel="recall@100 vs exact", title="Recall vs memory")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    cfg, _ = parse_cfg("Build + benchmark FAISS indexes")
    icfg = cfg["index"]
    fs = FeatureStore(cfg)
    mdir = path(cfg, "models", "retrieval")
    # Index the serving-time embeddings (as of the test cutoff = latest data).
    emb = np.load(mdir / "item_emb_test.npy")[1:]
    # query set: real user embeddings from the retrieval model for val users
    from ..retrieval import two_tower as tt

    ut, _ = tt.load_towers(cfg, fs.meta, fs.item_table("rank").genres, fs.item_table("rank").n_pop_buckets, mdir)
    states = fs.states("val").subsample(icfg["n_queries"], 0)
    q = tt.embed_users(ut, states)
    rows, chosen, _ = run_benchmark(emb, q, icfg, 100, icfg["latency_queries"])
    best = build(chosen["kind"], emb, icfg, **{k: v for k, v in chosen["params"].items() if k in ("efSearch", "nprobe")})
    sdir = path(cfg, "serving", create=True)
    sdir.mkdir(parents=True, exist_ok=True)
    best.save(sdir)
    plot(rows, path(cfg, "reports", "faiss_benchmark.png", create=True))
    save_json({"results": rows, "chosen": chosen, "faiss": HAVE_FAISS, "n_items": len(emb), "dim": emb.shape[1]}, path(cfg, "reports", "faiss_benchmark.json", create=True))
    log.info("chosen index: %s (recall@100 %.3f, p95 %.3f ms)", chosen["name"], chosen["recall@100"], chosen["p95"])


if __name__ == "__main__":
    main()
