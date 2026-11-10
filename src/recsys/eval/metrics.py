"""Ranking metrics with binary relevance. Definitions (documented in docs/results.md):

recall@K = |top-K ∩ relevant| / |relevant|      (macro-averaged over users)
NDCG@K   = DCG@K / IDCG@K,  DCG = Σ rel_i / log2(i+2),  IDCG uses min(|relevant|, K) ones
MAP@K    = mean over users of  Σ_{i<=K} P@i·rel_i / min(|relevant|, K)
hit@K    = 1[top-K ∩ relevant ≠ ∅]
"""
from __future__ import annotations

import numpy as np


def _rel_matrix(ranked: np.ndarray, targets: list) -> np.ndarray:
    rel = np.zeros(ranked.shape, dtype=bool)
    for i, t in enumerate(targets):
        if len(t):
            rel[i] = np.isin(ranked[i], t)
    return rel


def _n_rel(targets: list) -> np.ndarray:
    return np.array([len(t) for t in targets], dtype=np.float64)


def recall_at_k(ranked, targets, k: int) -> float:
    rel = _rel_matrix(ranked[:, :k], targets)
    n = _n_rel(targets)
    ok = n > 0
    return float((rel.sum(1)[ok] / n[ok]).mean()) if ok.any() else float("nan")


def hit_at_k(ranked, targets, k: int) -> float:
    rel = _rel_matrix(ranked[:, :k], targets)
    ok = _n_rel(targets) > 0
    return float(rel.any(1)[ok].mean()) if ok.any() else float("nan")


def ndcg_per_user(ranked, targets, k: int) -> np.ndarray:
    kk = min(k, ranked.shape[1])
    rel = _rel_matrix(ranked[:, :kk], targets).astype(np.float64)
    disc = 1.0 / np.log2(np.arange(kk) + 2)
    dcg = (rel * disc).sum(1)
    ideal = np.array([disc[: min(len(t), kk)].sum() for t in targets])
    return np.where(ideal > 0, dcg / np.maximum(ideal, 1e-12), np.nan)


def ndcg_at_k(ranked, targets, k: int) -> float:
    v = ndcg_per_user(ranked, targets, k)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def map_at_k(ranked, targets, k: int) -> float:
    kk = min(k, ranked.shape[1])
    rel = _rel_matrix(ranked[:, :kk], targets).astype(np.float64)
    prec = np.cumsum(rel, axis=1) / (np.arange(kk) + 1)
    n = np.minimum(_n_rel(targets), kk)
    ap = np.where(n > 0, (prec * rel).sum(1) / np.maximum(n, 1), np.nan)
    return float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan")


def evaluate_ranked(ranked: np.ndarray, targets: list, ks=(10, 50, 100)) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        kk = min(k, ranked.shape[1])
        out[f"recall@{k}"] = recall_at_k(ranked, targets, kk)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, targets, kk)
        out[f"hit@{k}"] = hit_at_k(ranked, targets, kk)
    out["map@10"] = map_at_k(ranked, targets, 10)
    out["n_users"] = int(sum(1 for t in targets if len(t)))
    return out


def auc_score(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def logloss(y, p, eps: float = 1e-7) -> float:
    y, p = np.asarray(y, dtype=np.float64), np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def calibration_table(y, p, n_bins: int = 10) -> dict:
    """Reliability bins + Expected Calibration Error."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            rows.append({"bin": i, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"bins": rows, "ece": float(ece), "mean_pred": float(p.mean()), "base_rate": float(y.mean())}
