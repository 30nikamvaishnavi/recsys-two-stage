"""Maximal Marginal Relevance re-ranking (relevance vs redundancy on content vectors)."""
from __future__ import annotations

import numpy as np


def mmr(items: np.ndarray, rel: np.ndarray, content: np.ndarray, k: int, lam: float = 0.7) -> np.ndarray:
    """Greedy MMR. items (n,), rel (n,) relevance, content (n_items+1, d) L2-normalised vectors.
    Returns the selected positions (indices into `items`) in pick order."""
    n = len(items)
    k = min(k, n)
    if n == 0:
        return np.empty(0, dtype=int)
    r = (rel - rel.min()) / (np.ptp(rel) + 1e-9)
    v = content[items]
    chosen: list[int] = []
    max_sim = np.zeros(n)
    avail = np.ones(n, bool)
    for _ in range(k):
        score = lam * r - (1 - lam) * max_sim
        score[~avail] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        avail[j] = False
        max_sim = np.maximum(max_sim, v @ v[j])
    return np.array(chosen, dtype=int)
