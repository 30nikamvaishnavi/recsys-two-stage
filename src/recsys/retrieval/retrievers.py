"""Candidate generators sharing one interface:

    retrieve(states: StateArrays, k, seen: SeenIndex|None) -> (items (B,k) int, scores (B,k) float)

Item indices are 1..n (0 is padding and never returned).
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import scipy.sparse as sp

from ..features.common import masked_mean
from ..features.item_table import ItemTable
from ..features.state import SeenIndex, StateArrays

CHUNK = 512


def topk_from_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1] - 1)
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(scores, part, 1)
    order = np.argsort(-ps, axis=1)
    return np.take_along_axis(part, order, 1), np.take_along_axis(ps, order, 1)


class Retriever(Protocol):
    def retrieve(
        self, states: StateArrays, k: int, seen: SeenIndex | None
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _ChunkedMixin:
    def score_chunk(self, states: StateArrays) -> np.ndarray:  # (b, n_items+1)
        raise NotImplementedError

    def retrieve(self, states: StateArrays, k: int, seen: SeenIndex | None = None):
        items, scs = [], []
        for s in range(0, len(states), CHUNK):
            sub = states.take(np.arange(s, min(s + CHUNK, len(states))))
            sc = self.score_chunk(sub).astype(np.float32, copy=False)
            sc[:, 0] = -np.inf
            if seen is not None:
                seen.mask_scores(sc, sub.user_idx)
            i, v = topk_from_scores(sc, k)
            items.append(i)
            scs.append(v)
        return np.vstack(items), np.vstack(scs)


class PopularityRetriever(_ChunkedMixin):
    """Non-personalised baseline: most popular items as of the cutoff."""

    def __init__(self, table: ItemTable):
        self.pop = table.pop[:, 0].copy()

    def score_chunk(self, states):
        return np.tile(self.pop, (len(states), 1))


class SVDRetriever(_ChunkedMixin):
    """Matrix-factorisation baseline: truncated SVD of the binary user-item matrix (stage-0
    positives). Users are folded in from their recent history: score = (x V) Vᵀ."""

    def __init__(self, v: np.ndarray):
        self.v = v.astype(np.float32)  # (n_items+1, r)

    @classmethod
    def fit(cls, user_idx, item_idx, n_users: int, n_items: int, rank: int, seed: int = 0):
        from scipy.sparse.linalg import svds

        m = sp.csr_matrix(
            (np.ones(len(user_idx), np.float32), (user_idx, item_idx)),
            shape=(n_users + 1, n_items + 1),
        )
        m.data[:] = 1.0
        m = sp.csr_matrix(m.multiply(1.0 / np.sqrt(np.maximum(m.sum(1), 1))))  # damp heavy users
        r = min(rank, min(m.shape) - 2)
        _, _, vt = svds(m, k=r, random_state=seed)
        return cls(vt.T)

    def score_chunk(self, states):
        b, n = states.hist.shape
        x = np.zeros((b, self.v.shape[0]), dtype=np.float32)
        rows = np.repeat(np.arange(b), n)
        mask = states.hist.ravel() != 0
        x[rows[mask], states.hist.ravel()[mask]] = 1.0
        return (x @ self.v) @ self.v.T


class ContentRetriever(_ChunkedMixin):
    """Content-based kNN: cosine(mean content vector of history, item content vectors), optionally
    blended with popularity (z-scored) -> used for users with 1-4 interactions."""

    def __init__(self, table: ItemTable, alpha: float = 1.0):
        self.c = table.content_vectors()
        self.alpha = alpha
        lp = table.pop[:, 0]
        self.pop_z = ((lp - lp[1:].mean()) / (lp[1:].std() + 1e-6)).astype(np.float32)

    def score_chunk(self, states):
        prof = masked_mean(states.hist, self.c)
        prof /= np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-6)
        cos = prof @ self.c.T
        cz = (cos - cos[:, 1:].mean(1, keepdims=True)) / (cos[:, 1:].std(1, keepdims=True) + 1e-6)
        return self.alpha * cz + (1 - self.alpha) * self.pop_z[None, :]


class EmbeddingRetriever(_ChunkedMixin):
    """Two-tower retrieval with exact inner-product search over item embeddings. `user_fn` maps a
    StateArrays chunk to (b, d) user embeddings (TF tower, ONNX session, ...)."""

    def __init__(self, user_fn: Callable[[StateArrays], np.ndarray], item_emb: np.ndarray):
        self.user_fn = user_fn
        self.e = item_emb.astype(np.float32)

    def user_embeddings(self, states: StateArrays) -> np.ndarray:
        return np.vstack(
            [
                self.user_fn(states.take(np.arange(s, min(s + 4096, len(states)))))
                for s in range(0, len(states), 4096)
            ]
        )

    def score_chunk(self, states):
        return self.user_fn(states) @ self.e.T


def evaluate_retriever(
    retriever,
    states: StateArrays,
    seen,
    ks=(10, 50, 100),
    max_users: int | None = None,
    seed: int = 0,
):
    from ..eval.metrics import evaluate_ranked

    st = states.subsample(max_users, seed)
    items, _ = retriever.retrieve(st, max(ks), seen)
    return evaluate_ranked(items, st.targets, ks)


def evaluate_retriever_detailed(
    retriever,
    states: StateArrays,
    seen,
    ks=(10, 50, 100),
    max_users: int | None = None,
    seed: int = 0,
):
    from ..eval.metrics import evaluate_ranked_detailed

    st = states.subsample(max_users, seed)
    items, _ = retriever.retrieve(st, max(ks), seen)
    return evaluate_ranked_detailed(items, st.targets, ks, n_bootstrap=2000, seed=seed)
