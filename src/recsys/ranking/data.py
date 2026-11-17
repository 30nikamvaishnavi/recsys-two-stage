"""Ranker feature layout + candidate-set construction.

`candidate_dense` is the ONE place the ranker's dense vector is built; the training pipeline and
the online gateway both call it, so there is no train/serve skew.

dense = [user_num(3) | user_genre_aff(G) | item_genres(G) | item_num(6) | genome(D) | cross(3) | retrieval(2)]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.common import NUM_USER, G, masked_mean
from ..features.item_table import ItemTable
from ..features.state import SeenIndex, StateArrays

N_ITEM_NUM, N_CROSS, N_RET = 6, 3, 2


@dataclass(frozen=True)
class Layout:
    genome_dim: int

    @property
    def groups(self) -> dict[str, tuple[int, int]]:
        o, g, d = 0, {}, self.genome_dim
        for name, w in (("user_num", NUM_USER), ("user_gen", G), ("item_gen", G), ("item_num", N_ITEM_NUM),
                        ("genome", d), ("cross", N_CROSS), ("ret", N_RET)):  # fmt: skip
            g[name] = (o, o + w)
            o += w
        g["_total"] = (0, o)
        return g

    @property
    def dim(self) -> int:
        return self.groups["_total"][1]

    def mask(self, drop: list[str]) -> np.ndarray:
        m = np.ones(self.dim, np.float32)
        for name in drop:
            a, b = self.groups[name]
            m[a:b] = 0.0
        return m


ABLATIONS = {
    "full": [],
    "no_cross": ["cross"],
    "no_retrieval_score": ["ret"],
    "no_item_content": ["item_gen", "item_num", "genome"],
    "no_user_features": ["user_num", "user_gen"],
}


def candidate_dense(
    states: StateArrays,
    items: np.ndarray,
    scores: np.ndarray,
    table: ItemTable,
    ret_rank: np.ndarray | None = None,
) -> np.ndarray:
    """states (B), items (B,K), scores (B,K) -> dense (B*K, D) float32 (row-major over B then K)."""
    B, K = items.shape
    ug = masked_mean(states.hist, table.genres)  # (B,G)
    hg = masked_mean(states.hist, table.genome)  # (B,D)
    hp = masked_mean(states.hist, table.pop[:, :1])  # (B,1)
    ig, gv = table.genres[items], table.genome[items]  # (B,K,G) (B,K,D)
    overlap = (ug[:, None, :] * ig).sum(-1) / np.maximum(ig.sum(-1), 1.0)
    cos = (hg[:, None, :] * gv).sum(-1) / (
        np.linalg.norm(hg, axis=1)[:, None] * np.linalg.norm(gv, axis=-1) + 1e-6
    )
    gap = hp - table.pop[items, 0]
    item_num = np.concatenate(
        [table.pop[items], table.pop_rank_pct[items][..., None], (table.year_bucket[items] / 10.0)[..., None], table.has_genome[items][..., None]], -1
    )  # fmt: skip
    cross = np.stack([overlap, cos, gap], -1)
    rank = np.broadcast_to(np.arange(K), (B, K)) if ret_rank is None else ret_rank
    ret = np.stack([scores, np.log1p(rank) / np.log(1000.0)], -1)
    un = np.broadcast_to(states.numeric[:, None, :], (B, K, NUM_USER))
    ugb = np.broadcast_to(ug[:, None, :], (B, K, G))
    dense = np.concatenate([un, ugb, ig, item_num, gv, cross, ret], -1).astype(np.float32)
    return dense.reshape(B * K, -1)


def build_candidates(retriever, states: StateArrays, seen: SeenIndex | None, k: int):
    return retriever.retrieve(states, k, seen)


def labels_for(items: np.ndarray, targets: list) -> np.ndarray:
    y = np.zeros(items.shape, dtype=np.float32)
    for i, t in enumerate(targets):
        if len(t):
            y[i] = np.isin(items[i], t)
    return y


@dataclass
class RankData:
    dense: np.ndarray  # (M, D)
    item_idx: np.ndarray  # (M,)
    label: np.ndarray  # (M,)
    group: np.ndarray  # (M,) user position (for user-level splits)


def build_rank_train(
    retriever, states, seen, table, k: int, pop_negs=None, n_items: int = 0
) -> RankData:
    """Positives = held-out true interactions among retrieved candidates; negatives = the other
    retrieved candidates (hard negatives) + optional popularity negatives (Spark job)."""
    items, scores = retriever.retrieve(states, k, seen)
    y = labels_for(items, states.targets)
    dense = candidate_dense(states, items, scores, table)
    item_flat, y_flat = items.reshape(-1), y.reshape(-1)
    group = np.repeat(np.arange(len(states)), k)
    parts = [(dense, item_flat, y_flat, group)]
    if pop_negs is not None and len(pop_negs):
        pos_of = {int(u): i for i, u in enumerate(states.user_idx)}
        rows = pop_negs[pop_negs["user_idx"].isin(pos_of)]
        ui = rows["user_idx"].map(pos_of).to_numpy()
        ii = rows["item_idx"].to_numpy()
        retrieved = {(a, int(b)) for a, row in enumerate(items) for b in row}
        keep = np.array(
            [(a, int(b)) not in retrieved for a, b in zip(ui, ii, strict=True)], dtype=bool
        )
        ui, ii = ui[keep], ii[keep]
        if len(ui):
            sub = states.take(ui)
            ev = getattr(retriever, "e", None)
            sc = (
                (retriever.user_embeddings(sub) * ev[ii]).sum(1)
                if ev is not None
                else np.zeros(len(ui), np.float32)
            )
            d = candidate_dense(
                sub,
                ii[:, None],
                sc[:, None],
                table,
                ret_rank=np.full((len(ui), 1), max(n_items, 1000)),
            )
            lab = np.array(
                [float(i in set(states.targets[a].tolist())) for a, i in zip(ui, ii, strict=True)],
                np.float32,
            )
            parts.append((d, ii.astype(items.dtype), lab, ui))
    return RankData(*[np.concatenate(p) for p in zip(*parts, strict=True)])
