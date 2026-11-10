"""Cold-start fallback router with logged reason codes.

    n_pos >= min_history                      -> two_stage          WARM_USER
    1 <= n_pos < min_history                  -> content_blend      SPARSE_HISTORY_CONTENT_BLEND
    n_pos == 0 and onboarding genres given    -> genre_popularity   NEW_USER_ONBOARDING_GENRES
    n_pos == 0, no signal                     -> global_popularity  NEW_USER_NO_SIGNAL
    models down / retrieval empty             -> global_popularity  MODEL_UNAVAILABLE / RETRIEVAL_EMPTY

New ITEMS are handled in the embedding layer: items with no interactions are embedded from content
only (item tower with id=0), so they are indexed immediately (see retrieval.two_tower.embed_items).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.common import GENRE_IDX
from ..features.item_table import ItemTable
from ..features.state import StateArrays
from ..retrieval.retrievers import ContentRetriever, topk_from_scores
from ..utils import get_logger
from .diversity import mmr

log = get_logger("coldstart")

WARM, SPARSE, ONBOARD, NO_SIGNAL = "two_stage", "content_blend", "genre_popularity", "global_popularity"


@dataclass
class Decision:
    strategy: str
    reason: str


class ColdStartRouter:
    def __init__(self, table: ItemTable, min_history: int = 5, content_alpha: float = 0.7, mmr_lambda: float = 0.7):
        self.t, self.min_history, self.mmr_lambda = table, min_history, mmr_lambda
        self.content = table.content_vectors()
        self.content_ret = ContentRetriever(table, alpha=content_alpha)
        self.pop = table.pop[:, 0].copy()
        self.pop[0] = -np.inf

    def route(self, n_pos: int, onboarding_genres: list[str] | None = None, models_ok: bool = True, retrieval_empty: bool = False) -> Decision:
        if not models_ok:
            d = Decision(NO_SIGNAL, "MODEL_UNAVAILABLE_POPULARITY")
        elif retrieval_empty:
            d = Decision(NO_SIGNAL, "RETRIEVAL_EMPTY_POPULARITY")
        elif n_pos >= self.min_history:
            d = Decision(WARM, "WARM_USER")
        elif n_pos >= 1:
            d = Decision(SPARSE, "SPARSE_HISTORY_CONTENT_BLEND")
        elif onboarding_genres:
            d = Decision(ONBOARD, "NEW_USER_ONBOARDING_GENRES")
        else:
            d = Decision(NO_SIGNAL, "NEW_USER_NO_SIGNAL")
        log.debug("route n_pos=%d -> %s (%s)", n_pos, d.strategy, d.reason)
        return d

    # ---- strategies ---------------------------------------------------------------------------
    def _finish(self, scores: np.ndarray, k: int, seen: np.ndarray, diversify: bool, pool: int = 200):
        scores = scores.copy()
        scores[0] = -np.inf
        if len(seen):
            scores[seen] = -np.inf
        items, sc = topk_from_scores(scores[None, :], pool if diversify else k)
        items, sc = items[0], sc[0]
        ok = np.isfinite(sc)
        items, sc = items[ok], sc[ok]
        if diversify and len(items):
            sel = mmr(items, sc, self.content, k, self.mmr_lambda)
            items, sc = items[sel], sc[sel]
        return items[:k], sc[:k]

    def global_popularity(self, k: int, seen: np.ndarray, trending_weight: float = 0.3):
        lp = self.t.pop
        score = (1 - trending_weight) * lp[:, 0] + trending_weight * lp[:, 1]
        return self._finish(score, k, seen, diversify=True)

    def genre_popularity(self, genres: list[str], k: int, seen: np.ndarray):
        gi = [GENRE_IDX[g] for g in genres if g in GENRE_IDX]
        if not gi:
            return self.global_popularity(k, seen)
        overlap = self.t.genres[:, gi].sum(1) / len(gi)
        score = np.where(overlap > 0, self.t.pop[:, 0] * (0.5 + overlap), -np.inf)
        return self._finish(score, k, seen, diversify=True)

    def content_blend(self, state_row: StateArrays, k: int, seen: np.ndarray):
        sc = self.content_ret.score_chunk(state_row)[0]
        return self._finish(sc, k, seen, diversify=True)

    def recommend(self, decision: Decision, states: StateArrays, row: int, k: int, seen: np.ndarray, onboarding_genres=None):
        if decision.strategy == SPARSE:
            return self.content_blend(states.take(np.array([row])), k, seen)
        if decision.strategy == ONBOARD:
            return self.genre_popularity(onboarding_genres or [], k, seen)
        return self.global_popularity(k, seen)

    def top_genres(self, hist: np.ndarray, n: int = 3) -> list[str]:
        from ..features.common import GENRES

        h = hist[hist != 0]
        if len(h) == 0:
            return []
        aff = self.t.genres[h].sum(0)
        return [GENRES[i] for i in np.argsort(-aff)[:n] if aff[i] > 0]
