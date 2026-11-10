"""Online recommendation pipeline.

 cache? -> user state -> router -> [user tower (Triton)] -> ANN top-N (FAISS) -> filter seen ->
 features -> ranker (Triton) -> MMR diversity -> cache -> response.  Every stage is timed.
Any model failure degrades to popularity (reason code MODEL_UNAVAILABLE_POPULARITY) instead of a 5xx.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..cache.redis_cache import RecCache
from ..coldstart.diversity import mmr
from ..coldstart.router import WARM, ColdStartRouter
from ..features.common import GENRES, GENRE_IDX
from ..features.item_table import ItemTable
from ..ranking.data import candidate_dense
from ..utils import Timer, get_logger
from .state_store import UserStateStore

log = get_logger("pipeline")
VARIANTS = ("wide_deep", "dcn_v2")


class Recommender:
    def __init__(self, backend, ann, item_emb: np.ndarray, table: ItemTable, items_df, store: UserStateStore,
                 cache: RecCache, router: ColdStartRouter, cfg, log_dir: Path | None = None):  # fmt: skip
        self.backend, self.ann, self.emb, self.table, self.store, self.cache, self.router = backend, ann, item_emb, table, store, cache, router
        self.cfg = cfg
        self.n_cand = cfg["ranking"]["n_candidates_serve"]
        self.mmr_lambda = cfg["serving"]["mmr_lambda"]
        self.titles = ["", *items_df.sort_values("item_idx")["title"].tolist()]
        self.genres_txt = ["", *items_df.sort_values("item_idx")["genres"].tolist()]
        self.movie_id = np.array([0, *items_df.sort_values("item_idx")["movie_id"].tolist()])
        self.idx_of_movie = {int(m): i for i, m in enumerate(self.movie_id) if i}
        self.content = router.content
        self.log_dir = Path(log_dir) if log_dir else None
        self._t0 = time.time()

    def clock(self) -> float:
        """Serving clock: starts at the data cutoff so recency features match offline training."""
        return self.store.cutoff + (time.time() - self._t0)

    # ------------------------------------------------------------------------------------------
    def _items_payload(self, items, ret, rank_sc, final):
        return [
            {
                "item_id": int(self.movie_id[i]), "title": self.titles[i], "genres": self.genres_txt[i],
                "score": float(s), "retrieval_score": None if r is None else float(r), "ranker_score": None if q is None else float(q),
            }
            for i, r, q, s in zip(items, ret, rank_sc, final, strict=True)
        ]  # fmt: skip

    def recommend(self, user_id: int, k: int = 10, variant: str = "dcn_v2", onboarding_genres: list[str] | None = None, use_cache: bool = True) -> dict:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        lat: dict[str, float] = {}
        total = Timer()
        total.__enter__()
        gkey = ",".join(sorted(onboarding_genres or []))
        vkey = f"{variant}|{gkey}" if gkey else variant
        if use_cache:
            with Timer() as t:
                hit = self.cache.get_recs(vkey, user_id, k)
            lat["cache"] = t.ms
            if hit:
                total.__exit__()
                hit["meta"] = {**hit["meta"], "cache_hit": True, "latency_ms": {**lat, "total": total.ms}}
                return hit

        with Timer() as t:
            state = self.store.get(user_id, onboarding_genres)
            seen = self.store.seen(state)
            arrs = self.store.arrays(state, self.clock())
            decision = self.router.route(state.n_pos, state.onboarding_genres or onboarding_genres)
        lat["state_router"] = t.ms

        items = ret = rk = final = None
        if decision.strategy == WARM:
            try:
                items, ret, rk, final = self._two_stage(arrs, seen, variant, k, lat)
                if len(items) == 0:
                    decision = self.router.route(state.n_pos, retrieval_empty=True)
            except Exception as e:  # model server down, timeout, shape mismatch...
                log.error("two-stage failed (%s); degrading to popularity", e)
                decision = self.router.route(state.n_pos, models_ok=False)
        if items is None or len(items) == 0:
            with Timer() as t:
                items, final = self.router.recommend(decision, arrs, 0, k, seen, state.onboarding_genres or onboarding_genres)
            lat["fallback"] = t.ms
            ret = rk = [None] * len(items)

        out = {
            "user_id": user_id, "k": k, "variant": variant,
            "items": self._items_payload(items, ret, rk, final),
            "meta": {"strategy": decision.strategy, "reason_code": decision.reason, "cache_hit": False, "backend": getattr(self.backend, "name", "none")},
        }  # fmt: skip
        if use_cache:
            self.cache.set_recs(vkey, user_id, k, out)
        total.__exit__()
        out["meta"]["latency_ms"] = {**lat, "total": total.ms}
        return out

    def _two_stage(self, arrs, seen, variant, k, lat):
        with Timer() as t:
            u = self.backend.user_embed(arrs.user_idx, arrs.hist, arrs.numeric)
        lat["user_tower"] = t.ms
        with Timer() as t:
            n_fetch = self.n_cand + min(len(seen), 1000)
            cand, sc = self.ann.search(u, n_fetch)
            cand, sc = cand[0], sc[0]
            keep = (cand != 0) & ~np.isin(cand, seen)
            cand, sc = cand[keep][: self.n_cand], sc[keep][: self.n_cand]
        lat["ann_search"] = t.ms
        if len(cand) == 0:
            return cand, [], [], []
        with Timer() as t:
            dense = candidate_dense(arrs, cand[None, :], sc[None, :].astype(np.float32), self.table)
        lat["features"] = t.ms
        with Timer() as t:
            p = self.backend.rank(variant, cand, dense)
        lat["ranker"] = t.ms
        with Timer() as t:
            order = np.argsort(-p)[:50]
            sel = order[mmr(cand[order], p[order], self.content, k, self.mmr_lambda)]
        lat["postprocess"] = t.ms
        return cand[sel], sc[sel], p[sel], p[sel]

    # ------------------------------------------------------------------------------------------
    def similar_items(self, item_id: int, k: int = 10) -> list[dict]:
        idx = self.idx_of_movie.get(int(item_id))
        if idx is None:
            raise KeyError(item_id)
        cand, sc = self.ann.search(self.emb[idx][None, :], k + 1)
        out = [(int(c), float(s)) for c, s in zip(cand[0], sc[0], strict=True) if c not in (0, idx)][:k]
        return [{"item_id": int(self.movie_id[c]), "title": self.titles[c], "genres": self.genres_txt[c], "similarity": s, "is_cold_item": bool(self.table.cold_mask[c])} for c, s in out]

    def feedback(self, user_id: int, item_id: int, event: str, rating: float | None = None, position: int | None = None, variant: str | None = None) -> dict:
        idx = self.idx_of_movie.get(int(item_id))
        rec = {"ts": time.time(), "user_id": user_id, "item_id": item_id, "event": event, "rating": rating, "position": position, "variant": variant}
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / "feedback.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        invalidated = 0
        if event in ("click", "rating") and idx is not None:
            st = self.store.get(user_id)
            if st.user_idx != 0 or st.is_new:
                self.store.add_interaction(st, idx, rating, self.clock())
                invalidated = self.cache.invalidate_user(user_id)
        return {"logged": True, "cache_entries_invalidated": invalidated}

    def genre_options(self) -> list[str]:
        return [g for g in GENRES if g in GENRE_IDX and g != "(no genres listed)"]
