"""Cold-start evaluation: simulate users with k in {0..4} interactions by truncating the history of
warm validation users (targets unchanged), and compare strategies on the SAME users.

k=0 users get 'onboarding genres' = top genres of their (hidden) full history, mimicking a signup form.
"""

from __future__ import annotations

import numpy as np

from ..eval.metrics import evaluate_ranked
from ..features.state import SeenIndex, StateArrays
from .router import ColdStartRouter


def _run_strategy(name, fn, states, results, kmax, ks):
    items = np.zeros((len(states), kmax), dtype=np.int64)
    for r in range(len(states)):
        it, _ = fn(r)
        items[r, : len(it)] = it
    results[name] = evaluate_ranked(items, states.targets, ks)


def evaluate_cold_start(router: ColdStartRouter, states: StateArrays, seen: SeenIndex, table, two_tower_retriever=None,
                        truncations=(0, 1, 2, 3, 4), kmax: int = 50, max_users: int = 1500, ks=(10, 50)) -> dict:  # fmt: skip
    base = states.take(np.where(states.n_pos >= max(truncations) + 1)[0]).subsample(max_users, 0)
    onboard = [router.top_genres(h) for h in base.hist]
    out: dict = {"n_users": int(len(base)), "by_history_len": {}}
    for k in truncations:
        tr = base.truncate_history(k)
        res = {}

        seen_of = lambda r, state=tr: seen.seen(int(state.user_idx[r]))  # noqa: E731
        _run_strategy(
            "global_popularity",
            lambda r, seen_fn=seen_of: router.global_popularity(kmax, seen_fn(r)),
            tr,
            res,
            kmax,
            ks,
        )
        if k == 0:
            _run_strategy(
                "genre_popularity",
                lambda r, genres=onboard, seen_fn=seen_of: router.genre_popularity(
                    genres[r], kmax, seen_fn(r)
                ),
                tr,
                res,
                kmax,
                ks,
            )
        if k >= 1:
            _run_strategy(
                "content_blend",
                lambda r, state=tr, seen_fn=seen_of: router.content_blend(
                    state.take(np.array([r])), kmax, seen_fn(r)
                ),
                tr,
                res,
                kmax,
                ks,
            )
        if two_tower_retriever is not None:
            it, _ = two_tower_retriever.retrieve(tr, kmax, seen)
            res["two_tower"] = evaluate_ranked(it, tr.targets, ks)

        def routed(r, state=tr, history_len=k, seen_fn=seen_of, genres=onboard):
            d = router.route(int(state.n_pos[r]), genres[r] if history_len == 0 else None)
            return router.recommend(d, state, r, kmax, seen_fn(r), genres[r])

        _run_strategy("router", routed, tr, res, kmax, ks)
        out["by_history_len"][str(k)] = res
    return out
