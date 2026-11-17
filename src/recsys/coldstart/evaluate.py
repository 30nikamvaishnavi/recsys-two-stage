"""Cold-start evaluation: simulate users with k in {0..4} interactions by truncating the history of
warm validation users (targets unchanged), and compare strategies on the SAME users.

k=0 users get 'onboarding genres' = top genres of their (hidden) full history, mimicking a signup form.
"""

from __future__ import annotations

import numpy as np

from ..eval.metrics import evaluate_ranked
from ..features.state import SeenIndex, StateArrays
from .router import ColdStartRouter


def evaluate_cold_start(router: ColdStartRouter, states: StateArrays, seen: SeenIndex, table, two_tower_retriever=None,
                        truncations=(0, 1, 2, 3, 4), kmax: int = 50, max_users: int = 1500, ks=(10, 50)) -> dict:  # fmt: skip
    base = states.take(np.where(states.n_pos >= max(truncations) + 1)[0]).subsample(max_users, 0)
    onboard = [router.top_genres(h) for h in base.hist]
    out: dict = {"n_users": int(len(base)), "by_history_len": {}}
    for k in truncations:
        tr = base.truncate_history(k)
        res = {}

        def run(name, fn):
            items = np.zeros((len(tr), kmax), dtype=np.int64)
            for r in range(len(tr)):
                it, _ = fn(r)
                items[r, : len(it)] = it
            res[name] = evaluate_ranked(items, tr.targets, ks)

        seen_of = lambda r: seen.seen(int(tr.user_idx[r]))  # noqa: E731
        run("global_popularity", lambda r: router.global_popularity(kmax, seen_of(r)))
        if k == 0:
            run("genre_popularity", lambda r: router.genre_popularity(onboard[r], kmax, seen_of(r)))
        if k >= 1:
            run(
                "content_blend",
                lambda r: router.content_blend(tr.take(np.array([r])), kmax, seen_of(r)),
            )
        if two_tower_retriever is not None:
            it, _ = two_tower_retriever.retrieve(tr, kmax, seen)
            res["two_tower"] = evaluate_ranked(it, tr.targets, ks)

        def routed(r):
            d = router.route(int(tr.n_pos[r]), onboard[r] if k == 0 else None)
            return router.recommend(d, tr, r, kmax, seen_of(r), onboard[r])

        run("router", routed)
        out["by_history_len"][str(k)] = res
    return out
