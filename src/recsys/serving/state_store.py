"""Online user-state store (in-process; production would back this with Redis/feature store)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..features.common import SECONDS_PER_DAY, pad_history, user_numeric


@dataclass
class UserState:
    user_idx: int  # 0 = unknown user
    hist: list[int] = field(default_factory=list)  # last-N positives, oldest -> newest
    n_pos: int = 0
    n_all: int = 0
    avg_rating: float = float("nan")
    last_ts: float = 0.0
    extra_seen: set = field(default_factory=set)
    onboarding_genres: list[str] = field(default_factory=list)
    is_new: bool = False


class UserStateStore:
    def __init__(
        self,
        user_map: pd.DataFrame,
        states: pd.DataFrame,
        seen_csr: sp.csr_matrix,
        history_len: int,
        cutoff_ts: int,
        pos_threshold: float = 4.0,
    ):
        self.ext2idx = dict(
            zip(user_map["user_id"].tolist(), user_map["user_idx"].tolist(), strict=True)
        )
        self.seen_csr, self.N, self.cutoff, self.thr = (
            seen_csr,
            history_len,
            cutoff_ts,
            pos_threshold,
        )
        self._lock = threading.Lock()
        self.states: dict[int, UserState] = {}
        for r in states.itertuples(index=False):
            h = [int(i) for i in r.hist_items if i != 0]
            self.states[int(r.user_idx)] = UserState(
                int(r.user_idx), h, int(r.n_pos), int(r.n_all),
                float(r.avg_rating) if r.avg_rating == r.avg_rating else float("nan"), cutoff_ts - float(r.days_since_last) * SECONDS_PER_DAY,
            )  # fmt: skip
        self.new_users: dict[int, UserState] = {}

    def get(self, ext_user_id: int, onboarding_genres: list[str] | None = None) -> UserState:
        idx = self.ext2idx.get(ext_user_id)
        if idx is None:  # never-seen user -> cold
            st = self.new_users.setdefault(ext_user_id, UserState(0, is_new=True))
            if onboarding_genres:
                st.onboarding_genres = onboarding_genres
            return st
        return self.states.get(idx) or self.states.setdefault(idx, UserState(idx))

    def seen(self, st: UserState) -> np.ndarray:
        base = (
            self.seen_csr.indices[
                self.seen_csr.indptr[st.user_idx] : self.seen_csr.indptr[st.user_idx + 1]
            ]
            if 0 < st.user_idx < self.seen_csr.shape[0]
            else np.empty(0, np.int32)
        )
        return (
            np.union1d(base, np.fromiter(st.extra_seen, dtype=np.int64)) if st.extra_seen else base
        )

    def arrays(self, st: UserState, now_ts: float | None = None):
        from ..features.state import StateArrays

        days = (now_ts - st.last_ts) / SECONDS_PER_DAY if now_ts and st.last_ts else 0.0
        return StateArrays(
            np.array([st.user_idx], np.int64), pad_history(st.hist, self.N)[None, :],
            user_numeric([st.n_all], [st.avg_rating], [days]), np.array([st.n_pos], np.int32), [],
        )  # fmt: skip

    def add_interaction(
        self, st: UserState, item_idx: int, rating: float | None, ts: float
    ) -> None:
        with self._lock:
            st.extra_seen.add(item_idx)
            n = st.n_all
            r = rating if rating is not None else self.thr
            st.avg_rating = (
                r if n == 0 or st.avg_rating != st.avg_rating else (st.avg_rating * n + r) / (n + 1)
            )
            st.n_all += 1
            st.last_ts = ts
            if r >= self.thr:
                st.hist = (st.hist + [item_idx])[-self.N :]
                st.n_pos += 1
