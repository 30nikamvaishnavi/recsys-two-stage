"""User-state arrays (as-of a cutoff) and the 'seen items' index used to filter recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import SECONDS_PER_DAY, pad_history, user_numeric


@dataclass
class StateArrays:
    user_idx: np.ndarray  # (B,) int64 (0 = unknown user)
    hist: np.ndarray  # (B,N) int32, oldest->newest, zero padded at the end
    numeric: np.ndarray  # (B,3) float32
    n_pos: np.ndarray  # (B,) int32 positives before cutoff
    targets: list = field(default_factory=list)  # list[np.ndarray] future positives (may be empty)

    def __len__(self) -> int:
        return len(self.user_idx)

    def take(self, idx: np.ndarray) -> StateArrays:
        return StateArrays(
            self.user_idx[idx], self.hist[idx], self.numeric[idx], self.n_pos[idx],
            [self.targets[i] for i in idx] if self.targets else [],
        )  # fmt: skip

    def subsample(self, n: int | None, seed: int = 0) -> StateArrays:
        if n is None or n >= len(self):
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self), n, replace=False))
        return self.take(idx)

    def truncate_history(self, k: int) -> StateArrays:
        """Keep only the k most recent history items (cold-start simulation); numeric feats follow."""
        hist = np.zeros_like(self.hist)
        valid = (self.hist != 0).sum(1)
        for i in range(len(self)):
            keep = min(k, valid[i])
            if keep:
                hist[i, :keep] = self.hist[i, valid[i] - keep : valid[i]]
        num = self.numeric.copy()
        num[:, 0] = np.log1p(np.minimum(self.n_pos, k).astype(np.float32)) / 6.0
        return StateArrays(
            self.user_idx, hist, num, np.minimum(self.n_pos, k).astype(np.int32), self.targets
        )


def states_from_frame(df: pd.DataFrame, history_len: int) -> StateArrays:
    """Convert a user_state parquet/DataFrame (Spark or pandas engine output) to arrays."""
    hist = (
        np.stack([pad_history(h if h is not None else [], history_len) for h in df["hist_items"]])
        if len(df)
        else np.zeros((0, history_len), np.int32)
    )
    return StateArrays(
        df["user_idx"].to_numpy(np.int64),
        hist.astype(np.int32),
        user_numeric(df["n_all"].fillna(0), df["avg_rating"], df["days_since_last"]),
        df["n_pos"].fillna(0).to_numpy(np.int32),
        [np.asarray(t, dtype=np.int32) for t in df["targets"]] if "targets" in df else [],
    )


def examples_to_arrays(df: pd.DataFrame, history_len: int):
    hist = np.stack([pad_history(h, history_len) for h in df["hist_items"]]).astype(np.int32)
    num = user_numeric(df["n_prior"], df["avg_prior_rating"], df["days_since_prev"])
    return df["user_idx"].to_numpy(np.int64), df["item_idx"].to_numpy(np.int64), hist, num


class SeenIndex:
    """CSR user x item of everything the user interacted with before a cutoff."""

    def __init__(self, inter: pd.DataFrame, cutoff_ts: int, n_users: int, n_items: int):
        d = inter[inter["ts"] < cutoff_ts]
        self.m = sp.csr_matrix(
            (np.ones(len(d), dtype=np.int8), (d["user_idx"].to_numpy(), d["item_idx"].to_numpy())),
            shape=(n_users + 1, n_items + 1),
        )

    def mask_scores(self, scores: np.ndarray, user_idx: np.ndarray) -> np.ndarray:
        """In-place: set scores of already-seen items to -inf. scores: (B, n_items+1)."""
        for r, u in enumerate(user_idx):
            if 0 < u < self.m.shape[0]:
                scores[r, self.m.indices[self.m.indptr[u] : self.m.indptr[u + 1]]] = -np.inf
        return scores

    def seen(self, user: int) -> np.ndarray:
        if user <= 0 or user >= self.m.shape[0]:
            return np.empty(0, dtype=np.int32)
        return self.m.indices[self.m.indptr[user] : self.m.indptr[user + 1]]


def days_between(a_ts, b_ts):
    return (
        np.asarray(a_ts, dtype=np.float64) - np.asarray(b_ts, dtype=np.float64)
    ) / SECONDS_PER_DAY
