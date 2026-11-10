"""Build a small, fully in-memory 'world' (synthetic MovieLens-format data -> split -> features)."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from recsys.config import load_config
from recsys.data.prepare import clean_and_split, load_raw
from recsys.data.synthetic import generate
from recsys.features import pandas_features as pf
from recsys.features.item_table import build_item_table
from recsys.features.state import SeenIndex, states_from_frame


@dataclass
class World:
    cfg: dict
    ds: object
    tables: dict
    states: dict
    seen: dict
    train_ex: object
    n_users: int
    n_items: int


@lru_cache(maxsize=2)
def build_world(n_users: int = 700, n_items: int = 450, seed: int = 0) -> World:
    cfg = load_config(mode="smoke")
    d = Path(tempfile.mkdtemp())
    generate(d, n_users, n_items, 40, 55, seed=seed)
    ds = clean_and_split(*load_raw(d), cfg)
    N, inter = cfg["features"]["history_len"], ds.interactions
    nu, ni = len(ds.user_map), len(ds.items)
    genome = pf.genome_vectors(ds.genome, cfg["features"]["genome_dim"])
    tables, states, seen = {}, {}, {}
    for stage, (cut, end) in pf.stage_windows(ds.boundaries).items():
        pop = pf.item_pop(inter, cut, cfg["features"]["pop_windows_days"], ni)
        tables[stage] = build_item_table(ds.items, pop, genome, cfg)
        states[stage] = states_from_frame(pf.user_state(inter, cut, end, N), N)
        seen[stage] = SeenIndex(inter, cut, nu, ni)
    return World(cfg, ds, tables, states, seen, pf.train_examples(inter, N), nu, ni)


class FakeBackend:
    """Deterministic numpy stand-in for Triton: user tower = normalised mean of item vectors."""

    name = "fake"

    def __init__(self, item_vecs: np.ndarray, fail: bool = False):
        self.v = item_vecs / np.maximum(np.linalg.norm(item_vecs, axis=1, keepdims=True), 1e-9)
        self.fail, self.calls = fail, 0

    def user_embed(self, user_idx, hist, numeric):
        if self.fail:
            raise ConnectionError("triton down")
        self.calls += 1
        m = (hist != 0)[..., None]
        e = (self.v[hist] * m).sum(1) / np.maximum(m.sum(1), 1)
        return (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)).astype(np.float32)

    def rank(self, variant, item_idx, dense):
        if self.fail:
            raise ConnectionError("triton down")
        bias = 0.02 if variant == "dcn_v2" else 0.0
        s = dense[:, -2] + 0.5 * dense[:, -N_CROSS_OVERLAP] + bias * dense[:, -1]
        return 1 / (1 + np.exp(-s))

    def ready(self):
        return not self.fail


N_CROSS_OVERLAP = 5  # position of `overlap` counted from the end: [overlap, cos, gap, score, rank]
