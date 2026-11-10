"""ANN index over item embeddings (inner product). FAISS when available, exact NumPy otherwise.

Row r of the index <-> item_idx = ids[r] (ids excludes the padding row 0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss

    HAVE_FAISS = True
except ImportError:  # pragma: no cover - exercised in the light CI image
    faiss = None
    HAVE_FAISS = False


class NumpyFlatIndex:
    """Exact search with the faiss-like API; used when faiss is not installed and in tests."""

    def __init__(self, d: int):
        self.d, self.x = d, np.zeros((0, d), np.float32)

    def add(self, x):
        self.x = np.vstack([self.x, np.asarray(x, np.float32)])

    def search(self, q, k):
        s = np.asarray(q, np.float32) @ self.x.T
        k = min(k, s.shape[1])
        i = np.argpartition(-s, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(s, i, 1)
        o = np.argsort(-v, axis=1)
        return np.take_along_axis(v, o, 1), np.take_along_axis(i, o, 1)

    @property
    def ntotal(self):
        return len(self.x)


class AnnIndex:
    """Wrapper holding the FAISS (or numpy) index + item-id mapping + save/load."""

    def __init__(self, index, ids: np.ndarray, kind: str, params: dict | None = None):
        self.index, self.ids, self.kind, self.params = index, ids.astype(np.int32), kind, params or {}

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """-> (item_idx (B,k), scores (B,k)); -1 rows from FAISS are mapped to item 0."""
        k = min(k, len(self.ids))
        s, r = self.index.search(np.ascontiguousarray(q, dtype=np.float32), k)
        items = np.where(r >= 0, self.ids[np.clip(r, 0, None)], 0)
        return items, s

    def save(self, d: Path, name: str = "ann") -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{name}_ids.npy", self.ids)
        meta = {"kind": self.kind, "params": self.params}
        if HAVE_FAISS and not isinstance(self.index, NumpyFlatIndex):
            faiss.write_index(self.index, str(d / f"{name}.faiss"))
        else:
            np.save(d / f"{name}_vectors.npy", self.index.x)
        import json

        (d / f"{name}.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, d: Path, name: str = "ann") -> AnnIndex:
        import json

        meta = json.loads((d / f"{name}.json").read_text())
        ids = np.load(d / f"{name}_ids.npy")
        if (d / f"{name}.faiss").exists() and HAVE_FAISS:
            idx = faiss.read_index(str(d / f"{name}.faiss"))
            _apply_params(idx, meta["kind"], meta["params"])
        else:
            idx = NumpyFlatIndex(np.load(d / f"{name}_vectors.npy").shape[1])
            idx.add(np.load(d / f"{name}_vectors.npy"))
        return cls(idx, ids, meta["kind"], meta["params"])


def _apply_params(idx, kind: str, params: dict) -> None:
    if kind == "hnsw" and "efSearch" in params:
        faiss.ParameterSpace().set_index_parameter(idx, "efSearch", params["efSearch"])
    if kind == "ivfpq" and "nprobe" in params:
        faiss.ParameterSpace().set_index_parameter(idx, "nprobe", params["nprobe"])


def build(kind: str, emb: np.ndarray, icfg, **params) -> AnnIndex:
    """emb: (n, d) float32 for items 1..n (row 0 excluded). kind in flat | hnsw | ivfpq."""
    x = np.ascontiguousarray(emb, dtype=np.float32)
    n, d = x.shape
    ids = np.arange(1, n + 1, dtype=np.int32)
    if not HAVE_FAISS:
        idx = NumpyFlatIndex(d)
        idx.add(x)
        return AnnIndex(idx, ids, "flat")
    ip = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        idx = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        idx = faiss.IndexHNSWFlat(d, icfg["hnsw_m"], ip)
        idx.hnsw.efConstruction = icfg["hnsw_ef_construction"]
        idx.hnsw.efSearch = params.get("efSearch", 64)
    elif kind == "ivfpq":
        nlist = int(min(icfg["ivf_nlist"], max(4, n // 39)))
        m = icfg["pq_m"]
        while d % m:
            m -= 1
        idx = faiss.index_factory(d, f"IVF{nlist},PQ{m}x{icfg['pq_nbits']}", ip)
        idx.train(x)
        idx.nprobe = params.get("nprobe", 16)
        params = {**params, "nlist": nlist, "pq_m": m}
    else:
        raise ValueError(kind)
    idx.add(x)
    return AnnIndex(idx, ids, kind, params)


def index_bytes(a: AnnIndex) -> int:
    if HAVE_FAISS and not isinstance(a.index, NumpyFlatIndex):
        return int(len(faiss.serialize_index(a.index)))
    return int(a.index.x.nbytes)
