"""Model backends with one interface. TritonBackend = production; OnnxBackend = same ONNX files run
in-process with onnxruntime (dev / fallback / tests)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class ModelBackend(Protocol):
    name: str

    def user_embed(self, user_idx: np.ndarray, hist: np.ndarray, numeric: np.ndarray) -> np.ndarray: ...
    def rank(self, variant: str, item_idx: np.ndarray, dense: np.ndarray) -> np.ndarray: ...
    def ready(self) -> bool: ...


def _tower_feed(meta: dict, user_idx, hist, numeric):
    return {"user_idx": user_idx.astype(np.int64).reshape(-1, 1), "hist_items": hist.astype(np.int64), "user_numeric": numeric.astype(np.float32)}


class OnnxBackend:
    name = "onnxruntime"

    def __init__(self, repo: Path, meta: dict):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.meta = meta
        self.s = {m: ort.InferenceSession(str(repo / m / "1" / "model.onnx"), so, providers=["CPUExecutionProvider"]) for m in meta["models"]}

    def user_embed(self, user_idx, hist, numeric):
        m = self.meta["models"][self.meta["roles"]["user_tower"]]
        feed = _tower_feed(m, user_idx, hist, numeric)
        return self.s[self.meta["roles"]["user_tower"]].run(m["outputs"][:1], {k: feed[k] for k in m["inputs"]})[0]

    def rank(self, variant, item_idx, dense):
        name = self.meta["roles"][variant]
        m = self.meta["models"][name]
        return self.s[name].run(m["outputs"][:1], {"item_idx": item_idx.astype(np.int64).reshape(-1, 1), "dense": dense.astype(np.float32)})[0].reshape(-1)

    def ready(self):
        return True


class TritonBackend:
    name = "triton"

    def __init__(self, url: str, meta: dict, timeout: float = 5.0):
        import tritonclient.http as http

        self.http, self.meta = http, meta
        self.c = http.InferenceServerClient(url, connection_timeout=timeout, network_timeout=timeout)

    def _infer(self, model: str, feed: dict[str, np.ndarray]) -> np.ndarray:
        m = self.meta["models"][model]
        ins = []
        for n in m["inputs"]:
            a = feed[n]
            t = self.http.InferInput(n, list(a.shape), "INT64" if a.dtype == np.int64 else "FP32")
            t.set_data_from_numpy(a, binary_data=True)
            ins.append(t)
        res = self.c.infer(model, ins, outputs=[self.http.InferRequestedOutput(m["outputs"][0], binary_data=True)])
        return res.as_numpy(m["outputs"][0])

    def user_embed(self, user_idx, hist, numeric):
        return self._infer(self.meta["roles"]["user_tower"], _tower_feed(self.meta, user_idx, hist, numeric))

    def rank(self, variant, item_idx, dense):
        out = self._infer(self.meta["roles"][variant], {"item_idx": item_idx.astype(np.int64).reshape(-1, 1), "dense": dense.astype(np.float32)})
        return out.reshape(-1)

    def ready(self):
        try:
            return bool(self.c.is_server_ready()) and all(self.c.is_model_ready(m) for m in self.meta["models"])
        except Exception:
            return False

    def reload(self) -> None:
        for m in self.meta["models"]:
            self.c.load_model(m)


def make_backend(cfg, repo: Path, meta_path: Path, kind: str | None = None):
    import os

    meta = json.loads(Path(meta_path).read_text())
    kind = kind or os.environ.get("RECSYS_BACKEND") or cfg["serving"]["backend"]
    if kind == "triton":
        return TritonBackend(os.environ.get("TRITON_URL", cfg["serving"]["triton_url"]), meta)
    return OnnxBackend(repo, meta)
