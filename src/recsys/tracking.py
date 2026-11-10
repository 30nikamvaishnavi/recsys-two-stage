"""MLflow wrapper that degrades to a no-op when MLflow is unavailable."""
from __future__ import annotations

import os
from contextlib import contextmanager

from .utils import get_logger

log = get_logger("tracking")


@contextmanager
def run(cfg, name: str, tags: dict | None = None):
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        log.warning("mlflow not installed; tracking disabled")
        yield _Noop()
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    try:
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        ctx = mlflow.start_run(run_name=f"{name}-{cfg['mode']}", tags=tags or {})
    except Exception as e:  # tracking server down must never break training
        log.warning("mlflow unavailable (%s); tracking disabled", e)
        yield _Noop()
        return
    with ctx:
        mlflow.log_params({"mode": cfg["mode"]})
        yield _Live(mlflow)


class _Noop:
    def params(self, d): ...
    def metrics(self, d, step=None): ...
    def artifact(self, p): ...


class _Live:
    def __init__(self, mlflow):
        self.m = mlflow

    def params(self, d: dict):
        flat = {}
        for k, v in d.items():
            flat[str(k)] = str(v)[:250]
        self.m.log_params(flat)

    def metrics(self, d: dict, step: int | None = None):
        clean = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and v == v}
        self.m.log_metrics(clean, step=step)

    def artifact(self, p):
        try:
            self.m.log_artifact(str(p))
        except Exception as e:  # pragma: no cover
            log.warning("artifact upload failed: %s", e)
