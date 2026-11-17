"""YAML configuration with mode overrides and attribute access."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(
    path: str | Path | None = None, mode: str | None = None, overrides: dict | None = None
) -> Cfg:
    """Load default.yaml, apply `modes[mode]` then explicit overrides."""
    with open(path or DEFAULT_CONFIG) as f:
        raw = yaml.safe_load(f)
    mode = mode or os.environ.get("RECSYS_MODE") or raw.get("mode", "sample")
    if mode not in raw.get("modes", {}):
        raise ValueError(f"unknown mode {mode!r}; choose from {list(raw.get('modes', {}))}")
    modes = raw.pop("modes")
    cfg = _merge(raw, modes[mode])
    cfg["mode"] = mode
    cfg = _merge(cfg, overrides or {})
    return Cfg(cfg)


def parse_cfg(description: str = "", extra_args: list[tuple[str, dict]] | None = None):
    """Standard CLI: --config, --mode. Returns (cfg, args)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None)
    p.add_argument("--mode", default=None, choices=["smoke", "sample", "full"])
    for name, kw in extra_args or []:
        p.add_argument(name, **kw)
    args = p.parse_args()
    return load_config(args.config, args.mode), args


def path(cfg: Cfg, key: str, *parts: str, create: bool = False) -> Path:
    """Resolve cfg.paths[key]/parts relative to the repo root (or env RECSYS_HOME)."""
    base = Path(os.environ.get("RECSYS_HOME", ROOT))
    p = base / cfg["paths"][key]
    if cfg["mode"] != "full" and key in {
        "data_processed",
        "features",
        "models",
        "serving",
        "reports",
    }:
        p = p / cfg["mode"]  # keep smoke/sample/full artifacts separate
    p = p.joinpath(*parts)
    if create:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p
