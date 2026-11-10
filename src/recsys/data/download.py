"""Download and verify MovieLens 25M (https://grouplens.org/datasets/movielens/25m/)."""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

from ..config import parse_cfg, path
from ..utils import get_logger
from .synthetic import generate

log = get_logger("download")
REQUIRED = ["ratings.csv", "movies.csv", "genome-scores.csv", "genome-tags.csv"]


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_dir(cfg) -> Path:
    name = "synthetic" if cfg["data"]["dataset"] == "synthetic" else "ml-25m"
    return Path(path(cfg, "data_raw")) / name


def verify(d: Path) -> bool:
    return all((d / f).exists() and (d / f).stat().st_size > 0 for f in REQUIRED)


def download_movielens(cfg) -> Path:
    d = raw_dir(cfg)
    if verify(d):
        log.info("dataset already present at %s", d)
        return d
    base = Path(path(cfg, "data_raw"))
    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / "ml-25m.zip"
    if not zip_path.exists():
        log.info("downloading %s (~250 MB)", cfg["data"]["url"])
        urllib.request.urlretrieve(cfg["data"]["url"], zip_path)
    try:  # GroupLens publishes an md5 next to the archive
        expected = urllib.request.urlopen(cfg["data"]["url"] + ".md5", timeout=30).read().decode().split()[0]
        actual = _md5(zip_path)
        if expected != actual:
            zip_path.unlink()
            raise RuntimeError(f"md5 mismatch: expected {expected}, got {actual}; re-run to re-download")
        log.info("md5 verified: %s", actual)
    except OSError as e:
        log.warning("could not fetch md5 for verification (%s); continuing", e)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(base)
    if not verify(d):
        raise RuntimeError(f"extracted dataset incomplete in {d}")
    log.info("ready: %s", d)
    return d


def ensure_raw(cfg) -> Path:
    if cfg["data"]["dataset"] == "synthetic":
        d = raw_dir(cfg)
        if not verify(d):
            s = cfg["data"]["synthetic"]
            log.info("generating synthetic MovieLens-format data in %s", d)
            generate(d, s["n_users"], s["n_items"], s["n_tags"], s["mean_events"], seed=cfg["data"]["seed"])
        return d
    return download_movielens(cfg)


if __name__ == "__main__":
    cfg, _ = parse_cfg("Download / generate the raw dataset")
    ensure_raw(cfg)
