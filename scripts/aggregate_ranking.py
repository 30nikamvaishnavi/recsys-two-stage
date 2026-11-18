"""Aggregate tagged ranker runs (ranking_metrics_<tag>.json) into mean +/- std over seeds.

    python -m scripts.aggregate_ranking --mode sample
Writes artifacts/reports/<mode>/ranking_experiments.json and prints a markdown table.
Only reads reports produced by real runs; it never invents or fills in numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("ndcg@10", "recall@10", "hit@10", "recall@50")


def _stats(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    return {"mean": float(arr.mean()), "std": std, "n_runs": int(len(arr))}


def aggregate(reports: list[dict]) -> dict:
    """Mean/std across runs for ranker vs retrieval-only, the paired difference and ablations."""
    out = {"n_runs": len(reports), "seeds": [r.get("seed") for r in reports], "detailed": {}}
    for split in ("val", "test"):
        out["detailed"][split] = {}
        for kind in reports[0]["detailed"][split]:
            runs = [r["detailed"][split][kind] for r in reports]
            row = {"ranker": {}, "retrieval_only": {}, "paired_diff": {}}
            for metric in METRICS:
                row["ranker"][metric] = _stats([x["ranker"][metric] for x in runs])
                row["retrieval_only"][metric] = _stats([x["retrieval_only"][metric] for x in runs])
            for metric in ("ndcg@10", "recall@10"):
                row["paired_diff"][metric] = _stats(
                    [x["paired_diff_vs_retrieval"][metric]["mean_diff"] for x in runs]
                )
            row["ranker"]["auc"] = _stats([x["ranker"]["auc"] for x in runs])
            out["detailed"][split][kind] = row
    out["ablations"] = {}
    for split, key in (("val", "ablations"), ("test", "ablations_test")):
        names = reports[0].get(key, {})
        out["ablations"][split] = {
            name: {m: _stats([r[key][name][m] for r in reports]) for m in ("ndcg@10", "recall@10")}
            for name in names
        }
    return out


def _fmt(s: dict) -> str:
    return f"{s['mean']:.4f} ± {s['std']:.4f}" if s["n_runs"] > 1 else f"{s['mean']:.4f}"


def render_markdown(summary: dict, split: str = "test") -> str:
    n = summary["n_runs"]
    how = f"mean ± std over {n} seeds" if n > 1 else "a single run (no std)"
    header = "| Model | Retrieval-only NDCG@10 | Ranker NDCG@10 | Paired diff NDCG@10 | Recall@10 |"
    lines = [
        f"Ranker results on the {split} window, {how} (seeds {summary['seeds']}).",
        "",
        header,
        "| --- | --- | --- | --- | --- |",
    ]
    for kind, row in summary["detailed"][split].items():
        cells = [
            _fmt(row["retrieval_only"]["ndcg@10"]),
            _fmt(row["ranker"]["ndcg@10"]),
            _fmt(row["paired_diff"]["ndcg@10"]),
            _fmt(row["ranker"]["recall@10"]),
        ]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    lines += ["", f"DCN-V2 ablations on the {split} window:", ""]
    lines += ["| Variant | NDCG@10 | Recall@10 |", "| --- | --- | --- |"]
    for name, row in summary["ablations"][split].items():
        lines.append(f"| {name} | {_fmt(row['ndcg@10'])} | {_fmt(row['recall@10'])} |")
    return "\n".join(lines)


def main() -> None:
    from recsys.config import load_config, path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=None, choices=["smoke", "sample", "full"])
    parser.add_argument("--pattern", default="ranking_metrics_s*.json")
    args = parser.parse_args()
    cfg = load_config(None, args.mode)
    folder = Path(path(cfg, "reports"))
    files = sorted(folder.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {folder}; run the seeds first")
    reports = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    summary = aggregate(reports)
    summary["files"] = [f.name for f in files]
    out_file = folder / "ranking_experiments.json"
    out_file.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(render_markdown(summary, "test"))
    print()
    print(render_markdown(summary, "val"))


if __name__ == "__main__":
    main()
