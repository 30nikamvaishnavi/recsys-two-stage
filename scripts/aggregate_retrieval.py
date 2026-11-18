"""Aggregate tagged retrieval runs by configuration and render a Markdown results table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from recsys.config import parse_cfg, path
from recsys.utils import load_json, save_json

MODELS = ("popularity", "svd_mf", "two_tower")
SPLITS = ("val", "test")
TABLE_METRICS = ("recall@100", "ndcg@10")


def _is_metric(name: str) -> bool:
    return name.startswith(("recall@", "ndcg@", "hit@", "map@"))


def aggregate_reports(report_dir: str | Path) -> dict:
    report_dir = Path(report_dir)
    files = sorted(report_dir.glob("retrieval_metrics_*.json"))
    if not files:
        raise FileNotFoundError(f"no tagged retrieval reports found in {report_dir}")

    groups = {}
    for report_path in files:
        report = load_json(report_path)
        experiment = report.get("experiment")
        if not experiment or "settings" not in experiment:
            raise ValueError(f"{report_path} is missing tagged experiment settings")
        settings = experiment["settings"]
        key = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        group = groups.setdefault(key, {"settings": settings, "runs": []})
        group["runs"].append(
            {
                "tag": experiment.get("tag"),
                "seed": experiment.get("seed"),
                "report": report,
            }
        )

    experiments = []
    for key in sorted(groups):
        group = groups[key]
        runs = sorted(group["runs"], key=lambda run: (run["seed"], run["tag"] or ""))
        metrics = {}
        for split in SPLITS:
            metrics[split] = {}
            for model in MODELS:
                available = sorted(
                    {
                        metric
                        for run in runs
                        for metric, value in run["report"].get(split, {}).get(model, {}).items()
                        if _is_metric(metric) and isinstance(value, (int, float))
                    }
                )
                metrics[split][model] = {}
                for metric in available:
                    values = [
                        float(run["report"][split][model][metric])
                        for run in runs
                        if metric in run["report"].get(split, {}).get(model, {})
                    ]
                    metrics[split][model][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "n": len(values),
                    }
        experiments.append(
            {
                "settings": group["settings"],
                "seeds": [run["seed"] for run in runs],
                "tags": [run["tag"] for run in runs],
                "metrics": metrics,
            }
        )
    return {"experiments": experiments}


def _summary_cell(summary: dict | None) -> str:
    if not summary:
        return "n/a"
    std = "n/a" if summary["std"] is None else f"{summary['std']:.4f}"
    return f"{summary['mean']:.4f} +/- {std}"


def render_markdown(aggregate: dict) -> str:
    headers = ["settings", "seeds"] + [
        f"{split} {model} {metric}"
        for split in SPLITS
        for model in MODELS
        for metric in TABLE_METRICS
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for experiment in aggregate["experiments"]:
        settings = json.dumps(experiment["settings"], sort_keys=True, separators=(",", ":"))
        seeds = ", ".join(str(seed) for seed in experiment["seeds"])
        cells = [f"`{settings}`", seeds]
        for split in SPLITS:
            for model in MODELS:
                for metric in TABLE_METRICS:
                    cells.append(
                        _summary_cell(
                            experiment["metrics"].get(split, {}).get(model, {}).get(metric)
                        )
                    )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main():
    cfg, _ = parse_cfg("Aggregate tagged retrieval experiments")
    report_dir = Path(path(cfg, "reports"))
    aggregate = aggregate_reports(report_dir)
    save_json(aggregate, report_dir / "retrieval_experiments.json")
    (report_dir / "retrieval_experiments.md").write_text(render_markdown(aggregate))
    print(f"wrote retrieval experiment aggregates to {report_dir}")


if __name__ == "__main__":
    main()
