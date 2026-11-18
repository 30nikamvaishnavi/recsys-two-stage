import pytest

from scripts.aggregate_retrieval import aggregate_reports, render_markdown


def _write_run(report_dir, tag, seed, temperature, recall):
    from recsys.utils import save_json

    save_json(
        {
            "experiment": {
                "tag": tag,
                "seed": seed,
                "settings": {"temperature": temperature},
            },
            "val": {"two_tower": {"recall@100": recall, "ndcg@10": recall / 2}},
            "test": {"two_tower": {"recall@100": recall, "ndcg@10": recall / 2}},
        },
        report_dir / f"retrieval_metrics_{tag}.json",
    )


def test_aggregate_keeps_sweep_settings_and_summarizes_seed_runs(tmp_path):
    _write_run(tmp_path, "same-a", 0, 0.05, 0.2)
    _write_run(tmp_path, "same-b", 1, 0.05, 0.4)
    _write_run(tmp_path, "sweep-worse", 0, 0.1, 0.1)

    result = aggregate_reports(tmp_path)
    by_temperature = {item["settings"]["temperature"]: item for item in result["experiments"]}

    seeded = by_temperature[0.05]["metrics"]["val"]["two_tower"]["recall@100"]
    assert seeded["mean"] == pytest.approx(0.3)
    assert seeded["std"] == pytest.approx(0.1414213562373095)
    assert seeded["n"] == 2
    assert by_temperature[0.1]["metrics"]["val"]["two_tower"]["recall@100"]["mean"] == 0.1
    assert "temperature" in render_markdown(result)
    assert "0.1" in render_markdown(result)
