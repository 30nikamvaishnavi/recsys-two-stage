"""Weekly retraining DAG.

download/validate -> Spark features -> train retrieval -> evaluate -> build FAISS index ->
train rankers -> EVALUATION GATE -> (export ONNX + Triton repo, promote, reload) -> warm Redis

The gate short-circuits the DAG: a candidate that is worse than production is never promoted.
Failures are logged via on_failure_callback (swap the body for Slack/email in production).
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

PY = os.environ.get("RECSYS_PY", "/opt/venv/bin/python")
MODE = os.environ.get("RECSYS_MODE", "sample")
HOME = os.environ.get("RECSYS_HOME", "/opt/recsys")
ENV = {"RECSYS_MODE": MODE, "RECSYS_HOME": HOME, "PYTHONPATH": f"{HOME}/src", "TF_USE_LEGACY_KERAS": "1", "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
       "REDIS_URL": os.environ.get("REDIS_URL", "redis://redis:6379/0"), "TRITON_URL": os.environ.get("TRITON_URL", "triton:8000")}  # fmt: skip
log = logging.getLogger("airflow.task")


def gate_passes() -> bool:
    """True if `recsys.promote gate` exits 0. Runs with the SAME env as the BashOperator tasks
    (PYTHONPATH, RECSYS_MODE...), otherwise `recsys` is not importable and the gate would silently
    block every promotion."""
    r = subprocess.run(
        [PY, "-m", "recsys.promote", "gate"],
        cwd=HOME,
        env={**os.environ, **ENV},
        capture_output=True,
        text=True,
    )
    log.info("gate output:\n%s%s", r.stdout, r.stderr)
    return r.returncode == 0


def alert(context):
    ti = context["task_instance"]
    log.error(
        "ALERT: task %s in DAG %s failed (try %s): %s",
        ti.task_id,
        ti.dag_id,
        ti.try_number,
        context.get("exception"),
    )


def run(module: str, extra: str = "") -> str:
    return f"cd {HOME} && {PY} -m {module} {extra}".strip()


default_args = {
    "owner": "recsys",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": alert,
}

with DAG(
    "recsys_weekly_retrain", default_args=default_args, schedule="@weekly", start_date=datetime(2026, 10, 1), catchup=False,
    max_active_runs=1, tags=["recsys"], doc_md=__doc__,
) as dag:  # fmt: skip
    download = BashOperator(
        task_id="download_data", bash_command=run("recsys.data.download"), env=ENV, append_env=True
    )
    prepare = BashOperator(
        task_id="prepare_split", bash_command=run("recsys.data.prepare"), env=ENV, append_env=True
    )
    validate = BashOperator(
        task_id="validate_data", bash_command=run("recsys.data.validate"), env=ENV, append_env=True
    )
    feats = BashOperator(
        task_id="spark_features",
        bash_command=run("recsys.features.build", "--engine spark"),
        env=ENV,
        append_env=True,
    )
    retr = BashOperator(
        task_id="train_retrieval",
        bash_command=run("recsys.retrieval.train"),
        env=ENV,
        append_env=True,
    )
    index = BashOperator(
        task_id="build_faiss_index",
        bash_command=run("recsys.index.build"),
        env=ENV,
        append_env=True,
    )
    rank = BashOperator(
        task_id="train_rankers", bash_command=run("recsys.ranking.train"), env=ENV, append_env=True
    )
    gate = ShortCircuitOperator(task_id="evaluation_gate", python_callable=gate_passes)
    export = BashOperator(
        task_id="export_to_triton_repo",
        bash_command=run("recsys.serving.export"),
        env=ENV,
        append_env=True,
    )
    promote = BashOperator(
        task_id="promote_and_reload_triton",
        bash_command=run("recsys.promote", "promote"),
        env=ENV,
        append_env=True,
    )
    warm = BashOperator(
        task_id="warm_redis",
        bash_command=f"cd {HOME} && {PY} scripts/warm_cache.py",
        env=ENV,
        append_env=True,
    )
    (
        download
        >> prepare
        >> validate
        >> feats
        >> retr
        >> index
        >> rank
        >> gate
        >> export
        >> promote
        >> warm
    )
