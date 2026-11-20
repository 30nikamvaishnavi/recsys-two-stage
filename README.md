# Large-Scale Personalized Recommendation Engine (two-stage: retrieval → ranking)

An end-to-end, production-style recommender in the architecture used by Netflix/Amazon:
**stage 1** quickly retrieves a few hundred candidates with a two-tower model + FAISS, **stage 2** re-scores them with a deep
ranking model; cold-start fallbacks, Redis caching, Triton serving, Airflow retraining with an evaluation gate and a
simulated A/B test between two rankers.

**Project page:** [30nikamvaishnavi.github.io/recsys-two-stage](https://30nikamvaishnavi.github.io/recsys-two-stage/).
GitHub Pages publishes from `docs/`; this is a static project overview, not a hosted interactive recommender. The
application stack currently runs locally with Docker Compose. See the [project summary](docs/project_summary.md) and
[phase progress log](docs/progress/phase-i3-to-l.md) for details and verification status.

| Layer | Tech |
|---|---|
| Data / features | MovieLens 25M, **Apache Spark** (point-in-time features), pandas reference engine |
| Retrieval | **TensorFlow Recommenders** two-tower, in-batch sampled softmax + logQ correction |
| ANN | **FAISS** (Flat / HNSW / IVF-PQ benchmark) |
| Ranking | DCN-V2 and Wide & Deep (Keras) |
| Cold start | popularity-by-genre + MMR, content kNN blend, content-only item embeddings; **Redis** cache |
| Serving | **Triton Inference Server** (ONNX, CPU, dynamic batching) behind a **FastAPI** gateway |
| MLOps | **Airflow** weekly DAG with evaluation gate, MLflow tracking, Docker Compose, CI |
| Demo | Streamlit (side-by-side models, latency breakdown, A/B dashboard) |

## Quickstart (WSL2 Ubuntu 22.04; see `docs/setup_wsl2.md`)

```bash
make setup && source .venv/bin/activate
make smoke                 # whole pipeline on synthetic data in minutes -> proves everything is wired
# real run (20% of MovieLens 25M users by default):
make data features train   # download+split -> Spark features -> retrieval, FAISS, rankers, cold start
make export                # ONNX + Triton repository + serving assets
docker compose up --build  # Redis + Triton + API + MLflow + Streamlit + Airflow
```
Then: API docs http://localhost:8080/docs · demo http://localhost:8501 · MLflow :5000 · Airflow :8081.
`MODE=full` uses all 25M ratings; `ENGINE=pandas` swaps Spark for the pandas engine if Java is unavailable.
The phase (k) A/B simulation and load test are currently **not verified**; their earlier results are invalid and must not
be cited. Do not rerun or publish those results until the sample models have been re-exported and the in-process A/B has
been rerun. See the [phase (k) status](docs/progress/phase-i3-to-l.md).
Cache benchmark: `make cache-bench`.

## Repository layout
```
src/recsys/{data,features,retrieval,ranking,index,coldstart,serving,cache,abtest,eval}
dags/recsys_weekly.py   configs/default.yaml   docker/ docker-compose.yml   app/streamlit_app.py
tests/ (pytest)         scripts/ (load test, cache bench, report, cache warming)   docs/   notebooks/
```

## Results
`make report` generates `docs/results_<mode>.md` from `artifacts/reports/` (retrieval vs baselines, FAISS trade-off,
ranker vs retrieval-only + ablation + calibration, cold-start curve, simulated A/B, and load/cache measurements).
The existing phase (k) A/B and load-test output is invalid because the shared model repository held SMOKE-mode ONNX
models during the sample run; do not cite or reuse those A/B/load-test values. The generated sample report marks these
sections as **NOT VERIFIED**. Do not regenerate or publish the sample report until verified A/B and load-test results
are available, because the current report inputs still contain the invalid run. See the
[phase progress log](docs/progress/phase-i3-to-l.md) for details.

## Correctness guarantees (tested on synthetic data; items marked * need TF/Spark and are verified by `make smoke`)
- Time-based split; **counterfactual leakage tests**: rewriting the future leaves every feature unchanged.
- Metric functions checked against hand-computed values; FAISS/NumPy index round-trip; cache TTL/invalidation; router reason codes.
- End-to-end online pipeline with per-stage latency, seen-item filtering, graceful degradation when the model server is down.
- A/B statistics (z-test, bootstrap CI, A/A false-positive rate, MDE) and deterministic bucketing.
- *Spark output parity vs the pandas reference engine (`tests/test_spark_parity.py`), *ONNX-vs-TF numerical check at export.

## Honest scope notes
The A/B test is a **simulation** (no real users) - see `docs/limitations_and_next_steps.md`. Cold start for *items* happens at
index-build time. Docs: `project_summary.md`, `architecture.md`, `api.md`, `data_dictionary.md`, `presentation_outline.md`, `reflection_template.md`.
