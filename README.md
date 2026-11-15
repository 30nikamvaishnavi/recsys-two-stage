# Large-Scale Personalized Recommendation Engine (two-stage: retrieval → ranking)

An end-to-end, production-style recommender in the architecture used by Netflix/Amazon:
**stage 1** quickly retrieves a few hundred candidates with a two-tower model + FAISS, **stage 2** re-scores them with a deep
ranking model; cold-start fallbacks, Redis caching, Triton serving, Airflow retraining with an evaluation gate and a (simulated)
online A/B test between two rankers.

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
make abtest report         # simulated A/B, writes docs/results_sample.md
docker compose up --build  # Redis + Triton + API + MLflow + Streamlit + Airflow
```
Then: API docs http://localhost:8080/docs · demo http://localhost:8501 · MLflow :5000 · Airflow :8081.
`MODE=full` uses all 25M ratings; `ENGINE=pandas` swaps Spark for the pandas engine if Java is unavailable.
Load test: `make load-test`; cache benchmark: `make cache-bench`.

## Repository layout
```
src/recsys/{data,features,retrieval,ranking,index,coldstart,serving,cache,abtest,eval}
dags/recsys_weekly.py   configs/default.yaml   docker/ docker-compose.yml   app/streamlit_app.py
tests/ (pytest)         scripts/ (load test, cache bench, report, cache warming)   docs/   notebooks/
```

## Results
Run `make report`; it generates `docs/results_<mode>.md` from `artifacts/reports/` (retrieval vs baselines, FAISS trade-off,
ranker vs retrieval-only + ablation + calibration, cold-start curve, A/B with 95% CIs, load/cache numbers).
**Paste your real numbers here after running** - this README intentionally contains none that were not measured on your machine.

## Correctness guarantees (tested on synthetic data; items marked * need TF/Spark and are verified by `make smoke`)
- Time-based split; **counterfactual leakage tests**: rewriting the future leaves every feature unchanged.
- Metric functions checked against hand-computed values; FAISS/NumPy index round-trip; cache TTL/invalidation; router reason codes.
- End-to-end online pipeline with per-stage latency, seen-item filtering, graceful degradation when the model server is down.
- A/B statistics (z-test, bootstrap CI, A/A false-positive rate, MDE) and deterministic bucketing.
- *Spark output parity vs the pandas reference engine (`tests/test_spark_parity.py`), *ONNX-vs-TF numerical check at export.

## Honest scope notes
The A/B test is a **simulation** (no real users) - see `docs/limitations_and_next_steps.md`. Cold start for *items* happens at
index-build time. Docs: `architecture.md`, `api.md`, `data_dictionary.md`, `presentation_outline.md`, `reflection_template.md`.
