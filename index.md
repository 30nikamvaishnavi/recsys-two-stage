---
title: Two-Stage Recommendation Engine
---

# Large-Scale Personalized Recommendation Engine

A CPU-oriented, two-stage MovieLens recommender: two-tower retrieval with FAISS followed by DCN-V2 or Wide & Deep
ranking. The project includes time-based feature generation, cold-start handling, Redis caching, Triton ONNX serving,
a FastAPI gateway, and Airflow retraining workflows.

This is the project's **static overview page**. The interactive Streamlit/API stack is not hosted here; run it locally
with Docker Compose using the [setup instructions](https://github.com/30nikamvaishnavi/recsys-two-stage/blob/main/docs/setup_wsl2.md).

## Project links

- [Source repository](https://github.com/30nikamvaishnavi/recsys-two-stage)
- [Architecture](https://github.com/30nikamvaishnavi/recsys-two-stage/blob/main/docs/architecture.md)
- [API documentation](https://github.com/30nikamvaishnavi/recsys-two-stage/blob/main/docs/api.md)
- [Phase progress and verification status](https://github.com/30nikamvaishnavi/recsys-two-stage/blob/main/docs/progress/phase-i3-to-l.md)
- [Sample results](https://github.com/30nikamvaishnavi/recsys-two-stage/blob/main/docs/results_sample.md)

## Current verification status

The previous sample A/B simulation and load-test results are **INVALID** because the shared model repository held
SMOKE-mode ONNX models. Both A/B arms fell back to popularity. The sample models are being re-exported before the
in-process A/B is rerun. These A/B and load-test results must not be cited; see the
[phase (k) status](https://github.com/30nikamvaishnavi/recsys-two-stage/blob/main/docs/progress/phase-i3-to-l.md).

Any A/B test in this project is a **simulation with no real users**.
