# Project Summary

## Overview

This project implements a two-stage personalized movie recommender using MovieLens 25M. A retrieval stage finds
candidate movies, then a ranking stage orders those candidates for each user. The project is designed to run on
CPU and includes data processing, model training, evaluation, and a local serving stack.

## System stages

1. **Time-aware data and features:** split interactions by time and build features from information available at
   each cutoff.
2. **Candidate retrieval:** train a TensorFlow Recommenders two-tower model and retrieve candidates with FAISS.
3. **Ranking:** score candidates with DCN-V2 or Wide & Deep using shared candidate features.
4. **Fallbacks and serving:** handle sparse or unknown users with cold-start strategies; serve ONNX models through
   Triton behind a FastAPI gateway, with Redis caching.
5. **Operations and demo:** use an Airflow workflow with an evaluation gate, MLflow tracking, Docker Compose, and a
   Streamlit interface.

## Evaluation and limitations

Evaluation uses time-based splits. The A/B test is a **simulation with no real users** and does not establish
real-world click-through performance.

The previous sample-mode A/B and load-test results are **INVALID**. The shared `model_repository/` contained
SMOKE-mode ONNX models instead of sample models, so both A/B arms fell back to popularity. The invalid metrics are
intentionally excluded. See the [phase progress log](progress/phase-i3-to-l.md); do not cite or regenerate those
results until the sample models have been re-exported and the in-process A/B has been rerun.

## Run locally

Follow the [WSL2 setup guide](setup_wsl2.md), then start with:

```bash
make setup
source .venv/bin/activate
make smoke
```

The GitHub Pages site is a static overview and does not host the Python application. Use Docker Compose locally for
the Streamlit, API, Triton, and Redis services; see the
[README](https://github.com/30nikamvaishnavi/recsys-two-stage/blob/main/README.md) for the full quickstart.
