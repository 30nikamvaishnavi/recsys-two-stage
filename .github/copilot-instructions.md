# Instructions for GitHub Copilot (Chat / agent) in this repo

Project: two-stage recommender (two-tower retrieval + FAISS -> DCN-V2 / Wide&Deep ranking), MovieLens 25M, CPU-only,
Python 3.10, TensorFlow 2.15 + Keras 2 + tensorflow-recommenders 0.7.3, PySpark 3.5, Triton (ONNX), FastAPI, Redis.

Hard rules
- NEVER invent, estimate or hard-code metric values (NDCG, recall, latency, CTR). Numbers in docs/README come only from
  files in `artifacts/reports/` produced by real runs (`make report`).
- The A/B test is a SIMULATION. Always say so; never call it a live or real-user experiment.
- Splits are by TIME. Never introduce random train/test splits or features computed with data after the cutoff.
- Ranker features are built ONLY by `recsys.ranking.data.candidate_dense` (shared by training and serving: no skew).
- Keep TF 2.15 / Keras 2 APIs (`tf.keras`); do not migrate to Keras 3. Do not add dependencies without asking.
- The target machine has 16 GB RAM and no GPU: prefer config knobs (`configs/default.yaml`, modes smoke|sample|full)
  over bigger models; never add GPU-only code.

Style
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`). Keep tests green: `make test`; lint: `make lint`.
- Small, focused changes. Explain the *why* of any non-obvious design choice in a short comment or docs/architecture.md.
- When fixing an error, state the root cause first, then the minimal change.
