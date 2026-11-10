# 10-minute presentation outline

| min | slide | content |
|---|---|---|
| 0:00 | 1. Problem | Netflix/Amazon-style recommendations: 60k items, 160k users; why two stages (latency vs precision). Goal + success metrics (recall@K, NDCG@10, CTR). |
| 0:45 | 2. Data and split | MovieLens 25M; **time-based** 70/10/10/10 split; show the timeline figure; leakage tests (counterfactual: scramble the future, features unchanged). |
| 1:45 | 3. Architecture | The Mermaid diagram: offline pipeline, online path, where Redis / FAISS / Triton / Airflow sit. |
| 3:00 | 4. Retrieval | Two-tower, logQ correction, ID dropout. Table: Popularity vs SVD-MF vs Two-tower (recall@10/50/100, NDCG@10). |
| 4:00 | 5. ANN index | Flat vs HNSW vs IVF-PQ plot (recall vs latency vs memory); chosen index and why. |
| 4:45 | 6. Ranking | DCN-V2 vs Wide&Deep vs retrieval-only; ablation table; calibration. Hard negatives from retrieval. |
| 5:45 | 7. Cold start + cache | Router reason codes, onboarding genres, content blend; cold-start curve by #interactions. Redis hit rate and latency with/without cache. |
| 6:45 | 8. Serving + MLOps | Triton (ONNX, dynamic batching), FastAPI per-stage latency, Airflow DAG with evaluation gate, MLflow. **Live demo** (Streamlit side-by-side, new user). |
| 8:15 | 9. A/B test | Hash bucketing, click model (state it is a **simulation**), CTR with 95% CI, z-test + bootstrap, A/A check, decision. |
| 9:15 | 10. Challenges, results, next steps | 3 hardest problems (leakage-safe features, train/serve skew, Triton/ONNX shapes), headline numbers, next steps. |
| 10:00 | Q&A | Prepared answers: why logQ, why time split, HNSW vs IVF-PQ, why ONNX, what would change at 100× scale. |

**Demo script (90 s):** open Streamlit → existing user → both models side by side + latency bars → repeat to show cache hit →
new user with "Sci-Fi, Thriller" → show reason code `NEW_USER_ONBOARDING_GENRES` → A/B tab.
