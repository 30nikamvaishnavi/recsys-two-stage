# Limitations and next steps

## Honest limitations
1. **The A/B test is a simulation.** There are no real users; clicks come from a position-biased model driven by held-out
   interactions. It validates the experimentation machinery (bucketing, logging, CIs, decision rule) and ranks models under a
   stated click model, but its CTR numbers are not predictive of real CTR. Held-out positives also under-count items a user
   would have enjoyed but never rated (exposure bias).
2. **Implicit-feedback proxy.** Positives are ratings ≥ 4. MovieLens ratings are an explicit, self-selected signal, not impressions.
3. **CPU-only training.** Embedding dim 64, 20% user sample by default; `full` mode uses all 25M ratings but takes hours on a laptop.
4. **Ranker base rate.** Trained on retrieved candidates + popularity negatives, so predicted probabilities are not calibrated to
   production CTR (the calibration plot/ECE quantifies this). Platt/isotonic calibration is a next step.
5. **Popularity features inside the training window** are computed as of the window end, not per example (no leakage across the
   split boundaries - proven by tests - but within-window the item tower can see slightly "future" popularity).
6. **Min-activity filter** (≥ 5 interactions) uses whole-history counts before the split.
7. **Single-node infra.** Triton on CPU, Redis single instance, in-process user-state store, Airflow `standalone`.
8. Cold *items* are embedded at index-build time (weekly/nightly), not instantly on creation.

## Next steps
- Sequence model for the user tower (SASRec / GRU4Rec) and multi-interest retrieval.
- Multi-task ranker (click, watch-time, rating) and listwise losses (LambdaRank).
- Counterfactual / off-policy evaluation (IPS, doubly-robust) to de-bias offline metrics; interleaving for faster A/B tests.
- Streaming feature updates (Kafka → Redis feature store) and an incremental index (FAISS `add`) for same-minute cold items.
- Calibration layer, exploration (Thompson sampling / ε-greedy slots) and fairness/coverage constraints in MMR.
- GPU Triton + TensorRT, ONNX int8 quantisation, request-level batching in the gateway.
- Prometheus + Grafana dashboards and drift monitoring (embedding / feature PSI) triggering retrains.
