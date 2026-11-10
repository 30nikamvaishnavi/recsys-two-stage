# Results

Results are **generated, never typed by hand**. After running the pipeline:

```bash
make report          # writes docs/results_<mode>.md from artifacts/reports/<mode>/*.json
```

Metric definitions: recall@K = hits in top-K / #relevant (macro over users); NDCG@K with binary relevance and
IDCG over min(#relevant, K); MAP@10; AUC/logloss on all candidate rows. Evaluation uses the **time-based validation
window** (retrieval, ranking, cold start) and the **test window** (A/B replay). Warm users = at least 5 prior positives.

Paste the generated tables into the README "Results" section once you have run `MODE=sample` (and optionally `MODE=full`).
