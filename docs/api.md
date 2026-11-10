# API reference

Base URL `http://localhost:8080`; interactive docs at `/docs` (OpenAPI at `/openapi.json`).

## `POST /recommend`
```json
{"user_id": 123, "k": 10, "model_variant": "dcn_v2", "onboarding_genres": ["Sci-Fi","Thriller"], "use_cache": true}
```
`model_variant`: `wide_deep` (arm A) or `dcn_v2` (arm B). `onboarding_genres` only matters for brand-new users.
`GET /recommend?user_id=123&k=10&model_variant=dcn_v2&genres=Comedy,Romance` is also supported.

Response
```json
{
  "user_id": 123, "k": 10, "variant": "dcn_v2",
  "items": [{"item_id": 296, "title": "Pulp Fiction (1994)", "genres": "Comedy|Crime|Drama|Thriller",
             "score": 0.81, "retrieval_score": 0.62, "ranker_score": 0.81}],
  "meta": {"strategy": "two_stage", "reason_code": "WARM_USER", "cache_hit": false, "backend": "triton",
           "latency_ms": {"state_router": 0.3, "user_tower": 3.1, "ann_search": 0.4, "features": 2.2,
                          "ranker": 6.5, "postprocess": 1.1, "total": 14.0}}
}
```
Strategies / reason codes: `two_stage/WARM_USER`, `content_blend/SPARSE_HISTORY_CONTENT_BLEND`,
`genre_popularity/NEW_USER_ONBOARDING_GENRES`, `global_popularity/NEW_USER_NO_SIGNAL`,
`global_popularity/MODEL_UNAVAILABLE_POPULARITY`, `global_popularity/RETRIEVAL_EMPTY_POPULARITY`.

`item_id` and `user_id` are the **original MovieLens ids**. Unknown user ids are treated as new users.
Errors: `422` invalid body/variant, `503` recommender not loaded.

## `GET /similar-items?item_id=296&k=10`
Nearest items in the two-tower embedding space (cold items included). `404` for unknown ids.

## `POST /feedback`
```json
{"user_id": 123, "item_id": 296, "event": "click", "rating": null, "position": 0, "variant": "dcn_v2"}
```
`event`: `impression` | `click` | `rating`. Always appended to `logs/feedback.jsonl`. A `click`/`rating`
(≥ threshold) updates the user's history and **invalidates that user's cached recommendations**.

## `GET /health`
`{"status":"ok|degraded|starting","model_backend":"triton","model_backend_ready":true,"cache":"redis","cache_healthy":true,"n_items":...}`

## `GET /metrics`
Prometheus text: `recsys_stage_latency_ms{stage=...}` histograms, `recsys_requests_total{variant,strategy,cache_hit}`.

## Other
`GET /cache/stats` (hits, misses, hit_rate, backend) · `GET /genres`

## Triton models (`model_repository/`)
| model | inputs | output |
|---|---|---|
| `user_tower` | `user_idx` int64[1], `hist_items` int64[20], `user_numeric` fp32[3] | `embedding` fp32[64] |
| `ranker_wd`, `ranker_dcn` | `item_idx` int64[1], `dense` fp32[D] | `score` fp32[1] |

`config.pbtxt` is generated at export time from the real ONNX graph (names/dtypes/dims), `KIND_CPU`, dynamic batching.
