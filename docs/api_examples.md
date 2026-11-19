# API response examples

These are shortened responses captured from the running sample API with the Triton backend and Redis cache. Item lists are truncated to one item and latency details are omitted; response values shown below came from actual API responses.

## Warm user

```json
{
  "user_id": 2084,
  "k": 10,
  "variant": "dcn_v2",
  "items": [
    {
      "item_id": 79132,
      "title": "Inception (2010)",
      "genres": "Action|Crime|Drama|Mystery|Sci-Fi|Thriller|IMAX",
      "score": 0.39360684156417847,
      "retrieval_score": 0.21211114525794983,
      "ranker_score": 0.39360684156417847
    }
  ],
  "meta": {
    "strategy": "two_stage",
    "reason_code": "WARM_USER",
    "cache_hit": true,
    "backend": "triton"
  }
}
```

## History, but no test-window positives

This user was present in `user_states_all.parquet` and absent from `user_states.parquet`; the API served the user as warm.

```json
{
  "user_id": 1,
  "k": 10,
  "variant": "dcn_v2",
  "items": [
    {
      "item_id": 293,
      "title": "L\u00e9on: The Professional (a.k.a. The Professional) (L\u00e9on) (1994)",
      "genres": "Action|Crime|Drama|Thriller",
      "score": 0.457250714302063,
      "retrieval_score": 0.15553194284439087,
      "ranker_score": 0.457250714302063
    }
  ],
  "meta": {
    "strategy": "two_stage",
    "reason_code": "WARM_USER",
    "cache_hit": false,
    "backend": "triton"
  }
}
```

## Sparse-history user

The tested user had three prior positive interactions and was routed to the content fallback.

```json
{
  "user_id": 380,
  "k": 10,
  "variant": "dcn_v2",
  "items": [
    {
      "item_id": 110,
      "title": "Braveheart (1995)",
      "genres": "Action|Drama|War",
      "score": 3.6669936180114746,
      "retrieval_score": null,
      "ranker_score": null
    }
  ],
  "meta": {
    "strategy": "content_blend",
    "reason_code": "SPARSE_HISTORY_CONTENT_BLEND",
    "cache_hit": false,
    "backend": "triton"
  }
}
```
