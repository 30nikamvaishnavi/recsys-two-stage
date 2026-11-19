# Phase (i-2): Triton

## Files changed

- [`phase-i-2.md`](./phase-i-2.md) — phase record. No model or Compose configuration was changed.

## Commands and exit codes

- Inspected only the three model `config.pbtxt` files, the Triton service block in `docker-compose.yml`, and `src/recsys/serving/backends.py`.
- ONNX Runtime metadata inspection for `user_tower`, `ranker_wd`, and `ranker_dcn` — exit 0.
- User-provided curl probe result: `/usr/bin/curl` — curl is present; no healthcheck change was needed.
- User-provided `docker images` result: `nvcr.io/nvidia/tritonserver:24.08-py3 26.1GB`.
- User-provided `time docker compose up -d triton` result: real `1.3s`; this measures container start only, not readiness. Triton was not started again by this assistant.
- User-provided `docker compose ps triton` result: `Up 7 seconds (healthy)`; user-provided `curl http://localhost:8000/v2/health/ready` returned HTTP 200.
- TritonClient versus ONNX Runtime comparison for the same generated arrays at batch sizes 1 and 64 — exit 0.
- No package installation was needed; `pip list | grep -i tritonclient` showed `tritonclient 2.47.0`.

## Config and ONNX metadata comparison

All three configs use `backend: "onnxruntime"`, `max_batch_size: 2048`, dynamic batching with preferred batch sizes, and `KIND_CPU`. Declared static shapes and dtypes agree with ONNX Runtime metadata:

| Model | Inputs | Output |
|---|---|---|
| `user_tower` | `user_idx`: int64 `[batch, 1]`; `hist_items`: int64 `[batch, 20]`; `user_numeric`: float32 `[batch, 3]` | `user_embedding`: float32 `[batch, 64]` |
| `ranker_wd` | `item_idx`: int64 `[batch, 1]`; `dense`: float32 `[batch, 86]` | `score`: float32 `[batch, 1]` |
| `ranker_dcn` | `item_idx`: int64 `[batch, 1]`; `dense`: float32 `[batch, 86]` | `score`: float32 `[batch, 1]` |

Both tested batch sizes fit the configured maximum. The comparison used identical generated ID arrays (int64) and feature arrays (float32) as inputs to ONNX Runtime and Triton.

## Measured results

Maximum absolute output difference:

| Model | Batch size 1 | Batch size 64 |
|---|---:|---:|
| `user_tower` | `0.0` | `0.0` |
| `ranker_wd` | `0.0` | `0.0` |
| `ranker_dcn` | `0.0` | `0.0` |

The `ranker_wd` and `ranker_dcn` Triton outputs were finite and within `[0, 1]` for both batch sizes. The tested output ranges were:

| Model | Batch size 1 | Batch size 64 |
|---|---:|---:|
| `ranker_wd` | `0.10681188106536865` to `0.10681188106536865` | `2.6226043701171875e-06` to `0.5684511065483093` |
| `ranker_dcn` | `0.5834212303161621` to `0.5834212303161621` | `7.471442222595215e-05` to `0.7905025482177734` |

Triton image size is `26.1GB` as reported by the user. Startup-to-readiness time is **not measured**; the reported `1.3s` is container-start time, and the health check was observed separately. No inference latency is reported in this phase.

The user-provided output reported Triton server version `2.49.0`, `MODE_EXPLICIT`, `strict_readiness 1`, HTTP `8000`, gRPC `8001`, and metrics `8002`. Model inference succeeded for all three models. Triton logs were not inspected, and no CUDA/GPU warnings were included in the provided output; all configs specify `KIND_CPU`.

## NOT VERIFIED

- Exact elapsed time from `docker compose up -d triton` until `/v2/health/ready`; only container-start time and a later healthy/HTTP-200 observation were supplied.
- Triton logs for CUDA/GPU warnings.
- Model behavior outside the two tested batch sizes and generated inputs.
- Inference latency; intentionally deferred to phase (k).

## Next exact command

Proceed to phase (i-3), API checks with Triton backend, only after the user says `go`.

## Additional user-observed results

- The user observed Triton logs reporting “successfully loaded” for `ranker_wd`, `user_tower`, and `ranker_dcn`.
- The user observed these CPU-only warnings: no NVIDIA driver detected, pinned memory pool unavailable, and CUDA memory pool disabled. All three models still loaded.
- The user observed Compose status `Up 7 seconds (healthy)`.
- Triton and ONNX Runtime outputs were identical on the tested random inputs (`max abs diff 0.0`); this validates the serving path only, not model quality.
