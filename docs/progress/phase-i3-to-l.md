# Phases (i-3)–(l) progress

## Phase (i-3): API

### Files changed

- [`api_examples.md`](../api_examples.md) — three shortened, live API responses from Triton/Redis.
- [`phase-i3-to-l.md`](./phase-i3-to-l.md) — ongoing multi-phase progress record.
- No API/model source code was changed.

### Commands and exit codes

- User reported `docker compose build api` completed without errors and built `recsys-two-stage-api:latest` (`1.15GB`).
- `cd ~/recsys-two-stage && source .venv/bin/activate && docker compose up -d redis triton api` — exit 0.
- `cd ~/recsys-two-stage && source .venv/bin/activate && docker compose ps redis triton api && curl -fsS http://localhost:8080/health` — exit 0; all three containers healthy and API health returned OK with Triton and Redis ready.
- First acceptance script — exit 1 because its assertion assumed the literal strategy string was `warm`; actual warm strategy is `two_stage`. No application code was changed.
- Revised API acceptance script — exit 0.
- Cache-miss/repeat-hit check after feedback invalidation — exit 0.

### API acceptance results

- `/health`: HTTP 200; `status=ok`, `model_backend=triton`, backend ready, cache `redis`, cache healthy.
- Warm-user sample: `n_pos=334`; strategy `two_stage`, reason `WARM_USER`.
- History with no test-window positives: `n_pos=39`, `n_all=70`; present in `user_states_all.parquet`, absent in `user_states.parquet`; strategy `two_stage`, reason `WARM_USER`.
- Sparse-history sample: `n_pos=3`, `n_all=27`; strategy `content_blend`, reason `SPARSE_HISTORY_CONTENT_BLEND`.
- Unknown user: strategy `global_popularity`, reason `NEW_USER_NO_SIGNAL`.
- Identical recommendation requests after invalidation: first `cache_hit=false`, second `cache_hit=true`.
- `/similar-items`: HTTP 200, returned 3 items.
- `/feedback` click: HTTP 200; invalidated 2 cache entries; next recommendation was a miss.
- `/metrics`: HTTP 200; Prometheus text included `recsys_requests_total`.
- Three shortened live responses are in [`api_examples.md`](../api_examples.md); their source is the corresponding live `/recommend` JSON responses. User-state counts came from `user_states_all.parquet` and `user_states.parquet`.

### NOT VERIFIED

- Compose stability for two minutes is pending phase (i-4).
- No source-code bug was identified or changed in phase (i-3).

## Phase (i-4): Compose

Pending.

## Phase (j): Airflow

Pending.

## Phase (k): A/B SIMULATION, load test, latency chart

Any A/B output must say **SIMULATION, no real users**. No A/B or load test has been run in this phase yet.

## Phase (l): Streamlit

Pending. A demo script must describe only pages and API behavior observed in the real demo.
