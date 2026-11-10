# Architecture

## System diagram

```mermaid
flowchart LR
  subgraph Offline["Offline (weekly, Airflow)"]
    RAW[(MovieLens 25M)] --> PREP[prepare: clean + TIME split]
    PREP --> SPARK[Spark features<br/>point-in-time]
    SPARK --> TT[Two-tower training<br/>TFRS, logQ sampled softmax]
    TT --> EMB[item embeddings]
    EMB --> FAISS[FAISS benchmark<br/>Flat / HNSW / IVF-PQ]
    TT --> RC[retrieval candidates]
    RC --> RK[Rankers: Wide&Deep + DCN-V2<br/>hard negatives]
    RK --> GATE{eval gate<br/>>= production?}
    GATE -- yes --> ONNX[ONNX export]
    ONNX --> REPO[(Triton model repository)]
    FAISS --> ASSETS[(serving assets)]
  end
  subgraph Online["Online serving (docker compose)"]
    C[Client / Streamlit] --> API[FastAPI gateway]
    API -->|1 cache?| REDIS[(Redis)]
    API -->|2 router| CS[Cold-start router]
    CS -->|warm| UT[Triton: user tower]
    UT --> ANN[FAISS top-300]
    ANN --> FEAT[feature builder]
    FEAT --> RKT[Triton: ranker A/B]
    RKT --> POST[filter seen + MMR]
    CS -->|cold| FB[genre-pop / content / popularity]
    POST --> REDIS
    FB --> REDIS
  end
  REPO --> UT
  REPO --> RKT
  ASSETS --> ANN
  AB[A/B simulator<br/>hash bucketing + click model] --> API
  MLF[(MLflow)] -.tracks.- TT
  MLF -.tracks.- RK
```

## Request flow (`POST /recommend`)

```mermaid
sequenceDiagram
  participant U as Client
  participant G as FastAPI
  participant R as Redis
  participant T as Triton
  participant F as FAISS
  U->>G: user_id, k, model_variant
  G->>R: GET rec:{variant}:{user}:{k}
  alt hit
    R-->>G: cached JSON
  else miss
    G->>G: user state + router (reason code)
    alt WARM_USER
      G->>T: user_tower(user_idx, history, numeric)
      T-->>G: 64-d embedding
      G->>F: search(emb, 300 + |seen|)
      F-->>G: candidates
      G->>G: drop seen, build dense features
      G->>T: ranker(item_idx, dense)
      T-->>G: P(positive) per candidate
      G->>G: MMR diversity -> top-k
    else cold / model down
      G->>G: genre-popularity | content blend | popularity
    end
    G->>R: SETEX + index key for invalidation
  end
  G-->>U: items + strategy + reason_code + per-stage latency
```

## Data flow and time windows

```
time ──────────────────────────────────────────────────────────────────────▶
|<──── retrieval-train 70% ────>|<─ ranker-train 10% ─>|<─ val 10% ─>|<─ test 10% ─>|
0                               t1                     t2            t3
```

| Stage | Window | State frozen at | What trains / evaluates |
|---|---|---|---|
| retrieval-train | [0, t1) | each example's own time | two-tower, SVD baseline |
| ranker-train | [t1, t2) | t1 | rankers learn on retrieval candidates; labels = true positives in window |
| validation | [t2, t3) | t2 | model selection, gate metrics, ablations |
| test | [t3, ∞) | t3 | A/B replay, serving assets (latest state) |

The ranker is trained on candidates produced by a retrieval model that **never saw those
labels**, so its score distribution matches serving (the usual failure of training a ranker on
the retrieval model's own training data is avoided).

## Key design decisions (interview answers)

| Decision | Why |
|---|---|
| Time split, three cutoffs | Random splits leak future behaviour and inflate metrics; production predicts the future. Tests prove invariance to future data. |
| In-batch sampled softmax + **logQ correction** | In-batch negatives are sampled ∝ popularity, which over-penalises popular items; subtracting log q(item) de-biases the logits. Accidental hits removed. |
| L2-normalised 64-d embeddings, temperature 0.05 | Inner product = cosine; bounded scores make ANN and the ranker's retrieval-score feature stable. |
| ID dropout in both towers | Trains the content-only path so **new items and new users** get usable embeddings with no retraining. |
| Index choice by measurement | Benchmark script picks the lowest-p95 index reaching target recall@100 vs exact (default 0.95). HNSW usually wins on latency, IVF-PQ on memory. |
| Two rankers (Wide&Deep, DCN-V2) | Needed for the A/B test; Wide&Deep memorises user-genre × item-genre crosses, DCN-V2 learns explicit feature crosses. |
| ONNX on Triton (not TF SavedModel) | Triton's TF backend lags TensorFlow versions; ONNX Runtime CPU is stable and verified numerically at export. |
| Shared `candidate_dense` | One function builds ranker features offline and online → no train/serve skew. |
| Graceful degradation | Model errors fall back to popularity with reason code `MODEL_UNAVAILABLE_POPULARITY`; the API never 5xx's because Triton is down. |
