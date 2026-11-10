# Data dictionary

Index 0 is reserved for padding / unknown in **both** user and item index spaces.

## `data/processed/<mode>/`
| file | columns |
|---|---|
| `interactions.parquet` | `user_idx` i32, `item_idx` i32, `rating` f32 (0.5–5), `ts` i64 (epoch s), `is_pos` bool (rating ≥ 4), `stage` i8 (0 retr, 1 rank, 2 val, 3 test) |
| `items.parquet` | `item_idx`, `movie_id`, `title`, `year` (nullable), `genres` (pipe string) |
| `genome.parquet` | `item_idx`, `tag_id`, `relevance` |
| `user_map.parquet` | `user_idx`, `user_id` (original MovieLens id) |
| `boundaries.json` | `t1 < t2 < t3` split timestamps |

Cleaning: de-duplicate (user,item) keeping the latest; optional deterministic user sample; drop users with < 5 interactions
(a count-only filter applied before splitting - documented minor leak, standard practice); drop ratings of unknown movies.

## `artifacts/features/<mode>/` (identical schema from the Spark and pandas engines)
| file | columns |
|---|---|
| `train_examples.parquet` | `user_idx,item_idx,ts`, `n_prior` (all earlier events), `avg_prior_rating`, `days_since_prev`, `hist_items` int[20] (previous positives, oldest→newest, zero-padded at end) |
| `user_state_{rank,val,test}.parquet` | `user_idx`, `targets` int[] (positives in the stage window), state **frozen at window start**: `n_all`, `n_pos`, `avg_rating`, `days_since_last`, `hist_items` int[20] |
| `item_pop_{rank,val,test}.parquet` | `item_idx`, `pop_all`, `pop_7d`, `pop_30d` (positives strictly before the cutoff) |
| `genome_vectors.parquet` | `item_idx`, `genome` float[D] (PCA of 1128-tag genome) |
| `cold_items.parquet`, `cold_users.parquet` | `item_idx|user_idx`, `n_pos` below the cold threshold at the test cutoff |
| `pop_negatives_rank.parquet` | `user_idx,item_idx` popularity hard negatives not seen by the user |
| `item_table_{stage}.npz` | dense arrays: `genres (n+1,20)`, `year_bucket`, `pop (n+1,3) log1p`, `pop_bucket`, `pop_rank_pct`, `genome`, `has_genome`, `n_pos` |
| `item_q.npy` | in-batch sampling probability per item (logQ correction) |

## Ranker dense vector (`ranking/data.py::Layout`)
`user_num(3) | user_genre_affinity(20) | item_genres(20) | item_num(6) | genome(D) | cross(3) | retrieval(2)`
- `user_num`: log activity, centred mean rating, log recency
- `item_num`: log pop (all/7d/30d), popularity percentile, year bucket/10, has_genome
- `cross`: genre overlap, genome cosine (history mean vs item), popularity gap
- `retrieval`: two-tower score, log retrieval rank

## Reports (`artifacts/reports/<mode>/`)
`retrieval_metrics.json`, `faiss_benchmark.{json,png}`, `ranking_metrics.json`, `ranker_calibration.png`,
`coldstart_metrics.json`, `ab_results.json`, `ab_impressions.jsonl`, `load_test.json`, `cache_benchmark.json`, `eda_summary.json`.
