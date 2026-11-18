"""Two-tower retrieval model with TensorFlow Recommenders.

User tower : user-id emb (+ID dropout) | mean-pooled history item emb | genre affinity | numeric
Item tower : item-id emb (+ID dropout) | genres | year bucket | genome vec | popularity bucket
Both L2-normalised (dim d). Loss: in-batch sampled softmax with logQ popularity correction
(tfrs.tasks.Retrieval + candidate_sampling_probability) and accidental-hit removal.

Towers are plain functional Keras models so they export cleanly to ONNX/Triton. ID dropout
(id -> 0) trains the content-only path used for cold users and cold items.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from ..features.common import N_YEAR_BUCKETS, NUM_USER, G
from ..features.item_table import ItemTable

keras = tf.keras


class MaskedMeanPool(keras.layers.Layer):
    """inputs=[emb (B,N,D), ids (B,N)] -> mean over ids != 0."""

    def call(self, inputs):
        emb, ids = inputs
        m = tf.cast(tf.not_equal(ids, 0), emb.dtype)[..., None]
        return tf.reduce_sum(emb * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)


class GenreAffinity(keras.layers.Layer):
    """Mean genre multi-hot of the history items; table is a constant baked into the graph."""

    def __init__(self, genre_matrix: np.ndarray, **kw):
        super().__init__(**kw)
        self.table = tf.constant(genre_matrix, tf.float32)

    def call(self, ids):
        g = tf.gather(self.table, ids)
        m = tf.cast(tf.not_equal(ids, 0), tf.float32)[..., None]
        return tf.reduce_sum(g * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)


class L2Norm(keras.layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=-1)


def _mlp(x, hidden, out_dim, name_prefix):
    for i, h in enumerate(hidden):
        x = keras.layers.Dense(h, activation="relu", name=f"{name_prefix}_hidden_{i}")(x)
    x = keras.layers.Dense(out_dim, name=f"{name_prefix}_projection")(x)
    return L2Norm(name=f"{name_prefix}_embedding")(x)


def build_user_tower(
    n_users: int, n_items: int, genre_matrix: np.ndarray, history_len: int, dim: int, hidden
) -> keras.Model:
    uid = keras.Input((1,), dtype=tf.int64, name="user_idx")
    hist = keras.Input((history_len,), dtype=tf.int64, name="hist_items")
    num = keras.Input((NUM_USER,), dtype=tf.float32, name="user_numeric")
    u = keras.layers.Flatten(name="user_id_flatten")(
        keras.layers.Embedding(n_users + 1, dim, name="user_id_embedding")(uid)
    )
    h = MaskedMeanPool(name="history_mean_pool")(
        [keras.layers.Embedding(n_items + 1, dim, name="history_item_embedding")(hist), hist]
    )
    g = keras.layers.Dense(32, activation="relu", name="history_genre_dense")(
        GenreAffinity(genre_matrix, name="history_genre_affinity")(hist)
    )
    n = keras.layers.Dense(16, activation="relu", name="user_numeric_dense")(num)
    out = _mlp(keras.layers.Concatenate(name="user_features")([u, h, g, n]), hidden, dim, "user")
    return keras.Model([uid, hist, num], out, name="user_tower")


def build_item_tower(
    n_items: int, genome_dim: int, n_pop_buckets: int, dim: int, hidden
) -> keras.Model:
    iid = keras.Input((1,), dtype=tf.int64, name="item_idx")
    gen = keras.Input((G,), dtype=tf.float32, name="item_genres")
    yb = keras.Input((1,), dtype=tf.int64, name="year_bucket")
    gv = keras.Input((genome_dim,), dtype=tf.float32, name="genome")
    pb = keras.Input((1,), dtype=tf.int64, name="pop_bucket")
    parts = [
        keras.layers.Flatten(name="item_id_flatten")(
            keras.layers.Embedding(n_items + 1, dim, name="item_id_embedding")(iid)
        ),
        keras.layers.Dense(32, activation="relu", name="item_genre_dense")(gen),
        keras.layers.Flatten(name="year_bucket_flatten")(
            keras.layers.Embedding(N_YEAR_BUCKETS, 8, name="year_bucket_embedding")(yb)
        ),
        keras.layers.Dense(32, activation="relu", name="item_genome_dense")(gv),
        keras.layers.Flatten(name="pop_bucket_flatten")(
            keras.layers.Embedding(n_pop_buckets + 1, 8, name="pop_bucket_embedding")(pb)
        ),
    ]
    out = _mlp(keras.layers.Concatenate(name="item_features")(parts), hidden, dim, "item")
    return keras.Model([iid, gen, yb, gv, pb], out, name="item_tower")


def item_inputs(
    table: ItemTable, item_idx: np.ndarray, id_override: int | None = None
) -> dict[str, np.ndarray]:
    return {
        "item_idx": (np.zeros_like(item_idx) if id_override == 0 else item_idx).astype(np.int64)[
            :, None
        ],
        "item_genres": table.genres[item_idx],
        "year_bucket": table.year_bucket[item_idx].astype(np.int64)[:, None],
        "genome": table.genome[item_idx],
        "pop_bucket": table.pop_bucket[item_idx].astype(np.int64)[:, None],
    }


class TwoTowerModel(tfrs.Model):
    def __init__(self, user_tower, item_tower, table: ItemTable, cfg):
        super().__init__()
        r = cfg["retrieval"]
        self.user_tower, self.item_tower = user_tower, item_tower
        self.p_item, self.p_user = r["item_id_dropout"], r["user_id_dropout"]
        self.t_gen, self.t_yb = tf.constant(table.genres), tf.constant(table.year_bucket, tf.int64)
        self.t_gv, self.t_pb = tf.constant(table.genome), tf.constant(table.pop_bucket, tf.int64)
        self.logq_correction = r["logq_correction"]
        self.task = tfrs.tasks.Retrieval(temperature=r["temperature"], remove_accidental_hits=True)

    def compute_loss(self, features, training=False):
        iid = features["item_idx"]
        uid = features["user_idx"]
        iid_in = (
            tf.where(tf.random.uniform(tf.shape(iid)) < self.p_item, tf.zeros_like(iid), iid)
            if training
            else iid
        )
        uid_in = (
            tf.where(tf.random.uniform(tf.shape(uid)) < self.p_user, tf.zeros_like(uid), uid)
            if training
            else uid
        )
        u = self.user_tower(
            {
                "user_idx": uid_in[:, None],
                "hist_items": features["hist_items"],
                "user_numeric": features["user_numeric"],
            }
        )
        i = self.item_tower(
            {
                "item_idx": iid_in[:, None], "item_genres": tf.gather(self.t_gen, iid),
                "year_bucket": tf.gather(self.t_yb, iid)[:, None],
                "genome": tf.gather(self.t_gv, iid), "pop_bucket": tf.gather(self.t_pb, iid)[:, None],
            }
        )  # fmt: skip
        task_args = {"candidate_ids": iid, "compute_metrics": False}
        if self.logq_correction:
            task_args["candidate_sampling_probability"] = features["item_q"]
        return self.task(u, i, **task_args)


def make_dataset(arrs: dict[str, np.ndarray], batch: int, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        {
            "user_idx": arrs["user_idx"].astype(np.int64), "item_idx": arrs["item_idx"].astype(np.int64),
            "hist_items": arrs["hist_items"].astype(np.int64), "user_numeric": arrs["user_numeric"], "item_q": arrs["item_q"],
        }
    )  # fmt: skip
    return (
        ds.shuffle(min(len(arrs["item_idx"]), 500_000), seed=seed)
        .batch(batch, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def embed_items(
    item_tower, table: ItemTable, content_only_cold: bool = True, batch: int = 8192
) -> np.ndarray:
    """(n+1, d) item embeddings. Items with zero positives so far are embedded from CONTENT ONLY
    (id := 0) so brand-new items can be indexed immediately. Row 0 is zeros."""
    n = table.n_items
    out = np.zeros((n + 1, item_tower.output_shape[-1]), dtype=np.float32)
    ids = np.arange(1, n + 1)
    for s in range(0, n, batch):
        sl = ids[s : s + batch]
        x = item_inputs(table, sl)
        if content_only_cold:
            cold = table.cold_mask[sl]
            x["item_idx"] = np.where(cold[:, None], 0, x["item_idx"]).astype(np.int64)
        out[sl] = item_tower.predict(x, batch_size=batch, verbose=0)
    return out


def embed_users(user_tower, states, batch: int = 4096) -> np.ndarray:
    return user_tower.predict(
        {"user_idx": states.user_idx.astype(np.int64)[:, None], "hist_items": states.hist.astype(np.int64), "user_numeric": states.numeric},
        batch_size=batch, verbose=0,
    )  # fmt: skip


def save_towers(user_tower, item_tower, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    user_tower.save_weights(str(out_dir / "user_tower.weights.h5"))
    item_tower.save_weights(str(out_dir / "item_tower.weights.h5"))


def load_towers(cfg, meta: dict, genre_matrix: np.ndarray, n_pop_buckets: int, out_dir):
    r = cfg["retrieval"]
    ut = build_user_tower(
        meta["n_users"],
        meta["n_items"],
        genre_matrix,
        meta["history_len"],
        r["embedding_dim"],
        r["hidden"],
    )
    it = build_item_tower(
        meta["n_items"], meta["genome_dim"], n_pop_buckets, r["embedding_dim"], r["hidden"]
    )
    ut.load_weights(str(out_dir / "user_tower.weights.h5"))
    it.load_weights(str(out_dir / "item_tower.weights.h5"))
    return ut, it
