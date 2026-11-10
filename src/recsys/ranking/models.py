"""Ranking models (Keras): DCN-V2 and Wide & Deep. Both take two tensors:
    item_idx int64 (B,1)   dense float32 (B, D)   ->   score float32 (B,1) in (0,1)
so they can be served by Triton as ONNX with a single, stable signature."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..features.common import G
from .data import Layout

keras = tf.keras


class FeatureMask(keras.layers.Layer):
    """Multiply by a constant 0/1 vector (feature-group ablations are baked into the model)."""

    def __init__(self, mask: np.ndarray, **kw):
        super().__init__(**kw)
        self.m = tf.constant(mask, tf.float32)

    def call(self, x):
        return x * self.m


class CrossLayerV2(keras.layers.Layer):
    """DCN-V2 full-rank cross layer: x_{l+1} = x0 ⊙ (W x_l + b) + x_l."""

    def build(self, shape):
        d = shape[0][-1]
        self.w = self.add_weight("w", (d, d), initializer="glorot_uniform")
        self.b = self.add_weight("b", (d,), initializer="zeros")

    def call(self, inputs):
        x0, x = inputs
        return x0 * (tf.matmul(x, self.w) + self.b) + x


class Outer(keras.layers.Layer):
    """Flattened outer product (B,a)x(B,b)->(B,a*b): the Wide part's user-genre x item-genre cross."""

    def call(self, inputs):
        a, b = inputs
        return tf.reshape(a[:, :, None] * b[:, None, :], (-1, G * G))


def _mlp(x, sizes, drop=0.1):
    for s in sizes:
        x = keras.layers.Dense(s, activation="relu")(x)
        x = keras.layers.Dropout(drop)(x)
    return x


def build_ranker(kind: str, layout: Layout, n_items: int, rcfg, drop_groups: list[str] | None = None) -> keras.Model:
    idx = keras.Input((1,), dtype=tf.int64, name="item_idx")
    dense = keras.Input((layout.dim,), dtype=tf.float32, name="dense")
    x = FeatureMask(layout.mask(drop_groups or []))(dense)
    x0 = keras.layers.Concatenate()([x, keras.layers.Flatten()(keras.layers.Embedding(n_items + 1, rcfg["item_emb_dim"])(idx))])
    deep = _mlp(x0, rcfg["mlp"])
    if kind == "dcn_v2":
        xl = x0
        for _ in range(rcfg["cross_layers"]):
            xl = CrossLayerV2()([x0, xl])
        logit = keras.layers.Dense(1)(keras.layers.Concatenate()([xl, deep]))
    elif kind == "wide_deep":
        sl = lambda name: keras.layers.Lambda(lambda t, a=layout.groups[name][0], b=layout.groups[name][1]: t[:, a:b])(x)  # noqa: E731
        wide_in = keras.layers.Concatenate()([Outer()([sl("user_gen"), sl("item_gen")]), sl("item_gen")])
        logit = keras.layers.Add()([keras.layers.Dense(1)(wide_in), keras.layers.Dense(1)(deep)])
    else:
        raise ValueError(kind)
    return keras.Model([idx, dense], keras.layers.Activation("sigmoid", name="score")(logit), name=f"ranker_{kind}")


def fit_ranker(model, train, val, rcfg, verbose=2):
    model.compile(
        optimizer=keras.optimizers.Adam(rcfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    cb = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True)]
    x = lambda d: {"item_idx": d.item_idx.astype(np.int64)[:, None], "dense": d.dense}  # noqa: E731
    return model.fit(
        x(train), train.label, validation_data=(x(val), val.label), epochs=rcfg["epochs"],
        batch_size=rcfg["batch_size"], callbacks=cb, verbose=verbose,
    )  # fmt: skip


def predict(model, item_idx: np.ndarray, dense: np.ndarray, batch: int = 8192) -> np.ndarray:
    return model.predict({"item_idx": item_idx.astype(np.int64)[:, None], "dense": dense}, batch_size=batch, verbose=0).reshape(-1)
