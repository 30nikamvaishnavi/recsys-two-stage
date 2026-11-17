"""TensorFlow / TFRS tests. Skipped automatically where TensorFlow is not installed (e.g. light CI)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_recommenders")

from recsys.features.common import NUM_USER  # noqa: E402
from recsys.features.state import examples_to_arrays  # noqa: E402
from recsys.ranking import models as RM  # noqa: E402
from recsys.ranking.data import ABLATIONS, Layout  # noqa: E402
from recsys.retrieval import two_tower as tt  # noqa: E402


def small_towers(world):
    tab, N = world.tables["rank"], world.cfg["features"]["history_len"]
    ut = tt.build_user_tower(world.n_users, world.n_items, tab.genres, N, 16, [32])
    it = tt.build_item_tower(
        world.n_items, world.cfg["features"]["genome_dim"], tab.n_pop_buckets, 16, [32]
    )
    return ut, it, tab, N


def test_towers_output_unit_norm_embeddings(world):
    ut, it, tab, N = small_towers(world)
    st = world.states["val"].take(np.arange(8))
    u = tt.embed_users(ut, st)
    assert u.shape == (8, 16) and np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-4)
    e = tt.embed_items(it, tab)
    assert e.shape == (world.n_items + 1, 16) and np.isfinite(e).all() and np.allclose(e[0], 0)


def test_cold_items_embedded_from_content_only(world):
    ut, it, tab, N = small_towers(world)
    cold = np.where(tab.cold_mask)[0]
    assert len(cold) > 0
    e = tt.embed_items(it, tab, content_only_cold=True)
    x = tt.item_inputs(tab, cold[:5])
    x["item_idx"] = np.zeros_like(x["item_idx"])
    assert np.allclose(it.predict(x, verbose=0), e[cold[:5]], atol=1e-5)


def test_two_tower_training_step_runs_and_loss_is_finite(world):
    ut, it, tab, N = small_towers(world)
    cfg = world.cfg
    u, i, h, num = examples_to_arrays(world.train_ex.head(2048), N)
    q = np.full(len(i), 1e-3, np.float32)
    arrs = {"user_idx": u, "item_idx": i, "hist_items": h, "user_numeric": num, "item_q": q}
    model = tt.TwoTowerModel(ut, it, tab, cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    hist = model.fit(tt.make_dataset(arrs, 256, 0), epochs=2, verbose=0)
    assert all(np.isfinite(hist.history["loss"]))


@pytest.mark.parametrize("kind", ["wide_deep", "dcn_v2"])
def test_rankers_forward_and_ablation_mask(world, kind):
    lay = Layout(world.cfg["features"]["genome_dim"])
    rc = world.cfg["ranking"]
    m = RM.build_ranker(kind, lay, world.n_items, rc)
    x = np.random.default_rng(0).normal(size=(6, lay.dim)).astype(np.float32)
    idx = np.arange(1, 7)
    p = RM.predict(m, idx, x)
    assert p.shape == (6,) and ((p > 0) & (p < 1)).all()
    m2 = RM.build_ranker(kind, lay, world.n_items, rc, ABLATIONS["no_retrieval_score"])
    m2.set_weights(m.get_weights())
    a, b = x.copy(), x.copy()
    b[:, lay.groups["ret"][0] : lay.groups["ret"][1]] += 5.0
    assert np.allclose(RM.predict(m2, idx, a), RM.predict(m2, idx, b), atol=1e-6)
    assert not np.allclose(RM.predict(m, idx, a), RM.predict(m, idx, b), atol=1e-6)


def test_ranker_learns_a_separable_signal(world):
    from recsys.ranking.data import RankData

    lay = Layout(world.cfg["features"]["genome_dim"])
    rng = np.random.default_rng(0)
    d = rng.normal(size=(4000, lay.dim)).astype(np.float32)
    y = (d[:, lay.groups["cross"][0]] > 0).astype(np.float32)
    rd = RankData(d, rng.integers(1, world.n_items, 4000), y, np.arange(4000))
    m = RM.build_ranker("dcn_v2", lay, world.n_items, {**world.cfg["ranking"], "epochs": 8})
    h = RM.fit_ranker(
        m,
        RankData(d[:3000], rd.item_idx[:3000], y[:3000], rd.group[:3000]),
        RankData(d[3000:], rd.item_idx[3000:], y[3000:], rd.group[3000:]),
        {**world.cfg["ranking"], "epochs": 8},
        verbose=0,
    )
    assert max(h.history["val_auc"]) > 0.9


def test_onnx_export_matches_tf_and_writes_triton_config(world, tmp_path):
    pytest.importorskip("tf2onnx")
    pytest.importorskip("onnxruntime")
    from recsys.serving.export import export_onnx

    ut, it, tab, N = small_towers(world)
    sample = {
        "user_idx": np.array([[1], [0]], np.int64),
        "hist_items": np.zeros((2, N), np.int64),
        "user_numeric": np.zeros((2, NUM_USER), np.float32),
    }
    meta = export_onnx(ut, tmp_path, "user_tower", sample)
    assert meta["max_abs_diff"] < 1e-3 and set(meta["inputs"]) == set(sample)
    cfg_txt = (tmp_path / "user_tower" / "config.pbtxt").read_text()
    assert "KIND_CPU" in cfg_txt and "dynamic_batching" in cfg_txt
