import numpy as np
from recsys.features.common import NUM_USER, G
from recsys.ranking.data import ABLATIONS, Layout, build_rank_train, candidate_dense, labels_for
from recsys.retrieval import retrievers as R


def test_layout_dimensions_and_masks():
    lay = Layout(8)
    assert lay.dim == NUM_USER + G + G + 6 + 8 + 3 + 2
    g = lay.groups
    assert g["user_num"][0] == 0 and g["ret"][1] == lay.dim
    m = lay.mask(ABLATIONS["no_cross"])
    assert m.sum() == lay.dim - 3 and (m[g["cross"][0] : g["cross"][1]] == 0).all()
    assert Layout(8).mask([]).all()


def test_candidate_dense_values_hand_check(world):
    t, st = world.tables["val"], world.states["val"].take(np.arange(3))
    items = np.array([[5, 6], [7, 8], [9, 10]])
    scores = np.array([[0.9, 0.1], [0.5, 0.4], [0.3, 0.2]], np.float32)
    d = candidate_dense(st, items, scores, t)
    lay = Layout(world.cfg["features"]["genome_dim"])
    assert d.shape == (6, lay.dim)
    g = lay.groups
    assert np.allclose(d[:, g["ret"][0]], scores.reshape(-1))  # retrieval score column
    assert np.allclose(
        d[:2, g["user_num"][0] : g["user_num"][1]], st.numeric[0]
    )  # user feats repeated per candidate
    assert np.allclose(d[0, g["item_gen"][0] : g["item_gen"][1]], t.genres[5])
    ug = d[0, g["user_gen"][0] : g["user_gen"][1]]
    overlap = d[0, g["cross"][0]]
    assert np.isclose(overlap, (ug * t.genres[5]).sum() / max(t.genres[5].sum(), 1))
    assert d[0, g["ret"][0] + 1] < d[1, g["ret"][0] + 1]  # rank feature increases with position


def test_labels_for():
    y = labels_for(np.array([[1, 2, 3], [4, 5, 6]]), [np.array([2, 9]), np.array([])])
    assert y.tolist() == [[0, 1, 0], [0, 0, 0]]


def test_build_rank_train_labels_and_pop_negatives(world):
    import pandas as pd

    st = world.states["rank"]
    st = st.take(np.where(st.n_pos >= 3)[0]).subsample(80, 0)
    E = np.random.default_rng(0).normal(size=(world.n_items + 1, 8)).astype(np.float32)
    U = np.random.default_rng(1).normal(size=(world.n_users + 1, 8)).astype(np.float32)

    class Ret(R.EmbeddingRetriever):
        pass

    ret = Ret(lambda s: U[s.user_idx], E)
    negs = pd.DataFrame(
        {"user_idx": np.repeat(st.user_idx[:5], 3), "item_idx": np.tile([1, 2, 3], 5)}
    )
    rd = build_rank_train(
        ret, st, world.seen["rank"], world.tables["rank"], 20, negs, world.n_items
    )
    assert (
        rd.dense.shape[0] == len(rd.label) == len(rd.item_idx) == len(rd.group)
        and rd.dense.shape[0] >= 80 * 20
    )
    assert set(np.unique(rd.label)) <= {0.0, 1.0}
    # each user's retrieved candidates exclude seen items
    seen = world.seen["rank"]
    first = rd.item_idx[rd.group == 0][:20]
    assert not (set(first.tolist()) & set(seen.seen(int(st.user_idx[0])).tolist()))
