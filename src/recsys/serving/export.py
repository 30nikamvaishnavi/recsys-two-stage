"""Export trained models to ONNX, write the Triton model repository (config.pbtxt, dynamic batching,
KIND_CPU) and the serving assets (item table, embeddings, ANN index inputs, user states, seen matrix).

    python -m recsys.serving.export --mode sample
ONNX output is verified numerically against TensorFlow before being accepted.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..config import parse_cfg, path
from ..features.loader import FeatureStore
from ..utils import get_logger, save_json

log = get_logger("export")

_DT = {1: "TYPE_FP32", 7: "TYPE_INT64", 6: "TYPE_INT32"}


def pbtxt(
    name: str,
    inputs: list[tuple[str, int, list[int]]],
    outputs: list[tuple[str, int, list[int]]],
    max_batch: int = 2048,
) -> str:
    def block(items):
        return ",\n".join(
            f'  {{ name: "{n}" data_type: {_DT[t]} dims: [ {", ".join(map(str, d))} ] }}'
            for n, t, d in items
        )

    return (
        f'name: "{name}"\nbackend: "onnxruntime"\nmax_batch_size: {max_batch}\n'
        f"input [\n{block(inputs)}\n]\noutput [\n{block(outputs)}\n]\n"
        "dynamic_batching { preferred_batch_size: [ 8, 32, 128 ] max_queue_delay_microseconds: 2000 }\n"
        "instance_group [ { kind: KIND_CPU count: 2 } ]\n"
    )


def export_onnx(
    model, repo: Path, name: str, sample: dict[str, np.ndarray], max_batch: int = 2048
) -> dict:
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    d = repo / name / "1"
    d.mkdir(parents=True, exist_ok=True)
    specs = [
        tf.TensorSpec((None, *t.shape[1:]), t.dtype, name=t.name.split(":")[0])
        for t in model.inputs
    ]
    tf2onnx.convert.from_keras(
        model, input_signature=specs, opset=15, output_path=str(d / "model.onnx")
    )
    g = onnx.load(str(d / "model.onnx")).graph

    def desc(vals):
        return [
            (
                v.name,
                v.type.tensor_type.elem_type,
                [dim.dim_value for dim in v.type.tensor_type.shape.dim][1:],
            )
            for v in vals
        ]

    ins, outs = desc(g.input), desc(g.output)
    (repo / name / "config.pbtxt").write_text(pbtxt(name, ins, outs, max_batch))
    # numerical verification vs TF
    ref = model.predict(sample, verbose=0)
    got = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {i[0]: sample[i[0]] for i in ins}
    )[0]
    err = float(np.abs(ref - got).max())
    if err > 1e-3:
        raise RuntimeError(f"ONNX export of {name} deviates from TF by {err}")
    log.info("exported %s (max abs diff vs TF %.2e)", name, err)
    return {"inputs": [i[0] for i in ins], "outputs": [o[0] for o in outs], "max_abs_diff": err}


def export_assets(cfg, fs: FeatureStore, sdir: Path) -> None:
    import pandas as pd
    import scipy.sparse as sp

    sdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(path(cfg, "features")) / "item_table_test.npz", sdir / "item_table.npz")
    shutil.copy(Path(path(cfg, "models", "retrieval")) / "item_emb_test.npy", sdir / "item_emb.npy")
    fs.items().to_parquet(sdir / "items.parquet", index=False)
    fs.user_map().to_parquet(sdir / "user_map.parquet", index=False)
    pd.read_parquet(Path(path(cfg, "features")) / "user_state_test.parquet").to_parquet(
        sdir / "user_states.parquet", index=False
    )
    from .all_states import user_state_all

    everyone = user_state_all(
        fs.interactions(["user_idx", "item_idx", "rating", "ts", "is_pos"]),
        fs.cutoff("test"),
        fs.N,
    )
    everyone.to_parquet(sdir / "user_states_all.parquet", index=False)
    seen = fs.seen("test")
    sp.save_npz(sdir / "seen.npz", seen.m)
    save_json(
        {
            **fs.meta,
            "cutoff_ts": fs.cutoff("test"),
            "positive_threshold": cfg["data"]["positive_threshold"],
        },
        sdir / "assets_meta.json",
    )


def main():
    from ..ranking import models as M
    from ..ranking.data import Layout
    from ..retrieval import two_tower as tt

    cfg, _ = parse_cfg("Export ONNX models + Triton repository + serving assets")
    fs = FeatureStore(cfg)
    repo = Path(path(cfg, "model_repository"))
    repo.mkdir(parents=True, exist_ok=True)
    sdir = path(cfg, "serving", create=True)
    sdir.mkdir(parents=True, exist_ok=True)
    tab = fs.item_table("rank")
    ut, _ = tt.load_towers(
        cfg, fs.meta, tab.genres, tab.n_pop_buckets, path(cfg, "models", "retrieval")
    )
    N, rc = fs.N, cfg["ranking"]
    layout = Layout(cfg["features"]["genome_dim"])
    names = cfg["serving"]["models"]
    meta = {
        "models": {},
        "roles": {
            "user_tower": names["user_tower"],
            "wide_deep": names["wide_deep"],
            "dcn_v2": names["dcn_v2"],
        },
    }
    sample_u = {
        "user_idx": np.array([[1], [0]], np.int64),
        "hist_items": np.zeros((2, N), np.int64),
        "user_numeric": np.zeros((2, 3), np.float32),
    }
    sample_u["hist_items"][0, :3] = [1, 2, 3]
    meta["models"][names["user_tower"]] = export_onnx(ut, repo, names["user_tower"], sample_u)
    for kind in ("wide_deep", "dcn_v2"):
        m = M.build_ranker(kind, layout, fs.n_items, rc)
        m.load_weights(str(path(cfg, "models", "ranking") / f"{kind}.weights.h5"))
        sample = {
            "item_idx": np.array([[1], [2], [3]], np.int64),
            "dense": np.random.default_rng(0).normal(size=(3, layout.dim)).astype(np.float32),
        }
        meta["models"][names[kind]] = export_onnx(m, repo, names[kind], sample)
    save_json(meta, sdir / "serving_meta.json")
    export_assets(cfg, fs, sdir)
    log.info("model repository: %s | assets: %s", repo, sdir)


if __name__ == "__main__":
    main()
