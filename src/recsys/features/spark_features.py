"""Apache Spark feature pipeline (PySpark, local mode).

Produces the same Parquet schema as pandas_features.py:
  train_examples.parquet       user_idx,item_idx,ts,n_prior,avg_prior_rating,days_since_prev,hist_items
  user_state_{rank,val,test}   user_idx,targets,n_all,avg_rating,n_pos,days_since_last,hist_items
  item_pop_{rank,val,test}     item_idx,pop_all,pop_7d,pop_30d     (point-in-time at each cutoff)
  genome_vectors.parquet       item_idx,genome
  cold_items.parquet / cold_users.parquet                           (as of the test cutoff)
  pop_negatives_rank.parquet   user_idx,item_idx                    (popularity hard negatives)
"""

from __future__ import annotations

from pathlib import Path

from ..config import path
from ..utils import get_logger, load_json, timed
from .pandas_features import stage_windows

log = get_logger("spark_features")


def get_spark(cfg):
    from pyspark.sql import SparkSession

    s = cfg["spark"]
    tmp = Path(path(cfg, "features")) / "_spark_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.master(s["master"])
        .appName("recsys-features")
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", str(tmp))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def _pad(df, col: str, n: int, out: str):
    """Right-pad an array<int> column with zeros to length n."""
    return df.withColumn(
        out,
        F_expr(
            f"concat(coalesce({col}, cast(array() as array<int>)), array_repeat(0, {n} - size(coalesce({col}, cast(array() as array<int>)))))"
        ),
    )


def F_expr(s):  # tiny indirection so the module imports without pyspark
    from pyspark.sql import functions as F

    return F.expr(s)


def run(cfg, spark=None) -> Path:
    from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    spark = spark or get_spark(cfg)
    proc = Path(path(cfg, "data_processed"))
    out = path(cfg, "features", create=True)
    out.mkdir(parents=True, exist_ok=True)
    N = cfg["features"]["history_len"]
    b = load_json(proc / "boundaries.json")
    wins = stage_windows(b)

    inter = spark.read.parquet(str(proc / "interactions.parquet")).cache()
    items = spark.read.parquet(str(proc / "items.parquet")).select("item_idx")
    pos = inter.filter(F.col("is_pos"))
    day = 86400.0

    # ---- 1. retrieval training examples (stage 0 positives) with per-example history ----------
    with timed(log, "train examples"):
        w_all = Window.partitionBy("user_idx").orderBy("ts", "item_idx")
        prior = (
            inter.filter(F.col("stage") == 0)
            .withColumn(
                "n_prior",
                F.count(F.lit(1))
                .over(w_all.rowsBetween(Window.unboundedPreceding, -1))
                .cast("int"),
            )
            .withColumn(
                "avg_prior_rating",
                F.avg("rating").over(w_all.rowsBetween(Window.unboundedPreceding, -1)),
            )
            .withColumn("days_since_prev", (F.col("ts") - F.lag("ts").over(w_all)) / day)
            .select("user_idx", "item_idx", "n_prior", "avg_prior_rating", "days_since_prev")
        )
        p0 = pos.filter(F.col("stage") == 0)
        w_pos = Window.partitionBy("user_idx").orderBy("ts", "item_idx")
        ex = p0.withColumn("hist", F.collect_list("item_idx").over(w_pos.rowsBetween(-N, -1)))
        ex = ex.join(prior, ["user_idx", "item_idx"], "inner")
        ex = _pad(ex, "hist", N, "hist_items").select(
            "user_idx",
            "item_idx",
            "ts",
            "n_prior",
            "avg_prior_rating",
            "days_since_prev",
            "hist_items",
        )
        ex.write.mode("overwrite").parquet(str(out / "train_examples.parquet"))

    # ---- 2. point-in-time user state + item popularity at every stage cutoff -----------------
    pw = Window.partitionBy("user_idx").orderBy(F.col("ts").desc(), F.col("item_idx").desc())
    for stage, (cut, end) in wins.items():
        with timed(log, f"state+pop @{stage}"):
            before = inter.filter(F.col("ts") < cut)
            agg = before.groupBy("user_idx").agg(
                F.count(F.lit(1)).alias("n_all"),
                F.avg("rating").alias("avg_rating"),
                F.max("ts").alias("last_ts"),
                F.sum(F.col("is_pos").cast("int")).alias("n_pos"),
            )
            hist = (
                before.filter(F.col("is_pos"))
                .withColumn("rn", F.row_number().over(pw))
                .filter(F.col("rn") <= N)
                .groupBy("user_idx")
                .agg(F.sort_array(F.collect_list(F.struct("rn", "item_idx")), asc=False).alias("h"))
                .select("user_idx", F.expr("transform(h, x -> x.item_idx)").alias("hist"))
            )
            tgt = (
                pos.filter((F.col("ts") >= cut) & (F.col("ts") < end))
                .groupBy("user_idx")
                .agg(F.collect_list("item_idx").alias("targets"))
            )
            st = tgt.join(agg, "user_idx", "left").join(hist, "user_idx", "left")
            st = st.withColumn("n_all", F.coalesce("n_all", F.lit(0)).cast("long"))
            st = st.withColumn("n_pos", F.coalesce("n_pos", F.lit(0)).cast("long"))
            st = st.withColumn(
                "days_since_last", F.coalesce((F.lit(cut) - F.col("last_ts")) / day, F.lit(0.0))
            )
            st = _pad(st, "hist", N, "hist_items").select(
                "user_idx",
                "targets",
                "n_all",
                "avg_rating",
                "n_pos",
                "days_since_last",
                "hist_items",
            )
            st.write.mode("overwrite").parquet(str(out / f"user_state_{stage}.parquet"))

            pb = pos.filter(F.col("ts") < cut)
            cols = [F.count(F.lit(1)).alias("pop_all")]
            for d, name in zip(
                cfg["features"]["pop_windows_days"], ["pop_7d", "pop_30d"], strict=False
            ):
                cols.append(
                    F.sum(F.when(F.col("ts") >= cut - d * 86400, 1).otherwise(0)).alias(name)
                )
            pop = (
                items.join(pb.groupBy("item_idx").agg(*cols), "item_idx", "left")
                .na.fill(0)
                .select(
                    "item_idx",
                    F.col("pop_all").cast("long"),
                    F.col("pop_7d").cast("long"),
                    F.col("pop_30d").cast("long"),
                )
            )
            pop.write.mode("overwrite").parquet(str(out / f"item_pop_{stage}.parquet"))

    # ---- 3. genome content vectors (PCA) --------------------------------------------------------
    with timed(log, "genome PCA"):
        g = spark.read.parquet(str(proc / "genome.parquet"))
        tag_ids = sorted(r["tag_id"] for r in g.select("tag_id").distinct().collect())
        piv = g.groupBy("item_idx").pivot("tag_id", tag_ids).agg(F.first("relevance")).na.fill(0.0)
        vec = VectorAssembler(inputCols=[str(t) for t in tag_ids], outputCol="raw").transform(piv)
        sc = StandardScaler(inputCol="raw", outputCol="centered", withMean=True, withStd=False).fit(
            vec
        )
        vec = sc.transform(vec)
        k = min(cfg["features"]["genome_dim"], len(tag_ids))
        pca = PCA(k=k, inputCol="centered", outputCol="pca").fit(vec)
        res = pca.transform(vec).select(
            "item_idx", vector_to_array("pca").cast("array<float>").alias("genome")
        )
        res.write.mode("overwrite").parquet(str(out / "genome_vectors.parquet"))

    # ---- 4. cold-start lists + popularity hard negatives ------------------------------------
    with timed(log, "cold lists + negatives"):
        c = cfg["coldstart"]
        t3 = b["t3"]
        ic = items.join(
            pos.filter(F.col("ts") < t3).groupBy("item_idx").agg(F.count(F.lit(1)).alias("n_pos")),
            "item_idx",
            "left",
        ).na.fill(0)
        ic.filter(F.col("n_pos") < c["cold_item_min_interactions"]).write.mode("overwrite").parquet(
            str(out / "cold_items.parquet")
        )
        umap = spark.read.parquet(str(proc / "user_map.parquet")).select("user_idx")
        uc = umap.join(
            pos.filter(F.col("ts") < t3).groupBy("user_idx").agg(F.count(F.lit(1)).alias("n_pos")),
            "user_idx",
            "left",
        ).na.fill(0)
        uc.filter(F.col("n_pos") < c["min_history"]).write.mode("overwrite").parquet(
            str(out / "cold_users.parquet")
        )

        t1 = b["t1"]
        pool = cfg["spark"]["pop_negatives_pool"]
        top = (
            pos.filter(F.col("ts") < t1)
            .groupBy("item_idx")
            .agg(F.count(F.lit(1)).alias("c"))
            .orderBy(F.desc("c"))
            .limit(pool)
            .select("item_idx")
        )
        rank_users = spark.read.parquet(str(out / "user_state_rank.parquet")).select("user_idx")
        seen = (
            inter.filter(F.col("ts") < t1)
            .select("user_idx", "item_idx")
            .withColumn("seen", F.lit(1))
        )
        cand = (
            rank_users.crossJoin(top)
            .join(seen, ["user_idx", "item_idx"], "left")
            .filter(F.col("seen").isNull())
            .select("user_idx", "item_idx")
        )
        wr = Window.partitionBy("user_idx").orderBy(F.rand(cfg["data"]["seed"]))
        (
            cand.withColumn("rn", F.row_number().over(wr))
            .filter(F.col("rn") <= cfg["spark"]["pop_negatives_per_user"])
            .select("user_idx", "item_idx")
            .write.mode("overwrite").parquet(str(out / "pop_negatives_rank.parquet"))
        )  # fmt: skip
    log.info("Spark features written to %s", out)
    return out
