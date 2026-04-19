"""c-TF-IDF per cluster -- in-house (no bertopic).

Reads ``clusters.parquet`` (from cluster_worker) and the text messages
view to build one pseudo-document per cluster, then runs a sklearn
``CountVectorizer`` and computes the c-TF-IDF weights used by BERTopic.
Writes top-N terms per cluster to ``cluster_terms.parquet``.

Public API
----------
run_terms(con, settings, *, force=False) -> dict[str, int]
    Compute ``cluster_terms.parquet``.  Returns ``{"clusters": K, "terms": N}``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

if TYPE_CHECKING:
    import duckdb

    from claude_sql.config import Settings


def run_terms(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Compute c-TF-IDF top terms per cluster and write the parquet output.

    Parameters
    ----------
    con
        Open DuckDB connection with ``messages_text`` registered.
    settings
        Runtime settings (``clusters_parquet_path``, ``cluster_terms_parquet_path``,
        and the ``tfidf_*`` hyperparameters).
    force
        If True, recompute even when the output parquet already exists.

    Returns
    -------
    dict[str, int]
        ``{"clusters": K, "terms": N}`` where K is the number of clusters
        processed and N is the total row count written to parquet.
    """
    out = settings.cluster_terms_parquet_path
    clusters_path = settings.clusters_parquet_path

    if not clusters_path.exists() or clusters_path.stat().st_size < 16:
        raise FileNotFoundError(
            f"Clusters parquet missing at {clusters_path}.  Run `claude-sql cluster` first."
        )
    if out.exists() and out.stat().st_size > 16 and not force:
        df = pl.read_parquet(out)
        logger.info("cluster_terms parquet already exists at {}", out)
        return {"clusters": int(df["cluster_id"].n_unique()), "terms": len(df)}

    from sklearn.feature_extraction.text import CountVectorizer

    t0 = time.monotonic()
    # Join clusters parquet to messages_text on uuid.  ``mt.uuid`` is DuckDB
    # UUID; the parquet column is VARCHAR -- cast to match.
    sql = """
        SELECT c.cluster_id,
               mt.text_content
          FROM read_parquet(?) c
          JOIN messages_text mt
            ON CAST(mt.uuid AS VARCHAR) = c.uuid
         WHERE c.cluster_id >= 0
           AND mt.text_content IS NOT NULL
    """
    df = con.execute(sql, [str(clusters_path)]).pl()
    logger.info(
        "Joined {} rows clusters x messages_text in {:.1f}s",
        len(df),
        time.monotonic() - t0,
    )

    # Build one pseudo-document per cluster.
    per_cluster = (
        df.group_by("cluster_id")
        .agg(pl.col("text_content").str.join("\n").alias("doc"))
        .sort("cluster_id")
    )
    cluster_ids = per_cluster["cluster_id"].to_list()
    docs = per_cluster["doc"].to_list()
    logger.info("Built {} cluster pseudo-docs", len(docs))

    cv = CountVectorizer(
        min_df=settings.tfidf_min_df,
        max_df=settings.tfidf_max_df,
        ngram_range=(settings.tfidf_ngram_min, settings.tfidf_ngram_max),
        lowercase=True,
        strip_accents="unicode",
    ).fit(docs)
    tf = cv.transform(docs).toarray().astype(np.float32)  # (n_clusters, vocab)
    vocab = cv.get_feature_names_out()
    logger.info("Vocabulary size: {} terms", len(vocab))

    # c-TF-IDF: term frequency in cluster x log(1 + avg_docs_per_term / col_sum).
    row_sum = tf.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    tf_norm = tf / row_sum
    col_sum = tf.sum(axis=0)
    total = tf.sum()
    avg = col_sum / max(total, 1.0)
    idf = np.log(1.0 + (avg.sum() / np.maximum(col_sum, 1e-9)))
    ctfidf = tf_norm * idf  # (n_clusters, vocab)

    top_n = settings.tfidf_top_n_terms
    rows: list[tuple[int, str, float, int]] = []
    for k, cid in enumerate(cluster_ids):
        idx = np.argsort(-ctfidf[k])[:top_n]
        for rank, i in enumerate(idx):
            w = float(ctfidf[k, i])
            if w <= 0:
                continue
            rows.append((int(cid), str(vocab[i]), w, rank + 1))

    out_df = pl.DataFrame(
        rows,
        schema={
            "cluster_id": pl.Int32,
            "term": pl.Utf8,
            "weight": pl.Float32,
            "rank": pl.Int32,
        },
        orient="row",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out)
    logger.info(
        "Wrote {} term-rows across {} clusters in {:.1f}s",
        len(out_df),
        len(cluster_ids),
        time.monotonic() - t0,
    )
    return {"clusters": len(cluster_ids), "terms": len(out_df)}
