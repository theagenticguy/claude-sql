"""c-TF-IDF per cluster -- in-house (no bertopic).

Orchestration for the ``terms`` command (MIGRATION Phase C / T-3-1). This
module owns the I/O: read ``clusters.parquet`` joined to the ``messages_text``
view, build one pseudo-document per cluster, and write the top-N terms to
``cluster_terms.parquet``. The pure ``CountVectorizer`` + c-TF-IDF weighting
math lives in :func:`claude_sql.domain.structure.terms.compute_ctfidf`.

Public API
----------
run_terms(con, settings, *, force=False) -> dict[str, int]
    Compute ``cluster_terms.parquet``.  Returns ``{"clusters": K, "terms": N}``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from claude_sql.domain.structure.terms import compute_ctfidf
from claude_sql.infrastructure.freshness import (
    is_output_fresh,
    newest_mtime_ns,
    sidecar_for,
    stamp_output,
)

if TYPE_CHECKING:
    import duckdb

    from claude_sql.infrastructure.settings import Settings


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

    # Freshness keys on the CLUSTERS parquet's mtime, not on this output's mere
    # existence. HDBSCAN re-mints ``cluster_id`` on every fit, so terms computed
    # against an earlier clustering label a partition that no longer exists —
    # an existence gate makes that state permanent and silent.
    sidecar = sidecar_for(out, input_name="clusters")
    clusters_mtime_ns = newest_mtime_ns(clusters_path)
    if not force and is_output_fresh(out, sidecar=sidecar, input_mtime_ns=clusters_mtime_ns):
        df = pl.read_parquet(out)
        logger.info("Clusters unchanged since last terms run; reusing {}.", out)
        return {"clusters": int(df["cluster_id"].n_unique()), "terms": len(df)}

    # Project the god-Settings down to the c-TF-IDF slice (T-2-4): the vectorizer
    # + weighting math reads only ``cfg`` — no Bedrock model ID or glob.
    cfg = settings.terms_config()

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
    docs_by_class = list(
        zip(per_cluster["cluster_id"].to_list(), per_cluster["doc"].to_list(), strict=True)
    )
    logger.info("Built {} cluster pseudo-docs", len(docs_by_class))

    rows = compute_ctfidf(docs_by_class, cfg)

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
    stamp_output(sidecar, clusters_mtime_ns)
    logger.info(
        "Wrote {} term-rows across {} clusters in {:.1f}s",
        len(out_df),
        len(docs_by_class),
        time.monotonic() - t0,
    )
    return {"clusters": len(docs_by_class), "terms": len(out_df)}


__all__ = ["run_terms"]
