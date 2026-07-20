"""DuckDB + Lance-backed :class:`~claude_sql.application.ports.SessionSearchPort`.

The importable semantic-search seam: embed a query with the injected
:class:`~claude_sql.domain.ports.EmbeddingProvider`, then run the same
inline cosine-kNN SQL the ``search`` CLI command uses (``cli.py:1814``) against
the Lance-backed ``message_embeddings`` view — with an optional
``session_id`` filter the CLI does not expose.

Registration is minimal: raw readers + derived views + VSS (and S3 config when a
transcript glob points at S3). Analytics parquets and macros are NEVER
registered — search does not need them and they add cold-start cost plus a
fragile dependency on caches that may not exist.

Library semantics differ from the CLI on one point: an empty embeddings store
returns ``[]`` rather than exiting. Callers that want the CLI's exit-code-2
behavior keep that in the CLI. The provider/dimension fail-loud guard
(``ensure_store_matches``) still fires through ``register_vss`` — a store
written by a different embedder raises rather than returning garbage scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
from loguru import logger

from claude_sql.domain.retrieval import SearchHit
from claude_sql.infrastructure.duckdb_s3 import configure_s3, settings_need_s3
from claude_sql.infrastructure.duckdb_views import register_raw, register_views, register_vss
from claude_sql.infrastructure.embedding import build_embedder
from claude_sql.infrastructure.settings import Settings

if TYPE_CHECKING:
    from claude_sql.domain.ports import EmbeddingProvider


class DuckDbSessionSearch:
    """Adapter: semantic top-k search over message embeddings via DuckDB + Lance.

    Implements :class:`~claude_sql.application.ports.SessionSearchPort`. The
    embedder and the bound ``message_embeddings`` view are adapter state, opened
    lazily on the first :meth:`search` / :meth:`embed_query` call.
    """

    def __init__(
        self, settings: Settings | None = None, *, embedder: EmbeddingProvider | None = None
    ):
        """Bind settings + embedder.

        Parameters
        ----------
        settings
            Active :class:`Settings`; a default instance is created when absent.
        embedder
            The :class:`EmbeddingProvider` used to embed queries. Defaults to
            :func:`claude_sql.infrastructure.embedding.build_embedder` over ``settings`` —
            constructed lazily so building a search adapter never imports the
            provider's heavy deps (botocore / httpx / fastembed) until a query
            actually runs.
        """
        self._settings = settings or Settings()
        self._embedder = embedder
        self._con: duckdb.DuckDBPyConnection | None = None

    def _get_embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = build_embedder(self._settings)
        return self._embedder

    def _expected_identity(self) -> tuple[str, int | None]:
        """Return the ``(model_id, dim)`` the store guard should enforce.

        When an embedder is injected it is the source of truth for the store's
        identity — its ``model_id`` / ``dimension`` gate ``register_vss``'s
        fail-loud guard, so a caller wiring a specific provider sees a mismatch
        against a store written by a different one. When no embedder is injected
        we fall back to ``Settings.expected_embedding_identity`` (the default
        provider path), which stays dependency-free — it does NOT build the
        embedder, so the search adapter can bind a connection without importing
        botocore / httpx / fastembed until a query actually runs.
        """
        if self._embedder is not None:
            return (self._embedder.model_id, self._embedder.dimension)
        return self._settings.expected_embedding_identity()

    def _connection(self) -> duckdb.DuckDBPyConnection:
        """Return the cached connection, opening + minimally registering on first use.

        raw + views + vss ONLY. The VSS guard reads the store's stamped
        ``(model, dim)`` and fails loud on a provider mismatch.
        """
        if self._con is None:
            con = duckdb.connect(":memory:")
            if settings_need_s3(self._settings):
                configure_s3(con, self._settings)
            register_raw(
                con,
                glob=self._settings.default_glob,
                subagent_glob=self._settings.subagent_glob,
                subagent_meta_glob=self._settings.subagent_meta_glob,
            )
            register_views(con)
            expected_model, expected_dim = self._expected_identity()
            register_vss(
                con,
                embeddings_parquet=self._settings.embeddings_parquet_path,
                lance_uri=self._settings.lance_uri,
                dim=int(self._settings.output_dimension),
                metric=self._settings.hnsw_metric,
                expected_model=expected_model,
                expected_dim=expected_dim,
            )
            self._con = con
        return self._con

    def close(self) -> None:
        """Close the cached connection if one was opened."""
        if self._con is not None:
            self._con.close()
            self._con = None

    # -- SessionSearchPort ---------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Return the query embedding vector (always float; the adapter's width)."""
        return self._get_embedder().embed_query(text)

    def search(self, query: str, *, k: int = 10, session_id: str | None = None) -> list[SearchHit]:
        """Return the top-``k`` nearest neighbors to ``query`` as typed hits.

        When ``session_id`` is given the cosine-kNN is confined to that session
        (a ``WHERE mt.session_id = ?`` on the join). An empty embeddings store
        returns ``[]``.
        """
        con = self._connection()
        row = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
        if not row or int(row[0]) == 0:
            logger.debug("session_search: empty embeddings store — returning no hits")
            return []

        qv = self.embed_query(query)
        # Cast to the query vector's own width, not settings.output_dimension:
        # local providers (ollama/onnx) emit a fixed native width unrelated to
        # the Cohere Matryoshka knob, and the store view is bound at the stored
        # width (see register_vss / the search CLI command).
        dim = len(qv)
        params: list[object] = [qv]
        session_filter = ""
        if session_id is not None:
            session_filter = "WHERE CAST(mt.session_id AS VARCHAR) = ?"
            params.append(session_id)
        params.append(k)
        # Rank by cosine similarity descending.  The HNSW index was built with
        # metric='cosine', so ORDER BY array_cosine_distance (== 1 - sim) ASC
        # is what triggers the index lookup.  Using array_distance here (L2)
        # would silently bypass the index AND give wrong ranks because the raw
        # int8-cast-to-float document vectors have magnitudes in the thousands
        # while the query vector is unit-normalized.
        sql = f"""
            WITH qv AS (SELECT CAST(? AS FLOAT[{dim}]) AS v)
            SELECT CAST(mt.uuid AS VARCHAR)        AS uuid,
                   CAST(mt.session_id AS VARCHAR)  AS session_id,
                   mt.ts                           AS ts,
                   mt.role                         AS role,
                   substr(mt.text_content, 1, 200) AS snippet,
                   array_cosine_similarity(me.embedding, (SELECT v FROM qv)) AS cosine_sim
            FROM message_embeddings me
            JOIN messages_text mt ON CAST(mt.uuid AS VARCHAR) = me.uuid
            {session_filter}
            ORDER BY array_cosine_distance(me.embedding, (SELECT v FROM qv)) ASC
            LIMIT ?
        """
        rows = con.execute(sql, params).fetchall()
        return [
            SearchHit(
                uuid=uuid,
                session_id=sid,
                ts=ts,
                role=role,
                snippet=snippet or "",
                cosine_sim=float(cosine_sim),
            )
            for uuid, sid, ts, role, snippet, cosine_sim in rows
        ]


__all__ = ["DuckDbSessionSearch"]
