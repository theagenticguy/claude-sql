"""Cross-use-case DuckDB helpers shared by the LLM-analytics pipelines.

``_count_pending_sessions`` is the pure-SQL dry-run counter used by the
classify + conflicts use-cases for ``--dry-run`` cost estimation. It moved
here from ``core/llm_shared.py`` in the v2 hexagonal final cut (T-5-2): it is a
DuckDB query that does not belong in the Bedrock transport adapter, and both
use-cases share it, so the application layer is its home.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb


def _count_pending_sessions(
    con: duckdb.DuckDBPyConnection,
    *,
    already: set[str],
    since_days: int | None,
    limit: int | None,
) -> int:
    """Return the count of sessions that have text messages but no classification yet.

    Pure SQL — does NOT materialize any session text.  This is the fast path for
    ``--dry-run`` cost estimation against the full corpus (the previous path
    iterated :func:`iter_session_texts`, which took ~15 min on 6K+ sessions).
    """
    where = ["mt.text_content IS NOT NULL", "length(mt.text_content) >= 1"]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    sql = f"""
        SELECT count(DISTINCT CAST(mt.session_id AS VARCHAR))
          FROM messages_text mt
         WHERE {" AND ".join(where)}
    """
    row = con.execute(sql).fetchone()
    total = int(row[0]) if row is not None else 0
    if already:
        # Subtract sessions that already have a classification.  We pull only
        # the overlap via a parameterized IN so we don't double-count sessions
        # in ``already`` that aren't actually in the corpus anymore.
        placeholders = ",".join("?" for _ in already)
        overlap_sql = f"""
            SELECT count(DISTINCT CAST(mt.session_id AS VARCHAR))
              FROM messages_text mt
             WHERE {" AND ".join(where)}
               AND CAST(mt.session_id AS VARCHAR) IN ({placeholders})
        """
        overlap_row = con.execute(overlap_sql, list(already)).fetchone()
        overlap = int(overlap_row[0]) if overlap_row is not None else 0
        total = max(0, total - overlap)
    if limit is not None:
        total = min(total, int(limit))
    return total
