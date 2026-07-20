"""S3-backed transcript source for claude-sql.

claude-sql's default source-of-truth is the local JSONL glob under
``~/.claude/projects/`` read zero-copy by DuckDB ``read_json``. This module
lets the same views read transcripts that live in S3 instead — e.g. sessions
mirrored there by the ``claude-agent-sdk`` ``S3SessionStore`` adapter, whose
key layout is::

    s3://{bucket}/{prefix}{project_key}/{session_id}/part-{epochMs13}-{rand6}.jsonl

The integration is deliberately thin: ``read_json`` already accepts an
``s3://`` URI as its glob argument, so once the DuckDB ``httpfs`` extension is
loaded and an S3 secret is configured on the connection, the entire existing
view/macro stack works unchanged against the remote corpus. No download step;
reads stay zero-copy over the wire with HTTP range requests.

Credentials come from DuckDB's ``credential_chain`` provider, which resolves
the standard AWS chain (env vars, shared config, instance/role profiles) —
the same chain boto3 uses elsewhere in claude-sql. No keys are ever embedded
in SQL. A custom ``endpoint`` / ``url_style`` / ``use_ssl`` triple is exposed
on :class:`~claude_sql.infrastructure.settings.Settings` for non-AWS S3 stores and for
pointing tests at a local mock server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import duckdb

    from claude_sql.infrastructure.settings import Settings

#: Name of the DuckDB secret claude-sql creates for S3 access. Stable so a
#: re-register (``CREATE OR REPLACE SECRET``) updates rather than duplicates.
#: This is a DuckDB catalog object name, not a credential — the actual keys
#: come from the credential_chain provider at query time.
S3_SECRET_NAME: str = "claude_sql_s3"  # noqa: S105 — DuckDB secret object name, not a password


def is_s3_uri(glob: str | None) -> bool:
    """Return ``True`` when ``glob`` is an ``s3://`` URI.

    Used to decide whether a connection needs :func:`configure_s3` before the
    raw readers bind. ``None`` and local paths return ``False``.
    """
    return bool(glob) and glob.startswith("s3://")


def settings_need_s3(settings: Settings) -> bool:
    """Return ``True`` when any of the three transcript globs points at S3.

    The subagent and meta globs are derived from the same root as the primary
    glob in every supported configuration, so in practice this tracks
    ``default_glob`` — but we check all three so a hand-pinned subagent glob on
    S3 still triggers httpfs setup.
    """
    return (
        is_s3_uri(settings.default_glob)
        or is_s3_uri(settings.subagent_glob)
        or is_s3_uri(settings.subagent_meta_glob)
    )


def _sql_str(value: str) -> str:
    """Escape a Python string as a single-quoted SQL literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def configure_s3(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Load ``httpfs`` and create the S3 secret so ``read_json`` can hit S3.

    Idempotent: ``INSTALL``/``LOAD`` are no-ops once satisfied and the secret
    uses ``CREATE OR REPLACE``. Safe to call on every connection open; callers
    gate it behind :func:`settings_need_s3` so local-only runs never pay the
    extension-load cost.

    The secret uses the ``credential_chain`` provider — credentials resolve
    from the standard AWS chain (env, shared config, instance/role profiles).
    No key material is embedded in the SQL. When ``settings.s3_endpoint`` is
    set, the endpoint / URL style / SSL flag are pinned for non-AWS stores and
    local mock servers.

    Parameters
    ----------
    con
        Open DuckDB connection.
    settings
        Active settings; supplies ``region`` and the optional S3 endpoint
        overrides.

    Raises
    ------
    duckdb.Error
        If the extension load or secret creation fails. Logged via
        ``logger.exception`` before re-raise so the caller sees the cause.
    """
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")

        clauses: list[str] = [
            "TYPE s3",
            "PROVIDER credential_chain",
            f"REGION {_sql_str(settings.region)}",
        ]
        if settings.s3_endpoint:
            clauses.append(f"ENDPOINT {_sql_str(settings.s3_endpoint)}")
            clauses.append(f"URL_STYLE {_sql_str(settings.s3_url_style)}")
            clauses.append(f"USE_SSL {'true' if settings.s3_use_ssl else 'false'}")

        body = ",\n                ".join(clauses)
        con.execute(
            f"""
            CREATE OR REPLACE SECRET {S3_SECRET_NAME} (
                {body}
            );
            """
        )
        logger.debug(
            "Configured S3 source: secret {} region {} endpoint {}",
            S3_SECRET_NAME,
            settings.region,
            settings.s3_endpoint or "(default AWS)",
        )
    except Exception:
        # register-or-fail-loud — an httpfs/secret failure must surface so the
        # caller doesn't proceed to a read that will fail with a cryptic
        # HTTP error instead of the real credential/extension cause.
        logger.exception("Failed to configure S3 transcript source")
        raise
