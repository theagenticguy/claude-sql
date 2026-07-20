"""DuckDB-backed :class:`~claude_sql.application.ports.TranscriptReaderPort`.

This is the importable retrieval seam downstream consumers use to pull one
session's transcript out of the local (or S3) corpus without
paying for the whole analytics stack. It registers ONLY the raw readers + the
derived business views — no VSS, no analytics parquets, no macros — so a reader
is cheap to open and never touches Bedrock or the embeddings store.

Two connection strategies live here:

* A **cached full connection** (lazy-opened, one glob scan over the whole
  corpus) backs :meth:`session_bounds` / :meth:`session_ids` and the
  slow-path fallback for :meth:`session_messages`.
* A **short-lived narrowed-glob connection** is opened per call for
  :meth:`session_messages` / :meth:`read_turn_text` when the requested
  session's on-disk file(s) can be isolated by narrowing the ``read_json``
  glob to that one session. This is the structural analogue of the ``peek``
  command's session-scoped materialization (``cli.py:1144``): the
  ``session_id`` predicate does NOT push below the ``content_blocks`` lateral
  ``UNNEST``, so scoping the *data* (a one-session glob) rather than the
  *predicate* is what keeps a single-session read off a full-corpus scan.

The pruned path is tried first (flat ``{project}/{sid}.jsonl`` then the
S3SessionStore part-dir ``{project}/{sid}/part-*.jsonl`` layout); a zero-match
glob raises ``duckdb.IOException`` at register time, which we treat as "not this
layout" and fall through — ultimately to the full-corpus connection filtered by
``WHERE session_id = ?``, which is always a correct superset.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from claude_sql.domain.transcript import (
    TranscriptRow,
    _timeline_sort_key,
    _TimelineRow,
    render_turn_text,
)
from claude_sql.infrastructure.duckdb_s3 import configure_s3, settings_need_s3
from claude_sql.infrastructure.duckdb_views import (
    _MAX_OBJECT_SIZE,
    _RAW_EVENT_COLUMNS,
    _render_columns_clause,
    _sql_str,
    register_raw,
    register_views,
)
from claude_sql.infrastructure.session_text_loader import (
    _iso_expr,
    _load_session_order,
    session_bounds as _load_session_bounds,
)
from claude_sql.infrastructure.settings import Settings

if TYPE_CHECKING:
    from datetime import datetime

#: Suffix of the primary transcript glob for the standard flat layout
#: (``{root}/projects/*/*.jsonl``). Session-scoped pruning rewrites this tail.
_FLAT_GLOB_SUFFIX = "/*.jsonl"

#: The four-field ``read_json`` projection the consumer collapse contract reads.
#: A rawer projection than the ``messages_text`` view (no 32-char HAVING floor,
#: no per-block UNNEST): the raw ``message`` JSON is folded message-at-a-time by
#: :func:`~claude_sql.domain.transcript.render_turn_text`, so short turns like
#: ``screenshot?`` render and the original ISO timestamp string is preserved
#: verbatim (carried as VARCHAR, no TIMESTAMP round-trip).
_COLLAPSE_COLUMNS: dict[str, str] = {
    "uuid": "VARCHAR",
    "sessionId": "VARCHAR",
    "type": "VARCHAR",
    "timestamp": "VARCHAR",
    "message": "JSON",
}


def _read_transcript_rows(
    con: duckdb.DuckDBPyConnection, glob: str, *, session_id: str | None = None
) -> list[TranscriptRow]:
    """Read raw message-envelope rows for the consumer collapse contract.

    Projects the four :data:`_COLLAPSE_COLUMNS` fields via ``read_json`` over
    ``glob``. A zero-match glob raises ``duckdb.IOException`` ("No files found");
    that is the empty-session signal, mapped to ``[]``. When ``session_id`` is
    given (the full-corpus fallback), rows are filtered to that session's
    ``sessionId`` field — a session-scoped glob needs no filter.
    """
    # ``cols`` is the trusted column-projection constant and the glob path is
    # $-bound below, so the interpolation here is not user-controlled SQL.
    cols = _render_columns_clause(_COLLAPSE_COLUMNS)
    sql = (
        "SELECT uuid, type, timestamp, message, sessionId "
        "FROM read_json(?, format='newline_delimited', union_by_name=true, "
        f"ignore_errors=true, columns={{{cols}}}, maximum_object_size={_MAX_OBJECT_SIZE})"
    )
    try:
        raw = con.execute(sql, [glob]).fetchall()
    except duckdb.IOException as exc:
        if "No files found" in str(exc):
            return []
        raise
    rows: list[TranscriptRow] = []
    for uuid, type_, timestamp, message, sid in raw:
        if session_id is not None and str(sid) != session_id:
            continue
        rows.append(TranscriptRow(uuid=uuid, type=type_, timestamp=timestamp, message=message))
    return rows


def _local_glob_has_match(glob: str) -> bool:
    """Return ``True`` when a local (non-S3) glob matches at least one file.

    ``read_json`` raises ``duckdb.IOException`` on a zero-match glob, so the
    reader probes the subagent globs up front and substitutes a known-good glob
    when a corpus simply has no subagent sidecars (the timeline views never read
    the subagent tables, so any real file that lets ``read_json`` infer a schema
    is a correct substitute). S3 globs can't be probed with ``pathlib``; those
    are handled on the S3 path directly, so this treats them as "assume present".
    """
    if glob.startswith("s3://"):
        return True
    # A glob is an absolute-ish pattern; split the first wildcard-free prefix off
    # and let pathlib walk the remainder.
    p = Path(glob)
    anchor = p.anchor or "/"
    parts = p.relative_to(anchor).parts
    base = Path(anchor)
    pattern_parts: list[str] = []
    for i, part in enumerate(parts):
        if any(ch in part for ch in "*?["):
            pattern_parts = list(parts[i:])
            break
        base = base / part
    else:
        # No wildcard at all — a literal path.
        return base.exists()
    return any(base.glob("/".join(pattern_parts)))


@dataclass(frozen=True, slots=True)
class TranscriptRef:
    """Address of one transcript living in an ``S3SessionStore`` bucket.

    Mirrors the key layout documented in
    :mod:`claude_sql.infrastructure.duckdb_s3`::

        s3://{bucket}/{prefix}{project_key}/{session_id}/part-{epochMs}-{rand}.jsonl

    ``prefix`` is expected to carry its own trailing slash (or be empty), matching
    the store's own convention, so ``{prefix}{project_key}`` concatenates cleanly.
    """

    bucket: str
    prefix: str
    project_key: str
    session_id: str


def _load_session_rows(con: duckdb.DuckDBPyConnection, session_id: str) -> list[_TimelineRow]:
    """Return one session's chronological timeline rows from a bound connection.

    Queries the three v1 timeline views (``messages_text`` / ``tool_calls`` /
    ``tool_results``) filtered to ``session_id`` and merges them into one
    :func:`~claude_sql.domain.transcript._timeline_sort_key`-ordered list. On a
    session-narrowed connection the ``WHERE`` is exact and cheap; on the
    full-corpus fallback it is a correct (if broader) scan.
    """
    rows: list[_TimelineRow] = []

    # Read text turns from ``content_blocks`` aggregated per message, NOT from
    # ``messages_text`` — the latter carries a 32-char HAVING floor for the
    # analytics/embeddings path, which would drop short turns like
    # "screenshot?" from this reader seam. Aggregating the raw text blocks here
    # (no floor) is the reader's own no-floor projection (DELTA-2); analytics
    # keep the floored ``messages_text`` view untouched.
    text_sql = f"""
        SELECT CASE WHEN cb.ts IS NULL THEN '' ELSE {_iso_expr("cb.ts")} END AS ts_iso,
               any_value(cb.role) AS role,
               string_agg(cb.text, '\n\n') AS text_content,
               CAST(cb.message_uuid AS VARCHAR) AS uuid
          FROM content_blocks cb
         WHERE CAST(cb.session_id AS VARCHAR) = ?
           AND cb.block_type = 'text'
           AND cb.text IS NOT NULL
           AND length(cb.text) > 0
           AND cb.message_type != 'attachment'
         GROUP BY cb.message_uuid, cb.ts
         ORDER BY cb.ts
    """
    for ts_iso, role, body, uuid in con.execute(text_sql, [session_id]).fetchall():
        rows.append(
            _TimelineRow(
                ts_iso=ts_iso, role=role or "user", kind="text", body=body, aux=None, uuid=uuid
            )
        )

    tool_use_sql = f"""
        SELECT CASE WHEN tc.ts IS NULL THEN '' ELSE {_iso_expr("tc.ts")} END AS ts_iso,
               tc.tool_name,
               CAST(tc.tool_input AS VARCHAR) AS tool_input_json
          FROM tool_calls tc
         WHERE CAST(tc.session_id AS VARCHAR) = ?
         ORDER BY tc.ts
    """
    for ts_iso, tool_name, tool_input_json in con.execute(tool_use_sql, [session_id]).fetchall():
        rows.append(
            _TimelineRow(
                ts_iso=ts_iso, role="tool", kind="tool_use", body=tool_input_json, aux=tool_name
            )
        )

    tool_result_sql = f"""
        SELECT CASE WHEN tr.ts IS NULL THEN '' ELSE {_iso_expr("tr.ts")} END AS ts_iso,
               tr.tool_use_id,
               CAST(tr.content AS VARCHAR) AS content_json
          FROM tool_results tr
         WHERE CAST(tr.session_id AS VARCHAR) = ?
         ORDER BY tr.ts
    """
    for ts_iso, tool_use_id, content_json in con.execute(tool_result_sql, [session_id]).fetchall():
        rows.append(
            _TimelineRow(
                ts_iso=ts_iso, role="tool", kind="tool_result", body=content_json, aux=tool_use_id
            )
        )

    rows.sort(key=_timeline_sort_key)
    return rows


def _row_to_dict(row: _TimelineRow) -> dict[str, Any]:
    """Project a timeline row into the plain dict shape the port returns."""
    return {
        "uuid": row.uuid,
        "ts": row.ts_iso,
        "role": row.role,
        "kind": row.kind,
        "body": row.body,
        "aux": row.aux,
    }


class DuckDbTranscriptReader:
    """Adapter: read assembled transcript text + session structure via DuckDB.

    Implements :class:`~claude_sql.application.ports.TranscriptReaderPort`. The
    ``read_json`` glob triple is adapter state, derived either from ``settings``
    or from an explicit ``config_root`` override.
    """

    def __init__(self, settings: Settings | None = None, *, config_root: str | Path | None = None):
        """Bind the transcript globs.

        Parameters
        ----------
        settings
            Active :class:`Settings`; a default instance is created when absent.
            Supplies the S3 region/endpoint and (when ``config_root`` is unset)
            the three transcript globs.
        config_root
            When given, overrides the transcript root: the three globs are
            derived as ``{root}/projects/*/*.jsonl`` (and the ``subagents/``
            siblings), mirroring
            :func:`claude_sql.infrastructure.settings._claude_config_root`. Accepts either a
            plain root (``/data/alice``) or a root that itself contains a ``*``
            wildcard for multi-config-dir setups
            (``~/agent-fleet/config-dirs/*``) — the extra glob level simply
            widens the scan.
        """
        self._settings = settings or Settings()
        if config_root is not None:
            root = str(Path(config_root).expanduser())
            self._glob = f"{root}/projects/*/*.jsonl"
            self._subagent_glob = f"{root}/projects/*/*/subagents/agent-*.jsonl"
            self._subagent_meta_glob = f"{root}/projects/*/*/subagents/agent-*.meta.json"
        else:
            self._glob = self._settings.default_glob
            self._subagent_glob = self._settings.subagent_glob
            self._subagent_meta_glob = self._settings.subagent_meta_glob
        self._con: duckdb.DuckDBPyConnection | None = None
        self._stub_tmpdir: tempfile.TemporaryDirectory[str] | None = None

    # -- connection lifecycle ------------------------------------------------

    def _subagent_stub_dir(self) -> Path:
        """Return a cached temp dir holding a subagent + meta stub.

        A corpus can legitimately have no subagent sidecars, and ``read_json``
        raises ``duckdb.IOException`` on a zero-match glob at register time. The
        timeline views this reader uses never read the subagent tables, but the
        derived ``subagent_*`` views expect the subagent JSONL / ``meta.json``
        SCHEMA (columns like ``agentType`` / ``description``) — so we can't just
        repoint the subagent glob at the transcript files. Instead we seed a
        one-file stub with the right shape (mirroring the test-suite's own
        ``_seed_subagent_stub``) and point the subagent globs at it. Cached for
        the reader's lifetime; the ``TemporaryDirectory`` is retained on the
        instance so it isn't GC'd out from under an open connection.
        """
        stub = self._stub_tmpdir
        if stub is None:
            stub = tempfile.TemporaryDirectory(prefix="claude_sql_reader_stub_")
            self._stub_tmpdir = stub
            sa_dir = (
                Path(stub.name)
                / "projects"
                / "proj-stub"
                / "00000000-0000-0000-0000-000000000000"
                / "subagents"
            )
            sa_dir.mkdir(parents=True, exist_ok=True)
            (sa_dir / "agent-placeholder.jsonl").write_text(
                json.dumps(
                    {
                        "uuid": "sa-stub",
                        "timestamp": "2026-01-01T00:00:00.000Z",
                        "sessionId": "placeholder",
                        "type": "user",
                        "message": {"role": "user", "content": [{"type": "text", "text": "stub"}]},
                    }
                )
                + "\n"
            )
            (sa_dir / "agent-placeholder.meta.json").write_text(
                json.dumps({"agentType": "stub", "description": "reader stub subagent"})
            )
        return Path(stub.name)

    def _part_dir_glob(self) -> str | None:
        """Return the S3SessionStore part-dir glob for the standard flat layout.

        ``{root}/projects/*/*.jsonl`` → ``{root}/projects/*/*/part-*.jsonl``.
        ``None`` when the primary glob is non-standard (e.g. an ``s3://`` glob),
        in which case there is no local part-dir layer to fold in.
        """
        if not self._glob.endswith(_FLAT_GLOB_SUFFIX) or self._glob.startswith("s3://"):
            return None
        stem = self._glob[: -len(_FLAT_GLOB_SUFFIX)]
        return f"{stem}/*/part-*.jsonl"

    def _ingest_part_dir_events(self, con: duckdb.DuckDBPyConnection, part_glob: str) -> None:
        """Supplement ``v_raw_events`` with the part-dir layout's sessions.

        ``register_raw`` binds ``v_raw_events`` over a SINGLE glob, so the flat
        ``{sid}.jsonl`` scan never sees ``{sid}/part-*.jsonl`` sessions (a brace
        glob isn't supported and a glob LIST raises when the part member matches
        nothing). This appends the part-dir rows via ``INSERT ... BY NAME``,
        recomputing ``source_file`` / ``session_id_file`` exactly as
        ``register_raw`` does, so ``sessions`` / enumeration see both layouts
        (DELTA-5). No-op when the part glob matches nothing.
        """
        if not _local_glob_has_match(part_glob):
            return
        cols = _render_columns_clause(_RAW_EVENT_COLUMNS)
        con.execute(
            f"""
            INSERT INTO v_raw_events BY NAME
            SELECT *,
                   filename AS source_file,
                   regexp_extract(filename, '/([^/]+)/part-[^/]*\\.jsonl$', 1) AS session_id_file
            FROM read_json(
                {_sql_str(part_glob)},
                format='newline_delimited',
                union_by_name=true,
                filename=true,
                ignore_errors=true,
                columns={{{cols}}},
                maximum_object_size={_MAX_OBJECT_SIZE}
            );
            """
        )

    def _register(
        self, con: duckdb.DuckDBPyConnection, main_glob: str, *, with_part_dirs: bool = False
    ) -> None:
        """Register raw + derived views over ``main_glob`` on ``con``.

        Substitutes a private stub for the subagent / meta globs when the
        configured ones match no files, so a subagent-free corpus still binds
        the derived views (see :meth:`_subagent_stub_dir`). When
        ``with_part_dirs`` is set (the full-corpus enumeration connection), also
        folds the part-dir layout's sessions into ``v_raw_events`` before the
        derived views bind (DELTA-5).
        """
        if settings_need_s3(self._settings) or main_glob.startswith("s3://"):
            configure_s3(con, self._settings)
        sub_glob = self._subagent_glob
        sub_meta_glob = self._subagent_meta_glob
        if not _local_glob_has_match(sub_glob) or not _local_glob_has_match(sub_meta_glob):
            stub_root = self._subagent_stub_dir()
            sub_glob = str(stub_root / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl")
            sub_meta_glob = str(
                stub_root / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json"
            )
        register_raw(con, glob=main_glob, subagent_glob=sub_glob, subagent_meta_glob=sub_meta_glob)
        if with_part_dirs:
            part_glob = self._part_dir_glob()
            if part_glob is not None:
                self._ingest_part_dir_events(con, part_glob)
        register_views(con)

    def _full_connection(self) -> duckdb.DuckDBPyConnection:
        """Return the cached full-corpus connection, opening it on first use.

        Registers only the raw readers + derived views (no VSS / analytics /
        macros). The single glob scan is amortized across every
        :meth:`session_bounds` / :meth:`session_ids` / fallback call. Both
        on-disk layouts (flat ``{sid}.jsonl`` + part-dir ``{sid}/part-*.jsonl``)
        are folded in so enumeration sees every session (DELTA-5).
        """
        if self._con is None:
            con = duckdb.connect(":memory:")
            self._register(con, self._glob, with_part_dirs=True)
            self._con = con
        return self._con

    def _open_narrow(self, main_glob: str) -> duckdb.DuckDBPyConnection | None:
        """Open a short-lived connection scoped to ``main_glob``, or ``None``.

        Returns ``None`` when the narrowed glob matches no files on disk
        (``read_json`` raises ``duckdb.IOException`` at register time), signaling
        the caller to try the next layout / fall back.
        """
        if not _local_glob_has_match(main_glob):
            return None
        con = duckdb.connect(":memory:")
        try:
            self._register(con, main_glob)
        except duckdb.IOException:
            con.close()
            return None
        return con

    def _narrow_globs(self, session_id: str) -> list[str]:
        """Return the candidate session-scoped globs for the standard layout.

        Flat ``{root}/projects/*/{sid}.jsonl`` first, then the S3SessionStore
        part-dir ``{root}/projects/*/{sid}/part-*.jsonl``. Empty when the
        primary glob is non-standard (e.g. an ``s3://`` part glob), in which
        case the caller uses the full-corpus fallback directly.
        """
        if not self._glob.endswith(_FLAT_GLOB_SUFFIX):
            return []
        stem = self._glob[: -len(_FLAT_GLOB_SUFFIX)]
        return [f"{stem}/{session_id}.jsonl", f"{stem}/{session_id}/part-*.jsonl"]

    def _local_session_rows(self, session_id: str) -> list[_TimelineRow]:
        """Load one local session's rows via the pruned path, else the fallback."""
        for narrow in self._narrow_globs(session_id):
            con = self._open_narrow(narrow)
            if con is None:
                continue
            try:
                rows = _load_session_rows(con, session_id)
            finally:
                con.close()
            if rows:
                return rows
        # Fallback: full-corpus connection filtered by session_id.
        return _load_session_rows(self._full_connection(), session_id)

    def _local_transcript_rows(self, session_id: str) -> list[TranscriptRow]:
        """Load one local session's raw message rows for the collapse.

        Reads the rawer four-field projection (no 32-char floor, raw timestamp
        string, message JSON folded per-message) via the same flat→part-dir
        pruned-glob discovery ``session_messages`` uses. Falls back to a
        session-filtered scan over the primary glob when no narrow glob matches.
        """
        for narrow in self._narrow_globs(session_id):
            if not _local_glob_has_match(narrow):
                continue
            con = duckdb.connect(":memory:")
            try:
                rows = _read_transcript_rows(con, narrow)
            finally:
                con.close()
            if rows:
                return rows
        # Fallback: scan the primary glob, filtered to this session id.
        con = duckdb.connect(":memory:")
        try:
            if settings_need_s3(self._settings) or self._glob.startswith("s3://"):
                configure_s3(con, self._settings)
            return _read_transcript_rows(con, self._glob, session_id=session_id)
        finally:
            con.close()

    def _s3_transcript_rows(self, ref: TranscriptRef) -> list[TranscriptRow]:
        """Load one S3 session's raw message rows over an httpfs connection."""
        base = f"s3://{ref.bucket}/{ref.prefix}{ref.project_key}/{ref.session_id}"
        main_glob = f"{base}/part-*.jsonl"
        con = duckdb.connect(":memory:")
        try:
            configure_s3(con, self._settings)
            return _read_transcript_rows(con, main_glob)
        finally:
            con.close()

    def close(self) -> None:
        """Close the cached full-corpus connection + clean up the subagent stub."""
        if self._con is not None:
            self._con.close()
            self._con = None
        if self._stub_tmpdir is not None:
            self._stub_tmpdir.cleanup()
            self._stub_tmpdir = None

    # -- TranscriptReaderPort ------------------------------------------------

    def session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return the ordered message rows for one session (chronological).

        Each row is ``{uuid, ts, role, kind, body, aux}`` where ``kind`` is one
        of ``text`` / ``tool_use`` / ``tool_result`` and ``aux`` carries the
        tool name (``tool_use``) or tool_use_id (``tool_result``).
        """
        return [_row_to_dict(r) for r in self._local_session_rows(session_id)]

    def read_turn_text(self, ref: str | TranscriptRef) -> str:
        """Return the consumer collapse transcript text for one session.

        ``ref`` is either a bare local ``session_id`` or a :class:`TranscriptRef`
        addressing an S3-mirrored session. Reads the rawer four-field projection
        (bypassing the ``messages_text`` 32-char floor so short turns render) and
        delegates to the pure
        :func:`~claude_sql.domain.transcript.render_turn_text` — one line per
        message, inline ``[tool_use:name]`` / ``[tool_result]`` markers, the raw
        timestamp string, ``(ts, kind_rank, uuid)`` ordering, and collapse
        truncation (per-turn `` …``, total hard-slice).
        """
        rows = (
            self._s3_transcript_rows(ref)
            if isinstance(ref, TranscriptRef)
            else self._local_transcript_rows(ref)
        )
        return render_turn_text(rows)

    def session_bounds(
        self, *, since_days: int | None = None, limit: int | None = None
    ) -> dict[str, tuple[datetime | None, datetime | None]]:
        """Return ``{session_id: (last_ts, transcript_mtime)}`` for the window."""
        return _load_session_bounds(self._full_connection(), since_days=since_days, limit=limit)

    def session_ids(self, *, since_days: int | None = None, limit: int | None = None) -> list[str]:
        """Return the newest-first session ids matching the window."""
        return _load_session_order(self._full_connection(), since_days=since_days, limit=limit)


__all__ = [
    "DuckDbTranscriptReader",
    "TranscriptRef",
]
