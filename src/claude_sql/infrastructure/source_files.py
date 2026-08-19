"""Filesystem watermark scanning over the transcript globs.

The raw readers (``v_raw_events`` / ``v_raw_subagents``) are DuckDB TEMP TABLEs
materialized from a glob, so advancing them incrementally needs to know which
source files moved. This module is the one place that stats the corpus:
:func:`scan_glob_mtimes` expands a glob and returns ``{path: mtime_ns}``, which
:func:`claude_sql.domain.watch.diff_source_mtimes` turns into a delta.

Cost: measured 88-101 ms for 6,744 files on the live interactive corpus, so a
scan is cheap enough to run per refresh and far cheaper than the ~6.4 s full
re-materialization it replaces.

**Remote globs are not scannable and say so.** An ``s3://`` glob has no local
``stat``, and issuing thousands of HeadObject calls to synthesize one would be
both slow and a lie about what the numbers mean. :func:`scan_globs` returns
``None`` for a remote corpus, and every caller treats ``None`` as "watermark
unobservable — fall back to a full rebuild" rather than as an empty scan (which
would read as "the corpus has no files" and delete every row).
"""

from __future__ import annotations

import glob as globlib
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from claude_sql.infrastructure.duckdb_s3 import is_s3_uri

if TYPE_CHECKING:
    from collections.abc import Iterable


def scan_glob_mtimes(pattern: str) -> dict[str, int]:
    """Expand one local glob and return ``{absolute_path: mtime_ns}``.

    ``mtime_ns`` is ``os.stat().st_mtime_ns`` — filesystem modification time in
    **epoch nanoseconds**, the same space DuckDB's ``read_json`` reads the file
    content from. It is NOT the monotonic clock the debounce uses.

    Files that vanish between the glob expansion and the ``stat`` are skipped
    rather than raised: an in-flight session can be renamed or rotated mid-scan,
    and a scan is a best-effort observation, not a transaction.

    Directories matching the pattern are skipped — only regular files carry
    transcript rows, and a directory mtime moves whenever any child changes,
    which would mark the whole tree dirty on every write.
    """
    found: dict[str, int] = {}
    # ``glob.iglob`` over ``Path.glob``: the patterns are absolute strings with
    # wildcards in interior segments, which ``Path.glob`` cannot take without
    # first being split into an anchor plus a relative pattern. ``os.stat`` over
    # ``Path.stat`` for the same reason the mode check reuses that one stat
    # result — this loop runs 14k times per sweep and every avoided object and
    # syscall shows up in the measured 88-101 ms.
    for path in globlib.iglob(pattern, recursive=True):  # noqa: PTH207 — interior-wildcard absolute pattern
        try:
            st = os.stat(path)  # noqa: PTH116 — one stat serves both the mtime and the mode check
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        found[path] = st.st_mtime_ns
    return found


def scan_globs(patterns: Iterable[str]) -> dict[str, int] | None:
    """Scan several globs into one ``{path: mtime_ns}`` map.

    Returns
    -------
    dict[str, int] | None
        The merged watermark, or ``None`` when ANY pattern is an ``s3://`` URI.
        ``None`` means "unobservable", never "empty": a caller that treated it
        as empty would compute a delta claiming every known file was removed.

    A pattern that matches nothing contributes no entries — that is a genuine
    empty result and is distinct from ``None``.
    """
    merged: dict[str, int] = {}
    for pattern in patterns:
        if is_s3_uri(pattern):
            logger.debug("watermark scan skipped: {} is remote", pattern)
            return None
        merged.update(scan_glob_mtimes(pattern))
    return merged


#: Characters that make a path component a glob pattern rather than a literal.
_WILDCARD_CHARS: str = "*?["


def glob_root(pattern: str) -> Path | None:
    """The deepest wildcard-free ancestor of ``pattern`` — a watchable directory.

    ``/home/u/.claude/projects/*/*.jsonl`` → ``/home/u/.claude/projects``. An OS
    watcher subscribes to directories, not patterns, and watches them
    recursively, so one root covers both the primary transcripts and the
    subagent sidecars nested under them.

    Returns ``None`` for a remote pattern or when the resolved root does not
    exist — a caller cannot watch either, and returning a plausible-looking
    path would produce a watcher that silently observes nothing.
    """
    if is_s3_uri(pattern):
        return None
    parts = Path(pattern).parts
    literal: list[str] = []
    for part in parts:
        if any(char in part for char in _WILDCARD_CHARS):
            break
        literal.append(part)
    if not literal:
        return None
    root = Path(*literal)
    return root if root.is_dir() else None


def glob_roots(patterns: Iterable[str]) -> tuple[Path, ...]:
    """Watchable roots for ``patterns``, de-duplicated and nesting-collapsed.

    A root that is already covered by a shallower root in the same set is
    dropped: OS watchers recurse, so subscribing to both a directory and its
    descendant would deliver every event under the descendant twice.
    """
    roots = {root for root in (glob_root(pattern) for pattern in patterns) if root is not None}
    minimal = {
        root
        for root in roots
        if not any(other != root and other in root.parents for other in roots)
    }
    return tuple(sorted(minimal))


__all__ = ["glob_root", "glob_roots", "scan_glob_mtimes", "scan_globs"]
