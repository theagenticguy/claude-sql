"""Pure value-objects for the semantic-retrieval seam.

``SearchHit`` is the typed row a ``SessionSearchPort`` returns — the columns the
``search`` CLI path projects (uuid, session_id, role, cosine similarity,
snippet) plus the message timestamp, so a use-case gets a typed value instead of
a raw DataFrame row.

It lives in ``domain`` (not ``application.ports``) so the concrete DuckDB+Lance
adapter in ``infrastructure`` can construct it without importing *up* into the
application layer. ``application.ports`` re-exports it for callers that name the
whole port surface from one module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One semantic-search result row.

    Mirrors the columns the ``search`` CLI path already projects (uuid,
    session_id, role, cosine similarity, snippet) plus the message timestamp,
    so a use-case gets a typed value instead of a raw DataFrame row.
    """

    uuid: str
    session_id: str
    ts: datetime | None
    role: str
    snippet: str
    cosine_sim: float
