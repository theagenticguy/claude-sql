"""Shared Hypothesis strategies for the domain-layer property tests.

Every generator here is *valid-by-design*: it composes inputs the domain
functions accept without any ``assume``-heavy filtering, so shrinking stays
cheap and no example is silently discarded. The row generators build
:class:`~claude_sql.domain.transcript._TimelineRow` instances whose sort keys
are total (unique per-row uuids) so ``render_turn_text``'s ordering is a pure
function of the row *set*, not the input list order.
"""

from __future__ import annotations

from hypothesis import strategies as st

from claude_sql.domain.transcript import TranscriptRow, _TimelineRow

# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------

#: Printable ASCII with no control characters. Codepoint 10 (newline) is < 32
#: and therefore excluded, so a generated body / name / aux never smuggles a
#: line break into a single rendered line — which would break the per-line
#: reasoning the transcript properties rely on.
SAFE_TEXT = st.text(
    alphabet=st.characters(codec="ascii", min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=200,
)
SAFE_TEXT_NONEMPTY = st.text(
    alphabet=st.characters(codec="ascii", min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=40,
)
#: Non-blank text: printable ASCII with no space, so ``str.strip()`` (the
#: collapse strips each text block) never empties it. Used for transcript-row
#: bodies where an empty collapsed body would drop the row and break the
#: one-line-per-row reasoning the properties rely on.
SAFE_TEXT_NONBLANK = st.text(
    alphabet=st.characters(codec="ascii", min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
)

ROLES = st.sampled_from(["user", "assistant", "tool", "system"])
KINDS = st.sampled_from(["text", "tool_use", "tool_result"])

#: A small pool of ISO-8601 timestamps. Sampling from a small set makes
#: same-timestamp ties frequent, which is exactly the case the role-rank +
#: uuid tiebreak exists to resolve. Lexicographic order equals chronological
#: order for these fixed-width strings, matching the domain sort key.
_TS_POOL = [
    "2026-01-01T00:00:00",
    "2026-01-01T00:00:01",
    "2026-03-14T09:26:53",
    "2026-06-15T12:30:45",
    "2027-12-31T23:59:59",
]
TIMESTAMPS = st.sampled_from(_TS_POOL)


# ---------------------------------------------------------------------------
# _TimelineRow generators
# ---------------------------------------------------------------------------


@st.composite
def timeline_rows(
    draw: st.DrawFn,
    *,
    min_size: int = 0,
    max_size: int = 8,
) -> list[_TimelineRow]:
    """A list of mixed-kind timeline rows with globally-unique uuids.

    Each row gets ``uuid=str(i)`` so the ``(ts, role_rank, uuid)`` sort key
    ``render_turn_text`` uses is a *total* order over the list — the render is
    then a deterministic function of the row set regardless of input order.
    Text rows always carry a body (so they render); tool rows may omit theirs.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    rows: list[_TimelineRow] = []
    for i in range(n):
        kind = draw(KINDS)
        role = draw(ROLES)
        ts = draw(TIMESTAMPS)
        if kind == "text":
            body: str | None = draw(SAFE_TEXT)
            aux: str | None = None
        elif kind == "tool_use":
            body = draw(st.one_of(st.none(), SAFE_TEXT))
            aux = draw(st.one_of(st.none(), SAFE_TEXT_NONEMPTY))
        else:  # tool_result
            body = draw(st.one_of(st.none(), SAFE_TEXT))
            aux = draw(st.one_of(st.none(), SAFE_TEXT_NONEMPTY))
        rows.append(_TimelineRow(ts_iso=ts, role=role, kind=kind, body=body, aux=aux, uuid=str(i)))
    return rows


@st.composite
def same_ts_text_rows(draw: st.DrawFn) -> list[_TimelineRow]:
    """Text rows that all share one timestamp, with unique uuids.

    Forces the ``render_turn_text`` collapse onto its tie-break path:
    ordering by role rank (user < assistant < tool < system) and then uuid.
    """
    ts = draw(TIMESTAMPS)
    n = draw(st.integers(min_value=1, max_value=8))
    rows: list[_TimelineRow] = []
    for i in range(n):
        role = draw(ROLES)
        body = draw(SAFE_TEXT)
        rows.append(
            _TimelineRow(ts_iso=ts, role=role, kind="text", body=body, aux=None, uuid=f"u-{i:03d}")
        )
    return rows


@st.composite
def tool_use_rows(draw: st.DrawFn) -> list[_TimelineRow]:
    """Non-empty list of ``tool_use`` rows with unique uuids and distinct ts.

    Distinct timestamps guarantee every row renders on its own line under a
    generous total cap, so each ``[tool_use:{name}]`` marker is observable.
    """
    n = draw(st.integers(min_value=1, max_value=6))
    rows: list[_TimelineRow] = []
    for i in range(n):
        aux = draw(st.one_of(st.none(), SAFE_TEXT_NONEMPTY))
        # Distinct, monotonically-increasing timestamps → distinct lines.
        ts = f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}"
        rows.append(
            _TimelineRow(ts_iso=ts, role="tool", kind="tool_use", body="{}", aux=aux, uuid=str(i))
        )
    return rows


# ---------------------------------------------------------------------------
# TranscriptRow generators (raw message-envelope rows for render_turn_text)
# ---------------------------------------------------------------------------
#: Envelope ``type`` vocabulary — the collapse ordering key. Kept distinct
#: from ``ROLES`` so a row's ``type`` (ordering) and inner role (rendered
#: prefix) can differ, exercising the role-fidelity fallback.
TYPES = st.sampled_from(["user", "assistant", "tool", "system"])


@st.composite
def transcript_rows(
    draw: st.DrawFn,
    *,
    min_size: int = 0,
    max_size: int = 8,
) -> list[TranscriptRow]:
    """A list of raw message rows with globally-unique uuids and text bodies.

    Each row gets ``uuid=str(i)`` so the ``(ts, kind_rank, uuid)`` collapse key
    is a *total* order over the list — the render is then a deterministic
    function of the row set regardless of input order. Every row carries at
    least one non-empty text block so it renders (empty-body rows are dropped by
    the collapse, which would make per-row line reasoning ambiguous).
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    rows: list[TranscriptRow] = []
    for i in range(n):
        type_ = draw(TYPES)
        ts = draw(TIMESTAMPS)
        text = draw(SAFE_TEXT_NONBLANK)
        rows.append(
            TranscriptRow(
                uuid=str(i),
                type=type_,
                timestamp=ts,
                message={"role": type_, "content": [{"type": "text", "text": text}]},
            )
        )
    return rows


@st.composite
def same_ts_transcript_rows(draw: st.DrawFn) -> list[TranscriptRow]:
    """Text-only rows sharing one timestamp, unique uuids, role == type.

    Forces the collapse onto its tie-break path: ordering by envelope-type rank
    (user < assistant < tool < system) then uuid.
    """
    ts = draw(TIMESTAMPS)
    n = draw(st.integers(min_value=1, max_value=8))
    rows: list[TranscriptRow] = []
    for i in range(n):
        type_ = draw(TYPES)
        text = draw(SAFE_TEXT_NONBLANK)
        rows.append(
            TranscriptRow(
                uuid=f"u-{i:03d}",
                type=type_,
                timestamp=ts,
                message={"role": type_, "content": [{"type": "text", "text": text}]},
            )
        )
    return rows


@st.composite
def tool_use_transcript_rows(draw: st.DrawFn) -> list[TranscriptRow]:
    """Assistant rows each carrying one ``tool_use`` block, distinct timestamps.

    Distinct timestamps guarantee every row renders on its own line under a
    generous total cap, so each ``[tool_use:{name}]`` marker is observable. Each
    row also carries a leading text block so the collapsed body is non-empty.
    """
    n = draw(st.integers(min_value=1, max_value=6))
    rows: list[TranscriptRow] = []
    for i in range(n):
        name = draw(SAFE_TEXT_NONBLANK)
        ts = f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}.000Z"
        rows.append(
            TranscriptRow(
                uuid=str(i),
                type="assistant",
                timestamp=ts,
                message={
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "step"},
                        {"type": "tool_use", "name": name},
                    ],
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Settings field strategies (valid-by-design ranges)
# ---------------------------------------------------------------------------

#: Per-field strategies for the numeric knobs the config projections copy.
#: ``Settings`` carries no field-level numeric constraints (verified against
#: core/config.py), so "valid" is just type-correct; ranges below are
#: deliberately sensible so the built object is realistic.
SETTINGS_NUMERIC_FIELDS: dict[str, st.SearchStrategy[object]] = {
    "umap_n_components_50": st.integers(min_value=2, max_value=200),
    "umap_n_components_2": st.integers(min_value=2, max_value=8),
    "umap_n_neighbors": st.integers(min_value=2, max_value=200),
    "umap_min_dist_cluster": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "umap_min_dist_viz": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "umap_metric": st.sampled_from(["cosine", "euclidean", "manhattan"]),
    "hdbscan_min_cluster_size": st.integers(min_value=2, max_value=500),
    "hdbscan_min_samples": st.integers(min_value=1, max_value=100),
    "leiden_knn_k": st.integers(min_value=2, max_value=100),
    "leiden_edge_floor": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "leiden_min_community_size": st.integers(min_value=1, max_value=50),
    "leiden_resolution": st.one_of(
        st.none(), st.floats(min_value=0.01, max_value=2.0, allow_nan=False)
    ),
    "leiden_resolution_range_lo": st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
    "leiden_resolution_range_hi": st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    "leiden_n_iterations": st.integers(min_value=-1, max_value=50),
    "seed": st.integers(min_value=0, max_value=2**31 - 1),
    "tfidf_min_df": st.integers(min_value=1, max_value=20),
    "tfidf_max_df": st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    "tfidf_ngram_min": st.integers(min_value=1, max_value=3),
    "tfidf_ngram_max": st.integers(min_value=1, max_value=5),
    "tfidf_top_n_terms": st.integers(min_value=1, max_value=100),
    "session_text_total_max_chars": st.integers(min_value=1, max_value=2_000_000),
    "session_text_tool_result_max_chars": st.integers(min_value=1, max_value=500_000),
}
