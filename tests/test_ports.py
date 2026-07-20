"""Conformance + import-hygiene tests for the ``application.ports`` surface.

Two guarantees:

1. **Runtime conformance.** Every existing adapter that a later wave will wrap
   behind a port already satisfies the ``@runtime_checkable`` Protocol, so the
   later lift is a no-op structurally. We assert this where it is cheap — the
   two provider ports (``EmbeddingProvider`` / ``LlmAnalyticsProvider``) whose
   adapters construct without touching the network.

2. **Import leanness.** Importing the ports module must not drag in
   duckdb/polars/lancedb/boto3. The port method signatures reference those types
   only under ``TYPE_CHECKING`` (``from __future__ import annotations`` keeps all
   annotations as strings), so a bare import stays free of the heavy closure.
   This mirrors ``test_cli_import_is_lean`` for the CLI fast path.

Hermetic: no network, no disk, sub-second.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from claude_sql.application.ports import (
    CachePort,
    CheckpointPort,
    Clock,
    EmbeddingProvider,
    LlmAnalyticsProvider,
    PortResult,
    ReaderPort,
    RetryQueuePort,
    SearchHit,
    SessionSearchPort,
    TranscriptReaderPort,
    VectorStorePort,
)
from claude_sql.domain.errors import DomainError, RefusalError
from claude_sql.infrastructure.settings import Settings


def test_all_ports_are_runtime_checkable() -> None:
    """Every port Protocol carries ``_is_runtime_protocol`` (i.e. is @runtime_checkable)."""
    for port in (
        Clock,
        TranscriptReaderPort,
        SessionSearchPort,
        VectorStorePort,
        EmbeddingProvider,
        LlmAnalyticsProvider,
        CheckpointPort,
        RetryQueuePort,
        CachePort,
        ReaderPort,
    ):
        # runtime_checkable sets this private flag; isinstance() against the
        # Protocol only works when it is set.
        assert getattr(port, "_is_runtime_protocol", False), port


def test_embedding_adapter_conforms_to_port() -> None:
    """The default Cohere adapter satisfies ``EmbeddingProvider`` structurally.

    Constructed via the factory (no network at construction), then checked with
    a runtime ``isinstance``. This is the re-exported provider port; the check
    pins that ports.py re-exports the same object the adapters implement.
    """
    from claude_sql.infrastructure.embedding import build_embedder

    embedder = build_embedder(Settings(output_dimension=256))
    assert isinstance(embedder, EmbeddingProvider)


def test_llm_analytics_adapter_conforms_to_port() -> None:
    """The default Sonnet adapter satisfies ``LlmAnalyticsProvider`` structurally."""
    from claude_sql.infrastructure.llm_analytics import build_llm_analytics_provider

    provider = build_llm_analytics_provider(Settings())
    assert isinstance(provider, LlmAnalyticsProvider)


def test_search_hit_is_frozen_dataclass() -> None:
    """``SearchHit`` is the typed semantic-search row (frozen, ordered fields)."""
    hit = SearchHit(
        uuid="u1",
        session_id="s1",
        ts=None,
        role="user",
        snippet="hello",
        cosine_sim=0.9,
    )
    assert hit.uuid == "u1"
    assert hit.cosine_sim == 0.9
    # frozen: setattr through a dynamic name raises FrozenInstanceError (the
    # dynamic name keeps the static type checker from rejecting the assignment
    # while still exercising the runtime frozen guarantee).
    attr = "uuid"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(hit, attr, "u2")


def test_port_result_alias_binds() -> None:
    """``PortResult[T]`` is a ``Result`` alias parameterized over ``DomainError``."""
    from returns.pipeline import is_successful
    from returns.result import Failure, Success

    ok: PortResult[int] = Success(1)
    err: PortResult[int] = Failure(RefusalError("terminal"))
    assert is_successful(ok)
    assert not is_successful(err)
    assert isinstance(err.failure(), DomainError)


def test_domain_error_hierarchy() -> None:
    """``RefusalError`` is a ``DomainError``; both are plain exceptions."""
    assert issubclass(RefusalError, DomainError)
    assert issubclass(DomainError, Exception)


def test_ports_import_is_lean() -> None:
    """Importing ports.py pulls no duckdb/polars/lancedb/boto3.

    Runs in a fresh interpreter so a heavy module imported by another test
    (or already resident in this process) cannot mask a real leak.
    """
    code = (
        "import sys, claude_sql.application.ports;"
        "heavy=[m for m in "
        "('duckdb','polars','lancedb','boto3','botocore','httpx','fastembed','pyarrow')"
        " if m in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = result.stdout.strip()
    assert leaked == "", f"ports import leaked heavy modules: {leaked}"
