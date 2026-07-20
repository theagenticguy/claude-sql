"""Adapter-parity tests for the pluggable ``EmbeddingProvider`` port.

Every adapter round-trips ``embed_query`` / ``embed_documents`` against a MOCK
backend (no live Ollama, no Bedrock, no model download): httpx.post is
monkeypatched for Ollama, the fastembed model class for ONNX BGE, and the boto3
client for Cohere. Each asserts a stable :attr:`dimension`, the correct
:attr:`provider` / :attr:`model_id`, and that :func:`ensure_store_matches`
raises :class:`EmbeddingProviderMismatch` when the store was stamped by a
different provider. The ONNX test skips when fastembed is absent.

Hermetic: the whole file runs in well under a second and touches no network or
disk beyond ``tmp_path``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from claude_sql.infrastructure.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMismatch,
    build_embedder,
    ensure_store_matches,
)
from claude_sql.infrastructure.settings import Settings

# ---------------------------------------------------------------------------
# Cohere on Bedrock (base install, no extra)
# ---------------------------------------------------------------------------


def test_cohere_adapter_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """CohereBedrockEmbedder embeds a query (float) and documents (int8->float)."""
    from claude_sql.infrastructure.embedding import cohere_bedrock
    from conftest import FakeBedrockClient

    settings = Settings(output_dimension=256, embedding_type="int8", embed_concurrency=2)
    embedder = build_embedder(settings)
    assert isinstance(embedder, EmbeddingProvider)
    assert embedder.provider == "cohere-bedrock"
    assert embedder.model_id == "global.cohere.embed-v4:0"
    assert embedder.dimension == 256

    # Query path forces embedding_type="float".
    fake_q = FakeBedrockClient({"embeddings": {"float": [[0.5, 0.25, 0.125]]}})
    monkeypatch.setattr(cohere_bedrock, "_build_bedrock_client", lambda _s: fake_q)
    qv = embedder.embed_query("what did I work on")
    assert qv == [pytest.approx(0.5), pytest.approx(0.25), pytest.approx(0.125)]
    assert fake_q.captured[0]["body"]["input_type"] == "search_query"
    assert fake_q.captured[0]["body"]["embedding_types"] == ["float"]

    # Document path uses search_document + settings.embedding_type, widened to float.
    fake_d = FakeBedrockClient({"embeddings": {"int8": [[1, 2, 3], [4, 5, 6]]}})
    monkeypatch.setattr(cohere_bedrock, "_build_bedrock_client", lambda _s: fake_d)
    docs = asyncio.run(embedder.embed_documents(["a", "b"]))
    assert docs == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert fake_d.captured[0]["body"]["input_type"] == "search_document"
    assert fake_d.captured[0]["body"]["embedding_types"] == ["int8"]


def test_cohere_adapter_empty_documents_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty document input returns [] without building a client."""
    from claude_sql.infrastructure.embedding import cohere_bedrock

    def _explode(_settings: Any) -> Any:  # pragma: no cover - failure trap
        raise AssertionError("client must not be built for empty input")

    monkeypatch.setattr(cohere_bedrock, "_build_bedrock_client", _explode)
    embedder = build_embedder(Settings())
    assert asyncio.run(embedder.embed_documents([])) == []


# ---------------------------------------------------------------------------
# Ollama (behind [ollama] extra: httpx)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal httpx.Response stand-in: raise_for_status no-op + json()."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    """Async context-manager stand-in for httpx.AsyncClient."""

    def __init__(self, payload: dict[str, Any], captured: list[dict[str, Any]]) -> None:
        self._payload = payload
        self._captured = captured

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        self._captured.append({"url": url, "json": json})
        n = len(json["input"])
        return _FakeResponse({"embeddings": [[0.1, 0.2, 0.3, 0.4] for _ in range(n)]})


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Monkeypatch httpx.post + httpx.AsyncClient to return a fixed embedding."""
    import httpx

    captured: list[dict[str, Any]] = []

    def _fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _FakeResponse:
        captured.append({"url": url, "json": json})
        n = len(json["input"])
        return _FakeResponse({"embeddings": [[0.1, 0.2, 0.3, 0.4] for _ in range(n)]})

    monkeypatch.setattr(httpx, "post", _fake_post)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _FakeAsyncClient({}, captured))
    return captured


def test_ollama_adapter_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """OllamaEmbedder posts to /api/embed and round-trips query + documents."""
    captured = _install_fake_httpx(monkeypatch)

    settings = Settings(embedding_provider="ollama", ollama_model="nomic-embed-text")
    embedder = build_embedder(settings)
    assert isinstance(embedder, EmbeddingProvider)
    assert embedder.provider == "ollama"
    assert embedder.model_id == "ollama:nomic-embed-text"

    # dimension probes once (sentinel) and caches the width.
    assert embedder.dimension == 4
    assert embedder.dimension == 4  # cached; no error on second read

    qv = embedder.embed_query("hello")
    assert qv == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3), pytest.approx(0.4)]

    docs = asyncio.run(embedder.embed_documents(["a", "b", "c"]))
    assert len(docs) == 3
    assert all(len(v) == 4 for v in docs)

    # The native /api/embed route was hit, not the deprecated /api/embeddings.
    assert captured, "expected at least one POST"
    assert all(c["url"].endswith("/api/embed") for c in captured)


def test_ollama_adapter_empty_documents_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch)
    embedder = build_embedder(Settings(embedding_provider="ollama"))
    assert asyncio.run(embedder.embed_documents([])) == []


def test_ollama_adapter_base_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom base_url is honored and the /api/embed suffix appended once."""
    captured = _install_fake_httpx(monkeypatch)
    settings = Settings(embedding_provider="ollama", ollama_base_url="http://remote:9999/")
    embedder = build_embedder(settings)
    embedder.embed_query("hi")
    assert captured[0]["url"] == "http://remote:9999/api/embed"


# ---------------------------------------------------------------------------
# ONNX BGE (behind [onnx] extra: fastembed)
# ---------------------------------------------------------------------------


class _FakeNdarray:
    """Minimal ndarray stand-in exposing .tolist() and __len__."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values

    def __len__(self) -> int:
        return len(self._values)


class _FakeTextEmbedding:
    """Stand-in for fastembed.TextEmbedding.

    Records whether query_embed (prefix path) or passage_embed (no prefix) was
    used and yields fixed 384-wide vectors, mirroring bge-small-en-v1.5.
    """

    last_method: str = ""

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def query_embed(self, texts: list[str]) -> list[_FakeNdarray]:
        type(self).last_method = "query"
        return [_FakeNdarray([0.01] * 384) for _ in texts]

    def passage_embed(self, texts: list[str]) -> list[_FakeNdarray]:
        type(self).last_method = "passage"
        return [_FakeNdarray([0.02] * 384) for _ in texts]


def test_onnx_bge_adapter_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """OnnxBgeEmbedder uses query_embed for queries, passage_embed for docs."""
    pytest.importorskip("fastembed", reason="[onnx] extra not installed")
    # The adapter does `from fastembed import TextEmbedding` inside __init__;
    # patch it at the import source so the local rebind picks up the fake.
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)

    settings = Settings(embedding_provider="onnx-bge", onnx_model="BAAI/bge-small-en-v1.5")
    embedder = build_embedder(settings)
    assert isinstance(embedder, EmbeddingProvider)
    assert embedder.provider == "onnx-bge"
    assert embedder.model_id == "onnx:bge-small-en-v1.5"
    assert embedder.dimension == 384

    qv = embedder.embed_query("hello")
    assert len(qv) == 384
    assert _FakeTextEmbedding.last_method == "query"

    docs = asyncio.run(embedder.embed_documents(["a", "b"]))
    assert len(docs) == 2
    assert all(len(v) == 384 for v in docs)
    assert _FakeTextEmbedding.last_method == "passage"


def test_onnx_bge_adapter_empty_documents_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastembed", reason="[onnx] extra not installed")
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)
    embedder = build_embedder(Settings(embedding_provider="onnx-bge"))
    assert asyncio.run(embedder.embed_documents([])) == []


# ---------------------------------------------------------------------------
# The fail-loud provider/dimension guard
# ---------------------------------------------------------------------------


def test_guard_raises_on_provider_mismatch() -> None:
    """A store stamped by one provider rejects another provider's identity."""
    with pytest.raises(EmbeddingProviderMismatch, match="different provider/model"):
        ensure_store_matches(
            stored_model="global.cohere.embed-v4:0",
            stored_dim=1024,
            expected_model="ollama:nomic-embed-text",
            expected_dim=768,
        )


def test_guard_raises_on_dim_mismatch_same_provider() -> None:
    """Same Cohere model at a different Matryoshka width is still a mismatch."""
    with pytest.raises(EmbeddingProviderMismatch):
        ensure_store_matches(
            stored_model="global.cohere.embed-v4:0",
            stored_dim=1024,
            expected_model="global.cohere.embed-v4:0",
            expected_dim=512,
        )


def test_guard_noop_on_match() -> None:
    """Matching identity is a silent no-op."""
    ensure_store_matches(
        stored_model="global.cohere.embed-v4:0",
        stored_dim=1024,
        expected_model="global.cohere.embed-v4:0",
        expected_dim=1024,
    )


def test_guard_noop_on_empty_store() -> None:
    """An empty store (None identity) accepts any provider."""
    ensure_store_matches(
        stored_model=None,
        stored_dim=None,
        expected_model="ollama:nomic-embed-text",
        expected_dim=768,
    )


def test_guard_trusts_model_id_when_expected_dim_none() -> None:
    """Probe-only providers (expected_dim=None) gate on model_id alone."""
    # Same model, unknown expected width -> pass.
    ensure_store_matches(
        stored_model="ollama:nomic-embed-text",
        stored_dim=768,
        expected_model="ollama:nomic-embed-text",
        expected_dim=None,
    )
    # Different model, unknown expected width -> still fail.
    with pytest.raises(EmbeddingProviderMismatch):
        ensure_store_matches(
            stored_model="onnx:bge-small-en-v1.5",
            stored_dim=384,
            expected_model="ollama:nomic-embed-text",
            expected_dim=None,
        )


def test_run_backfill_fails_loud_on_provider_switch(
    tmp_corpus: dict[str, Any], tmp_settings: Settings
) -> None:
    """run_backfill refuses to append into a store written by another provider."""
    from datetime import UTC, datetime

    import duckdb
    import polars as pl

    from claude_sql.application.use_cases.embed import run_backfill
    from claude_sql.infrastructure import lance_store
    from claude_sql.infrastructure.duckdb_views import register_raw, register_views

    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    register_views(con)

    # Seed the Lance store with a foreign provider's stamp at a matching dim.
    dim = int(tmp_settings.output_dimension)
    db = lance_store.connect_db(tmp_settings.lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=dim)
    df = pl.DataFrame(
        {
            "uuid": ["foreign-1"],
            "model": ["ollama:nomic-embed-text"],
            "dim": [dim],
            "embedding": [[0.0] * dim],
            "embedded_at": [datetime.now(UTC)],
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.Int32,
            "embedding": pl.Array(pl.Float32, dim),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    lance_store.add_chunk(tbl, df)

    # Active provider is the default cohere-bedrock -> mismatch -> fail loud.
    try:
        with pytest.raises(EmbeddingProviderMismatch):
            asyncio.run(run_backfill(con=con, settings=tmp_settings))
    finally:
        con.close()
