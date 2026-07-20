"""Ollama local-server ``EmbeddingProvider`` adapter ([ollama] extra: httpx).

Talks to a local Ollama server via the native ``POST /api/embed`` route (NOT the
deprecated ``/api/embeddings``): ``/api/embed`` is batch-capable (``input``
accepts ``str | list[str]``) and returns L2-normalized unit-length float32
vectors server-side, which is exactly what the cosine-metric HNSW index wants.
Ollama has no document-vs-query asymmetry, so both port methods hit the same
route and differ only in cardinality. The dimension is fixed per model and
discovered by probing once with a sentinel then caching the width (model tags can
change width across releases, so a hard-coded table is not the source of truth).
"""

from __future__ import annotations

from loguru import logger

#: One-token text used to probe the model's fixed output width once at first use.
_SENTINEL = "dimension probe"

#: Timeout for embed calls; a cold model load on the Ollama server can take
#: tens of seconds, so give it generous headroom.
_TIMEOUT = 120.0


class OllamaEmbedder:
    """``EmbeddingProvider`` over a local Ollama server via ``/api/embed``.

    ``base_url`` defaults to ``http://localhost:11434`` and ``model`` to
    ``nomic-embed-text`` (768-dim, Ollama's recommended default embedder). The
    ``httpx`` import is deferred into ``__init__`` so the base install never
    needs it; a missing extra raises a clear install hint.
    """

    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "OllamaEmbedder requires the [ollama] extra: uv add 'claude-sql[ollama]'"
            ) from exc
        self._httpx = httpx
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model
        self._dim: int | None = None  # probe-once cache

    @property
    def model_id(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Probe once and cache: never assume a hard-coded width.
            self._dim = len(self._post([_SENTINEL])[0])
            logger.debug("Ollama model {} probed at dimension {}", self._model, self._dim)
        return self._dim

    def _post(self, inputs: list[str]) -> list[list[float]]:
        """POST one ``/api/embed`` call. ``embeddings`` is always
        ``list[list[float]]`` (one inner vector per input), L2-normalized."""
        resp = self._httpx.post(
            self._url,
            json={"model": self._model, "input": inputs},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch corpus embedding in one ``/api/embed`` call (``input`` accepts
        an array). Vectors are already float and L2-normalized; the cast is
        defensive."""
        if not texts:
            return []
        async with self._httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(self._url, json={"model": self._model, "input": texts})
            resp.raise_for_status()
            vectors = resp.json()["embeddings"]
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Single query vector. Ollama has no input_type / query prefix concept,
        so this is the same route as documents with one input."""
        return [float(x) for x in self._post([text])[0]]


__all__ = ["OllamaEmbedder"]
