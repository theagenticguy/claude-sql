"""Provider port Protocols that infrastructure adapters implement.

The two pluggable-provider seams — :class:`EmbeddingProvider` (text → vectors)
and :class:`LlmAnalyticsProvider` (one structured-output classification call) —
are pure ``@runtime_checkable`` Protocols with no runtime dependency beyond
``typing``. They live in ``domain`` (the innermost hexagon) so the concrete
adapters in ``infrastructure`` can be typed against them WITHOUT importing *up*
into ``application`` — exactly the pattern :class:`~claude_sql.domain.retrieval.SearchHit`
uses. :mod:`claude_sql.application.ports` re-exports both so use-cases name the
whole port surface from one module.

Rehomed here from ``core/embedding/base.py`` and ``core/llm_analytics/base.py``
in the v2 hexagonal final cut (T-8-2), which dissolved the transitional ``core``
package. The ``build_*`` factories and the concrete adapters moved into
``infrastructure/{embedding,llm_analytics}/``; the terminal errors moved to
:mod:`claude_sql.domain.errors`; the dimension guard to
:mod:`claude_sql.domain.embedding_guard`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

#: A structured-output schema. Bound to ``BaseModel`` so both the Sonnet path
#: (``model_validate`` on the parsed dict) and the Luna path
#: (``structured_output_model=Schema``) single-source the domain contract.
SchemaT = TypeVar("SchemaT", bound="BaseModel")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Port: turn text into vectors. One adapter per backend.

    The document/query asymmetry is preserved as two methods rather than a
    single ``input_type`` parameter, because each backend handles it
    differently: Cohere uses distinct ``search_document`` / ``search_query``
    input types, BGE prepends a query-only instruction prefix, and Ollama uses
    neither. Each adapter internalizes its own asymmetry handling, batching,
    retry, and float-widening.
    """

    @property
    def model_id(self) -> str:
        """Stable identity string stamped into the Lance ``model`` column and
        enforced on rebind (e.g. ``global.cohere.embed-v4:0``,
        ``ollama:nomic-embed-text``, ``onnx:bge-small-en-v1.5``). Globally
        unique across providers, so it doubles as the provider discriminator in
        the store guard."""
        ...

    @property
    def provider(self) -> str:
        """The provider tag: ``cohere-bedrock``, ``ollama``, or ``onnx-bge``."""
        ...

    @property
    def dimension(self) -> int:
        """Fixed output width this provider emits. Replaces the free
        ``Settings.output_dimension`` as the contract source for the Lance
        schema, the DuckDB ``FLOAT[dim]`` view cast, and the query-time cast."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch corpus embedding. Float vectors, input order preserved, ``[]``
        on empty input. The adapter owns batching, concurrency, retry, and (if
        the backend stores quantized) float-widening."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Single query vector, always float, length ``== self.dimension``.
        Synchronous: the query path is one call on the hot path of ``search``."""
        ...


@runtime_checkable
class LlmAnalyticsProvider(Protocol):
    """Port: one structured-output classification call. One adapter per backend.

    The seam is deliberately narrow: a system prompt, a user prompt, and a
    Pydantic schema in; a validated instance of that schema out. Everything
    provider-specific (model id, region, max tokens, thinking / reasoning
    effort, the boto3 / Strands client, and the concurrency limiter) is owned by
    the adapter and fixed at construction, exactly like the embedding adapters
    own their batching and retry.
    """

    @property
    def provider(self) -> str:
        """Provider tag: ``sonnet-bedrock`` or ``strands-luna``."""
        ...

    async def classify_structured(
        self, *, system: str, prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Run one structured-output call and return a validated ``schema`` instance.

        ``system`` is the (byte-stable, cacheable) task framing; ``prompt`` is
        the per-call user payload. The adapter raises the provider's terminal
        failure signal on error: :class:`~claude_sql.domain.errors.BedrockRefusalError`
        for a Sonnet content-policy refusal, or
        :class:`~claude_sql.domain.errors.LlmAnalyticsUnavailable` for a Luna
        transport / structured-output failure.
        """
        ...


__all__ = [
    "EmbeddingProvider",
    "LlmAnalyticsProvider",
    "SchemaT",
]
