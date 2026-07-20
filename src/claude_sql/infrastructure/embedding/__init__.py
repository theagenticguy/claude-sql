"""Pluggable embeddings infrastructure: the ``build_embedder`` factory + adapters.

The :class:`~claude_sql.domain.ports.EmbeddingProvider` port, the
:class:`~claude_sql.domain.errors.EmbeddingProviderMismatch` terminal error, and
the :func:`~claude_sql.domain.embedding_guard.ensure_store_matches` guard are
pure and live in ``domain``. This package holds the concrete adapters (one per
backend) and the :func:`build_embedder` factory that selects one from
``Settings``.

The adapters are NOT re-exported here: importing one is :func:`build_embedder`'s
job (lazily, inside the selected branch), so a bare
``import claude_sql.infrastructure.embedding`` never drags in botocore / httpx /
fastembed. The port / error / guard ARE re-exported for call-site stability
(historic ``from claude_sql.core.embedding import ...`` call sites moved here).
Import an adapter class directly from its module (e.g.
``claude_sql.infrastructure.embedding.cohere_bedrock``) only when you need the
concrete type.

Rehomed from ``core/embedding/`` in the v2 hexagonal final cut (T-8-2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claude_sql.domain.embedding_guard import ensure_store_matches
from claude_sql.domain.errors import EmbeddingProviderMismatch
from claude_sql.domain.ports import EmbeddingProvider

if TYPE_CHECKING:
    from claude_sql.infrastructure.settings import Settings


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Return the ``EmbeddingProvider`` adapter selected by ``settings``.

    Switches on ``settings.embedding_provider``. Adapter modules are imported
    lazily inside each branch so selecting one backend never imports another's
    heavy dependency, and a bare import of this module stays free of
    botocore / httpx / fastembed (pinned by ``test_cli_import_is_lean``).

    The default ``cohere-bedrock`` reproduces the v1.2.1 behavior exactly.
    """
    provider = settings.embedding_provider
    if provider == "cohere-bedrock":
        from claude_sql.infrastructure.embedding.cohere_bedrock import CohereBedrockEmbedder

        return CohereBedrockEmbedder(settings)
    if provider == "ollama":
        from claude_sql.infrastructure.embedding.ollama import OllamaEmbedder

        return OllamaEmbedder(base_url=settings.ollama_base_url, model=settings.ollama_model)
    if provider == "onnx-bge":
        from claude_sql.infrastructure.embedding.onnx_bge import OnnxBgeEmbedder

        return OnnxBgeEmbedder(model=settings.onnx_model)
    raise ValueError(
        f"Unknown embedding provider {provider!r}; "
        "expected one of 'cohere-bedrock', 'ollama', 'onnx-bge'."
    )


__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderMismatch",
    "build_embedder",
    "ensure_store_matches",
]
