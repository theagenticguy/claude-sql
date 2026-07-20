"""Local ONNX BGE ``EmbeddingProvider`` adapter ([onnx] extra: fastembed).

Runs a BAAI ``bge`` v1.5 ONNX model fully locally via fastembed (Qdrant), which
wraps onnxruntime + tokenizers + huggingface-hub with NO torch. fastembed is the
correctness-by-default path: it does CLS pooling (BGE v1.5 English models use the
[CLS] token, NOT mean pooling: mean pooling significantly degrades BGE per the
BAAI card) and L2-normalizes internally, and its ``query_embed()`` auto-applies
the BGE query instruction prefix while ``passage_embed()`` does not, mapping 1:1
onto the port's query-vs-document split. onnxruntime is CPU-blocking, so document
batches run under ``asyncio.to_thread``. Model default ``BAAI/bge-small-en-v1.5``
(384-dim); downloaded and cached on first embed. Dimension is probed once.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

#: One-token text used to probe the model's fixed output width once at first use.
_SENTINEL = "dimension probe"


class OnnxBgeEmbedder:
    """``EmbeddingProvider`` over a local BAAI ``bge`` ONNX model via fastembed.

    ``model`` defaults to ``BAAI/bge-small-en-v1.5`` (384-dim). The ``fastembed``
    import is deferred into ``__init__`` so the base install never needs it; a
    missing extra raises a clear install hint. The model downloads + caches on
    first embed (fastembed manages the on-disk cache), not at import time.
    """

    provider = "onnx-bge"

    def __init__(self, *, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "OnnxBgeEmbedder requires the [onnx] extra: uv add 'claude-sql[onnx]'"
            ) from exc
        self._model: Any = TextEmbedding(model_name=model)  # lazy download + cache on first embed
        self._repo = model
        self._dim: int | None = None  # probe-once cache

    @property
    def model_id(self) -> str:
        # e.g. 'onnx:bge-small-en-v1.5'
        return f"onnx:{self._repo.split('/')[-1]}"

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Probe once and cache via query_embed (prefix is irrelevant to width).
            vec = next(iter(self._model.query_embed([_SENTINEL])))
            self._dim = len(vec)
            logger.debug("ONNX BGE model {} probed at dimension {}", self._repo, self._dim)
        return self._dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch corpus embedding via ``passage_embed`` (NO query prefix).
        onnxruntime is CPU-blocking, so the batch runs off the event loop."""
        if not texts:
            return []

        def _run() -> list[list[float]]:
            return [[float(x) for x in v.tolist()] for v in self._model.passage_embed(texts)]

        return await asyncio.to_thread(_run)

    def embed_query(self, text: str) -> list[float]:
        """Single query vector via ``query_embed``, which auto-prepends the BGE
        instruction prefix 'Represent this sentence for searching relevant
        passages: ' to queries only."""
        vec = next(iter(self._model.query_embed([text])))
        return [float(x) for x in vec.tolist()]


__all__ = ["OnnxBgeEmbedder"]
