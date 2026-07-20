"""Cohere Embed v4 on Bedrock: the base-install ``EmbeddingProvider`` adapter.

A pure lift of the v1.2.1 embedding path that lived in
``analytics/embed_worker.py``: the raw ``invoke_model`` call
(:func:`_invoke_bedrock_sync`), its tenacity retry, the batch orchestration, and
the document-``int8`` / query-``float`` asymmetry. No new dependency: ``boto3``
is already core. Selecting any other provider never imports this module because
:func:`claude_sql.infrastructure.embedding.build_embedder` imports it lazily.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from botocore.exceptions import (
    ClientError,
    ConnectionError as BotoConnectionError,
    EndpointConnectionError,
    ReadTimeoutError,
    SSLError,
)
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from claude_sql.infrastructure.bedrock.client import _build_bedrock_client
from claude_sql.infrastructure.logging_setup import loguru_before_sleep

if TYPE_CHECKING:
    from claude_sql.infrastructure.settings import Settings

#: Conservative per-text character cap before sending to Bedrock. The real
#: model limit is 128K tokens per text; this cap keeps total payload below the
#: Bedrock 20 MB body ceiling even with a full batch of 96 large texts.
MAX_CHARS_PER_TEXT = 50_000

#: Bedrock error codes that tenacity should retry.
_RETRY_CODES: set[str] = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelErrorException",
}


def _is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` is a Bedrock error worth retrying.

    Two buckets:
    * ``ClientError`` with a code in :data:`_RETRY_CODES`: service-level
      throttling and transient model failures.
    * Network-layer errors (SSL, connection, endpoint, read-timeout) that
      surface when long-running batches hit flaky TCP connections.
    """
    if isinstance(exc, SSLError | BotoConnectionError | EndpointConnectionError | ReadTimeoutError):
        return True
    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code")
    return code in _RETRY_CODES


@retry(
    # Cohere Embed v4 on Bedrock has a strict TPM bucket that replenishes over
    # tens of seconds; wait up to 60s between attempts and try up to 10 times
    # before surfacing the ThrottlingException.
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(_is_retryable),
    before_sleep=loguru_before_sleep("WARNING"),
    reraise=True,
)
def _invoke_bedrock_sync(
    client: Any,
    model_id: str,
    texts: list[str],
    *,
    input_type: str,
    output_dimension: int,
    embedding_type: str,
) -> list[list[int]] | list[list[float]]:
    """Make one synchronous ``invoke_model`` call and return the vectors.

    Parameters
    ----------
    client
        A boto3 ``bedrock-runtime`` client.
    model_id
        Cohere Embed v4 model ID (direct or CRIS profile).
    texts
        Up to 96 strings; each is clipped to ``MAX_CHARS_PER_TEXT``.
    input_type
        Either ``"search_document"`` (corpus) or ``"search_query"``.
    output_dimension
        Target Matryoshka dimension: 256, 512, 1024, or 1536.
    embedding_type
        One of ``"int8"``, ``"float"``, ``"uint8"``, ``"binary"``, ``"ubinary"``.

    Returns
    -------
    list of list of int or float
        Flat list of vectors matching the order of ``texts``.
    """
    body = json.dumps(
        {
            "texts": [t[:MAX_CHARS_PER_TEXT] for t in texts],
            "input_type": input_type,
            "output_dimension": output_dimension,
            "embedding_types": [embedding_type],
            "truncate": "RIGHT",
        }
    )
    resp = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload["embeddings"][embedding_type]


async def _embed_one_batch(
    client: Any,
    texts: list[str],
    model_id: str,
    *,
    input_type: str,
    output_dimension: int,
    embedding_type: str,
    sem: asyncio.Semaphore,
) -> list[list[int]] | list[list[float]]:
    """Embed a single batch under a concurrency-limiting semaphore."""
    async with sem:
        return await asyncio.to_thread(
            _invoke_bedrock_sync,
            client,
            model_id,
            texts,
            input_type=input_type,
            output_dimension=output_dimension,
            embedding_type=embedding_type,
        )


class CohereBedrockEmbedder:
    """``EmbeddingProvider`` over Cohere Embed v4 on Amazon Bedrock.

    Preserves the v1.2.1 behavior exactly: documents embed at
    ``settings.embedding_type`` (``int8`` by default) and are float-widened on
    the way out; queries force ``embedding_type="float"`` regardless of the
    setting because the HNSW distance math needs float vectors. The Matryoshka
    ``output_dimension`` knob (256/512/1024/1536) is Cohere-specific and is the
    provider's fixed :attr:`dimension`.
    """

    provider = "cohere-bedrock"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def model_id(self) -> str:
        return self._settings.active_model_id

    @property
    def dimension(self) -> int:
        return int(self._settings.output_dimension)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus documents in parallel and return ``float`` vectors.

        Uses ``input_type="search_document"``. The Bedrock response (int8 or
        other type per ``settings.embedding_type``) is cast to ``float`` so the
        downstream Lance ``FLOAT[dim]`` column is directly consumable by the
        HNSW index.
        """
        if not texts:
            return []

        settings = self._settings
        client = _build_bedrock_client(settings)
        batch_size = settings.batch_size
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        sem = asyncio.Semaphore(settings.embed_concurrency)

        logger.info(
            "Embedding {} texts in {} batches (batch_size={}, concurrency={}, model={})",
            len(texts),
            len(batches),
            batch_size,
            settings.embed_concurrency,
            self.model_id,
        )

        t0 = time.monotonic()
        coros = [
            _embed_one_batch(
                client,
                batch,
                self.model_id,
                input_type="search_document",
                output_dimension=self.dimension,
                embedding_type=settings.embedding_type,
                sem=sem,
            )
            for batch in batches
        ]
        results = await asyncio.gather(*coros)
        elapsed = time.monotonic() - t0

        vectors: list[list[float]] = [
            [float(x) for x in v] for batch_vecs in results for v in batch_vecs
        ]
        logger.info(
            "Embedded {} vectors across {} batches in {:.2f}s ({:.1f} vec/s)",
            len(vectors),
            len(batches),
            elapsed,
            len(vectors) / elapsed if elapsed > 0 else 0.0,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for HNSW nearest-neighbor search.

        Uses ``input_type="search_query"`` and forces ``embedding_type="float"``
        regardless of ``settings.embedding_type`` because HNSW distance math
        needs float vectors. Returns a single vector of length :attr:`dimension`.
        """
        client = _build_bedrock_client(self._settings)
        vectors = _invoke_bedrock_sync(
            client,
            self.model_id,
            [text],
            input_type="search_query",
            output_dimension=self.dimension,
            embedding_type="float",
        )
        return [float(x) for x in vectors[0]]


__all__ = [
    "MAX_CHARS_PER_TEXT",
    "CohereBedrockEmbedder",
    "_embed_one_batch",
    "_invoke_bedrock_sync",
    "_is_retryable",
]
