"""The domain error hierarchy.

These are pure, dependency-free exception types that the application layer's
``Result[T, DomainError]`` port surface (``application/ports.py``) is
parameterized over. They live in the innermost hexagon so every other layer may
import them without inverting the dependency arrow.

Scope: pure, duckdb-free error types belong in the innermost hexagon.
:data:`EXIT_CODES`, :class:`ClassifiedError`, and :class:`InputValidationError`
were lifted here in T-2-2 (the duckdb classifier lives in
``infrastructure/duckdb_errors.py`` and the emit/format helpers in
``interfaces/cli/output.py``). The two pluggable-provider terminal errors —
:class:`EmbeddingProviderMismatch` (embedding provider/dim switch) and
:class:`LlmAnalyticsUnavailable` (opt-in analytics provider outage, fail-open) —
were rehomed here in the v2 hexagonal final cut (T-8-2) when the transitional
``core`` package was dissolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Exit codes that agents can rely on.  Keep them stable -- wire protocols
# always rot fastest at the boundary.
EXIT_CODES: dict[str, int] = {
    "ok": 0,
    "no_embeddings": 2,
    "invalid_input": 64,  # malformed user-supplied flags (e.g. --glob)
    "parse_error": 64,  # malformed SQL
    "catalog_error": 65,  # unknown view/macro/column
    "runtime_error": 70,  # everything else from duckdb.Error
    "duckdb_missing": 127,  # system `duckdb` binary not on PATH
}


class DomainError(Exception):
    """Base for every claude-sql domain error.

    The ``Result[T, DomainError]`` port contract in ``application/ports.py`` is
    parameterized over this type, so any failure a port surfaces as a
    ``Failure`` is a subclass of this. Concrete adapters may raise their own
    transport-specific exceptions internally, but what crosses a port boundary
    as an error value is a ``DomainError``.
    """


class RefusalError(DomainError):
    """Terminal, non-retryable refusal from an LLM provider.

    Generalizes :class:`BedrockRefusalError` (a content-policy refusal from
    Bedrock Sonnet, defined below). Terminal means the pipelines
    stamp a neutral placeholder row and clear the retry queue rather than
    cycling the unit forever (CLAUDE.md: "BedrockRefusalError is terminal").

    T-2-1 lifts the Bedrock transport into infrastructure and makes the concrete
    ``BedrockRefusalError`` a subclass of this base; this wave only defines the
    base so the port surface can name it without a cross-layer import into
    ``core``.
    """


class BedrockRefusalError(RefusalError):
    """Bedrock declined to classify the input under its content policy.

    Raised when the response has ``stop_reason == "refusal"`` and no
    content blocks. Callers treat this as a terminal, non-retryable
    outcome and can write a neutral placeholder row so the message is
    not re-tried in every future run.

    Lifted here in T-2-1 from ``core/llm_shared.py`` (where it was a bare
    ``Exception`` subclass). It now subclasses :class:`RefusalError` so the
    ``Result[T, DomainError]`` port surface can carry it. The name is kept for
    back-compat: the Bedrock infra transport re-exports it, and existing
    ``except BedrockRefusalError`` / ``isinstance`` sites in the workers keep
    matching because it is the same class object.
    """


class EmbeddingProviderMismatch(DomainError):  # noqa: N818 — named per the v2 DESIGN spec, not "*Error"
    """Raised when a Lance store's stamped ``(model, dim)`` differs from the
    active embedder's identity.

    Different embedding models produce vectors in incompatible spaces, so
    appending or querying across a provider switch yields numerically valid but
    semantically garbage cosine scores. This error is terminal: the store must
    be dropped and re-embedded under the new provider.

    Rehomed here from ``core/embedding/base.py`` in T-8-2. The guard function
    that raises it lives in :mod:`claude_sql.domain.embedding_guard`.
    """


class LlmAnalyticsUnavailable(DomainError):  # noqa: N818 — terminal "unavailable" signal, mirrors a fail-open reviewer adapter
    """The opt-in LLM-analytics provider could not produce structured output.

    Raised by the Strands/Luna adapter on any transport / structured-output /
    empty-result failure after logging a warning. It is a terminal, fail-open
    signal: the trajectory worker treats it like any recoverable per-chunk
    failure (enqueue for a later run, stamp neutral placeholders, or skip) so an
    opt-in analytics run degrading to "provider unavailable" NEVER crashes the
    core SQL / embedding pipeline. Mirrors a fail-open reviewer adapter's
    posture.

    Rehomed here from ``core/llm_analytics/base.py`` in T-8-2.
    """


@dataclass(frozen=True, slots=True)
class ClassifiedError:
    """The structured shape of a CLI error after classification."""

    kind: str  # "parse_error" | "catalog_error" | "runtime_error"
    exit_code: int
    message: str
    hint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "kind": self.kind,
                "message": self.message,
                "hint": self.hint,
            }
        }


class InputValidationError(ValueError):
    """Raised when a user-supplied flag (e.g. ``--glob``) is malformed.

    Carries its own ``hint`` so ``run_or_die`` can surface the fix alongside
    the failure. Maps to exit code 64 (``invalid_input``).
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


__all__ = [
    "EXIT_CODES",
    "BedrockRefusalError",
    "ClassifiedError",
    "DomainError",
    "EmbeddingProviderMismatch",
    "InputValidationError",
    "LlmAnalyticsUnavailable",
    "RefusalError",
]
