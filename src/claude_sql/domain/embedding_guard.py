"""The fail-loud embedding provider/dimension guard (pure domain rule).

The dimension contract (DESIGN.md §4.4): every embeddings-store row is stamped
with the embedder's ``model_id`` and ``dimension``, and both the write path
(:func:`claude_sql.application.use_cases.embed.run_backfill`) and the read/bind
path (:func:`claude_sql.infrastructure.duckdb_views.register_vss`) read that
stamp back and call :func:`ensure_store_matches` before touching vectors. A
provider switch is destructive (different models live in incompatible vector
spaces even at matching widths), so the guard fails loud rather than silently
corrupting the kNN index.

Rehomed here from ``core/embedding/base.py`` in T-8-2: this is a pure rule
(string/int comparison, no I/O), so it belongs in the domain hexagon. It raises
:class:`~claude_sql.domain.errors.EmbeddingProviderMismatch`.
"""

from __future__ import annotations

from claude_sql.domain.errors import EmbeddingProviderMismatch


def ensure_store_matches(
    *,
    stored_model: str | None,
    stored_dim: int | None,
    expected_model: str,
    expected_dim: int | None,
) -> None:
    """Fail loud if a store's stamped identity differs from the active embedder.

    ``stored_model`` / ``stored_dim`` come from the Lance ``model`` / ``dim``
    columns (both stamped on every row since v1). ``None`` for either means the
    store is empty (fresh install) and any provider may claim it, so the check
    is a no-op.

    ``model_id`` is the primary identity: it is globally unique and encodes the
    provider + model, so a match guarantees a compatible vector space. Cohere is
    the one provider whose single ``model_id`` can emit different Matryoshka
    widths, so ``expected_dim`` is also checked when supplied. For probe-only
    providers (ollama / onnx) the expected width is unknown at bind time, so
    ``expected_dim=None`` trusts ``model_id`` alone (a given local model always
    emits the same width, so a width change implies a model change).

    On a genuine mismatch this raises :class:`EmbeddingProviderMismatch` naming
    both sides and the exact recovery command.
    """
    if stored_model is None or stored_dim is None:
        return
    model_ok = stored_model == expected_model
    dim_ok = expected_dim is None or stored_dim == expected_dim
    if model_ok and dim_ok:
        return
    raise EmbeddingProviderMismatch(
        "Embedding store was written by a different provider/model. "
        f"stored=(model={stored_model!r}, dim={stored_dim}) "
        f"active=(model={expected_model!r}, dim={expected_dim}). "
        "Vectors from different models live in incompatible spaces (even at "
        "matching dimensions), so mixing them silently corrupts kNN search. "
        "Re-embed under the new provider: rm -rf the Lance store directory "
        "(default ~/.claude/embeddings_lance/) and re-run `claude-sql embed`."
    )


__all__ = ["ensure_store_matches"]
