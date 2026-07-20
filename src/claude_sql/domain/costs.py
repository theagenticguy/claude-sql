"""Pure cost arithmetic for the LLM-analytics pipelines.

MIGRATION Phase C (T-3-4): :func:`estimate_cost` is the back-of-envelope
dollar projection every ``--dry-run`` path uses to price a pending batch of
Sonnet classification calls. It is pure arithmetic over a token count and a
``(input_rate, output_rate)`` $/MTok tuple — no DuckDB, no Bedrock, no
``Settings`` — so it belongs in the domain hexagon.

It moved here from the pre-hexagonal ``core.llm_shared`` module (deleted in
the v2 reshape); workers and tests now import ``estimate_cost`` directly from
this module.

Note: the model-id *dated-suffix* normalization (``-YYYYMMDD`` strip) is a
separate concern that lives in SQL — the ``cost_estimate`` DuckDB macro in
``infrastructure.duckdb_views`` — and is not part of this pure-Python estimator.
"""

from __future__ import annotations


def estimate_cost(
    n_items: int,
    avg_in_tokens: int,
    avg_out_tokens: int,
    pricing: tuple[float, float],
) -> float:
    """Back-of-envelope dollar estimate for ``n_items`` classification calls.

    ``pricing`` is ``(input_rate, output_rate)`` in $/MTok. Cost is
    ``n * (in_tokens * in_rate + out_tokens * out_rate) / 1e6`` — a flat linear
    projection with no minimums, tiers, or cache accounting.
    """
    in_rate, out_rate = pricing
    return (n_items * avg_in_tokens * in_rate + n_items * avg_out_tokens * out_rate) / 1_000_000


__all__ = ["estimate_cost"]
