"""Bedrock infrastructure adapter — transport + structured-output flattening.

Split out of ``core/llm_shared.py`` in T-2-1:

* :mod:`~claude_sql.infrastructure.bedrock.client` — the ``bedrock-runtime``
  transport (client build/cache, retryable ``invoke_model``, cache-stat
  accumulator, structured-payload parsing).
* :mod:`~claude_sql.infrastructure.bedrock.structured_output` — the pydantic →
  Bedrock JSON-Schema-subset flattening and the four live ``*_SCHEMA`` constants.

This package imports boto3/botocore at module top (via ``client``), so it must
stay off any eager import path reachable from ``import claude_sql.interfaces.cli.app``.
"""

from __future__ import annotations
