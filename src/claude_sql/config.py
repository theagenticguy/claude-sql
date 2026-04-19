"""Runtime configuration for claude-sql.

Pydantic v2 ``BaseSettings`` populated from env vars prefixed with ``CLAUDE_SQL_``.
Defaults are picked for a single-user devbox install pointing at
``~/.claude/projects/**/*.jsonl``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_glob() -> str:
    # Top-level session transcripts only.  Subagent side-files live one level
    # deeper under ``<session>/subagents/`` and are discovered via SUBAGENT_GLOB.
    return os.path.expanduser("~/.claude/projects/*/*.jsonl")


def _default_subagent_glob() -> str:
    return os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.jsonl")


def _default_subagent_meta_glob() -> str:
    return os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.meta.json")


def _default_embeddings_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/embeddings.parquet"))


# Model pricing per 1M tokens (in_rate, out_rate).  Mirrors claude-mine/transform.py.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
}


class Settings(BaseSettings):
    """Environment-driven settings for claude-sql.

    All fields are overridable via env vars prefixed ``CLAUDE_SQL_`` (e.g.
    ``CLAUDE_SQL_REGION=us-west-2``) or via ``.env`` in the working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_SQL_",
        env_file=".env",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Data discovery
    # ------------------------------------------------------------------
    default_glob: str = Field(default_factory=_default_glob)
    subagent_glob: str = Field(default_factory=_default_subagent_glob)
    subagent_meta_glob: str = Field(default_factory=_default_subagent_meta_glob)

    # ------------------------------------------------------------------
    # Bedrock / embedding
    # ------------------------------------------------------------------
    region: str = "us-east-1"
    #: Cohere Embed v4 global CRIS profile. Sustained 223 vec/s with zero
    #: throttling at concurrency=8 in testing; US-only and direct on-demand
    #: both throttle hard at low TPM. No reason to expose the knob.
    model_id: str = "global.cohere.embed-v4:0"

    output_dimension: Literal[256, 512, 1024, 1536] = 1024
    embedding_type: Literal["int8", "float", "uint8", "binary", "ubinary"] = "int8"
    concurrency: int = 8
    batch_size: int = 96

    embeddings_parquet_path: Path = Field(default_factory=_default_embeddings_parquet)

    # ------------------------------------------------------------------
    # VSS / HNSW
    # ------------------------------------------------------------------
    hnsw_metric: Literal["cosine", "l2sq", "ip"] = "cosine"
    hnsw_ef_construction: int = 128
    hnsw_ef_search: int = 64
    hnsw_m: int = 16
    hnsw_m0: int = 32

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    pricing: dict[str, tuple[float, float]] = Field(default_factory=lambda: dict(DEFAULT_PRICING))

    @property
    def active_model_id(self) -> str:
        """Return the Bedrock model ID (kept as a property for call-site stability)."""
        return self.model_id
