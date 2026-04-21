"""Pre-registration: freeze + replay study manifests.

The ``freeze`` subcommand hashes the full study spec (rubric YAML,
judge panel, commit SHA, embed model, session-scoping rule) into a
deterministic manifest SHA and writes it under ``~/.claude/studies/<sha>/``.

The ``replay`` subcommand reads a manifest by SHA and rebuilds a
``Study`` object so downstream workers (``judge``, ``ungrounded-claim``,
``kappa``) can execute with the exact locked parameters.

This is the IRR study's audit trail.  Every parquet the workers write
carries the manifest SHA in a ``freeze_sha`` column.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from claude_sql import judges as judge_catalog


@dataclass(frozen=True)
class SessionScope:
    """Session-scoping rule for the study."""

    min_turns: int = 10
    max_turns: int = 40
    max_interrupt_minutes: int = 15
    kind: str = "mechanical"


@dataclass(frozen=True)
class Study:
    """A pre-registered IRR study specification."""

    rubric_path: str
    rubric_content_hash: str
    panel_shortnames: tuple[str, ...]
    panel_model_ids: tuple[str, ...]
    embed_model_id: str
    commit_sha: str
    session_scope: SessionScope
    seed: int
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def manifest_sha(self) -> str:
        """Deterministic SHA256 over the parameterised fields (excludes created_at)."""
        payload = {
            "rubric_content_hash": self.rubric_content_hash,
            "panel_model_ids": list(self.panel_model_ids),
            "embed_model_id": self.embed_model_id,
            "commit_sha": self.commit_sha,
            "session_scope": asdict(self.session_scope),
            "seed": self.seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["manifest_sha"] = self.manifest_sha
        return d


def _read_rubric(rubric_path: Path) -> tuple[str, str]:
    """Read rubric YAML/JSON from disk and return (content, SHA256 hash).

    Supports YAML (loaded as text, no parsing) and JSON; we hash the
    raw bytes so whitespace differences produce different manifests,
    which is what we want — even a reformatted rubric is a new study.
    """
    if not rubric_path.exists():
        raise FileNotFoundError(f"rubric not found: {rubric_path}")
    content = rubric_path.read_text(encoding="utf-8")
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, h


def _git_commit_sha(repo: Path) -> str:
    """Return the current ``git rev-parse HEAD`` for ``repo``, or ``"<dirty>"``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = out.stdout.strip()
        # Flag dirty working trees so the commit hash doesn't over-claim reproducibility
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return sha if not dirty.stdout.strip() else f"{sha}-dirty"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "<no-git>"


def _studies_root() -> Path:
    root = Path(os.path.expanduser("~/.claude/studies"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def freeze(
    rubric_path: Path,
    panel_shortnames: tuple[str, ...],
    embed_model_id: str = "global.cohere.embed-v4:0",
    session_scope: SessionScope | None = None,
    seed: int = 42,
    repo: Path | None = None,
) -> Study:
    """Create and persist a study manifest.

    Writes ``~/.claude/studies/<sha>/manifest.json`` plus a copy of the
    rubric so the manifest is self-contained.  Returns the ``Study``.
    """
    rubric_content, rubric_hash = _read_rubric(rubric_path)
    judges_resolved = judge_catalog.panel(list(panel_shortnames))
    panel_model_ids = tuple(j.model_id for j in judges_resolved)
    scope = session_scope or SessionScope()
    study = Study(
        rubric_path=str(rubric_path),
        rubric_content_hash=rubric_hash,
        panel_shortnames=tuple(panel_shortnames),
        panel_model_ids=panel_model_ids,
        embed_model_id=embed_model_id,
        commit_sha=_git_commit_sha(repo or Path.cwd()),
        session_scope=scope,
        seed=seed,
    )
    study_dir = _studies_root() / study.manifest_sha
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "manifest.json").write_text(
        json.dumps(study.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (study_dir / "rubric.yaml").write_text(rubric_content, encoding="utf-8")
    return study


def replay(manifest_sha: str) -> Study:
    """Load a previously-frozen study by its manifest SHA."""
    study_dir = _studies_root() / manifest_sha
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    d = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = SessionScope(**d["session_scope"])
    return Study(
        rubric_path=d["rubric_path"],
        rubric_content_hash=d["rubric_content_hash"],
        panel_shortnames=tuple(d["panel_shortnames"]),
        panel_model_ids=tuple(d["panel_model_ids"]),
        embed_model_id=d["embed_model_id"],
        commit_sha=d["commit_sha"],
        session_scope=scope,
        seed=d["seed"],
        created_at_utc=d["created_at_utc"],
    )


def list_studies() -> list[dict[str, object]]:
    """Return a summary of every frozen study under ``~/.claude/studies/``."""
    root = _studies_root()
    out: list[dict[str, object]] = []
    for d in sorted(root.iterdir()):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        payload = json.loads(mf.read_text(encoding="utf-8"))
        out.append(
            {
                "manifest_sha": payload["manifest_sha"],
                "created_at_utc": payload["created_at_utc"],
                "commit_sha": payload["commit_sha"],
                "n_judges": len(payload["panel_shortnames"]),
            }
        )
    return out
