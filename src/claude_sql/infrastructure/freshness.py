"""Source-mtime sidecars: skip a derived stage when its input has not moved.

Three analytics stages are expensive and derived from exactly one input:

======================  ====================================  ===============
stage                   input                                 output
======================  ====================================  ===============
``cluster``             the LanceDB embeddings dataset        clusters.parquet
``terms``               clusters.parquet (the cluster ids)    cluster_terms.parquet
``community``           the LanceDB embeddings dataset        session_communities.parquet
======================  ====================================  ===============

Each writes a sidecar next to its output holding the newest ``mtime_ns`` of the
input it was built from, and skips itself when that value has not changed. This
module is the single implementation of that rule.

**Why not "skip when the output exists".** That is what ``terms`` and
``community`` used to do, and it inverted the pipeline: ``cluster`` re-fit on
every tick (its input, the embeddings, genuinely moved), while the two stages
downstream of it skipped forever because their outputs existed. On the live
corpus that left ``cluster_terms`` and ``session_communities`` pinned at
2026-07-27 while ``clusters.parquet`` was rebuilt roughly every two hours —
c-TF-IDF labels describing a ``cluster_id`` partition that no longer existed,
because HDBSCAN re-mints those ids on every fit. An existence gate cannot
express "my input changed"; an mtime gate can.

The comparison is exact equality, not ``>``: an input whose mtime moved
backwards (a restored backup, a re-materialized cache) is also a reason to
recompute, for the same reason the raw watermark treats it as a change.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

#: Minimum byte size for an output to count as present. A zero-byte or
#: truncated parquet is a failed write, not a cache hit.
_MIN_OUTPUT_BYTES: int = 16


def newest_mtime_ns(path: Path) -> int | None:
    """The newest ``mtime_ns`` at or under ``path``; ``None`` when absent.

    Walks the tree for a directory because the LanceDB dataset writes new
    fragment files on append — the directory's own mtime does not always move,
    so the deepest child mtime is what tracks the data.

    Unreadable children are skipped rather than raised: a freshness probe must
    not be the thing that fails a stage.
    """
    if not path.exists():
        return None
    candidates = [path.stat().st_mtime_ns]
    if path.is_dir():
        for child in path.rglob("*"):
            with contextlib.suppress(OSError):
                candidates.append(child.stat().st_mtime_ns)
    return max(candidates)


def sidecar_for(output_path: Path, *, input_name: str) -> Path:
    """The sidecar path for ``output_path``'s freshness stamp.

    ``input_name`` is part of the filename so a stage whose input changes
    identity (embeddings → clusters) cannot silently read the previous input's
    stamp and conclude it is fresh.
    """
    return output_path.with_suffix(f"{output_path.suffix}.{input_name}_mtime")


def is_output_fresh(
    output_path: Path,
    *,
    sidecar: Path,
    input_mtime_ns: int | None,
) -> bool:
    """True when ``output_path`` was built from exactly this input state.

    Requires all four: a present and non-truncated output, a present sidecar, a
    readable input mtime, and an exact match. Any missing piece means recompute
    — a freshness claim that cannot be substantiated must fail closed.
    """
    if input_mtime_ns is None:
        return False
    if not output_path.exists() or output_path.stat().st_size <= _MIN_OUTPUT_BYTES:
        return False
    if not sidecar.exists():
        return False
    try:
        stamped = sidecar.read_text().strip()
    except OSError:
        return False
    return stamped == str(input_mtime_ns)


def stamp_output(sidecar: Path, input_mtime_ns: int | None) -> None:
    """Record the input mtime this output was built from.

    A ``None`` input mtime writes nothing: stamping an unreadable input would
    make the next run's exact-match comparison succeed against a placeholder.
    """
    if input_mtime_ns is None:
        return
    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(str(input_mtime_ns))
    except OSError as exc:
        # A failed stamp costs a recompute next run, which is correct-but-slow.
        # Failing the stage that just succeeded would be worse.
        logger.warning("could not write freshness sidecar {}: {}", sidecar, exc)


__all__ = [
    "is_output_fresh",
    "newest_mtime_ns",
    "sidecar_for",
    "stamp_output",
]
