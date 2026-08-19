"""The shared source-mtime sidecar gate (``infrastructure/freshness.py``).

One implementation serves ``cluster``, ``terms``, and ``community``, so its
edge cases are tested here once instead of three times over. The rule that
matters: freshness must fail CLOSED. Every branch that cannot substantiate "this
output was built from exactly this input" has to recompute, because the failure
mode on the other side is an output that silently describes a state that no
longer exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claude_sql.infrastructure.freshness import (
    is_output_fresh,
    newest_mtime_ns,
    sidecar_for,
    stamp_output,
)

if TYPE_CHECKING:
    from pathlib import Path


def _output(tmp_path: Path, *, name: str = "out.parquet", size: int = 128) -> Path:
    path = tmp_path / name
    path.write_bytes(b"x" * size)
    return path


# ---------------------------------------------------------------------------
# newest_mtime_ns
# ---------------------------------------------------------------------------


def test_newest_mtime_is_none_for_an_absent_path(tmp_path: Path) -> None:
    assert newest_mtime_ns(tmp_path / "nope") is None


def test_newest_mtime_of_a_file_is_its_own(tmp_path: Path) -> None:
    path = _output(tmp_path)
    assert newest_mtime_ns(path) == path.stat().st_mtime_ns


def test_newest_mtime_of_a_directory_walks_the_tree(tmp_path: Path) -> None:
    """LanceDB appends new fragment files; the directory's own mtime can lag."""
    store = tmp_path / "store"
    (store / "nested" / "deep").mkdir(parents=True)
    leaf = store / "nested" / "deep" / "fragment.lance"
    leaf.write_bytes(b"data")

    assert newest_mtime_ns(store) == leaf.stat().st_mtime_ns


# ---------------------------------------------------------------------------
# sidecar_for
# ---------------------------------------------------------------------------


def test_sidecar_names_the_input_it_stamps(tmp_path: Path) -> None:
    """Two inputs must not share one stamp file.

    ``community`` keys on the embeddings store and ``terms`` on the clusters
    parquet; a shared sidecar name would let one stage read the other's stamp
    and conclude it was fresh.
    """
    out = tmp_path / "session_communities.parquet"

    embeddings = sidecar_for(out, input_name="embeddings")
    clusters = sidecar_for(out, input_name="clusters")

    assert embeddings != clusters
    assert embeddings.name.endswith(".embeddings_mtime")
    assert clusters.name.endswith(".clusters_mtime")


# ---------------------------------------------------------------------------
# is_output_fresh — every branch fails closed
# ---------------------------------------------------------------------------


def test_fresh_when_the_stamp_matches_exactly(tmp_path: Path) -> None:
    out = _output(tmp_path)
    sidecar = sidecar_for(out, input_name="src")
    stamp_output(sidecar, 999)

    assert is_output_fresh(out, sidecar=sidecar, input_mtime_ns=999)


def test_stale_when_the_input_moved(tmp_path: Path) -> None:
    out = _output(tmp_path)
    sidecar = sidecar_for(out, input_name="src")
    stamp_output(sidecar, 999)

    assert not is_output_fresh(out, sidecar=sidecar, input_mtime_ns=1000)


def test_stale_when_the_input_moved_backwards(tmp_path: Path) -> None:
    """A restored backup is a change too — the comparison is equality, not ``>``."""
    out = _output(tmp_path)
    sidecar = sidecar_for(out, input_name="src")
    stamp_output(sidecar, 999)

    assert not is_output_fresh(out, sidecar=sidecar, input_mtime_ns=500)


def test_stale_when_no_sidecar_exists(tmp_path: Path) -> None:
    """The pre-fix behaviour: an existing output is NOT a freshness claim."""
    out = _output(tmp_path)

    assert not is_output_fresh(out, sidecar=sidecar_for(out, input_name="src"), input_mtime_ns=1)


def test_stale_when_the_output_is_absent(tmp_path: Path) -> None:
    out = tmp_path / "missing.parquet"
    sidecar = sidecar_for(out, input_name="src")
    stamp_output(sidecar, 1)

    assert not is_output_fresh(out, sidecar=sidecar, input_mtime_ns=1)


def test_stale_when_the_output_is_truncated(tmp_path: Path) -> None:
    """A zero-or-tiny parquet is a failed write, not a cache hit."""
    out = _output(tmp_path, size=4)
    sidecar = sidecar_for(out, input_name="src")
    stamp_output(sidecar, 1)

    assert not is_output_fresh(out, sidecar=sidecar, input_mtime_ns=1)


def test_stale_when_the_input_mtime_is_unreadable(tmp_path: Path) -> None:
    """``None`` input mtime cannot substantiate anything."""
    out = _output(tmp_path)
    sidecar = sidecar_for(out, input_name="src")
    stamp_output(sidecar, 1)

    assert not is_output_fresh(out, sidecar=sidecar, input_mtime_ns=None)


# ---------------------------------------------------------------------------
# stamp_output
# ---------------------------------------------------------------------------


def test_stamping_a_none_mtime_writes_nothing(tmp_path: Path) -> None:
    """Otherwise the next run's exact-match would succeed against a placeholder."""
    out = _output(tmp_path)
    sidecar = sidecar_for(out, input_name="src")

    stamp_output(sidecar, None)

    assert not sidecar.exists()


def test_stamping_creates_missing_parents(tmp_path: Path) -> None:
    sidecar = tmp_path / "nested" / "dir" / "out.parquet.src_mtime"

    stamp_output(sidecar, 7)

    assert sidecar.read_text() == "7"


def test_a_stamp_round_trips(tmp_path: Path) -> None:
    out = _output(tmp_path)
    sidecar = sidecar_for(out, input_name="src")
    mtime = newest_mtime_ns(out)

    stamp_output(sidecar, mtime)

    assert is_output_fresh(out, sidecar=sidecar, input_mtime_ns=mtime)
