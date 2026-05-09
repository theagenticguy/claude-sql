"""Coverage top-up for :mod:`claude_sql.judges`.

The four ``all_*`` accessors are simple wrappers over module-level
tuples. They are part of the public API (downstream callers pick a
panel without re-importing the underlying constants), so we exercise
each one to lock in the contract.
"""

from __future__ import annotations

from claude_sql import judges


def test_all_primary_returns_primary_panel_tuple() -> None:
    """``all_primary`` is the public accessor for ``PRIMARY_PANEL`` (line 219)."""
    out = judges.all_primary()
    assert out is judges.PRIMARY_PANEL
    # 8 non-Anthropic, non-Amazon judges.
    assert len(out) == 8


def test_all_within_family_returns_holdout_panel() -> None:
    """``all_within_family`` is the public accessor for ``WITHIN_FAMILY_HOLDOUT`` (line 224)."""
    out = judges.all_within_family()
    assert out is judges.WITHIN_FAMILY_HOLDOUT
    for j in out:
        assert j.family == "anthropic"


def test_all_bulk_returns_amazon_lane() -> None:
    """``all_bulk`` is the public accessor for ``BULK_PANEL`` (line 229)."""
    out = judges.all_bulk()
    assert out is judges.BULK_PANEL
    for j in out:
        assert j.family == "amazon"


def test_all_excluded_returns_dropped_judges() -> None:
    """``all_excluded`` returns the evaluated-and-dropped judges (line 234)."""
    out = judges.all_excluded()
    assert out is judges.EXCLUDED_JUDGES
    shortnames = {j.shortname for j in out}
    assert "mistral-large-3" in shortnames
    assert "magistral-small" in shortnames
