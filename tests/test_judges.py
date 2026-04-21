"""Unit tests for the cross-provider judge catalog."""

from __future__ import annotations

import pytest

from claude_sql import judges


def test_catalog_has_ten_primary_judges() -> None:
    """Primary panel must span 10 non-within-family models."""
    assert len(judges.PRIMARY_PANEL) == 10
    for j in judges.PRIMARY_PANEL:
        assert j.family == "non-anthropic-non-amazon"
        assert j.role == "judge"


def test_within_family_holdout_is_anthropic_only() -> None:
    for j in judges.WITHIN_FAMILY_HOLDOUT:
        assert j.family == "anthropic"
        assert j.role == "within-family-holdout"


def test_bulk_panel_is_amazon_current_gen() -> None:
    """Bulk lane is Nova 2 Lite + Nova 2 embeddings. Nova Pro v1 is excluded."""
    shortnames = {j.shortname for j in judges.BULK_PANEL}
    assert "nova-2-lite" in shortnames
    assert "nova-2-mm-embed" in shortnames
    # Nova Pro v1 is stale gen; must not appear anywhere in catalog.
    for j in judges.catalog():
        assert "nova-pro" not in j.shortname, "Nova Pro v1 must not be in the panel"


def test_resolve_by_shortname() -> None:
    assert judges.resolve("kimi-k2.5").model_id == "moonshotai.kimi-k2.5"
    assert judges.resolve("deepseek-v3.2").model_id == "deepseek.v3.2"


def test_resolve_by_model_id() -> None:
    assert judges.resolve("moonshotai.kimi-k2.5").shortname == "kimi-k2.5"


def test_resolve_unknown_raises_with_catalog_hint() -> None:
    with pytest.raises(KeyError) as exc:
        judges.resolve("gpt-4o")
    # Error payload must include the available shortnames so agents see options.
    assert "kimi-k2.5" in str(exc.value)


def test_panel_preserves_order() -> None:
    got = judges.panel(["mistral-large-3", "kimi-k2.5", "deepseek-v3.2"])
    assert [j.shortname for j in got] == ["mistral-large-3", "kimi-k2.5", "deepseek-v3.2"]


def test_catalog_shortnames_unique() -> None:
    shortnames = [j.shortname for j in judges.catalog()]
    assert len(shortnames) == len(set(shortnames)), "shortnames must be unique"
