"""Unit tests for kappa_worker."""

from __future__ import annotations

import numpy as np
import polars as pl

from claude_sql import kappa_worker as kw


def test_cohens_kappa_perfect_agreement() -> None:
    a = np.array([1, 1, 0, 0, 1])
    b = np.array([1, 1, 0, 0, 1])
    assert kw.cohens_kappa(a, b) == 1.0


def test_cohens_kappa_no_agreement_beyond_chance() -> None:
    a = np.array([1, 0, 1, 0, 1, 0])
    b = np.array([0, 1, 0, 1, 0, 1])
    k = kw.cohens_kappa(a, b)
    assert k == -1.0


def test_cohens_kappa_all_same_category() -> None:
    # pe == 1.0 edge case must return 0, not NaN
    a = np.array([1, 1, 1, 1])
    b = np.array([1, 1, 1, 1])
    assert kw.cohens_kappa(a, b) == 0.0


def test_fleiss_kappa_perfect_agreement() -> None:
    # 3 judges, 3 items, all agree on category 0
    ratings = np.array([[3, 0], [3, 0], [3, 0]])
    # pe_bar = 1.0 -> returns 0.0 (edge case, all one category)
    assert kw.fleiss_kappa(ratings) == 0.0


def test_fleiss_kappa_with_variation() -> None:
    # 3 judges, 4 items, mixed agreement across 2 categories
    ratings = np.array([[3, 0], [0, 3], [2, 1], [3, 0]])
    k = kw.fleiss_kappa(ratings)
    assert 0.0 < k < 1.0


def test_bootstrap_ci_is_bounded() -> None:
    a = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    b = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    lo, hi = kw.bootstrap_kappa_ci(a, b, n_bootstrap=200, seed=42)
    assert lo <= hi
    assert -1.0 <= lo <= 1.0
    assert -1.0 <= hi <= 1.0


def test_compute_pairwise_smoke() -> None:
    df = pl.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "axis": ["correction_required"] * 6,
            "judge_shortname": ["kimi-k2.5", "deepseek-v3.2"] * 3,
            "score": [1, 1, 0, 0, 1, 1],
        }
    )
    out = kw.compute_pairwise(df, n_bootstrap=50)
    assert len(out) == 1
    assert out[0].judge_a == "deepseek-v3.2"
    assert out[0].judge_b == "kimi-k2.5"
    assert out[0].n_items == 3


def test_compute_fleiss_needs_three_judges() -> None:
    df = pl.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "axis": ["fab"] * 6,
            "judge_shortname": ["a", "b", "c"] * 2,
            "score": [1, 1, 0, 0, 0, 1],
        }
    )
    out = kw.compute_fleiss(df, n_bootstrap=50)
    assert len(out) == 1
    assert out[0].n_judges == 3
    assert out[0].n_items == 2


def test_delta_gate_excludes_zero_for_big_gap() -> None:
    current = kw.FleissKappa(axis="x", n_judges=3, n_items=10, kappa=0.8, ci_low=0.75, ci_high=0.85)
    prior = kw.FleissKappa(axis="x", n_judges=3, n_items=10, kappa=0.3, ci_low=0.25, ci_high=0.35)
    assert kw.delta_gate_excludes_zero(current, prior) is True


def test_delta_gate_does_not_exclude_zero_for_overlap() -> None:
    current = kw.FleissKappa(axis="x", n_judges=3, n_items=10, kappa=0.6, ci_low=0.4, ci_high=0.8)
    prior = kw.FleissKappa(axis="x", n_judges=3, n_items=10, kappa=0.5, ci_low=0.3, ci_high=0.7)
    assert kw.delta_gate_excludes_zero(current, prior) is False


def test_load_scores_rejects_missing_columns(tmp_path) -> None:
    # Write a parquet missing 'axis'
    p = tmp_path / "bad.parquet"
    pl.DataFrame({"session_id": ["s1"], "judge_shortname": ["x"], "score": [1]}).write_parquet(p)
    try:
        kw.load_scores(p)
    except ValueError as e:
        assert "axis" in str(e)
    else:
        raise AssertionError("expected ValueError on missing column")
