"""Unit tests for kappa_worker."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from claude_sql.evals import kappa_worker as kw


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


def test_cohens_kappa_precomputed_categories_match() -> None:
    # Passing the category support explicitly must be byte-identical to
    # letting cohens_kappa derive it — including categories absent from
    # the arrays (they contribute pa*pb with pa==0).
    a = np.array([1, 1, 0, 0, 1, 0])
    b = np.array([1, 0, 0, 1, 1, 0])
    auto = kw.cohens_kappa(a, b)
    explicit = kw.cohens_kappa(a, b, sorted({0, 1}))
    superset = kw.cohens_kappa(a, b, sorted({0, 1, 2, 3}))  # extra cats are no-ops
    assert explicit == auto
    assert superset == auto


def test_bootstrap_hoisted_categories_byte_identical() -> None:
    # The hoisted-category bootstrap must match a per-call recompute
    # exactly. Bootstrap resamples draw with replacement from the same
    # arrays, so their support is always a subset of the full arrays'.
    rng = np.random.default_rng(7)
    a = rng.integers(0, 4, size=300)
    b = np.where(rng.random(300) < 0.7, a, rng.integers(0, 4, size=300))

    def bootstrap_per_call(aa, bb, n_bootstrap, seed=kw.RNG_SEED):
        r = np.random.default_rng(seed)
        n = len(aa)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = r.integers(0, n, size=n)
            # recompute support inside the loop (the pre-optimization shape)
            cats = sorted(set(aa[idx].tolist()) | set(bb[idx].tolist()))
            samples[i] = kw.cohens_kappa(aa[idx], bb[idx], cats)
        return (
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        )

    lo_h, hi_h = kw.bootstrap_kappa_ci(a, b, n_bootstrap=500, seed=42)
    lo_p, hi_p = bootstrap_per_call(a, b, n_bootstrap=500, seed=42)
    assert lo_h == lo_p
    assert hi_h == hi_p


def test_bootstrap_vectorized_matches_per_call_multicat() -> None:
    # The bootstrap is vectorized over the resample axis (one 2-D index draw
    # instead of n_bootstrap sequential 1-D draws). This must stay byte-
    # identical to a sequential per-call reference even at multi-category
    # scale (C=5), where np.sum over the category axis would reassociate and
    # drift the last ULP — the implementation accumulates the C products
    # sequentially to avoid exactly that.
    rng = np.random.default_rng(99)
    a = rng.integers(0, 5, size=250)
    b = np.where(rng.random(250) < 0.6, a, rng.integers(0, 5, size=250))

    def bootstrap_per_call(aa, bb, n_bootstrap, seed=kw.RNG_SEED):
        r = np.random.default_rng(seed)
        n = len(aa)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = r.integers(0, n, size=n)
            cats = sorted(set(aa[idx].tolist()) | set(bb[idx].tolist()))
            samples[i] = kw.cohens_kappa(aa[idx], bb[idx], cats)
        return (
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        )

    lo_v, hi_v = kw.bootstrap_kappa_ci(a, b, n_bootstrap=1000, seed=42)
    lo_p, hi_p = bootstrap_per_call(a, b, n_bootstrap=1000, seed=42)
    assert lo_v == lo_p
    assert hi_v == hi_p


def test_bootstrap_all_same_category_no_nan() -> None:
    # Degenerate resamples (every draw is one category) give pe == 1.0; the
    # vectorized element-wise guard must yield 0.0, never NaN/inf.
    a = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    lo, hi = kw.bootstrap_kappa_ci(a, b, n_bootstrap=300, seed=42)
    assert lo == 0.0
    assert hi == 0.0


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
    with pytest.raises(ValueError, match="axis"):
        kw.load_scores(p)
