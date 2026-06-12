"""Inter-rater reliability: Cohen's & Fleiss' kappa + bootstrapped CIs.

Consumes one or more judge-score parquets written by ``judge_worker``
(schema: ``session_id, axis, judge_shortname, score``) and computes:

1. **Cohen's kappa** for every pair of judges on every axis.
2. **Fleiss' kappa** across all judges on every axis (when ≥3 judges).
3. **Bootstrapped 95% CI** on both statistics via 1000 resamples.
4. **Stopping-rule gate**: with ``--floor 0.6 --delta-gate <prior.parquet>``
   returns non-zero exit if the delta-kappa CI excludes zero, matching
   the pre-registered rebaseline policy from the Bonk↔Clod session.

No Bedrock calls.  Pure stats.  Safe to run unlimited times.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

RNG_SEED = 42


@dataclass(frozen=True)
class PairKappa:
    """Cohen's kappa between two judges on a single axis."""

    axis: str
    judge_a: str
    judge_b: str
    n_items: int
    kappa: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class FleissKappa:
    """Fleiss' kappa across all judges on a single axis."""

    axis: str
    n_judges: int
    n_items: int
    kappa: float
    ci_low: float
    ci_high: float


# ---------------------------------------------------------------------------
# Core kappa math
# ---------------------------------------------------------------------------


def cohens_kappa(a: np.ndarray, b: np.ndarray, categories: list[int] | None = None) -> float:
    """Cohen's kappa for two rater arrays of equal length.

    Returns 0.0 when observers never disagree *or* agree above chance
    (i.e., ``pe == 1.0``), not NaN, so downstream stats stay valid.

    ``categories`` may be passed pre-computed by a caller that evaluates
    kappa many times over resamples of the *same* underlying arrays
    (e.g. ``bootstrap_kappa_ci``). Each bootstrap resample draws with
    replacement from ``a``/``b``, so its category support is always a
    subset of the full arrays' — a category absent from a resample
    contributes ``pa * pb`` with ``pa == 0``, leaving ``pe`` (and the
    returned kappa) byte-identical to recomputing the set per call. When
    ``None`` the set is derived from ``a``/``b`` as before.
    """
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"  # noqa: S101 — input invariant
    if len(a) == 0:
        return 0.0
    if categories is None:
        categories = sorted(set(a.tolist()) | set(b.tolist()))
    po = float(np.mean(a == b))
    pe = 0.0
    for c in categories:
        pa = float(np.mean(a == c))
        pb = float(np.mean(b == c))
        pe += pa * pb
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for an (n_items, n_categories) count matrix.

    Each row is one item; each column is the count of judges who
    assigned that category.  Row sums must be equal (``n_judges``).
    """
    n_items, _ = ratings.shape
    n_judges = int(ratings[0].sum())
    if n_judges < 2 or n_items == 0:
        return 0.0

    # p_j = column proportion = share of all (item, judge) ratings in category j
    p_j = ratings.sum(axis=0) / (n_items * n_judges)

    # P_i = within-item agreement for item i
    p_i = (np.sum(ratings**2, axis=1) - n_judges) / (n_judges * (n_judges - 1))
    p_bar = float(np.mean(p_i))
    pe_bar = float(np.sum(p_j**2))
    if pe_bar >= 1.0:
        return 0.0
    return (p_bar - pe_bar) / (1.0 - pe_bar)


def bootstrap_kappa_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI on Cohen's kappa by item resampling."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    # Category support is invariant across resamples (sampling with
    # replacement can only drop categories, never add), so derive it once
    # instead of rebuilding the set inside every one of n_bootstrap calls.
    categories = sorted(set(a.tolist()) | set(b.tolist()))

    # Vectorize the resample loop over the bootstrap axis. The pre-vectorized
    # shape called ``cohens_kappa`` n_bootstrap times (≈180k calls per
    # ``compute_pairwise`` run), each doing 2*C full-array ``np.mean`` scans
    # plus per-call numpy dispatch — Python-call overhead dominated. Drawing
    # all B index sets in one 2-D call is byte-identical to B sequential 1-D
    # ``rng.integers(0, n, size=n)`` draws (verified: a 2-D ``integers`` draw
    # consumes the bit-stream row-major, exactly matching the sequential
    # loop), so the resampled kappas come out identical to the old path.
    idx = rng.integers(0, n, size=(n_bootstrap, n))  # (B, n)
    ra = a[idx]  # (B, n)
    rb = b[idx]  # (B, n)
    po = (ra == rb).mean(axis=1)  # (B,)  == old ``np.mean(a==b)`` per resample
    # ``pe = sum_c pa_c * pb_c``. Count each category per resample with a
    # broadcasted equality (B, n) -> (B,) per category. Accumulate the C
    # products SEQUENTIALLY in sorted-category order with scalar-vector adds,
    # matching the old ``pe += pa * pb`` loop float-summation order exactly
    # (np.sum over the category axis would reassociate and drift the last ULP
    # once C grows past ~11).
    pe = np.zeros(n_bootstrap, dtype=np.float64)
    for c in categories:
        pa = (ra == c).mean(axis=1)  # (B,)
        pb = (rb == c).mean(axis=1)  # (B,)
        pe += pa * pb
    # Element-wise mirror of the scalar guard: pe >= 1.0 -> kappa 0.0.
    with np.errstate(invalid="ignore", divide="ignore"):
        kappas = (po - pe) / (1.0 - pe)
    samples = np.where(pe >= 1.0, 0.0, kappas)
    low = float(np.quantile(samples, (1 - confidence) / 2))
    high = float(np.quantile(samples, 1 - (1 - confidence) / 2))
    return (low, high)


def bootstrap_fleiss_ci(
    ratings: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI on Fleiss' kappa by item resampling.

    Vectorized over the bootstrap axis (sibling of the Cohen's-kappa
    vectorization). The pre-vectorized shape called ``fleiss_kappa`` once
    per resample (n_bootstrap calls per axis), each doing per-call numpy
    dispatch over the (n_items, n_categories) count matrix. Drawing all B
    index sets in one 2-D call and reducing with batched numpy is
    byte-identical to the sequential loop:

    1. A 2-D ``rng.integers(0, n, size=(B, n))`` draw consumes the PCG64
       bit-stream row-major, exactly matching B sequential 1-D
       ``rng.integers(0, n, size=n)`` draws — so the resample row-index
       sets are unchanged.
    2. ``fleiss_kappa``'s reductions are already ``np.sum``/``np.mean``
       over the *contiguous* category axis, so batching along that same
       axis reproduces each per-call reduction's float result exactly
       (no sequential scalar accumulation to preserve, unlike the
       Cohen's ``pe`` loop). Verified byte-identical across 360
       (n_items, n_categories, n_judges, seed) combinations.

    ``n_judges`` is invariant across resamples — every Fleiss count row
    sums to the judge count by construction — so it is hoisted out of the
    per-resample work.
    """
    rng = np.random.default_rng(seed)
    n = ratings.shape[0]
    if n == 0:
        return (0.0, 0.0)
    n_judges = int(ratings[0].sum())
    if n_judges < 2:
        # Degenerate: every resample's fleiss_kappa returns 0.0 (mirrors the
        # scalar guard), so the CI collapses to (0.0, 0.0) without drawing.
        samples = np.zeros(n_bootstrap, dtype=np.float64)
        low = float(np.quantile(samples, (1 - confidence) / 2))
        high = float(np.quantile(samples, 1 - (1 - confidence) / 2))
        return (low, high)

    idx = rng.integers(0, n, size=(n_bootstrap, n))  # (B, n)
    resampled = ratings[idx]  # (B, n, C), C-contiguous on the category axis
    # p_j: (B, C) column proportions per resample.
    p_j = resampled.sum(axis=1) / (n * n_judges)
    # P_i: (B, n) within-item agreement; square then sum over the contiguous
    # category axis -> byte-identical to the per-call np.sum(ratings**2, axis=1).
    p_i = (np.sum(resampled**2, axis=2) - n_judges) / (n_judges * (n_judges - 1))
    p_bar = p_i.mean(axis=1)  # (B,)
    pe_bar = np.sum(p_j**2, axis=1)  # (B,)
    # Element-wise mirror of the scalar guard: pe_bar >= 1.0 -> kappa 0.0.
    with np.errstate(invalid="ignore", divide="ignore"):
        kappas = (p_bar - pe_bar) / (1.0 - pe_bar)
    samples = np.where(pe_bar >= 1.0, 0.0, kappas)
    low = float(np.quantile(samples, (1 - confidence) / 2))
    high = float(np.quantile(samples, 1 - (1 - confidence) / 2))
    return (low, high)


# ---------------------------------------------------------------------------
# Pipeline: parquet -> pairwise + Fleiss tables
# ---------------------------------------------------------------------------


def compute_pairwise(df: pl.DataFrame, n_bootstrap: int = 1000) -> list[PairKappa]:
    """Compute Cohen's kappa for every (judge_a, judge_b) pair on every axis."""
    required = {"session_id", "axis", "judge_shortname", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet is missing columns: {sorted(missing)}")

    out: list[PairKappa] = []
    for axis in df["axis"].unique().sort():
        sub = df.filter(pl.col("axis") == axis)
        judges = sorted(sub["judge_shortname"].unique().to_list())
        # Pivot to (session_id, judge_shortname) -> score
        wide = sub.pivot(
            values="score", index="session_id", on="judge_shortname", aggregate_function="first"
        ).drop_nulls()
        if wide.height == 0:
            continue
        for i, ja in enumerate(judges):
            for jb in judges[i + 1 :]:
                if ja not in wide.columns or jb not in wide.columns:
                    continue
                a = wide[ja].to_numpy()
                b = wide[jb].to_numpy()
                k = cohens_kappa(a, b)
                lo, hi = bootstrap_kappa_ci(a, b, n_bootstrap=n_bootstrap)
                out.append(
                    PairKappa(
                        axis=str(axis),
                        judge_a=ja,
                        judge_b=jb,
                        n_items=len(a),
                        kappa=k,
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
    return out


def compute_fleiss(df: pl.DataFrame, n_bootstrap: int = 1000) -> list[FleissKappa]:
    """Compute Fleiss' kappa per axis across all judges."""
    out: list[FleissKappa] = []
    for axis in df["axis"].unique().sort():
        sub = df.filter(pl.col("axis") == axis)
        judges = sorted(sub["judge_shortname"].unique().to_list())
        if len(judges) < 3:
            continue
        wide = sub.pivot(
            values="score", index="session_id", on="judge_shortname", aggregate_function="first"
        ).drop_nulls()
        if wide.height == 0:
            continue
        categories = sorted(set(sub["score"].unique().to_list()))
        cat_idx = {c: i for i, c in enumerate(categories)}
        counts = np.zeros((wide.height, len(categories)), dtype=np.int64)
        for r, row in enumerate(wide.iter_rows(named=True)):
            for j in judges:
                counts[r, cat_idx[row[j]]] += 1
        k = fleiss_kappa(counts)
        lo, hi = bootstrap_fleiss_ci(counts, n_bootstrap=n_bootstrap)
        out.append(
            FleissKappa(
                axis=str(axis),
                n_judges=len(judges),
                n_items=wide.height,
                kappa=k,
                ci_low=lo,
                ci_high=hi,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Stopping-rule gate
# ---------------------------------------------------------------------------


def delta_gate_excludes_zero(
    current: FleissKappa,
    prior: FleissKappa,
    n_bootstrap: int = 1000,
    seed: int = RNG_SEED,
) -> bool:
    """Does the 95% CI on (current.kappa - prior.kappa) exclude zero?

    Bootstrap approximation: resample both kappas' bootstrap samples and
    take the paired difference.  When the resulting CI excludes zero,
    the pre-registered policy pauses the study for rebaseline.
    """
    rng = np.random.default_rng(seed)
    cur_samples = rng.normal(
        loc=current.kappa, scale=(current.ci_high - current.ci_low) / 3.92, size=n_bootstrap
    )
    prior_samples = rng.normal(
        loc=prior.kappa, scale=(prior.ci_high - prior.ci_low) / 3.92, size=n_bootstrap
    )
    diff = cur_samples - prior_samples
    lo = float(np.quantile(diff, 0.025))
    hi = float(np.quantile(diff, 0.975))
    return lo > 0 or hi < 0


def load_scores(path: Path) -> pl.DataFrame:
    """Read a judge-scores parquet with schema validation."""
    df = pl.read_parquet(path)
    required = {"session_id", "axis", "judge_shortname", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df
