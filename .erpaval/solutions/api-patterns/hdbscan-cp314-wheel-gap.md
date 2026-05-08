# hdbscan 0.8.42 is the lone Python 3.14 blocker in the scientific-python closure

**Category:** api-patterns
**Tags:** python-3.14, hdbscan, wheels, uv-tool-install, cp314, scientific-python
**Applies to:** any project shipping hdbscan via `uv tool install`
**Date:** 2026-05-08

## Fact

As of 2026-05-08, `hdbscan 0.8.42` (latest on PyPI) ships **no cp314
wheels**. Its wheel set stops at cp313. `sdist` has no
`python_requires` upper bound or classifier lockout, so pip/uv will
fall back to building from source on Python 3.14 — requiring Cython,
numpy, and a C toolchain at install time.

**Why:** `uv tool install <pkg>` is the canonical end-user path for
CLI tools built on hdbscan. End users don't have (and shouldn't need)
a C toolchain. Sdist-fallback means the install silently gets slow and
brittle, or outright fails on machines without compilers.

## Blast radius

Every other native dep in the scientific-python closure **does** ship
cp314 wheels as of this date:

- `duckdb 1.5.2` — 7 cp314 wheels
- `numpy 2.4.4` — 21 cp314 wheels
- `pyarrow 24.0.0` — 14 cp314 wheels
- `pydantic-core 2.46.4` — 30 cp314 wheels
- `scipy 1.17.1` — 20 cp314 wheels
- `scikit-learn 1.8.0` — 12 cp314 wheels
- `pyyaml 6.0.3` — 18 cp314 wheels
- `polars 1.40.1` — universal `py3-none-any` wheel (runtime native
  dispatch)
- `numba 0.65.1` — 8 cp314 wheels (no longer the historical blocker)
- `llvmlite 0.47.0` — 8 cp314 wheels

So hdbscan is literally the only gate for a scientific-python CLI
closure today.

## Decision tree

**If your project hard-depends on hdbscan AND ships via
`uv tool install`:**
- Target `requires-python = ">=3.13"` — not 3.14.
- Pin `.python-version = 3.13` and `mise.toml [tools].python = "3.13"`.

**If hdbscan is optional / server-side only:**
- You can target 3.14; users installing hdbscan on 3.14 are expected to
  have a build toolchain.

**If your project uses a hdbscan alternative (scikit-learn ≥1.3
ships `cluster.HDBSCAN`):**
- Drop the hdbscan dep. scikit-learn's port covers ~95% of hdbscan's
  surface; has cp314 wheels; smaller install footprint.

## Watch list

Active build-pipeline overhaul commits on `scikit-learn-contrib/hdbscan`
(2026-05-07: "Try to use pyproject more and use uv in build pipeline")
suggest `0.8.43` is imminent with cp314 wheels. When it ships:

```bash
# Verify cp314 wheels exist
curl -s https://pypi.org/pypi/hdbscan/json | jq '.urls[].filename' | grep cp314

# If present, one-line flip:
#   pyproject.toml   requires-python = ">=3.14"
#   .python-version  3.14
#   mise.toml        python = "3.14"
# Then: uv lock --upgrade
```

Note `networkx 3.6.1` excludes `!=3.14.1` (stdlib regression window) —
benign at 3.14.0 / 3.14.2+. Resolve to one of those, never 3.14.1.

## See also

- ADR `docs/adr/0015-stack-modernization.md` in claude-sql
  (session-638df1).
- Upstream: https://github.com/scikit-learn-contrib/hdbscan/issues/688
  (cp313 issue, resolved by 0.8.42 — template for filing cp314).
