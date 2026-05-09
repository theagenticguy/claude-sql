# CycloneDX SBOM for uv projects: `cyclonedx-py environment .venv`

**Category:** api-patterns
**Tags:** cyclonedx, sbom, uv, python, supply-chain, github-actions
**Applies to:** any Python project managed by uv that needs a CycloneDX SBOM
**Date:** 2026-05-08 (updated 2026-05-09 — flag-shape fix)

## Situation

`uv.lock` is the authoritative dep graph for uv-based Python projects,
but three common SBOM tools don't speak uv:

- **`cdxgen`** — lists `requirements.txt`, `setup.py`, `pyproject.toml`,
  `poetry.lock` as supported Python manifests. `uv.lock` is not in the
  list. Running `cdxgen -t python` will either read pyproject.toml
  partial info or fail.
- **`uv export`** — supports `requirements.txt` and a few others, but
  as of 2026-05-08 has no `--format cyclonedx` flag.
- **`cyclonedx-python`'s lockfile parsers** — the docs explicitly
  state "uv manifest and lockfile are not explicitly supported"
  alongside its poetry/pipenv/etc. parsers.

## The working path

`cyclonedx-python` *does* ship a first-class `environment` adapter that
walks the installed `.venv` and emits the resolved dep graph as
CycloneDX JSON. uv's `.venv` is a standard Python venv, so this Just
Works:

```bash
uv sync --frozen    # or --locked --all-extras --all-groups
uvx --from cyclonedx-bom cyclonedx-py environment .venv \
  --output-format JSON \
  --output-file SBOM.cdx.json
```

The `--from cyclonedx-bom` tells `uvx` which PyPI package ships the
`cyclonedx-py` entry point (the package name and CLI name differ).
Run inside a workflow or locally the same way.

**Flag-shape gotcha (caught 2026-05-09 on the v0.3.0 release).** The
`environment` subcommand only accepts `--output-format` (`--of`) and
`--output-file` (`-o`). It does **NOT** accept `--schema-version` or
`--outfile`. Earlier `--schema-version 1.6 --outfile ...` invocations
silently passed local checks because `uvx --help` resolves at the
top-level parser; the failure shows up only when the subcommand actually
runs ("unrecognized arguments"). Schema version defaults to the latest
spec the installed `cyclonedx-bom` supports — leave it implicit unless
you have a specific consumer that needs a pinned older spec, in which
case use `--validate`/`--no-validate` for the surrounding sanity check
rather than reaching for a non-existent flag.

## Release workflow

```yaml
# .github/workflows/sbom.yml
name: SBOM
on:
  release: { types: [published] }
  workflow_dispatch:
permissions: { contents: write }
jobs:
  sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: jdx/mise-action@v4
      - run: uv sync --locked --all-extras --all-groups
      - name: Generate CycloneDX SBOM from .venv
        run: |
          uvx --from cyclonedx-bom cyclonedx-py environment .venv \
            --output-format JSON \
            --output-file SBOM.cdx.json
      - uses: actions/upload-artifact@v7
        with: { name: sbom, path: SBOM.cdx.json }
      - name: Attach SBOM to release
        if: github.event_name == 'release'
        env: { GH_TOKEN: "${{ secrets.GITHUB_TOKEN }}" }
        run: gh release upload "${{ github.event.release.tag_name }}" SBOM.cdx.json --clobber
```

## Tradeoffs vs lockfile parsing

- **+** Reflects the actually-installed graph, not a resolver prediction.
  If `uv sync` applies any platform-specific markers or optional-dep
  selections, those are captured.
- **+** Works transparently across dep managers — same tool for uv,
  poetry, pip, Pipenv projects.
- **−** Requires a populated `.venv` (runs after `uv sync`), which adds
  a couple of seconds vs parsing a static lockfile.
- **−** Emits only what's *installed*, so unused optional groups don't
  appear. For an SBOM of all possible deps, re-run with different
  group selections.

## When to revisit

If `cyclonedx-python` or `cdxgen` adds native `uv.lock` support, flip
the workflow over — a static parser is faster than sync+walk. Watch
for `--from cyclonedx-bom` to add a `cyclonedx-py uv` subcommand.

## See also

- cyclonedx-python docs: https://github.com/CycloneDX/cyclonedx-python
- cdxgen (for comparison): https://github.com/CycloneDX/cdxgen
- claude-sql commit `add5117` on `chore/ci-hardening` (session-c4635d).
