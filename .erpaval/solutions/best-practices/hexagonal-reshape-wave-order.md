# Hexagonal reshape of a live codebase: wave order that survives 826 tests

**Category:** best-practices
**Tags:** hexagonal, refactoring, migration, import-linter, waves, subagents
**Session:** session-4fb0fd (claude-sql v2: core/analytics/app → domain/application/infrastructure/interfaces, ~50 files, 751 tests, zero red gates at every wave boundary)

## The order that worked

0. **Independent unblocks first** (toolchain bump, dead-code prune, env-var
   plumbing, spec line-number re-pin) — parallel, file-disjoint.
1. **Skeleton + ports + shims-in**: create layer dirs, `application/ports.py`
   Protocols verified against real call sites, move only clean adapters with
   re-export shims at old paths. Old import-linter contract stays valid because
   shims keep the old packages alive; new packages aren't in the contract yet.
2. **The splits** (multi-target modules → per-name routing into 2-4 layers),
   file-disjoint so 4 agents run concurrently.
3. **Workers → use_cases** with pure-math lifts to domain.
4. **The stateful hazard LAST, tests-first**: extend the regression suite for
   the known lifecycle bug (here: DuckDB register→write→rebind), prove it green
   on the OLD code, then cut. Injectable fn refs let existing module-object
   monkeypatches keep biting.
5. **Final cut alone (no concurrency)**: rewrite all imports per-name from each
   shim's actual re-export list, hand-fix string-coupled surfaces (monkeypatch
   strings, forbidden-eager-import lists, sys.modules probes, `python -m`
   subprocess paths, the import-linter contract itself), delete shims, flip the
   contract to the new DAG.

## Load-bearing details

- **Every wave ends `mise run check` green** — shims make this possible; the
  suite never goes red between waves, so any wedged agent costs one packet.
- **String-coupled surfaces are the sed-misses**: enumerate them in Explore
  (monkeypatch-by-string, forbidden-import lists, subprocess `-m` paths,
  contract package names, conftest collection imports) and hand them to the
  final-cut agent as an explicit checklist.
- **A strict linear layers DAG may be unreachable** while transitional packages
  remain (core↔infrastructure cycles). Two contracts — a layers DAG over the
  clean four + a `forbidden` fence on the transitional package — is sound and
  self-documenting; a `"domain | core"` sibling layer is NOT (hard cycle).
- **Byte-stability fences**: prompts and LLM-input renderers that checkpointing
  depends on move verbatim, with pre/post SHA comparison in validation — never
  reformatted.
- Concurrent-wave packets must name sibling-owned files explicitly ("do NOT
  edit X, sibling owns it") and share a conflict rule for co-extended files.

Related: [[alias-idiom-shim-for-monkeypatched-modules]].
