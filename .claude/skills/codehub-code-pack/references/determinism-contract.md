# Determinism contract — auditor reference

Ground truth for the `codehub-code-pack` skill. Cite this file when the
user disputes a `packHash` mismatch, when a CI determinism gate fails,
or when a future contributor proposes adding a non-deterministic emitter
to `@opencodehub/pack`. The reference implementation in
`packages/pack/src/` is authoritative; this document describes the
contract that the implementation enforces.

## 9-item code-pack BOM

Every `codehub code-pack` invocation produces a directory of nine BOM
items plus a manifest. Same `(commit, tokenizer, budget)` → byte-
identical output:

1. `manifest.json` — pack_hash, commit SHA, tokenizer ID, schema version, counts
2. PageRank-ranked symbol skeleton
3. File tree with framework labels
4. Dependency graph / lockfile slice (exact versions)
5. Top-N AST-chunked files with byte offsets
6. SCIP-grounded cross-refs (community clusters + call graph)
7. Optional embeddings sidecar (`.parquet`)
8. Salient docstrings / SARIF findings by severity + rule
9. LICENSES / NOTICES + README.md + full determinism contract

## Invariants

- **graphHash byte-identity** holds before and after every pack-
  affecting commit — the `DuckDbStore` / `GraphDbStore` parity suite
  stays green.
- **packHash byte-identity** — same
  `(commit, tokenizer, budget, chonkie_version, duckdb_version,
  grammar_commits)` → same `packHash`. Verified by the determinism
  suite at `packages/pack/src/pack-determinism.test.ts`.
- **No banned literals** in tracked source —
  `bash scripts/check-banned-strings.sh` exits 0 post-commit.
- **`mise run check`** exits 0 after every commit.
- **Naming + license** — every new package carries `@opencodehub/<name>`
  naming, Apache-2.0 license, `type: module`, `tsc --noEmit` clean.
- **No LLM calls** outside `@opencodehub/summarizer`.
- **Deterministic output** — every MCP tool and CLI output is
  alpha-sorted with a lex-stable tiebreak.

## Behavior

### Pack invocation

- `codehub code-pack <repo> --budget <N>` produces a directory
  containing all 9 BOM items plus `manifest.json` at
  `<repo>/.codehub/packs/<pack_hash>/`.
- The `pack_codebase` MCP tool routes through `@opencodehub/pack`. The
  legacy `repomix` path remains available under an `--engine repomix`
  opt-in flag for one milestone before removal.
- Two invocations of `codehub code-pack` with the same
  `(commit, tokenizer, budget)` produce byte-identical output (`cmp -s`
  on every file under the output directory).
- `manifest.json` carries
  `{commit, repo_origin_url, tokenizer_id, determinism_class,
  budget_tokens, grammar_commits, chonkie_version, duckdb_version,
  files[], pack_hash}` with
  `pack_hash = sha256(canonicalJson(all-other-fields))`.
- PageRank is computed at request time from the loaded
  `KnowledgeGraph` via `@opencodehub/analysis` — never at index time.

### Degraded modes

- When `@chonkiejs/core` fails to install or load (native binding
  unavailable on a CI platform), pack degrades to a line-split
  fallback and stamps `determinism_class: degraded` in the manifest —
  it does NOT silently emit byte-different output claiming strict
  determinism.
- When `tokenizer_id` names a Claude model, the manifest sets
  `determinism_class: best_effort`. The BOM verifier warns when asked
  to check byte-identity against such a pack.
- When the target repo has no embeddings computed, BOM item #7 (the
  Parquet sidecar) is absent entirely (not an empty file) and
  `manifest.files[]` does NOT list a path to it.

### Forbidden

- No LLM calls in `@opencodehub/pack` (enforced by
  `scripts/check-banned-strings.sh`-style audit + a
  `no-bedrock-outside-summarizer` test).
- No writer metadata (DuckDB `created_by`, chonkie writer tags) as
  top-level fields in `manifest.json` — all tool-version pins live in
  a single nested `pins: {}` object so the BOM schema is stable across
  tool upgrades.
- No tolerance-based PageRank convergence — fixed iterations only.
- CRLF files on Windows checkouts MUST NOT produce a different
  `pack_hash` than LF on Linux — ingest normalizes to LF before
  hashing content.

## packHash construction algorithm

The exact preimage shape that produces `packHash`:

1. Compute `fileHash = sha256_hex(raw_bytes)` for every emitted BOM
   file (items 2-9 from the contract above). CRLF files are
   normalized to LF **at ingest** before hashing content — the
   on-disk bytes after normalization are the bytes that get hashed.
2. Construct the manifest object with `packHash: ""` as a placeholder
   and `files[]` populated with `{kind, path, fileHash}` rows in the
   order they appear in `BomItem.kind` (the type union enumerates a
   stable order).
3. Serialize the manifest to RFC 8785-shaped canonical JSON (sorted
   keys, no whitespace, no trailing newline). All tool-version pins
   live in a single nested `pins: {}` object — the top-level
   `manifest.json` schema does not carry writer metadata.
4. `packHash = sha256_hex(canonicalJson(manifest_with_packHash_omitted))`.
5. Replace the placeholder. Write `manifest.json` with `packHash` set
   and `files[]` unchanged. The wire form serializes camelCase TS
   fields to snake_case keys (`pack_hash`, `determinism_class`,
   `repo_origin_url`, `tokenizer_id`, `budget_tokens`, `schema_version`)
   per `packages/pack/src/manifest.ts:84-90`.

The reference implementation is `packages/pack/src/manifest.ts` (the
`buildManifest()` helper). The serializer reuses
`packages/core-types/src/graph-hash.ts` `writeCanonicalJson` — the
same canonical-JSON pattern that `graphHash` uses.

## Determinism class triage

The manifest's `determinism_class` (snake_case on disk, `determinismClass`
in TS) takes one of three values:

| Class | Trigger | Implication |
|-------|---------|-------------|
| `strict` | None of the degraded triggers fire | The byte-identity invariant holds in full: same `(commit, tokenizer, budget, chonkie_version, duckdb_version, grammar_commits)` → same `pack_hash`. |
| `best_effort` | `tokenizer_id` resolves to a Claude model | The verifier MUST warn callers checking byte-identity. |
| `degraded` | `@chonkiejs/core` native binding fails to load | Line-split fallback used; pack still self-consistent locally but not portable. |

## Determinism suite location

The byte-identity test suite lives at
`packages/pack/src/pack-determinism.test.ts`. It runs `generatePack`
twice against a fixture repo, computes `cmp -s` over every output
file, and asserts manifest `pack_hash` equality. CI gates on this
suite.

When debugging a `pack_hash` drift:

1. Re-run with `engine: "pack"` and capture both manifests.
2. Compare `pins` first — a chonkie or duckdb upgrade in node_modules
   is the most common cause.
3. Compare `files[i].file_hash` row-by-row — the first mismatch
   identifies which BOM emitter is non-deterministic.
4. Inspect the offending emitter under `packages/pack/src/` (one
   module per BOM item: `manifest.ts`, `skeleton.ts`, `file-tree.ts`,
   `deps.ts`, `ast-chunker.ts`, `xrefs.ts`, `embeddings-sidecar.ts`,
   `findings.ts`, `licenses.ts`, `readme.ts`).
