---
name: codehub-code-pack
description: |
  Use when the user asks for a deterministic code pack of a repo or
  group — a 9-item BOM (manifest, skeleton, file-tree, deps,
  ast-chunks, xrefs, optional embeddings sidecar, findings,
  licenses + readme) that is byte-identical given the same
  (commit, tokenizer, budget). Examples: "pack this repo for an
  LLM", "deterministic code pack", "build a reproducible context
  pack", "pack the platform group". DO NOT use for one-off repo
  packing without determinism — `pack_codebase --engine repomix`
  is the bandwidth-saving fallback for that case (no packHash, no
  9-item BOM, no reproducibility contract).
argument-hint: "[<repo-or-group>] [--budget <N>] [--tokenizer <id>]"
allowed-tools: pack_codebase, list_repos, project_profile, list_findings
model: sonnet
---

# codehub-code-pack

Surface the `pack_codebase` MCP tool to a Claude Code agent. Produces a
**deterministic, 9-item Bill of Materials (BOM)** at `<repo>/.codehub/packs/<packHash>/`
that is byte-identical given the same `(commit, tokenizer, budget,
chonkie_version, duckdb_version, grammar_commits)`. The pack is the
durable artifact agents hand to long-context LLMs, archive in S3 for
later replay, or diff between commits to prove invariants did not
change.

## Purpose

The 9-item BOM is the smallest faithful representation of a repo for
LLM consumption: a manifest pinning every input that could change
output, a PageRank-ranked skeleton (top symbols first), a file tree,
a dependency lockfile slice, AST-chunked top files, SCIP-grounded
cross-refs, an optional embeddings Parquet sidecar, salient SARIF
findings, and a `LICENSES + README` pair. Determinism is the headline
property: re-running with identical inputs MUST produce identical
output bytes (verified by `cmp -s` and the determinism suite — see
`references/determinism-contract.md`).

`packHash` is `sha256(canonicalJson(manifest_with_packHash_omitted))` —
it commits to every other field in the manifest, including the
`fileHash` of every BOM item. Two packs share a `packHash` iff every
input that the pack emitter looked at is identical.

**When to use this skill vs `pack_codebase --engine repomix`:**

- Use **this skill** when the user wants reproducibility, archival, a
  pack to feed to a CI replay job, or a pack to compare across
  commits. Default for any "pack the repo" request unless the user
  explicitly asks to skip determinism.
- Use **`pack_codebase --engine repomix`** (no skill required) when
  the user wants a one-shot bandwidth-saving dump for a single LLM
  call and explicitly does not need byte-identity. The repomix path
  remains opt-in through M6 then sunsets in M7.

## Single-repo mode

1. **Pre-check** — call `list_repos`. If the target repo is not
   indexed, instruct the user to run `codehub analyze` and stop. If
   `≥ 2` repos are indexed and no `repo` argument was supplied, the
   per-repo tool will return `AMBIGUOUS_REPO`; retry with one of the
   `structuredContent.error.choices[].repo_uri` values verbatim
   (Sourcegraph-style URI, e.g. `github.com/org/repo`, or
   `local:<hash>`).
2. **Confirm graph freshness** — call `project_profile` on the
   resolved repo. If the response carries a `_meta.codehub/staleness`
   envelope, surface it: tell the user the pack will reflect the last
   `codehub analyze` run, not HEAD.
3. **Optional findings preview** — if the user asks for findings in
   the pack, call `list_findings` to confirm SARIF rows exist.
4. **Pack** — call `pack_codebase` with `engine: "pack"` (the
   default). The tool resolves `outDir` to
   `<repoRoot>/.codehub/packs/<packHash>/` and writes the 9 items
   plus `manifest.json`.
5. **Report back** — surface the `packHash`, the `determinismClass`,
   and the absolute output directory. If `determinismClass` is
   `best_effort` or `degraded`, name the cause (Anthropic tokenizer
   rotation hazard, or chonkie native binding unavailable).

The manifest schema is fixed at `schemaVersion: 1`. Required fields:
`commit`, `repoOriginUrl`, `tokenizerId`, `determinismClass`,
`budgetTokens`, `pins` (`chonkieVersion`, `duckdbVersion`,
`grammarCommits`), `files[]`, `packHash`, `schemaVersion`.

## Group mode

1. **Pre-check** — call `list_repos` and `mcp__codehub__group_list` to
   confirm the named group exists and every member is fresh.
2. **Fan out** — for each group member, run the single-repo flow
   above. The orchestrator does this with one `pack_codebase` call
   per member; pack runs are independent and parallelizable up to the
   Claude Code subagent ceiling.
3. **Aggregate** — emit a per-member table of
   `(repoUri, packHash, determinismClass, outDir)` so the caller can
   archive or replay each member individually.

`packHash` is **per-repo, not per-group, in v1**. There is no
`groupPackHash` — a group "pack" is the union of N per-repo BOMs. A
later milestone may introduce a group-level manifest aggregating
member packHashes; until then, the v1 contract is N independent
packs.

## Determinism class

The manifest stamps one of three values; agents must report it
verbatim when surfacing the pack to the user.

| Class | Meaning | When emitted |
|-------|---------|--------------|
| `strict` | Same `(commit, tokenizer, budget, chonkieVersion, duckdbVersion, grammarCommits)` → same `packHash`. The full reproducibility contract holds. | Default path: chonkie native binding loaded, deterministic tokenizer (e.g. local HF tokenizer with pinned hash). |
| `best_effort` | The tokenizer is an Anthropic API tokenizer (Claude family) — Anthropic may rotate the tokenizer pin behind the model name. Other inputs are still strictly pinned, but a future tokenizer rotation can change the output. | When `tokenizerId` resolves to a Claude model. The BOM verifier MUST warn callers checking byte-identity. |
| `degraded` | A primitive fallback was used (e.g. line-split chunker because `@chonkiejs/core` failed to load). The pack is still self-consistent and re-runs match locally, but **does not** match a `strict` pack on a different machine. | When chonkie native binding is unavailable on CI platform. |

## 9-item BOM contract

| # | File | Source module | Determinism contract |
|---|------|---------------|----------------------|
| 1 | `manifest.json` | `manifest.ts` | RFC 8785 canonical JSON; pack-hash field omitted from preimage; CRLF normalized to LF before hashing content |
| 2 | `skeleton.jsonl` | `skeleton.ts` | PageRank score DESC, then `id` ASC tiebreak |
| 3 | `file-tree.jsonl` | `file-tree.ts` | `path` ASC |
| 4 | `deps.jsonl` | `deps.ts` | `(ecosystem, name, version, id)` lexicographic ASC |
| 5 | `ast-chunks.jsonl` | `ast-chunker.ts` | chonkie chunker; LF-normalized; degrades to line-split with `determinismClass: degraded` |
| 6 | `xrefs.jsonl` | `xrefs.ts` | community rows first (`id` ASC), then call rows (`from`, `to`, `id` ASC) |
| 7 | `embeddings.parquet` | `embeddings-sidecar.ts` | OPTIONAL — absent entirely when no embeddings exist; ZSTD; `ORDER BY (node_id, granularity, chunk_index)` |
| 8 | `findings.jsonl` | `findings.ts` | severity rank then `ruleId` ASC |
| 9 | `licenses.md` + `readme.md` | `licenses.ts` + `readme.ts` | alpha-sorted dependency list; static template with manifest-derived header |

`manifest.files[]` lists every emitted item as `{kind, path, fileHash}`
where `fileHash` is `sha256` hex of the raw bytes. Item 7 is omitted
from `files[]` when no embeddings exist; do not emit an empty Parquet
file.

## Verification recipe — proving the pack is deterministic

A caller proves byte-identity by re-running and diffing:

```bash
# 1. Pin the environment so chonkie/duckdb match.
node --version
cat packages/pack/package.json | jq '.dependencies."@chonkiejs/core", .dependencies."@duckdb/node-api"'

# 2. Run the pack twice with identical args.
codehub code-pack --budget 200000 --tokenizer cl100k_base --out /tmp/packA
codehub code-pack --budget 200000 --tokenizer cl100k_base --out /tmp/packB

# 3. Tree-diff: this MUST produce no output.
diff -r /tmp/packA /tmp/packB

# 4. Hashes match.
jq -r '.pack_hash' /tmp/packA/manifest.json
jq -r '.pack_hash' /tmp/packB/manifest.json

# 5. Tool-version pins are identical (these MUST match across runs).
jq '.pins' /tmp/packA/manifest.json
jq '.pins' /tmp/packB/manifest.json
```

If `diff -r` reports any byte-level difference, do NOT silently retry
— inspect `manifest.determinism_class`. `degraded` means chonkie was
unavailable on at least one run; `best_effort` means the Anthropic
tokenizer rotated; `strict` mismatch is a determinism bug, file it.

## next_steps

When `packHash` drifts unexpectedly between two runs you believe are
identical:

1. Compare the two `manifest.json` files field-by-field — the first
   field that differs identifies the offending input.
2. Run `mcp__codehub__project_profile` to confirm the index has not
   been re-analyzed under you (an `analyze` invalidates the previous
   pack's `commit` field).
3. If `pins` differs, the local toolchain has changed — pin
   `@chonkiejs/core` and `@duckdb/node-api` in `package.json`.
4. If only `files[i].fileHash` differs for a single BOM item, that
   item's emitter has a determinism bug; raise it in the determinism
   suite under `packages/pack/src/`.
5. For deeper review, consult `references/determinism-contract.md`
   (the spec excerpt) and the determinism test suite at
   `packages/pack/src/pack-determinism.test.ts`.
